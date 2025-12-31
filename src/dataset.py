"""
SAR Sea Ice Dataset for SOD Classification.

Memmap + Precomputed Coordinates Approach (v2 with optimizations):
1. Store each scene as separate .npy files (memmap-able, float16)
2. Precompute valid crop coordinates (train: shuffled, val: fixed grid)
3. Cache open memmap handles (not data) per worker
4. Scene locality sampling for better page cache hits

Benefits:
- Minimal disk space (scenes stored once, ~25GB vs ~135GB for patches)
- Flexible crop size (change without re-preprocessing)
- OS page cache handles caching automatically
- Deterministic validation (fixed coordinate grid)
- Low RAM usage (workers don't load full scenes)

3-channel input: HH, HV, Incidence Angle
Target: SOD (6 classes), ignore_index=-100
"""

import csv
import gc
import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Try to import xarray (only needed for preprocessing)
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# Try to import OpenCV (faster resizing for full-scene mode)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

INPUT_CHANNELS = [
    'nersc_sar_primary',      # SAR HH
    'nersc_sar_secondary',    # SAR HV
    'sar_incidenceangle',     # Incidence angle
]
TARGET_VAR = 'SOD'

# SOD class names per dataset manual (Stage of Development)
CLASS_NAMES = [
    'OpenWater',         # 0
    'NewIce',            # 1
    'YoungIce',          # 2
    'ThinFirstYearIce',  # 3
    'ThickFirstYearIce', # 4
    'OldIce',            # 5 (older than 1 year)
]

NUM_CLASSES = 6
IGNORE_INDEX = -100

# Preprocessing settings
DEFAULT_CROP_SIZE = 512  # 512 gives better context for ~5000x5200 scenes
DEFAULT_STRIDE = 256     # 50% overlap for more training diversity
MIN_VALID_TRAIN = 0.3    # 30% valid label pixels for train
MIN_VALID_VAL = 0.5      # 50% valid label pixels for val (stricter)
MIN_SAR_VALID = 0.9      # 90% valid SAR pixels
MIN_SAR_VARIANCE = 1e-6  # Minimum variance to detect valid SAR (not constant/black)

# Winner's recipe settings (MMSeaIce paper)
DEFAULT_DOWNSAMPLE_SIZE = 128  # Downsample 512→128 for macro structure
SAR_DB_MIN = -30.0  # Clip SAR dB to this minimum
SAR_DB_MAX = 0.0    # Clip SAR dB to this maximum

# Full-scene mode settings (correct Winner's Recipe interpretation)
# 10x downsampling: 80m -> 800m resolution
# 5000x5200 scene -> 500x520 (fits in GPU memory easily)
SCENE_DOWNSAMPLE_FACTOR = 10
SCENE_OUTPUT_SIZE = 512  # Pad/resize to this for consistent batching


# ============================================================================
# Memmap Handle Cache (lightweight, per-worker)
# ============================================================================

class MemmapHandleCache:
    """
    LRU cache for open memmap handles.

    This caches the FILE HANDLES, not the data itself.
    Much lighter than caching full arrays - OS page cache handles data.
    """

    def __init__(self, max_handles: int = 16):
        self.max_handles = max_handles
        self.cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, scene_dir: Path, scene_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get memmap handles for a scene, opening if needed."""
        key = scene_name

        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]

        # Open memmap handles
        image_path = scene_dir / f"{scene_name}_image.npy"
        label_path = scene_dir / f"{scene_name}_label.npy"

        image_mmap = np.load(image_path, mmap_mode='r')
        label_mmap = np.load(label_path, mmap_mode='r')

        self.cache[key] = (image_mmap, label_mmap)

        # Evict oldest if over capacity
        while len(self.cache) > self.max_handles:
            self.cache.popitem(last=False)

        return self.cache[key]


# ============================================================================
# Memmap Dataset with Precomputed Coordinates
# ============================================================================

class SARMemmapDataset(Dataset):
    """
    Fast dataset using memory-mapped scenes and precomputed coordinates.

    Optimizations:
    - Caches open memmap handles (not data) per worker
    - Coordinates sorted by scene for locality (optional shuffling)
    - OS page cache handles actual data caching

    Winner's Recipe (MMSeaIce):
    - Downsample input for larger receptive field coverage
    - Convert SAR to dB and clip to [-30, 0] for stable gradients
    - Add month encoding for seasonal ice behavior
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        crop_size: int = DEFAULT_CROP_SIZE,
        augment: bool = False,
        handle_cache_size: int = 16,
        # Winner's recipe options
        downsample_size: int = 0,  # 0 = no downsampling, 128 = recommended
        use_db_normalization: bool = False,  # Convert SAR to dB and clip
        use_month_encoding: bool = False,  # Add month as 4th channel
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment
        self.downsample_size = downsample_size
        self.use_db_normalization = use_db_normalization
        self.use_month_encoding = use_month_encoding

        # Load normalization stats
        stats_path = self.data_dir / 'normalization_stats.json'
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            self.mean = np.array(stats['mean'], dtype=np.float32)
            self.std = np.array(stats['std'], dtype=np.float32)
        else:
            raise ValueError(f"Normalization stats not found: {stats_path}")

        # Load precomputed coordinates (numpy format for speed)
        coords_path = self.data_dir / f'{split}_coords.npy'
        if not coords_path.exists():
            # Fallback to JSON for backwards compatibility
            json_path = self.data_dir / f'{split}_coords.json'
            if json_path.exists():
                with open(json_path) as f:
                    coords_list = json.load(f)
                # Convert to structured array
                self.scene_names = list(set(c[0] for c in coords_list))
                self.scene_to_idx = {name: i for i, name in enumerate(self.scene_names)}
                self.coords = np.array(
                    [(self.scene_to_idx[c[0]], c[1], c[2]) for c in coords_list],
                    dtype=np.int32
                )
            else:
                raise ValueError(f"Coordinates not found: {coords_path}")
        else:
            # Load numpy coords
            data = np.load(coords_path, allow_pickle=True).item()
            self.scene_names = data['scene_names']
            self.scene_to_idx = {name: i for i, name in enumerate(self.scene_names)}
            self.coords = data['coords']  # [N, 3] array of (scene_idx, r, c)
            # Load classes present in each crop (for balanced sampling)
            self.crop_classes = data.get('classes', None)  # List of class lists per crop

        # Scene directory
        self.scene_dir = self.data_dir / split
        if not self.scene_dir.exists():
            raise ValueError(f"Scene directory not found: {self.scene_dir}")

        # Memmap handle cache (per worker, created fresh)
        self.handle_cache = MemmapHandleCache(max_handles=handle_cache_size)

        print(f"[{split}] {len(self.coords):,} crops, {len(self.scene_names)} scenes")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        scene_idx, r, c = self.coords[idx]
        scene_name = self.scene_names[scene_idx]

        # Get memmap handles (cached)
        image_mmap, label_mmap = self.handle_cache.get(self.scene_dir, scene_name)

        # Extract crop (only reads needed bytes via OS page cache)
        image_crop = image_mmap[:, r:r+self.crop_size, c:c+self.crop_size].copy()
        label_crop = label_mmap[r:r+self.crop_size, c:c+self.crop_size].copy()

        # Convert float16 -> float32 for processing
        image_crop = image_crop.astype(np.float32)

        # Handle NaN (rare after preprocessing)
        image_crop = np.nan_to_num(image_crop, nan=0.0)

        # Convert label: 255 -> IGNORE_INDEX
        label_crop = label_crop.astype(np.int64)
        label_crop[label_crop == 255] = IGNORE_INDEX

        # =====================================================================
        # Winner's Recipe: SAR dB normalization (before standard normalization)
        # =====================================================================
        if self.use_db_normalization:
            # SAR channels (HH, HV) are in linear scale (sigma0)
            # Convert to dB: dB = 10 * log10(sigma0)
            # Clip to [-30, 0] dB range and normalize to [0, 1]
            hh = image_crop[0]
            hv = image_crop[1]

            # Avoid log(0) by adding small epsilon
            hh_db = 10.0 * np.log10(np.maximum(hh, 1e-10))
            hv_db = 10.0 * np.log10(np.maximum(hv, 1e-10))

            # Clip to [-30, 0] dB
            hh_db = np.clip(hh_db, SAR_DB_MIN, SAR_DB_MAX)
            hv_db = np.clip(hv_db, SAR_DB_MIN, SAR_DB_MAX)

            # Normalize to [0, 1]
            hh_norm = (hh_db - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN)
            hv_norm = (hv_db - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN)

            # Incidence angle: normalize using stored stats (already reasonable range)
            inc_norm = (image_crop[2] - self.mean[2]) / (self.std[2] + 1e-8)

            image_crop = np.stack([hh_norm, hv_norm, inc_norm], axis=0)
        else:
            # Standard mean/std normalization
            image_crop = (image_crop - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)

        # =====================================================================
        # Winner's Recipe: Month encoding (seasonal ice behavior)
        # =====================================================================
        if self.use_month_encoding:
            # Extract month from scene name (format: YYYYMMDD...)
            try:
                month = int(scene_name[4:6])  # Extract MM from YYYYMMDD
                # Normalize month to [0, 1]: (month - 1) / 11
                month_value = (month - 1) / 11.0
            except (ValueError, IndexError):
                month_value = 0.5  # Default to middle of year if parsing fails

            # Create month channel (same spatial size as image)
            month_channel = np.full((1, image_crop.shape[1], image_crop.shape[2]),
                                    month_value, dtype=np.float32)
            image_crop = np.concatenate([image_crop, month_channel], axis=0)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_crop)
        label_tensor = torch.from_numpy(label_crop)

        # =====================================================================
        # Winner's Recipe: Downsample for larger effective receptive field
        # =====================================================================
        if self.downsample_size > 0 and self.downsample_size != self.crop_size:
            # Downsample image (bilinear for smooth gradients)
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.downsample_size, self.downsample_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Downsample label (nearest neighbor to preserve class indices)
            label_tensor = F.interpolate(
                label_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=(self.downsample_size, self.downsample_size),
                mode='nearest'
            ).squeeze(0).squeeze(0).long()

        # Augmentation (train only)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                label_tensor = torch.flip(label_tensor, dims=[1])

            # Random vertical flip
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1])
                label_tensor = torch.flip(label_tensor, dims=[0])

            # NOTE: rot90 disabled - incidence angle has geometric gradient across swath,
            # rotating breaks physics consistency. Flips are OK, rotations are not.

        return {'image': image_tensor, 'label': label_tensor}

    def get_sample_weights(self, class_frequencies: List[float] = None) -> np.ndarray:
        """
        Compute sample weights for class-balanced sampling.

        Crops containing rare classes get higher weights.
        Weight = max(inverse_frequency of classes in crop)

        Args:
            class_frequencies: List of class frequencies [0-1]. If None, uses uniform.

        Returns:
            weights: [N] array of sample weights
        """
        if self.crop_classes is None:
            # No class info, return uniform weights
            return np.ones(len(self.coords), dtype=np.float32)

        if class_frequencies is None:
            class_frequencies = [1.0 / NUM_CLASSES] * NUM_CLASSES

        # Compute inverse frequency weights per class
        class_weights = np.array([1.0 / max(f, 1e-6) for f in class_frequencies], dtype=np.float32)
        # Normalize so mean weight = 1
        class_weights = class_weights / class_weights.mean()

        # For each crop, weight = max weight of classes present (favor rare classes)
        sample_weights = np.zeros(len(self.coords), dtype=np.float32)
        for i, classes in enumerate(self.crop_classes):
            if classes:
                sample_weights[i] = max(class_weights[c] for c in classes)
            else:
                sample_weights[i] = 1.0

        return sample_weights


# ============================================================================
# Scene Locality Sampler (for better page cache hits)
# ============================================================================

class SceneLocalitySampler(torch.utils.data.Sampler):
    """
    Sampler that groups crops by scene for better I/O locality.

    Instead of fully random sampling, draws K consecutive crops from
    the same scene before switching. This dramatically improves page
    cache hit rate while maintaining epoch-level randomness.
    """

    def __init__(self, coords: np.ndarray, crops_per_scene: int = 8, shuffle: bool = True):
        """
        Args:
            coords: [N, 3] array with (scene_idx, r, c)
            crops_per_scene: How many crops to draw from same scene before switching
            shuffle: Whether to shuffle scenes and crops within scenes
        """
        self.coords = coords
        self.crops_per_scene = crops_per_scene
        self.shuffle = shuffle

        # Group indices by scene
        self.scene_to_indices: Dict[int, List[int]] = {}
        for idx, (scene_idx, _, _) in enumerate(coords):
            if scene_idx not in self.scene_to_indices:
                self.scene_to_indices[scene_idx] = []
            self.scene_to_indices[scene_idx].append(idx)

    def __iter__(self):
        # Build batches of indices grouped by scene
        all_indices = []

        scene_ids = list(self.scene_to_indices.keys())
        if self.shuffle:
            random.shuffle(scene_ids)

        for scene_id in scene_ids:
            indices = self.scene_to_indices[scene_id].copy()
            if self.shuffle:
                random.shuffle(indices)

            # Yield in chunks of crops_per_scene
            for i in range(0, len(indices), self.crops_per_scene):
                chunk = indices[i:i + self.crops_per_scene]
                all_indices.extend(chunk)

        return iter(all_indices)

    def __len__(self):
        return len(self.coords)


# ============================================================================
# Full-Scene Dataset (Winner's Recipe: 400km x 400km context)
# ============================================================================

class SARFullSceneDataset(Dataset):
    """
    Full-scene dataset with 10x downsampling for global context.

    Winner's Recipe (correct interpretation):
    - Instead of 512x512 crops at 80m (40km x 40km)
    - Use entire scene at 800m (400km x 400km after 10x downsample)
    - Model sees coastlines, ocean boundaries, transition zones

    Scene sizes: ~5000x5200 @ 80m -> ~500x520 @ 800m -> pad to 512x512
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        downsample_factor: int = SCENE_DOWNSAMPLE_FACTOR,
        output_size: int = SCENE_OUTPUT_SIZE,
        augment: bool = False,
        use_month_encoding: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.downsample_factor = downsample_factor
        self.output_size = output_size
        self.augment = augment
        self.use_month_encoding = use_month_encoding

        # Check OpenCV availability
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for full-scene mode: pip install opencv-python")

        # Load normalization stats
        stats_path = self.data_dir / 'normalization_stats.json'
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            self.mean = np.array(stats['mean'], dtype=np.float32)
            self.std = np.array(stats['std'], dtype=np.float32)
        else:
            raise ValueError(f"Normalization stats not found: {stats_path}")

        # Find all scenes in the split directory
        self.scene_dir = self.data_dir / split
        if not self.scene_dir.exists():
            raise ValueError(f"Scene directory not found: {self.scene_dir}")

        # Get list of scene names (from _image.npy files)
        image_files = sorted(self.scene_dir.glob("*_image.npy"))
        self.scene_names = [f.stem.replace("_image", "") for f in image_files]

        if not self.scene_names:
            raise ValueError(f"No scenes found in {self.scene_dir}")

        print(f"[{split}] {len(self.scene_names)} full scenes, {downsample_factor}x downsample -> {output_size}x{output_size}")

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        scene_name = self.scene_names[idx]

        # Load full scene
        image_path = self.scene_dir / f"{scene_name}_image.npy"
        label_path = self.scene_dir / f"{scene_name}_label.npy"

        image = np.load(image_path).astype(np.float32)  # [C, H, W]
        label = np.load(label_path)  # [H, W]

        # Handle NaN
        image = np.nan_to_num(image, nan=0.0)

        # Convert label: 255 -> IGNORE_INDEX
        label = label.astype(np.int64)
        label[label == 255] = IGNORE_INDEX

        # Downsample using OpenCV (INTER_AREA is best for reduction)
        # OpenCV expects [H, W, C] so we need to transpose
        C, H, W = image.shape
        new_h, new_w = H // self.downsample_factor, W // self.downsample_factor

        # Downsample each channel
        image_down = np.zeros((C, new_h, new_w), dtype=np.float32)
        for c in range(C):
            image_down[c] = cv2.resize(
                image[c], (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

        # Downsample label (nearest neighbor to preserve class indices)
        label_down = cv2.resize(
            label.astype(np.float32), (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int64)

        # Normalize
        image_down = (image_down - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)

        # Add month encoding if requested
        if self.use_month_encoding:
            try:
                month = int(scene_name[4:6])
                month_value = (month - 1) / 11.0
            except (ValueError, IndexError):
                month_value = 0.5

            month_channel = np.full((1, new_h, new_w), month_value, dtype=np.float32)
            image_down = np.concatenate([image_down, month_channel], axis=0)

        # Pad to output_size (center padding)
        padded_image, padded_label = self._pad_to_size(image_down, label_down)

        # Convert to tensors
        image_tensor = torch.from_numpy(padded_image)
        label_tensor = torch.from_numpy(padded_label)

        # Augmentation (train only)
        if self.augment:
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                label_tensor = torch.flip(label_tensor, dims=[1])
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1])
                label_tensor = torch.flip(label_tensor, dims=[0])

        return {
            'image': image_tensor,
            'label': label_tensor,
            'scene_name': scene_name,
            'original_size': (new_h, new_w),  # Size before padding
        }

    def _pad_to_size(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize/pad image and label to exact output_size."""
        C, H, W = image.shape

        # Use OpenCV resize to get exact output size (handles any input size)
        # This is simpler and more robust than crop+pad logic
        if H != self.output_size or W != self.output_size:
            # Resize image channels
            resized_image = np.zeros((C, self.output_size, self.output_size), dtype=np.float32)
            for c in range(C):
                resized_image[c] = cv2.resize(
                    image[c], (self.output_size, self.output_size),
                    interpolation=cv2.INTER_LINEAR
                )
            image = resized_image

            # Resize label (nearest neighbor to preserve class indices)
            label = cv2.resize(
                label.astype(np.float32), (self.output_size, self.output_size),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)

        return image, label


# ============================================================================
# Preprocessing: NetCDF -> Memmap .npy + Coordinates
# ============================================================================

class RunningStats:
    """Per-channel running mean/std (Welford's algorithm, FIXED)."""

    def __init__(self, num_channels=3):
        self.n = np.zeros(num_channels, dtype=np.int64)
        self.mean = np.zeros(num_channels, dtype=np.float64)
        self.M2 = np.zeros(num_channels, dtype=np.float64)

    def update(self, image: np.ndarray):
        """Update with image [C, H, W]. Excludes NaN values."""
        C = image.shape[0]
        for c in range(C):
            data = image[c].ravel()
            valid = data[~np.isnan(data)]
            if valid.size == 0:
                continue

            batch_n = valid.size
            batch_mean = float(valid.mean())
            batch_var = float(valid.var())

            n0 = self.n[c]
            if n0 == 0:
                self.mean[c] = batch_mean
                self.M2[c] = batch_var * batch_n
                self.n[c] = batch_n
            else:
                delta = batch_mean - self.mean[c]
                n1 = n0 + batch_n
                self.mean[c] = (n0 * self.mean[c] + batch_n * batch_mean) / n1
                self.M2[c] += batch_var * batch_n + (delta ** 2) * n0 * batch_n / n1
                self.n[c] = n1

    def get_stats(self):
        std = np.sqrt(self.M2 / np.maximum(self.n, 1))
        std = np.where(std < 1e-8, 1.0, std)
        return {
            'mean': self.mean.tolist(),
            'std': std.tolist(),
            'n_samples': self.n.tolist(),
            'channel_names': ['HH', 'HV', 'IncidenceAngle'],
        }


def extract_valid_coordinates(
    image: np.ndarray,
    label: np.ndarray,
    crop_size: int,
    stride: int,
    min_valid_fraction: float,
    min_sar_valid_fraction: float = 0.9,
    min_classes: int = 1,
    min_sar_variance: float = MIN_SAR_VARIANCE,
) -> List[Tuple[int, int, set]]:
    """
    Extract coordinates of valid crops from image and label arrays.

    Checks:
    1. Label validity (not ignore, valid class)
    2. SAR data validity (not NaN, has variance - real SAR can include zeros)
    3. Minimum class diversity (for training quality)

    Returns:
        List of (row, col, classes_present) tuples
    """
    H, W = label.shape
    coords = []

    for r in range(0, H - crop_size + 1, stride):
        for c in range(0, W - crop_size + 1, stride):
            # Check label validity
            crop_label = label[r:r+crop_size, c:c+crop_size]
            label_valid_mask = (crop_label != 255) & (crop_label < NUM_CLASSES)
            label_valid_frac = label_valid_mask.mean()

            if label_valid_frac < min_valid_fraction:
                continue

            # Check SAR data validity (HH and HV channels)
            crop_hh = image[0, r:r+crop_size, c:c+crop_size].astype(np.float32)
            crop_hv = image[1, r:r+crop_size, c:c+crop_size].astype(np.float32)

            # SAR valid = not NaN (real SAR CAN include zeros, so we use variance check)
            hh_valid = ~np.isnan(crop_hh)
            hv_valid = ~np.isnan(crop_hv)
            sar_valid_frac = (hh_valid & hv_valid).mean()

            if sar_valid_frac < min_sar_valid_fraction:
                continue

            # Variance check: reject constant/black crops (likely invalid data regions)
            # This is more robust than checking for zeros (real SAR can have zero values)
            hh_var = np.nanvar(crop_hh)
            hv_var = np.nanvar(crop_hv)
            if hh_var < min_sar_variance or hv_var < min_sar_variance:
                continue

            # Check class diversity
            valid_labels = crop_label[label_valid_mask]
            classes_present = set(np.unique(valid_labels).tolist())

            if len(classes_present) >= min_classes:
                coords.append((r, c, classes_present))

    return coords


def preprocess_memmap(
    nc_dir: str,
    output_dir: str,
    crop_size: int = DEFAULT_CROP_SIZE,
    stride: int = DEFAULT_STRIDE,
    train_ratio: float = 0.85,
    seed: int = 42,
):
    """
    Convert NetCDF files to memmap-ready .npy files + precomputed coordinates.

    Output structure:
        output_dir/
            train/
                scene_001_image.npy  (float16, memmap-able)
                scene_001_label.npy  (uint8, memmap-able)
            val/
                ...
            train_coords.npy   {scene_names: [...], coords: [N,3] array}
            val_coords.npy
            normalization_stats.json
            metadata.json
    """
    if not XARRAY_AVAILABLE:
        raise ImportError("xarray required: pip install xarray netcdf4")

    nc_dir = Path(nc_dir)
    output_dir = Path(output_dir)

    print("=" * 60)
    print("PREPROCESSING: NetCDF -> Memmap NPY + Coordinates")
    print("=" * 60)
    print(f"Input:  {nc_dir}")
    print(f"Output: {output_dir}")
    print(f"Crop: {crop_size}x{crop_size}, Stride: {stride}")
    print(f"Valid fraction: train >= {MIN_VALID_TRAIN}, val >= {MIN_VALID_VAL}")

    # Find all NetCDF files
    train_nc_dir = nc_dir / "train"
    all_files = sorted(train_nc_dir.glob("*.nc"))

    if not all_files:
        raise ValueError(f"No .nc files found in {train_nc_dir}")

    print(f"Found {len(all_files)} NetCDF scenes")

    # Split strategy: group by region_id if metadata is available, else random scene split
    split_strategy = "random"
    splits = None

    metadata_path = nc_dir / "metadata.csv"
    if metadata_path.exists():
        try:
            with open(metadata_path, newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                if 'input_path' in fieldnames and 'region_id' in fieldnames:
                    all_files_set = {p.resolve() for p in all_files}
                    region_to_files: Dict[str, List[Path]] = {}

                    for row in reader:
                        if row.get('split') and row['split'] != 'train':
                            continue
                        rel_path = row.get('input_path')
                        if not rel_path:
                            continue
                        nc_path = (nc_dir / rel_path).resolve()
                        if nc_path not in all_files_set:
                            continue
                        region_id = row.get('region_id') or "UNKNOWN"
                        region_to_files.setdefault(region_id, []).append(nc_path)

                    if region_to_files:
                        mapped_files = set()
                        for files in region_to_files.values():
                            mapped_files.update(files)

                        missing_files = [p for p in all_files_set if p not in mapped_files]
                        if missing_files:
                            region_to_files.setdefault("UNASSIGNED", []).extend(sorted(missing_files))
                            print(f"⚠️  {len(missing_files)} scenes missing region_id; grouped as UNASSIGNED")

                        region_ids = sorted(region_to_files.keys())
                        if len(region_ids) >= 2:
                            rng = random.Random(seed)
                            rng.shuffle(region_ids)
                            n_train_regions = int(len(region_ids) * train_ratio)
                            n_train_regions = max(1, min(len(region_ids) - 1, n_train_regions))
                            train_regions = set(region_ids[:n_train_regions])

                            splits = {'train': [], 'val': []}
                            for region_id in region_ids:
                                target = 'train' if region_id in train_regions else 'val'
                                splits[target].extend(sorted(region_to_files[region_id]))

                            split_strategy = "region"
                            print(f"Region split: {len(train_regions)} train regions, {len(region_ids) - len(train_regions)} val regions")
        except Exception as e:
            print(f"⚠️  Failed to read metadata.csv for region split: {e}")

    if splits is None:
        random.seed(seed)
        shuffled = list(all_files)
        random.shuffle(shuffled)

        n_train = int(len(shuffled) * train_ratio)
        splits = {
            'train': shuffled[:n_train],
            'val': shuffled[n_train:],
        }

    print(f"Split strategy: {split_strategy}")
    print(f"Train: {len(splits['train'])} scenes")
    print(f"Val:   {len(splits['val'])} scenes")

    # Create output directories
    for split in ['train', 'val']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Stats tracker
    stats = RunningStats(num_channels=3)

    # Class pixel counter (for computing class weights)
    class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_valid_pixels = 0

    # Coordinate data
    all_coords = {'train': [], 'val': []}
    all_classes = {'train': [], 'val': []}  # Classes present in each crop (for balanced sampling)
    scene_names = {'train': [], 'val': []}

    # Process each split
    for split_name, file_list in splits.items():
        print(f"\nProcessing {split_name.upper()}...")

        min_valid = MIN_VALID_TRAIN if split_name == 'train' else MIN_VALID_VAL
        scene_idx = 0

        for nc_path in tqdm(file_list, desc=split_name):
            scene_name = nc_path.stem

            try:
                # Load NetCDF
                ds = xr.open_dataset(nc_path)

                # Extract channels as float16 (half memory/disk)
                ch0 = ds[INPUT_CHANNELS[0]].values.astype(np.float16)
                ch1 = ds[INPUT_CHANNELS[1]].values.astype(np.float16)
                ch2 = ds[INPUT_CHANNELS[2]].values.astype(np.float16)
                label = ds[TARGET_VAR].values.astype(np.uint8)

                ds.close()

                # Stack channels [3, H, W]
                image = np.stack([ch0, ch1, ch2], axis=0)
                del ch0, ch1, ch2

                # Update normalization stats and class counts (train only)
                if split_name == 'train':
                    stats.update(image.astype(np.float32))
                    # Count pixels per class for weight computation
                    for c in range(NUM_CLASSES):
                        class_pixel_counts[c] += np.sum(label == c)
                    # Count valid pixels (not ignore value 255)
                    total_valid_pixels += np.sum(label != 255)

                # Save as .npy (memmap-able, NOT .npz)
                image_path = output_dir / split_name / f"{scene_name}_image.npy"
                label_path = output_dir / split_name / f"{scene_name}_label.npy"

                np.save(image_path, image)
                np.save(label_path, label)

                # Extract valid crop coordinates (checks label, SAR validity)
                # Require at least 1 valid class (same for train and val)
                # Previously used min_classes=2 for train but this was too restrictive
                min_classes = 1
                coords = extract_valid_coordinates(image, label, crop_size, stride, min_valid, min_classes=min_classes)

                # Add to coordinate list with scene index and class info
                scene_names[split_name].append(scene_name)
                for r, c, classes_present in coords:
                    all_coords[split_name].append((scene_idx, r, c))
                    all_classes[split_name].append(classes_present)

                scene_idx += 1

                # Memory cleanup
                del image, label
                gc.collect()

            except Exception as e:
                tqdm.write(f"Error processing {nc_path.name}: {e}")

        print(f"  {split_name}: {len(all_coords[split_name]):,} valid crops from {len(scene_names[split_name])} scenes")

    # Save coordinates as numpy (faster than JSON)
    for split_name in ['train', 'val']:
        coords_path = output_dir / f'{split_name}_coords.npy'
        coords_array = np.array(all_coords[split_name], dtype=np.int32)

        # Convert class sets to list of lists for saving
        classes_list = [list(c) for c in all_classes[split_name]]

        np.save(coords_path, {
            'scene_names': scene_names[split_name],
            'coords': coords_array,
            'classes': classes_list,  # Classes present in each crop
        }, allow_pickle=True)

        print(f"Saved {coords_path.name}: {len(coords_array):,} coordinates")

    # Compute class weights: only upweight rare classes, don't downweight mid-frequency
    # This is more stable than median frequency balancing which can hurt mid-frequency classes
    class_freqs = class_pixel_counts / total_valid_pixels

    # Simple upweighting: rare classes (< 5% frequency) get boosted, others stay at 1.0
    # This prevents destabilizing mid-frequency classes like ThickFirstYearIce (SOD_4)
    RARE_THRESHOLD = 0.05  # 5% frequency threshold (includes SOD_1/2/3)
    RARE_BOOST = 3.0       # Boost factor for rare classes (mild, not extreme)

    class_weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        if class_freqs[c] > 0 and class_freqs[c] < RARE_THRESHOLD:
            # Upweight rare classes, but cap to avoid instability
            class_weights[c] = min(RARE_BOOST, RARE_THRESHOLD / class_freqs[c])
        # Classes >= 2% frequency keep weight = 1.0 (no downweighting)

    # Save normalization stats and class weights
    norm_stats = stats.get_stats()
    norm_stats['class_pixel_counts'] = class_pixel_counts.tolist()
    norm_stats['class_frequencies'] = class_freqs.tolist()
    norm_stats['class_weights'] = class_weights.tolist()

    with open(output_dir / 'normalization_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\nNormalization stats (per-channel):")
    for i, name in enumerate(norm_stats['channel_names']):
        print(f"  {name}: mean={norm_stats['mean'][i]:.4f}, std={norm_stats['std'][i]:.4f}")

    print(f"\nClass distribution (training set):")
    for i, name in enumerate(CLASS_NAMES):
        pct = class_freqs[i] * 100
        print(f"  {name}: {class_pixel_counts[i]:,} pixels ({pct:.2f}%) -> weight={class_weights[i]:.3f}")

    # Save metadata
    metadata = {
        'version': 'v5_memmap_optimized',
        'crop_size': crop_size,
        'stride': stride,
        'min_valid_train': MIN_VALID_TRAIN,
        'min_valid_val': MIN_VALID_VAL,
        'split_strategy': split_strategy,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'ignore_index': IGNORE_INDEX,
        'image_dtype': 'float16',
        'label_dtype': 'uint8',
        'normalization': norm_stats,
        'splits': {
            'train': {
                'num_scenes': len(scene_names['train']),
                'num_crops': len(all_coords['train']),
            },
            'val': {
                'num_scenes': len(scene_names['val']),
                'num_crops': len(all_coords['val']),
            },
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Train: {len(all_coords['train']):,} crops from {len(scene_names['train'])} scenes")
    print(f"Val:   {len(all_coords['val']):,} crops from {len(scene_names['val'])} scenes")
    print(f"Output: {output_dir}")

    return metadata


# ============================================================================
# Main API
# ============================================================================

def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    crop_size: int = DEFAULT_CROP_SIZE,
    stride: int = DEFAULT_STRIDE,
    train_ratio: float = 0.85,
    seed: int = 42,
    prefetch_factor: int = 2,
    use_class_balanced: bool = False,  # Disabled by default - use simpler random sampling
    use_scene_locality: bool = True,   # Scene locality for better I/O
    crops_per_scene: int = 8,
    force_preprocess: bool = False,    # Force re-preprocessing even if data exists
    # Winner's recipe options (MMSeaIce paper)
    downsample_size: int = 0,          # 0 = no downsampling, 128 = recommended
    use_db_normalization: bool = False,  # Convert SAR to dB and clip
    use_month_encoding: bool = False,  # Add month as extra channel
):
    """
    Get train and validation dataloaders.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: DataLoader workers
        crop_size: Crop size for training
        stride: Stride for crop extraction (256 = 50% overlap with 512 crops)
        train_ratio: Train/val split ratio
        seed: Random seed
        prefetch_factor: Batches to prefetch per worker
        use_class_balanced: Use class-balanced sampling (oversample rare classes)
        use_scene_locality: Use scene locality sampler for better I/O
        crops_per_scene: Crops to draw from same scene before switching
        force_preprocess: Force re-preprocessing even if data exists
        downsample_size: Downsample crops to this size (0=disabled, 128=recommended)
        use_db_normalization: Convert SAR to dB, clip [-30,0], normalize to [0,1]
        use_month_encoding: Add month as 4th input channel
    """
    data_dir = Path(data_dir)
    memmap_dir = data_dir / "npy_memmap"

    # Check if preprocessing needed or params changed
    needs_preprocessing = force_preprocess or not (
        memmap_dir.exists() and
        (memmap_dir / "metadata.json").exists() and
        ((memmap_dir / "train_coords.npy").exists() or (memmap_dir / "train_coords.json").exists())
    )

    # Check if preprocessing params match (version check)
    if not needs_preprocessing:
        with open(memmap_dir / "metadata.json") as f:
            meta = json.load(f)

        # Compare key params - if mismatch, need to re-preprocess
        if meta.get('crop_size') != crop_size or meta.get('stride') != stride:
            print(f"\n⚠️  Preprocessing params mismatch!")
            print(f"   Existing: crop_size={meta.get('crop_size')}, stride={meta.get('stride')}")
            print(f"   Requested: crop_size={crop_size}, stride={stride}")
            print(f"   Re-preprocessing required. Delete {memmap_dir} or use force_preprocess=True")
            print(f"   Using existing data for now...\n")

    if needs_preprocessing:
        print("\n" + "=" * 60)
        print("Running preprocessing...")
        print(f"  crop_size={crop_size}, stride={stride}")
        print("=" * 60 + "\n")

        if not (data_dir / "train").exists():
            raise ValueError(f"Cannot find train/ directory in {data_dir}")

        preprocess_memmap(
            data_dir, memmap_dir,
            crop_size=crop_size,
            stride=stride,  # Use the passed stride, not crop_size!
            train_ratio=train_ratio,
            seed=seed,
        )

        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print("=" * 60 + "\n")
    else:
        # Load metadata to show info
        with open(memmap_dir / "metadata.json") as f:
            meta = json.load(f)
        print(f"Using memmap data: {memmap_dir}")
        print(f"  Params: crop_size={meta.get('crop_size')}, stride={meta.get('stride')}")
        print(f"  Train: {meta['splits']['train']['num_crops']:,} crops")
        print(f"  Val:   {meta['splits']['val']['num_crops']:,} crops")

    # Create datasets with winner's recipe options
    train_dataset = SARMemmapDataset(
        memmap_dir,
        split='train',
        crop_size=crop_size,
        augment=True,
        downsample_size=downsample_size,
        use_db_normalization=use_db_normalization,
        use_month_encoding=use_month_encoding,
    )

    val_dataset = SARMemmapDataset(
        memmap_dir,
        split='val',
        crop_size=crop_size,
        augment=False,
        downsample_size=downsample_size,
        use_db_normalization=use_db_normalization,
        use_month_encoding=use_month_encoding,
    )

    # Log winner's recipe settings if enabled
    if downsample_size > 0 or use_db_normalization or use_month_encoding:
        print(f"Winner's Recipe enabled:")
        if downsample_size > 0:
            print(f"  - Downsampling: {crop_size} -> {downsample_size}")
        if use_db_normalization:
            print(f"  - SAR dB normalization: clip [{SAR_DB_MIN}, {SAR_DB_MAX}] dB -> [0, 1]")
        if use_month_encoding:
            print(f"  - Month encoding: adding 4th channel")

    # Worker init for reproducibility
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # Train sampler
    # Priority: class-balanced > scene locality > random shuffle
    if use_class_balanced:
        # Load class frequencies for computing sample weights
        stats_path = memmap_dir / 'normalization_stats.json'
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            class_freqs = stats.get('class_frequencies', None)
        else:
            class_freqs = None

        # Compute sample weights (crops with rare classes get higher weights)
        sample_weights = train_dataset.get_sample_weights(class_freqs)
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(train_dataset),
            replacement=True,  # Allow resampling for balance
        )
        train_shuffle = False
        print(f"Using class-balanced sampling (weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f})")
    elif use_scene_locality:
        train_sampler = SceneLocalitySampler(
            train_dataset.coords,
            crops_per_scene=crops_per_scene,
            shuffle=True,
        )
        train_shuffle = False  # Sampler handles shuffling
    else:
        train_sampler = None
        train_shuffle = True

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic validation
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


def get_fullscene_dataloaders(
    data_dir: str,
    batch_size: int = 4,  # Smaller batch for full scenes
    num_workers: int = 4,
    downsample_factor: int = SCENE_DOWNSAMPLE_FACTOR,
    output_size: int = SCENE_OUTPUT_SIZE,
    use_month_encoding: bool = False,
    seed: int = 42,
):
    """
    Get train and validation dataloaders for full-scene mode.

    Winner's Recipe (correct interpretation):
    - 10x downsampling: 80m -> 800m resolution
    - Full scene fits in ~512x512 after downsampling
    - Model sees 400km x 400km context (coastlines, ocean, transitions)

    Args:
        data_dir: Path to data directory containing npy_memmap/
        batch_size: Batch size (use smaller values, scenes are larger)
        num_workers: DataLoader workers
        downsample_factor: Downsampling factor (10 = 80m->800m)
        output_size: Output size after padding (512 recommended)
        use_month_encoding: Add month as 4th channel
        seed: Random seed
    """
    data_dir = Path(data_dir)
    memmap_dir = data_dir / "npy_memmap"

    if not memmap_dir.exists():
        raise ValueError(f"Memmap directory not found: {memmap_dir}. Run preprocessing first.")

    # Create datasets
    train_dataset = SARFullSceneDataset(
        memmap_dir,
        split='train',
        downsample_factor=downsample_factor,
        output_size=output_size,
        augment=True,
        use_month_encoding=use_month_encoding,
    )

    val_dataset = SARFullSceneDataset(
        memmap_dir,
        split='val',
        downsample_factor=downsample_factor,
        output_size=output_size,
        augment=False,
        use_month_encoding=use_month_encoding,
    )

    # Worker init for reproducibility
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    print(f"\nFull-scene mode enabled:")
    print(f"  Resolution: 80m -> {80 * downsample_factor}m ({downsample_factor}x downsample)")
    print(f"  Physical coverage: ~{output_size * 80 * downsample_factor / 1000:.0f}km x {output_size * 80 * downsample_factor / 1000:.0f}km")
    print(f"  Output size: {output_size}x{output_size}")
    if use_month_encoding:
        print(f"  Month encoding: enabled (4 channels)")

    return train_loader, val_loader


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing memmap dataset (v2 optimized)...\n")

    train_loader, val_loader = get_dataloaders(
        data_dir="data/ai4arctic_hugging face",
        batch_size=8,
        num_workers=0,  # 0 for testing
        crop_size=256,
    )

    print(f"\nTrain: {len(train_loader)} batches")
    print(f"Val:   {len(val_loader)} batches")

    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Image: {batch['image'].shape}, dtype: {batch['image'].dtype}")
    print(f"  Label: {batch['label'].shape}, dtype: {batch['label'].dtype}")
    print(f"  Image range: [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")
    print(f"  Label unique: {torch.unique(batch['label']).tolist()}")

    print("\nDataset test passed!")
