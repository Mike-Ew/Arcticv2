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

import gc
import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Try to import xarray (only needed for preprocessing)
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

INPUT_CHANNELS = [
    'nersc_sar_primary',      # SAR HH
    'nersc_sar_secondary',    # SAR HV
    'sar_incidenceangle',     # Incidence angle
]
TARGET_VAR = 'SOD'

# Generic names until we verify RTT SOD semantics
CLASS_NAMES = [
    'SOD_0',  # Likely Open Water
    'SOD_1',  # Likely New Ice
    'SOD_2',  # Likely Young Ice
    'SOD_3',  # Likely First-Year Ice
    'SOD_4',  # Likely Multi-Year Ice
    'SOD_5',  # Likely Glacial Ice
]

NUM_CLASSES = 6
IGNORE_INDEX = -100

# Preprocessing settings
DEFAULT_CROP_SIZE = 256
DEFAULT_STRIDE = 256  # No overlap for efficiency
MIN_VALID_TRAIN = 0.3  # 30% valid pixels for train
MIN_VALID_VAL = 0.5    # 50% valid pixels for val (stricter)


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
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        crop_size: int = DEFAULT_CROP_SIZE,
        augment: bool = False,
        handle_cache_size: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment

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
        # For 256x256 float16 3-channel: ~384KB image + ~64KB label = ~448KB
        image_crop = image_mmap[:, r:r+self.crop_size, c:c+self.crop_size].copy()
        label_crop = label_mmap[r:r+self.crop_size, c:c+self.crop_size].copy()

        # Convert float16 -> float32 for processing
        image_crop = image_crop.astype(np.float32)

        # Handle NaN (rare after preprocessing)
        image_crop = np.nan_to_num(image_crop, nan=0.0)

        # Convert label: 255 -> IGNORE_INDEX
        label_crop = label_crop.astype(np.int64)
        label_crop[label_crop == 255] = IGNORE_INDEX

        # Normalize
        image_crop = (image_crop - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_crop)
        label_tensor = torch.from_numpy(label_crop)

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

            # Random 90-degree rotation
            k = random.randint(0, 3)
            if k > 0:
                image_tensor = torch.rot90(image_tensor, k, dims=[1, 2])
                label_tensor = torch.rot90(label_tensor, k, dims=[0, 1])

        return {'image': image_tensor, 'label': label_tensor}


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
    label: np.ndarray,
    crop_size: int,
    stride: int,
    min_valid_fraction: float,
) -> List[Tuple[int, int]]:
    """Extract coordinates of valid crops from a label array."""
    H, W = label.shape
    coords = []

    for r in range(0, H - crop_size + 1, stride):
        for c in range(0, W - crop_size + 1, stride):
            crop = label[r:r+crop_size, c:c+crop_size]
            valid_mask = (crop != 255) & (crop < NUM_CLASSES)
            valid_frac = valid_mask.mean()

            if valid_frac >= min_valid_fraction:
                coords.append((r, c))

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

    # Scene-level split
    random.seed(seed)
    shuffled = list(all_files)
    random.shuffle(shuffled)

    n_train = int(len(shuffled) * train_ratio)
    splits = {
        'train': shuffled[:n_train],
        'val': shuffled[n_train:],
    }

    print(f"Train: {len(splits['train'])} scenes")
    print(f"Val:   {len(splits['val'])} scenes")

    # Create output directories
    for split in ['train', 'val']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Stats tracker
    stats = RunningStats(num_channels=3)

    # Coordinate data
    all_coords = {'train': [], 'val': []}
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

                # Update normalization stats (train only, use float32 for accuracy)
                if split_name == 'train':
                    stats.update(image.astype(np.float32))

                # Save as .npy (memmap-able, NOT .npz)
                image_path = output_dir / split_name / f"{scene_name}_image.npy"
                label_path = output_dir / split_name / f"{scene_name}_label.npy"

                np.save(image_path, image)
                np.save(label_path, label)

                # Extract valid crop coordinates
                coords = extract_valid_coordinates(label, crop_size, stride, min_valid)

                # Add to coordinate list with scene index
                scene_names[split_name].append(scene_name)
                for r, c in coords:
                    all_coords[split_name].append((scene_idx, r, c))

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

        np.save(coords_path, {
            'scene_names': scene_names[split_name],
            'coords': coords_array,
        }, allow_pickle=True)

        print(f"Saved {coords_path.name}: {len(coords_array):,} coordinates")

    # Save normalization stats
    norm_stats = stats.get_stats()
    with open(output_dir / 'normalization_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\nNormalization stats (per-channel):")
    for i, name in enumerate(norm_stats['channel_names']):
        print(f"  {name}: mean={norm_stats['mean'][i]:.4f}, std={norm_stats['std'][i]:.4f}")

    # Save metadata
    metadata = {
        'version': 'v5_memmap_optimized',
        'crop_size': crop_size,
        'stride': stride,
        'min_valid_train': MIN_VALID_TRAIN,
        'min_valid_val': MIN_VALID_VAL,
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
    train_ratio: float = 0.85,
    seed: int = 42,
    prefetch_factor: int = 2,
    use_scene_locality: bool = True,
    crops_per_scene: int = 8,
):
    """
    Get train and validation dataloaders.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: DataLoader workers
        crop_size: Crop size (must match preprocessing)
        train_ratio: Train/val split ratio
        seed: Random seed
        prefetch_factor: Batches to prefetch per worker
        use_scene_locality: Use scene locality sampler for better I/O
        crops_per_scene: Crops to draw from same scene before switching
    """
    data_dir = Path(data_dir)
    memmap_dir = data_dir / "npy_memmap"

    # Check if preprocessing needed
    needs_preprocessing = not (
        memmap_dir.exists() and
        (memmap_dir / "metadata.json").exists() and
        ((memmap_dir / "train_coords.npy").exists() or (memmap_dir / "train_coords.json").exists())
    )

    if needs_preprocessing:
        print("\n" + "=" * 60)
        print("Memmap data not found. Running one-time preprocessing...")
        print("=" * 60 + "\n")

        if not (data_dir / "train").exists():
            raise ValueError(f"Cannot find train/ directory in {data_dir}")

        preprocess_memmap(
            data_dir, memmap_dir,
            crop_size=crop_size,
            stride=crop_size,  # No overlap
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
        print(f"  Train: {meta['splits']['train']['num_crops']:,} crops")
        print(f"  Val:   {meta['splits']['val']['num_crops']:,} crops")

    # Create datasets
    train_dataset = SARMemmapDataset(
        memmap_dir,
        split='train',
        crop_size=crop_size,
        augment=True,
    )

    val_dataset = SARMemmapDataset(
        memmap_dir,
        split='val',
        crop_size=crop_size,
        augment=False,
    )

    # Worker init for reproducibility
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # Train sampler (with optional scene locality)
    if use_scene_locality:
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
