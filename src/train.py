"""
Training Script for SAR Sea Ice Segmentation.

Features:
- Auto-preprocessing (NetCDF → Scene NPY on first run)
- TensorBoard logging
- RTX 5080 optimizations (TF32, BF16, cudnn.benchmark)
- Class-weighted loss for imbalance
- Checkpointing & early stopping
"""

import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Enable optimizations before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np

from dataset import get_dataloaders, get_fullscene_dataloaders, NUM_CLASSES, CLASS_NAMES
from model import get_model, count_parameters


# ============================================================================
# Focal + Dice Loss for Class Imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p) = -α * (1-p)^γ * log(p)

    - γ (gamma): focusing parameter, down-weights easy examples (default: 2.0)
    - α (alpha): class weights for balancing
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights [C]
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        B, C, H, W = inputs.shape

        # Compute softmax probabilities
        p = torch.softmax(inputs, dim=1)  # [B, C, H, W]

        # Reshape for gathering
        p = p.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        targets_flat = targets.reshape(-1)  # [B*H*W]

        # Create mask for valid pixels
        valid_mask = targets_flat != self.ignore_index

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Filter valid pixels
        p_valid = p[valid_mask]  # [N_valid, C]
        targets_valid = targets_flat[valid_mask]  # [N_valid]

        # Get probability of true class
        p_t = p_valid.gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # [N_valid]

        # Compute focal weight: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy: -log(p_t)
        ce = -torch.log(p_t.clamp(min=1e-8))

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets_valid]
            focal_loss = alpha_t * focal_weight * ce
        else:
            focal_loss = focal_weight * ce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Dice = 2 * intersection / (sum of areas)
    Loss = 1 - Dice

    Handles class imbalance by computing per-class dice and averaging.
    """

    def __init__(self, num_classes=6, ignore_index=-100, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        # Softmax to get probabilities
        probs = torch.softmax(inputs, dim=1)  # [B, C, H, W]

        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index).unsqueeze(1)  # [B, 1, H, W]

        # One-hot encode targets
        targets_clamped = targets.clone()
        targets_clamped[targets == self.ignore_index] = 0  # Temporary, will be masked
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets_clamped.unsqueeze(1), 1)  # [B, C, H, W]

        # Apply valid mask
        probs = probs * valid_mask
        targets_onehot = targets_onehot * valid_mask

        # Compute per-class dice
        dims = (0, 2, 3)  # Sum over batch and spatial dims
        intersection = (probs * targets_onehot).sum(dim=dims)
        cardinality = (probs + targets_onehot).sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average over classes that are present
        present_classes = (targets_onehot.sum(dim=dims) > 0).float()
        if present_classes.sum() > 0:
            dice_loss = 1 - (dice_per_class * present_classes).sum() / present_classes.sum()
        else:
            dice_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)

        return dice_loss


class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice Loss.

    Loss = focal_weight * FocalLoss + dice_weight * DiceLoss

    - Focal: handles class imbalance by down-weighting easy pixels
    - Dice: directly optimizes overlap, good for boundaries
    """

    def __init__(self, alpha=None, gamma=2.0, num_classes=6, ignore_index=-100,
                 focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


class CEDiceLoss(nn.Module):
    """
    Combined CrossEntropy + Dice Loss (simpler baseline, no Focal).

    Loss = ce_weight * CE + dice_weight * Dice

    This is more stable than Focal+Dice for moderately imbalanced data.
    CE provides pixel-wise supervision while Dice handles region overlap.
    """

    def __init__(self, weight=None, num_classes=6, ignore_index=-100,
                 ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAR Sea Ice Segmentation")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/ai4arctic_hugging face",
                        help="Path to data directory")

    # Model
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Encoder backbone (resnet34, resnet50, efficientnet-b0)")

    # Training
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (4 recommended)")
    parser.add_argument("--crop_size", type=int, default=512,
                        help="Crop size for training (512 recommended for ~5000x5200 scenes)")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Sanity check
    parser.add_argument("--overfit_batches", type=int, default=0,
                        help="Overfit test: train on N batches only (0=disabled). "
                             "Should drive loss→0, IoU→1. Use 1-4 for sanity check.")
    parser.add_argument("--force_preprocess", action="store_true",
                        help="Force re-preprocessing even if memmap data exists")
    parser.add_argument("--val_batches", type=int, default=0,
                        help="Limit validation to N batches (0=all). Useful for sanity runs.")

    # Winner's recipe (MMSeaIce paper)
    parser.add_argument("--downsample", type=int, default=0,
                        help="Downsample input to this size (0=disabled, 128=recommended). "
                             "Increases effective receptive field for ice floe structure.")
    parser.add_argument("--use_db_norm", action="store_true",
                        help="Convert SAR to dB, clip to [-30,0], normalize to [0,1]. "
                             "Reduces outlier impact and stabilizes gradients.")
    parser.add_argument("--use_month", action="store_true",
                        help="Add month as 4th input channel for seasonal ice behavior.")

    # Full-scene mode (correct Winner's Recipe: 400km x 400km context)
    parser.add_argument("--full_scene", action="store_true",
                        help="Use full-scene mode with 10x downsampling. "
                             "Processes entire scene (~500x520) instead of crops. "
                             "Gives model 400km x 400km context (coastlines, ocean boundaries).")
    parser.add_argument("--scene_downsample", type=int, default=10,
                        help="Downsampling factor for full-scene mode (default: 10 = 80m->800m)")

    # Class imbalance handling
    parser.add_argument("--enhanced_weights", action="store_true",
                        help="Use enhanced class weights based on inverse frequency. "
                             "Strongly upweights rare classes (NewIce: 25x, YoungIce: 18x, ThinFY: 20x).")
    parser.add_argument("--weight_preset", type=str, default="balanced",
                        choices=["weak", "balanced", "aggressive", "extreme"],
                        help="Class weight preset: "
                             "weak=current (max 2.7x), "
                             "balanced=moderate (max 20x), "
                             "aggressive=strong (max 35x), "
                             "extreme=full inverse freq (max 50x)")

    return parser.parse_args()


def enable_gpu_optimizations():
    """Enable RTX 5080 / modern GPU optimizations."""
    if not torch.cuda.is_available():
        return False

    # TF32 for faster matmul (RTX 30xx+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN autotuning
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # High precision matmul
    torch.set_float32_matmul_precision('high')

    print("\nGPU Optimizations enabled:")
    print("  - TF32 matmul: ON")
    print("  - cuDNN benchmark: ON")
    print("  - BF16 support:", "YES" if torch.cuda.is_bf16_supported() else "NO")

    return True


def get_class_weights(data_dir: str, enhanced: bool = False, preset: str = "balanced"):
    """
    Get class weights for loss function.

    Args:
        data_dir: Path to data directory (for loading frequencies)
        enhanced: If True, use enhanced weights based on inverse frequency
        preset: Weight preset - "weak", "balanced", "aggressive", "extreme"

    Weight Presets (based on DATA_PROFILE.md analysis):
        - weak: Original preprocessing weights (max 2.74x) - too weak
        - balanced: Moderate upweighting (max ~20x) - recommended start
        - aggressive: Strong upweighting (max ~35x) - for stubborn rare classes
        - extreme: Full inverse frequency (max ~50x) - may cause instability

    Class frequencies from dataset:
        OpenWater: 61.79%, NewIce: 1.82%, YoungIce: 2.86%,
        ThinFirstYearIce: 2.52%, ThickFirstYearIce: 17.06%, OldIce: 13.95%
    """
    # Pre-defined weight presets based on DATA_PROFILE.md analysis
    # Format: [OpenWater, NewIce, YoungIce, ThinFirstYearIce, ThickFirstYearIce, OldIce]
    WEIGHT_PRESETS = {
        # Original weak weights (max 2.74x) - insufficient for 33.9x imbalance
        "weak": [1.0, 2.74, 1.75, 1.98, 1.0, 1.0],

        # Balanced weights (max ~20x) - good starting point
        # Rationale: NewIce is 33.9x underrepresented, but 20x is more stable
        "balanced": [1.0, 20.0, 15.0, 15.0, 2.5, 3.0],

        # Aggressive weights (max ~35x) - for when balanced isn't enough
        # Close to true inverse frequency for rare classes
        "aggressive": [1.0, 30.0, 20.0, 22.0, 3.5, 4.0],

        # Extreme weights (max ~50x) - full inverse frequency
        # Risk: May cause training instability, use with lower LR
        "extreme": [1.0, 40.0, 25.0, 28.0, 4.0, 5.0],
    }

    if enhanced:
        if preset not in WEIGHT_PRESETS:
            print(f"⚠️ Unknown preset '{preset}', using 'balanced'")
            preset = "balanced"

        weights = torch.tensor(WEIGHT_PRESETS[preset], dtype=torch.float32)
        print(f"\n{'='*50}")
        print(f"📊 ENHANCED CLASS WEIGHTS (preset: {preset})")
        print(f"{'='*50}")
        print(f"{'Class':<20} {'Frequency':>10} {'Weight':>10} {'Boost':>10}")
        print(f"{'-'*50}")

        # Load frequencies for display
        stats_path = Path(data_dir) / 'npy_memmap' / 'normalization_stats.json'
        freqs = [61.79, 1.82, 2.86, 2.52, 17.06, 13.95]  # defaults
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
                if 'class_frequencies' in stats:
                    freqs = [f * 100 for f in stats['class_frequencies']]

        for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
            boost = "baseline" if w == 1.0 else f"{w:.1f}x"
            print(f"{name:<20} {freqs[i]:>9.2f}% {w:>10.1f} {boost:>10}")

        print(f"{'='*50}\n")
        return weights

    # Original behavior: load from preprocessing
    stats_path = Path(data_dir) / 'npy_memmap' / 'normalization_stats.json'

    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

        if 'class_weights' in stats:
            weights = torch.tensor(stats['class_weights'], dtype=torch.float32)
            print(f"Loaded class weights from preprocessing (weak - consider --enhanced_weights):")
            for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
                freq = stats['class_frequencies'][i] * 100
                print(f"  {name}: {freq:.2f}% -> weight={w:.3f}")
            return weights

    # Fallback to default weights
    print("Using default class weights (preprocessing stats not found)")
    return torch.tensor([0.2, 1.5, 1.5, 1.5, 1.5, 1.5])


def compute_iou_from_counts(intersections, unions):
    """Compute per-class and mean IoU from accumulated counts."""
    ious = []
    for inter, union in zip(intersections, unions):
        if union > 0:
            ious.append((inter / union).item())
        else:
            ious.append(float('nan'))
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    return ious, mean_iou


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, writer, use_amp):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    epoch_start = time.perf_counter()
    log_start = time.perf_counter()
    log_samples = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast('cuda', dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        log_samples += images.size(0)

        # Log every 50 batches (using monotonic timer)
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - log_start
            samples_per_sec = log_samples / elapsed if elapsed > 0 else 0
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | {samples_per_sec:.1f} samples/sec")
            log_start = time.perf_counter()
            log_samples = 0

    avg_loss = total_loss / num_batches
    epoch_time = time.perf_counter() - epoch_start

    writer.add_scalar("Loss/train", avg_loss, epoch)

    return avg_loss, epoch_time


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, use_amp, num_classes=6, max_batches=0):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    intersections = torch.zeros(num_classes, dtype=torch.long)
    unions = torch.zeros(num_classes, dtype=torch.long)
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.long)
    pred_pixel_counts = torch.zeros(num_classes, dtype=torch.long)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        if use_amp:
            with autocast('cuda', dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        num_batches += 1

        preds = outputs.argmax(dim=1)

        # Count pixels per class (for diagnostics + IoU)
        for c in range(num_classes):
            label_c = (labels == c)
            pred_c = (preds == c)
            class_pixel_counts[c] += label_c.sum().item()
            pred_pixel_counts[c] += pred_c.sum().item()
            intersections[c] += (label_c & pred_c).sum().item()
            unions[c] += (label_c | pred_c).sum().item()

        if max_batches > 0 and num_batches >= max_batches:
            break

    avg_loss = total_loss / num_batches

    class_ious, mean_iou = compute_iou_from_counts(intersections, unions)

    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Metrics/mIoU", mean_iou, epoch)

    for i, name in enumerate(CLASS_NAMES):
        if not np.isnan(class_ious[i]):
            writer.add_scalar(f"IoU/{name}", class_ious[i], epoch)

    # Log class distribution (ground truth vs predicted)
    total_gt_pixels = class_pixel_counts.sum().item()
    total_pred_pixels = pred_pixel_counts.sum().item()

    # Log to TensorBoard
    for i, name in enumerate(CLASS_NAMES):
        gt_pct = class_pixel_counts[i].item() / total_gt_pixels * 100 if total_gt_pixels > 0 else 0
        pred_pct = pred_pixel_counts[i].item() / total_pred_pixels * 100 if total_pred_pixels > 0 else 0
        writer.add_scalar(f"ClassDist_GT/{name}", gt_pct, epoch)
        writer.add_scalar(f"ClassDist_Pred/{name}", pred_pct, epoch)

    # Print detailed distribution every epoch (helps diagnose class collapse)
    print(f"\n  Class distribution (GT vs Pred):")
    print(f"    {'Class':<15} {'GT %':>8} {'Pred %':>8} {'Ratio':>8}")
    print(f"    {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    for i, name in enumerate(CLASS_NAMES):
        gt_pct = class_pixel_counts[i].item() / total_gt_pixels * 100 if total_gt_pixels > 0 else 0
        pred_pct = pred_pixel_counts[i].item() / total_pred_pixels * 100 if total_pred_pixels > 0 else 0
        ratio = pred_pct / gt_pct if gt_pct > 0 else 0
        status = "⚠️" if pred_pct < 0.1 and gt_pct > 0.5 else ""  # Warn if class collapsed
        print(f"    {name:<15} {gt_pct:>7.2f}% {pred_pct:>7.2f}% {ratio:>7.2f}x {status}")

    return avg_loss, mean_iou, class_ious


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }, path)
    print(f"  Saved checkpoint: {path}")


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
        enable_gpu_optimizations()
        use_amp = True
        # BF16 doesn't need scaler, FP16 does
        scaler = None if torch.cuda.is_bf16_supported() else GradScaler('cuda')
    else:
        use_amp = False
        scaler = None

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {run_dir}")

    # TensorBoard
    writer = SummaryWriter(log_dir=run_dir / "tensorboard")

    # Data (auto-preprocesses if needed)
    print(f"\nLoading data from: {args.data_dir}")

    if args.full_scene:
        # Full-scene mode: 10x downsampling, 400km x 400km context
        train_loader, val_loader = get_fullscene_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            downsample_factor=args.scene_downsample,
            use_month_encoding=args.use_month,
            seed=args.seed,
        )
    else:
        # Crop mode: 512x512 crops at native resolution
        train_loader, val_loader = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            crop_size=args.crop_size,
            seed=args.seed,
            force_preprocess=args.force_preprocess,
            # Winner's recipe options
            downsample_size=args.downsample,
            use_db_normalization=args.use_db_norm,
            use_month_encoding=args.use_month,
        )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Determine input channels (3 base + 1 if month encoding)
    in_channels = 4 if args.use_month else 3

    # Overfit test mode: use only N batches for both train and val
    if args.overfit_batches > 0:
        print(f"\n{'='*60}")
        print(f"🧪 OVERFIT TEST MODE: {args.overfit_batches} batch(es)")
        print(f"   Expected: loss → 0, IoU → 1")
        print(f"   If this fails, debug label mapping / augmentation / model")
        print(f"{'='*60}")

        # Get batches with class diversity (not just OpenWater)
        overfit_batches = []
        classes_seen = set()
        max_search = min(100, len(train_loader))  # Search up to 100 batches

        for i, batch in enumerate(train_loader):
            labels = batch['label']
            # Check what classes are in this batch (excluding ignore=-100)
            batch_classes = set(labels[labels >= 0].unique().tolist())

            # Prefer batches with multiple classes or rare classes (1-5)
            has_rare = any(c in batch_classes for c in [1, 2, 3, 4, 5])
            has_diversity = len(batch_classes) >= 2

            if has_rare or has_diversity or len(overfit_batches) < args.overfit_batches:
                overfit_batches.append(batch)
                classes_seen.update(batch_classes)
                print(f"   Batch {len(overfit_batches)}: classes {sorted(batch_classes)}")

            if len(overfit_batches) >= args.overfit_batches:
                break

            if i >= max_search:
                print(f"   ⚠️ Searched {max_search} batches, using what we found")
                break

        print(f"   Total classes in overfit set: {sorted(classes_seen)}")

        # Create simple list-based loader for overfit batches
        class OverfitLoader:
            def __init__(self, batches):
                self.batches = batches
            def __iter__(self):
                return iter(self.batches)
            def __len__(self):
                return len(self.batches)

        train_loader = OverfitLoader(overfit_batches)
        val_loader = OverfitLoader(overfit_batches)  # Validate on same data

        print(f"   Using {len(overfit_batches)} batch(es) for train AND val")

    # Model
    print(f"\nCreating model: U-Net with {args.encoder} encoder")
    model = get_model(
        encoder_name=args.encoder,
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
    )
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Input channels: {in_channels}")

    # Loss: CE + Dice (simpler and more stable than Focal+Dice)
    class_weights = get_class_weights(
        args.data_dir,
        enhanced=args.enhanced_weights,
        preset=args.weight_preset
    ).to(device)
    criterion = CEDiceLoss(
        weight=class_weights,
        num_classes=NUM_CLASSES,
        ignore_index=-100,
        ce_weight=0.5,
        dice_weight=0.5,
    )
    print(f"Loss: CE + Dice (0.5 + 0.5 weighting)")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # Log hyperparams
    writer.add_hparams(
        {
            'encoder': args.encoder,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
            'crop_size': args.crop_size,
        },
        {'hparam/placeholder': 0}
    )

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer, use_amp
        )

        # Validate
        val_loss, miou, class_ious = validate(
            model, val_loader, criterion, device, epoch, writer, use_amp,
            max_batches=args.val_batches,
        )

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("LR", current_lr, epoch)

        # Print summary
        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  mIoU:       {miou:.4f}")
        print(f"  LR:         {current_lr:.2e}")
        print(f"  Time:       {train_time:.1f}s")

        # Print per-class IoU
        print(f"  Per-class IoU:")
        for i, name in enumerate(CLASS_NAMES):
            iou_str = f"{class_ious[i]:.3f}" if not np.isnan(class_ious[i]) else "N/A"
            print(f"    {name}: {iou_str}")

        # Save best model
        if miou > best_miou:
            best_miou = miou
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': val_loss, 'miou': miou},
                run_dir / "best_model.pth"
            )

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': val_loss, 'miou': miou},
                run_dir / f"checkpoint_epoch_{epoch}.pth"
            )

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs,
        {'val_loss': val_loss, 'miou': miou},
        run_dir / "final_model.pth"
    )

    writer.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Output: {run_dir}")
    print(f"\nView results:")
    print(f"  tensorboard --logdir={run_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
