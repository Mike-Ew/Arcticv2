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
import time
from datetime import datetime
from pathlib import Path

# Enable optimizations before importing torch
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np

from dataset import get_dataloaders, NUM_CLASSES, CLASS_NAMES
from model import get_model, count_parameters


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
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Random crop size for training")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

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


def get_class_weights():
    """
    Class weights for imbalanced data.
    Open Water (0) is dominant, so we down-weight it.
    """
    # Based on typical distribution
    weights = torch.tensor([0.2, 1.5, 1.5, 1.5, 1.5, 1.5])
    return weights


def compute_metrics(preds, labels, num_classes=6, ignore_index=-100):
    """Compute per-class and mean IoU."""
    ious = []

    preds = preds.view(-1)
    labels = labels.view(-1)

    # Mask out ignore pixels
    valid = labels != ignore_index
    preds = preds[valid]
    labels = labels[valid]

    for c in range(num_classes):
        pred_c = (preds == c)
        label_c = (labels == c)

        intersection = (pred_c & label_c).sum().float()
        union = (pred_c | label_c).sum().float()

        if union > 0:
            ious.append((intersection / union).item())
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

    start_time = time.time()
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

        # Log every 50 batches
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * images.size(0) / elapsed
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | {samples_per_sec:.1f} samples/sec")

    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time

    writer.add_scalar("Loss/train", avg_loss, epoch)

    return avg_loss, epoch_time


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, use_amp, num_classes=6):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_preds = []
    all_labels = []

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for batch in loader:
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
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / num_batches

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    class_ious, mean_iou = compute_metrics(all_preds, all_labels, num_classes)

    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Metrics/mIoU", mean_iou, epoch)

    for i, name in enumerate(CLASS_NAMES):
        if not np.isnan(class_ious[i]):
            writer.add_scalar(f"IoU/{name}", class_ious[i], epoch)

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
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        seed=args.seed,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    print(f"\nCreating model: U-Net with {args.encoder} encoder")
    model = get_model(
        encoder_name=args.encoder,
        in_channels=3,
        num_classes=NUM_CLASSES,
    )
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Loss with class weights
    class_weights = get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    print(f"Class weights: {class_weights.tolist()}")

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
            model, val_loader, criterion, device, epoch, writer, use_amp
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
