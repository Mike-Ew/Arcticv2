"""
U-Net Baseline Model for SOD Classification.

Uses segmentation_models_pytorch for a battle-tested implementation.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_model(
    encoder_name: str = "resnet34",
    in_channels: int = 3,
    num_classes: int = 6,
    encoder_weights: str = None,  # None for SAR (no ImageNet pretrain)
):
    """
    Create a U-Net model for semantic segmentation.

    Args:
        encoder_name: Backbone encoder (resnet34, resnet50, efficientnet-b0, etc.)
        in_channels: Number of input channels (3 for SAR baseline)
        num_classes: Number of output classes (5 for SOD)
        encoder_weights: Pretrained weights ('imagenet' or None)

    Returns:
        model: nn.Module
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # Raw logits (we apply softmax in loss)
    )

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Available encoders for reference
AVAILABLE_ENCODERS = [
    "resnet18",      # 11M params - fast
    "resnet34",      # 21M params - good balance (default)
    "resnet50",      # 25M params - more capacity
    "efficientnet-b0",  # 5M params - efficient
    "efficientnet-b2",  # 9M params
    "mobilenet_v2",     # 3M params - lightweight
]


# Quick test
if __name__ == "__main__":
    print("Testing U-Net model...")
    print("=" * 50)

    model = get_model(
        encoder_name="resnet34",
        in_channels=3,
        num_classes=6,
    )

    print(f"Encoder: resnet34")
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.2f}, {y.max():.2f}]")

    # Test with GPU if available
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        model = model.cuda()
        x = x.cuda()
        with torch.no_grad():
            y = model(x)
        print(f"GPU forward pass: OK")

    print("\n✓ Model test passed!")
