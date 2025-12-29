"""
GPU Verification Script

Verifies that PyTorch can use the GPU and runs basic benchmarks.
"""

import torch
import time


def check_cuda():
    """Check CUDA availability."""
    print("=" * 50)
    print("GPU CHECK")
    print("=" * 50)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n❌ CUDA not available!")
        return False

    print(f"CUDA version:    {torch.version.cuda}")
    print(f"cuDNN version:   {torch.backends.cudnn.version()}")
    print(f"Device count:    {torch.cuda.device_count()}")
    print(f"Current device:  {torch.cuda.current_device()}")
    print(f"Device name:     {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"Compute cap:     {props.major}.{props.minor}")
    print(f"Total memory:    {props.total_memory / 1e9:.1f} GB")

    return True


def test_tensor_operations():
    """Test basic tensor operations on GPU."""
    print("\n" + "=" * 50)
    print("TENSOR OPERATIONS TEST")
    print("=" * 50)

    device = torch.device('cuda')

    # Create tensors
    print("\n1. Creating tensors on GPU...")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    print(f"   Tensor shape: {a.shape}")
    print(f"   Tensor device: {a.device}")
    print("   ✓ OK")

    # Matrix multiplication
    print("\n2. Matrix multiplication...")
    c = torch.matmul(a, b)
    print(f"   Result shape: {c.shape}")
    print("   ✓ OK")

    # Synchronize and check
    torch.cuda.synchronize()
    print("\n3. CUDA synchronize...")
    print("   ✓ OK")

    return True


def benchmark_matmul():
    """Benchmark matrix multiplication."""
    print("\n" + "=" * 50)
    print("BENCHMARK: Matrix Multiplication")
    print("=" * 50)

    device = torch.device('cuda')
    sizes = [1000, 2000, 4000]

    for size in sizes:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warmup
        torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        gflops = (2 * size**3 * 10) / elapsed / 1e9
        print(f"  {size}x{size}: {elapsed*100:.1f}ms avg, {gflops:.1f} GFLOPS")


def benchmark_conv2d():
    """Benchmark 2D convolution (common in image models)."""
    print("\n" + "=" * 50)
    print("BENCHMARK: Conv2D (Image Processing)")
    print("=" * 50)

    device = torch.device('cuda')

    # Simulate image batch: [B, C, H, W]
    batch_sizes = [1, 4, 16]

    conv = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)

    for bs in batch_sizes:
        x = torch.randn(bs, 64, 256, 256, device=device)

        # Warmup
        _ = conv(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = conv(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"  Batch {bs:2d}, 256x256: {elapsed*100:.1f}ms avg")


def test_memory():
    """Test GPU memory allocation."""
    print("\n" + "=" * 50)
    print("MEMORY TEST")
    print("=" * 50)

    device = torch.device('cuda')

    # Clear cache
    torch.cuda.empty_cache()

    allocated_before = torch.cuda.memory_allocated() / 1e9
    print(f"\n  Memory before: {allocated_before:.2f} GB")

    # Allocate large tensor
    x = torch.randn(5000, 5000, device=device)
    allocated_after = torch.cuda.memory_allocated() / 1e9
    print(f"  After 5000x5000 tensor: {allocated_after:.2f} GB")

    # Free
    del x
    torch.cuda.empty_cache()
    allocated_freed = torch.cuda.memory_allocated() / 1e9
    print(f"  After free: {allocated_freed:.2f} GB")
    print("  ✓ OK")


def main():
    print("\n" + "=" * 50)
    print("🔥 GPU VERIFICATION SCRIPT")
    print("=" * 50)

    if not check_cuda():
        return

    test_tensor_operations()
    test_memory()
    benchmark_matmul()
    benchmark_conv2d()

    print("\n" + "=" * 50)
    print("✓ ALL GPU TESTS PASSED")
    print("=" * 50)
    print("\nYour GPU is ready for training!")


if __name__ == "__main__":
    main()
