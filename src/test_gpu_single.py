"""
GPU test script for single model training
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path


def test_gpu_setup():
    """Test GPU setup and training"""
    print("="*60)
    print("GPU SETUP TEST")
    print("="*60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("Install PyTorch with CUDA:")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"✅ GPU Count: {torch.cuda.device_count()}")
    print(f"✅ Current GPU: {torch.cuda.current_device()}")
    print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # Force GPU device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    # Test GPU computation
    print("\nTesting GPU computation...")
    try:
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.matmul(x, y)

        memory_used = torch.cuda.memory_allocated(device) / 1024**3
        print(f"✅ GPU computation successful")
        print(f"✅ GPU memory used: {memory_used:.2f} GB")

        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False


def run_single_model_test():
    """Run single model training test"""
    if not test_gpu_setup():
        return

    print("\n" + "="*60)
    print("RUNNING SINGLE MODEL TEST")
    print("="*60)

    # Check if files exist
    if not Path("data/metadata.csv").exists():
        print("❌ data/metadata.csv not found!")
        print("Please make sure your data is in the correct location.")
        return

    if not Path("data/danbooru_images").exists():
        print("❌ data/danbooru_images folder not found!")
        print("Please make sure your images are in the correct location.")
        return

    if not Path("src/main_train.py").exists():
        print("❌ src/main_train.py not found!")
        print("Please make sure the training script exists.")
        return

    # Import training modules
    try:
        # Add src to Python path
        src_path = str(Path("src").absolute())
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from main_train import main

        # Set command line arguments
        original_argv = sys.argv.copy()
        sys.argv = [
            'test_gpu_single.py',
            '--model_name', 'resnet34',
            '--data_csv', 'data/metadata.csv',
            '--img_root', 'data/danbooru_images/',
            '--batch_size', '16',
            '--epochs', '2',
            '--top_k_tags', '50',
            '--use_amp',
            '--experiment_name', f'gpu_test_{torch.cuda.get_device_name(0).replace(" ", "_").replace("-", "_")}'
        ]

        print("🚀 Starting GPU training test...")
        main()

        # Restore original argv
        sys.argv = original_argv
        print("✅ GPU training test completed successfully!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

        # Restore original argv in case of error
        try:
            sys.argv = original_argv
        except:
            pass


if __name__ == "__main__":
    run_single_model_test()