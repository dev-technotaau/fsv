"""
Quick Start Script for YOLO Training
=====================================
Automates the entire setup and training process.

Usage:
    python quick_start_yolo.py
"""

import subprocess
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_dataset():
    """Check if dataset exists."""
    images_dir = Path("data/images")
    masks_dir = Path("data/masks")
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not masks_dir.exists():
        print(f"❌ Masks directory not found: {masks_dir}")
        return False
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    mask_files = list(masks_dir.glob("*.png"))
    
    # Filter out JSON files from image count
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"✅ Found {len(image_files)} images")
    print(f"✅ Found {len(mask_files)} masks")
    
    if len(image_files) == 0:
        print("❌ No images found!")
        return False
    
    if len(mask_files) == 0:
        print("❌ No masks found!")
        return False
    
    return True

def main():
    """Main quick start function."""
    
    print_header("YOLO TRAINING - QUICK START")
    
    print("This script will:")
    print("  1. Check Python version")
    print("  2. Install required packages")
    print("  3. Verify dataset")
    print("  4. Check GPU availability")
    print("  5. Start training")
    print()
    
    # Step 1: Check Python version
    print_header("STEP 1: CHECK PYTHON VERSION")
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required!")
        return
    
    print("✅ Python version OK")
    
    # Step 2: Install packages
    print_header("STEP 2: INSTALL PACKAGES")
    
    packages_to_check = [
        ("ultralytics", "ultralytics>=8.0.0"),
        ("torch", "torch>=2.0.0"),
        ("cv2", "opencv-python>=4.8.0"),
        ("albumentations", "albumentations>=1.3.0"),
        ("tqdm", "tqdm>=4.65.0"),
        ("yaml", "pyyaml>=6.0"),
    ]
    
    packages_to_install = []
    
    for module_name, package_spec in packages_to_check:
        try:
            __import__(module_name)
            print(f"✅ {module_name} already installed")
        except ImportError:
            print(f"⚠️  {module_name} not found - will install")
            packages_to_install.append(package_spec)
    
    if packages_to_install:
        print(f"\n📦 Installing {len(packages_to_install)} packages...")
        install_cmd = f"{sys.executable} -m pip install {' '.join(packages_to_install)}"
        if not run_command(install_cmd, "Package installation"):
            print("❌ Installation failed! Try manually:")
            print(f"   pip install {' '.join(packages_to_install)}")
            return
    else:
        print("\n✅ All packages already installed!")
    
    # Step 3: Verify dataset
    print_header("STEP 3: VERIFY DATASET")
    if not check_dataset():
        print("\n❌ Dataset check failed!")
        print("\nMake sure you have:")
        print("  - data/images/ folder with .jpg or .png images")
        print("  - data/masks/ folder with .png mask files")
        return
    
    print("✅ Dataset verified!")
    
    # Step 4: Check GPU
    print_header("STEP 4: CHECK GPU")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"✅ CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  No GPU found - will use CPU (very slow!)")
            response = input("\nContinue with CPU training? (y/n): ")
            if response.lower() != 'y':
                print("❌ Training cancelled")
                return
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
    
    # Step 5: Start training
    print_header("STEP 5: START TRAINING")
    
    print("📊 Training Configuration:")
    print("  - Model: YOLOv8x-seg (best accuracy)")
    print("  - Epochs: 300")
    print("  - Batch size: Auto-adjusted")
    print("  - Image size: 640x640")
    print("  - Expected time: 15-18 hours (RTX 3060)")
    print("  - Expected accuracy: 92-95% mAP@50")
    print()
    
    response = input("Start training now? (y/n): ")
    if response.lower() != 'y':
        print("❌ Training cancelled")
        print("\nTo start manually, run:")
        print("   python train_YOLO.py")
        return
    
    print("\n🚀 Starting training...")
    print("=" * 80)
    
    # Run training script
    try:
        subprocess.run([sys.executable, "train_YOLO.py"], check=True)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED!")
        print("=" * 80)
        print("\n📁 Results saved in: checkpoints/yolo/")
        print("📊 Check: checkpoints/yolo/<run_name>/results.png")
        print("🎯 Best model: checkpoints/yolo/<run_name>/weights/best.pt")
        print("📦 ONNX export: checkpoints/yolo/<run_name>/weights/best.onnx")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Training can be resumed automatically on next run!")
    except subprocess.CalledProcessError as e:
        print("\n\n❌ Training failed!")
        print(f"Error: {e}")
        print("\n🔍 Check log file: logs/yolo/training_*.log")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
