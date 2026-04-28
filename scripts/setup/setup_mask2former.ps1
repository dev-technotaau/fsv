# Mask2Former + SegFormer-B5 Environment Setup Script
# PowerShell script to set up the training environment
# Run with: powershell -ExecutionPolicy Bypass -File .\setup_mask2former.ps1

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Mask2Former + SegFormer-B5 Environment Setup" -ForegroundColor Cyan
Write-Host "  Production-Grade Fence Segmentation Training" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Function to check command existence
function Test-Command {
    param($Command)
    try {
        if (Get-Command $Command -ErrorAction Stop) {
            return $true
        }
    }
    catch {
        return $false
    }
}

# Check Python
Write-Host "[1/8] Checking Python installation..." -ForegroundColor Yellow
if (Test-Command python) {
    $pythonVersion = python --version
    Write-Host "  ✓ $pythonVersion found" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python not found! Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Check CUDA
Write-Host "`n[2/8] Checking CUDA availability..." -ForegroundColor Yellow
if (Test-Command nvidia-smi) {
    Write-Host "  ✓ NVIDIA GPU detected" -ForegroundColor Green
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} else {
    Write-Host "  ⚠ CUDA not detected. Training will use CPU (much slower)." -ForegroundColor Yellow
}

# Check Conda environment
Write-Host "`n[3/8] Checking conda environment..." -ForegroundColor Yellow
if (Test-Command conda) {
    $condaEnv = conda env list | Select-String -Pattern "\*" | ForEach-Object { $_.ToString().Split()[0] }
    Write-Host "  ✓ Active conda environment: $condaEnv" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Conda not found. Using system Python." -ForegroundColor Yellow
}

# Install PyTorch with CUDA
Write-Host "`n[4/8] Installing PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray
try {
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    Write-Host "  ✓ PyTorch installed successfully" -ForegroundColor Green
} catch {
    Write-Host "  ✗ PyTorch installation failed: $_" -ForegroundColor Red
    Write-Host "  Trying pip upgrade..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
}

# Install Transformers and dependencies
Write-Host "`n[5/8] Installing Transformers library..." -ForegroundColor Yellow
try {
    python -m pip install transformers>=4.30.0 timm>=0.9.0
    Write-Host "  ✓ Transformers installed successfully" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Transformers installation failed: $_" -ForegroundColor Red
}

# Install requirements
Write-Host "`n[6/8] Installing requirements from requirements/mask2former.txt..." -ForegroundColor Yellow
if (Test-Path "requirements/mask2former.txt") {
    try {
        python -m pip install -r requirements/mask2former.txt
        Write-Host "  ✓ All requirements installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Some requirements failed to install: $_" -ForegroundColor Red
        Write-Host "  Continuing anyway..." -ForegroundColor Yellow
    }
} else {
    Write-Host "  ✗ requirements/mask2former.txt not found!" -ForegroundColor Red
    exit 1
}

# Verify installations
Write-Host "`n[7/8] Verifying installations..." -ForegroundColor Yellow

# Test PyTorch
Write-Host "  Testing PyTorch..." -ForegroundColor Gray
python -c @"
import torch
print(f'    PyTorch version: {torch.__version__}')
print(f'    CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    CUDA version: {torch.version.cuda}')
    print(f'    GPU: {torch.cuda.get_device_name(0)}')
    print(f'    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"@

# Test Transformers
Write-Host "  Testing Transformers..." -ForegroundColor Gray
python -c @"
try:
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
    print(f'    ✓ Mask2Former available')
except ImportError as e:
    print(f'    ✗ Mask2Former import failed: {e}')
"@

# Test other packages
Write-Host "  Testing other packages..." -ForegroundColor Gray
python -c @"
try:
    import albumentations as A
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib
    import tqdm
    print(f'    ✓ All core packages available')
except ImportError as e:
    print(f'    ✗ Package import failed: {e}')
"@

# Check data directory
Write-Host "`n[8/8] Checking data directory..." -ForegroundColor Yellow
if (Test-Path "data/images" -and (Test-Path "data/masks")) {
    $imageCount = (Get-ChildItem "data/images" -Filter *.jpg, *.png | Measure-Object).Count
    $maskCount = (Get-ChildItem "data/masks" -Filter *.png | Measure-Object).Count
    Write-Host "  ✓ Data directory found" -ForegroundColor Green
    Write-Host "    Images: $imageCount" -ForegroundColor Gray
    Write-Host "    Masks: $maskCount" -ForegroundColor Gray
    
    if ($imageCount -eq 0 -or $maskCount -eq 0) {
        Write-Host "  ⚠ Warning: No data found! Please add images and masks." -ForegroundColor Yellow
    }
} else {
    Write-Host "  ✗ Data directory not found! Creating directories..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "data/images" | Out-Null
    New-Item -ItemType Directory -Force -Path "data/masks" | Out-Null
    Write-Host "  ✓ Directories created. Please add your data." -ForegroundColor Green
}

# Create output directories
Write-Host "`nCreating output directories..." -ForegroundColor Yellow
$directories = @(
    "checkpoints/mask2former",
    "logs/mask2former",
    "training_visualizations/mask2former"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "  ✓ Created $dir" -ForegroundColor Green
    } else {
        Write-Host "  ✓ $dir already exists" -ForegroundColor Gray
    }
}

# Final summary
Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Ensure your images are in data/images/" -ForegroundColor White
Write-Host "  2. Ensure your masks are in data/masks/" -ForegroundColor White
Write-Host "  3. Run training: python train_Mask2Former.py" -ForegroundColor White
Write-Host ""
Write-Host "Features enabled:" -ForegroundColor Yellow
Write-Host "  ✓ Mask2Former architecture with transformer-based decoder" -ForegroundColor Green
Write-Host "  ✓ SegFormer-B5 backbone (SOTA hierarchical vision transformer)" -ForegroundColor Green
Write-Host "  ✓ Mixed precision training (AMP)" -ForegroundColor Green
Write-Host "  ✓ Exponential Moving Average (EMA)" -ForegroundColor Green
Write-Host "  ✓ Advanced augmentation pipeline (20+ transforms)" -ForegroundColor Green
Write-Host "  ✓ Multi-task loss (Mask + Dice + Boundary + Lovász)" -ForegroundColor Green
Write-Host "  ✓ OneCycleLR scheduler with warmup" -ForegroundColor Green
Write-Host "  ✓ Gradient accumulation (effective batch: 16)" -ForegroundColor Green
Write-Host "  ✓ Early stopping with patience" -ForegroundColor Green
Write-Host "  ✓ TensorBoard logging with visualizations" -ForegroundColor Green
Write-Host "  ✓ GPU memory optimization (6GB laptop GPU ready)" -ForegroundColor Green
Write-Host ""
Write-Host "Advanced features:" -ForegroundColor Yellow
Write-Host "  • Query-based segmentation (100 object queries)" -ForegroundColor Cyan
Write-Host "  • Hierarchical pixel decoder" -ForegroundColor Cyan
Write-Host "  • Masked attention mechanism" -ForegroundColor Cyan
Write-Host "  • Layer-wise learning rate decay" -ForegroundColor Cyan
Write-Host "  • Stochastic depth regularization" -ForegroundColor Cyan
Write-Host "  • Label smoothing" -ForegroundColor Cyan
Write-Host "  • Comprehensive metrics (IoU, Dice, F1, Boundary F1)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Model specifications:" -ForegroundColor Yellow
Write-Host "  • Architecture: Mask2Former + SegFormer-B5" -ForegroundColor Cyan
Write-Host "  • Input size: 512x512" -ForegroundColor Cyan
Write-Host "  • Batch size: 2 (with 8x accumulation)" -ForegroundColor Cyan
Write-Host "  • Effective batch: 16" -ForegroundColor Cyan
Write-Host "  • Epochs: 150" -ForegroundColor Cyan
Write-Host "  • Learning rate: 1e-4 (backbone: 1e-5)" -ForegroundColor Cyan
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
