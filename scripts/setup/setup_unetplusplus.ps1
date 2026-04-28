# ============================================================================
# UNet++ Training Environment Setup Script (PowerShell)
# ============================================================================
# Application: Fence Detection with UNet++ Architecture
# Author: VisionGuard Team
# Date: November 14, 2025
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "UNet++ Training Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/7] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "   $pythonVersion" -ForegroundColor Green

if ($LASTEXITCODE -ne 0) {
    Write-Host "   ERROR: Python not found! Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Check if Python version is 3.8+
$versionMatch = [regex]::Match($pythonVersion, "Python (\d+)\.(\d+)")
if ($versionMatch.Success) {
    $major = [int]$versionMatch.Groups[1].Value
    $minor = [int]$versionMatch.Groups[2].Value
    
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
        Write-Host "   ERROR: Python 3.8+ required. Found: Python $major.$minor" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Check CUDA availability
Write-Host "[2/7] Checking CUDA/GPU availability..." -ForegroundColor Yellow
$cudaCheck = python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "   $cudaCheck" -ForegroundColor Green
} else {
    Write-Host "   PyTorch not installed yet - will install with CUDA support" -ForegroundColor Yellow
}

Write-Host ""

# Create virtual environment (optional but recommended)
Write-Host "[3/7] Virtual Environment Setup (Optional)" -ForegroundColor Yellow
$createVenv = Read-Host "   Create a virtual environment? (y/n) [Recommended: y]"

if ($createVenv -eq "y" -or $createVenv -eq "Y") {
    $venvName = "unetplusplus_env"
    
    if (Test-Path $venvName) {
        Write-Host "   Virtual environment '$venvName' already exists" -ForegroundColor Yellow
        $recreate = Read-Host "   Recreate it? (y/n)"
        
        if ($recreate -eq "y" -or $recreate -eq "Y") {
            Remove-Item -Recurse -Force $venvName
            Write-Host "   Creating virtual environment..." -ForegroundColor Yellow
            python -m venv $venvName
        }
    } else {
        Write-Host "   Creating virtual environment..." -ForegroundColor Yellow
        python -m venv $venvName
    }
    
    Write-Host "   Activating virtual environment..." -ForegroundColor Yellow
    & ".\$venvName\Scripts\Activate.ps1"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ Virtual environment activated" -ForegroundColor Green
    } else {
        Write-Host "   WARNING: Could not activate virtual environment automatically" -ForegroundColor Yellow
        Write-Host "   Please run: .\$venvName\Scripts\Activate.ps1" -ForegroundColor Yellow
    }
} else {
    Write-Host "   Skipping virtual environment creation" -ForegroundColor Yellow
}

Write-Host ""

# Upgrade pip
Write-Host "[4/7] Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ pip upgraded successfully" -ForegroundColor Green
} else {
    Write-Host "   WARNING: Could not upgrade pip" -ForegroundColor Yellow
}

Write-Host ""

# Install PyTorch with CUDA
Write-Host "[5/7] Installing PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host "   Detecting CUDA version..." -ForegroundColor Yellow

# Check if nvidia-smi is available
$nvidiaCheck = nvidia-smi 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "   NVIDIA GPU detected" -ForegroundColor Green
    
    # Install PyTorch with CUDA 11.8 (most compatible)
    Write-Host "   Installing PyTorch with CUDA 11.8..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ PyTorch with CUDA installed successfully" -ForegroundColor Green
    } else {
        Write-Host "   ERROR: Failed to install PyTorch" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "   No NVIDIA GPU detected - installing CPU-only PyTorch" -ForegroundColor Yellow
    pip install torch torchvision torchaudio
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ PyTorch (CPU) installed successfully" -ForegroundColor Green
        Write-Host "   WARNING: Training will be VERY slow without GPU" -ForegroundColor Red
    } else {
        Write-Host "   ERROR: Failed to install PyTorch" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Install requirements
Write-Host "[6/7] Installing UNet++ dependencies..." -ForegroundColor Yellow

if (Test-Path "requirements/unetplusplus.txt") {
    pip install -r requirements/unetplusplus.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ All dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "   ERROR: Failed to install some dependencies" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "   ERROR: requirements/unetplusplus.txt not found!" -ForegroundColor Red
    Write-Host "   Please ensure the file is in the current directory" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Verify installation
Write-Host "[7/7] Verifying installation..." -ForegroundColor Yellow

$verification = python -c @"
import sys
print('Python:', sys.version.split()[0])

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
except ImportError as e:
    print(f'PyTorch: NOT INSTALLED - {e}')
    sys.exit(1)

try:
    import segmentation_models_pytorch as smp
    print(f'Segmentation Models PyTorch: {smp.__version__}')
except ImportError as e:
    print(f'SMP: NOT INSTALLED - {e}')
    sys.exit(1)

try:
    import albumentations as A
    print(f'Albumentations: {A.__version__}')
except ImportError as e:
    print(f'Albumentations: NOT INSTALLED - {e}')
    sys.exit(1)

try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'OpenCV: NOT INSTALLED - {e}')
    sys.exit(1)

try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except ImportError as e:
    print(f'NumPy: NOT INSTALLED - {e}')
    sys.exit(1)

try:
    import matplotlib
    print(f'Matplotlib: {matplotlib.__version__}')
except ImportError as e:
    print(f'Matplotlib: NOT INSTALLED - {e}')
    sys.exit(1)

try:
    from tensorboard import __version__ as tb_version
    print(f'TensorBoard: {tb_version}')
except ImportError as e:
    print(f'TensorBoard: NOT INSTALLED - {e}')
    sys.exit(1)

print('\n✓ All core packages installed successfully!')
"@

Write-Host $verification -ForegroundColor Green

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "   ERROR: Installation verification failed!" -ForegroundColor Red
    Write-Host "   Please check the error messages above" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Ensure your dataset is in:" -ForegroundColor White
Write-Host "   - data/images/ (JPG/PNG images)" -ForegroundColor White
Write-Host "   - data/masks/ (PNG masks, 255=fence)" -ForegroundColor White
Write-Host ""
Write-Host "2. Start training:" -ForegroundColor White
Write-Host "   python train_UNetPlusPlus.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Monitor training:" -ForegroundColor White
Write-Host "   tensorboard --logdir=logs/unetplusplus" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Check visualizations:" -ForegroundColor White
Write-Host "   training_visualizations/unetplusplus/" -ForegroundColor White
Write-Host ""
Write-Host "5. Find checkpoints:" -ForegroundColor White
Write-Host "   checkpoints/unetplusplus/best_model.pth" -ForegroundColor White
Write-Host ""
Write-Host "For detailed instructions, see: UNETPLUSPLUS_TRAINING_GUIDE.md" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
