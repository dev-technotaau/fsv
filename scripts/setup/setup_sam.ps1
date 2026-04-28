# SAM Training - Quick Start Setup Script
# VisionGuard Team - November 12, 2025

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SAM Training - Quick Start Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python OK" -ForegroundColor Green
Write-Host ""

# Check CUDA
Write-Host "[2/6] Checking CUDA availability..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PyTorch already installed" -ForegroundColor Green
} else {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: CUDA installation failed, falling back to CPU" -ForegroundColor Red
        pip install torch torchvision torchaudio
    }
}
Write-Host ""

# Install requirements
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
if (Test-Path "requirements/sam.txt") {
    pip install -r requirements/sam.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "WARNING: requirements/sam.txt not found" -ForegroundColor Red
}
Write-Host ""

# Verify installations
Write-Host "[4/6] Verifying installations..." -ForegroundColor Yellow
python -c "import torch; import cv2; import albumentations; print('✓ Core packages OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Core packages verification failed" -ForegroundColor Red
    exit 1
}

python -c "from segment_anything import sam_model_registry; print('✓ SAM installed')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing SAM..." -ForegroundColor Yellow
    pip install git+https://github.com/facebookresearch/segment-anything.git
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: SAM installation failed" -ForegroundColor Red
        Write-Host "Try manual installation: pip install git+https://github.com/facebookresearch/segment-anything.git" -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "✓ All packages verified" -ForegroundColor Green
Write-Host ""

# Check data structure
Write-Host "[5/6] Checking data structure..." -ForegroundColor Yellow
$dataDir = "data"
$imagesDir = "$dataDir/images"
$masksDir = "$dataDir/masks"

if (-not (Test-Path $dataDir)) {
    Write-Host "WARNING: data/ directory not found" -ForegroundColor Red
    Write-Host "Creating directories..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $imagesDir | Out-Null
    New-Item -ItemType Directory -Force -Path $masksDir | Out-Null
    Write-Host "Please add your images and masks to these directories" -ForegroundColor Yellow
} else {
    $imageCount = (Get-ChildItem -Path $imagesDir -File | Measure-Object).Count
    $maskCount = (Get-ChildItem -Path $masksDir -File | Measure-Object).Count
    
    Write-Host "Found $imageCount images and $maskCount masks" -ForegroundColor Cyan
    
    if ($imageCount -eq 0 -or $maskCount -eq 0) {
        Write-Host "WARNING: No images or masks found" -ForegroundColor Red
    } else {
        Write-Host "✓ Data structure OK" -ForegroundColor Green
    }
}
Write-Host ""

# Create necessary directories
Write-Host "[6/6] Creating output directories..." -ForegroundColor Yellow
$dirs = @("checkpoints/sam", "logs/sam", "training_visualizations/sam")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
}
Write-Host "✓ Directories created" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Ensure your data is in:" -ForegroundColor White
Write-Host "   - data/images/ (input images)" -ForegroundColor White
Write-Host "   - data/masks/ (segmentation masks)" -ForegroundColor White
Write-Host ""
Write-Host "2. Start training:" -ForegroundColor White
Write-Host "   python train_SAM.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Monitor training (in separate terminal):" -ForegroundColor White
Write-Host "   tensorboard --logdir logs/sam" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Check outputs:" -ForegroundColor White
Write-Host "   - Best model: checkpoints/sam/best_model.pth" -ForegroundColor White
Write-Host "   - Logs: logs/sam/" -ForegroundColor White
Write-Host "   - Visualizations: training_visualizations/sam/" -ForegroundColor White
Write-Host ""
Write-Host "For detailed guide, see: SAM_TRAINING_GUIDE.md" -ForegroundColor Yellow
Write-Host ""

# System info
Write-Host "System Information:" -ForegroundColor Cyan
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}')"
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
