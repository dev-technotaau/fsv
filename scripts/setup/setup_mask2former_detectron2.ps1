# Mask2Former + Detectron2 + SegFormer-B5 Setup Script (PowerShell)
# Ultra Enterprise Training Setup for Windows
# Run with: .\setup_mask2former_detectron2.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Mask2Former + Detectron2 + SegFormer-B5" -ForegroundColor Cyan
Write-Host "Ultra Enterprise Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/10] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor Green

if ($pythonVersion -notmatch "Python 3\.(8|9|10|11)") {
    Write-Host "ERROR: Python 3.8-3.11 required!" -ForegroundColor Red
    exit 1
}

# Check CUDA availability
Write-Host ""
Write-Host "[2/10] Checking CUDA installation..." -ForegroundColor Yellow
try {
    $nvccVersion = nvcc --version 2>&1
    Write-Host "Found CUDA: $nvccVersion" -ForegroundColor Green
} catch {
    Write-Host "WARNING: CUDA not found. GPU training will not be available." -ForegroundColor Yellow
    Write-Host "Download CUDA from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
}

# Create virtual environment
Write-Host ""
Write-Host "[3/10] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv_detectron2") {
    Write-Host "Virtual environment already exists. Skipping..." -ForegroundColor Yellow
} else {
    python -m venv venv_detectron2
    Write-Host "Virtual environment created: venv_detectron2" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "[4/10] Activating virtual environment..." -ForegroundColor Yellow
.\venv_detectron2\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "[5/10] Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
Write-Host "Pip upgraded successfully" -ForegroundColor Green

# Install PyTorch with CUDA
Write-Host ""
Write-Host "[6/10] Installing PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
Write-Host ""
Write-Host "Verifying PyTorch installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install basic requirements first
Write-Host ""
Write-Host "[7/10] Installing basic dependencies..." -ForegroundColor Yellow
python -m pip install numpy opencv-python pillow matplotlib tqdm pyyaml
python -m pip install albumentations transformers timm einops
python -m pip install tensorboard wandb psutil scipy pandas scikit-learn seaborn
Write-Host "Basic dependencies installed" -ForegroundColor Green

# Install pycocotools (Windows compatible)
Write-Host ""
Write-Host "[8/10] Installing pycocotools..." -ForegroundColor Yellow
try {
    python -m pip install pycocotools
    Write-Host "pycocotools installed successfully" -ForegroundColor Green
} catch {
    Write-Host "Standard pycocotools failed. Installing Windows-compatible version..." -ForegroundColor Yellow
    python -m pip install pycocotools-windows
}

# Install Detectron2
Write-Host ""
Write-Host "[9/10] Installing Detectron2..." -ForegroundColor Yellow
Write-Host "This is a large package and may take 10-15 minutes..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Attempting pre-built Detectron2 installation..." -ForegroundColor Yellow

# Try pre-built wheel first (faster)
try {
    python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
    Write-Host "Detectron2 installed from pre-built wheel" -ForegroundColor Green
} catch {
    Write-Host "Pre-built wheel failed. Building from source..." -ForegroundColor Yellow
    Write-Host "NOTE: This requires Visual Studio Build Tools!" -ForegroundColor Red
    Write-Host "Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Yellow
    
    # Build from source
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
}

# Verify Detectron2
Write-Host ""
Write-Host "Verifying Detectron2 installation..." -ForegroundColor Yellow
python -c "import detectron2; print(f'Detectron2 version: {detectron2.__version__}')"

# Install Mask2Former
Write-Host ""
Write-Host "[10/10] Installing Mask2Former..." -ForegroundColor Yellow
Write-Host "Cloning Mask2Former repository..." -ForegroundColor Yellow

if (Test-Path "Mask2Former") {
    Write-Host "Mask2Former directory exists. Skipping clone..." -ForegroundColor Yellow
} else {
    git clone https://github.com/facebookresearch/Mask2Former.git
}

Write-Host "Installing Mask2Former dependencies..." -ForegroundColor Yellow
Set-Location Mask2Former
python -m pip install -r requirements.txt

Write-Host "Compiling CUDA operators for Mask2Former..." -ForegroundColor Yellow
Write-Host "NOTE: This requires CUDA toolkit and Visual Studio!" -ForegroundColor Red

Set-Location mask2former\modeling\pixel_decoder\ops

# For Windows, we need to use a different approach
Write-Host "Compiling MSDeformAttn CUDA extension..." -ForegroundColor Yellow
python setup.py build install

Set-Location ..\..\..\..\..\

# Verify Mask2Former
Write-Host ""
Write-Host "Verifying Mask2Former installation..." -ForegroundColor Yellow
python -c "from mask2former import add_maskformer2_config; print('Mask2Former: OK')"

# Install remaining requirements
Write-Host ""
Write-Host "Installing remaining requirements..." -ForegroundColor Yellow
python -m pip install -r requirements/mask2former_detectron2.txt

# Final verification
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FINAL VERIFICATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Testing all imports..." -ForegroundColor Yellow

$testScript = @"
import sys
import torch
import torchvision
import detectron2
from detectron2.config import get_cfg
from mask2former import add_maskformer2_config
from transformers import SegformerModel
import cv2
import numpy as np
import albumentations

print('\n✓ All imports successful!')
print(f'\nPyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Detectron2: {detectron2.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Albumentations: {albumentations.__version__}')
print('\n✓ Environment ready for training!')
"@

$testScript | python

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate environment: .\venv_detectron2\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Prepare dataset (if not done): The script will auto-generate COCO annotations" -ForegroundColor White
Write-Host "3. Start training: python train_Mask2Former_Detectron2.py" -ForegroundColor White
Write-Host ""
Write-Host "For GPU info: python -c `"import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')`"" -ForegroundColor White
Write-Host ""
Write-Host "Training tips:" -ForegroundColor Yellow
Write-Host "- 6GB GPU: Keep BATCH_SIZE=2, ACCUMULATION_STEPS=8" -ForegroundColor White
Write-Host "- 12GB GPU: Increase BATCH_SIZE=4, ACCUMULATION_STEPS=4" -ForegroundColor White
Write-Host "- 24GB GPU: Increase BATCH_SIZE=8, ACCUMULATION_STEPS=2" -ForegroundColor White
Write-Host ""
Write-Host "Monitor training:" -ForegroundColor Yellow
Write-Host "- TensorBoard: tensorboard --logdir=logs/mask2former_detectron2" -ForegroundColor White
Write-Host "- Weights & Biases: Set USE_WANDB=True in Config" -ForegroundColor White
Write-Host ""
Write-Host "Happy training! 🚀" -ForegroundColor Green
