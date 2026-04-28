# UNet++ Training Guide for Fence Detection

**Ultra Enterprise Edition - Complete Training Documentation**

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Configuration](#configuration)
6. [Training](#training)
7. [Monitoring](#monitoring)
8. [Evaluation](#evaluation)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)
11. [Performance Optimization](#performance-optimization)
12. [Comparison with Mask2Former](#comparison-with-mask2former)

---

## Overview

### What is UNet++?

UNet++ (Nested UNet) is an advanced semantic segmentation architecture that improves upon the original UNet with:

- **Nested Skip Pathways**: Better feature propagation between encoder and decoder
- **Dense Skip Connections**: Fine-grained detail preservation
- **Deep Supervision**: Multi-level auxiliary losses for faster convergence
- **Attention Mechanisms**: SCSE (Spatial & Channel Squeeze-Excitation) gates

### Key Features of This Implementation

✅ **EfficientNet-B7 Encoder** - State-of-the-art backbone (66M parameters)  
✅ **512×512 Resolution** - Higher than Mask2Former for better detail capture  
✅ **6 Advanced Loss Functions** - Focal, Dice, Tversky, Boundary, Lovász, SSIM  
✅ **Enhanced Augmentation** - 15+ augmentation techniques (Albumentations++)  
✅ **Mixed Precision Training** - AMP for faster training on modern GPUs  
✅ **EMA (Exponential Moving Average)** - Stable predictions  
✅ **Deep Supervision** - 5-level weighted auxiliary losses  
✅ **SCSE Attention Gates** - Spatial & channel squeeze-excitation  
✅ **OneCycleLR Scheduler** - Optimal learning rate scheduling  
✅ **6GB GPU Optimized** - Works on laptop GPUs  
✅ **TensorBoard Integration** - Real-time training visualization  

### Expected Performance

| Scene Type | Expected IoU | Confidence |
|------------|--------------|------------|
| Clean fences (lawn, garden) | 92-95% | High ✅ |
| Fences with vegetation | 85-90% | Medium ⚠️ |
| Complex backgrounds (trees, decks) | 70-85% | Medium ⚠️ |
| Occluded fences (pets, humans) | 60-75% | Low ⚠️ |

---

## System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS (CPU only)
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 6GB VRAM (GTX 1660 Ti, RTX 2060, RTX 3050)
- **CUDA**: 11.8 or higher
- **RAM**: 16GB system memory
- **Storage**: 10GB free disk space

### Recommended Requirements

- **GPU**: NVIDIA RTX 3060 (12GB), RTX 3070 (8GB), RTX 4060 Ti (16GB)
- **CUDA**: 12.1 or higher
- **RAM**: 32GB system memory
- **Storage**: SSD with 50GB free space

### Supported GPUs

| GPU Model | VRAM | Batch Size | Training Speed |
|-----------|------|------------|----------------|
| GTX 1660 Ti | 6GB | 3 | ~2 min/epoch |
| RTX 2060 | 6GB | 3 | ~1.8 min/epoch |
| RTX 3050 | 8GB | 4 | ~1.5 min/epoch |
| RTX 3060 | 12GB | 6 | ~1.2 min/epoch |
| RTX 3070 | 8GB | 5 | ~1.0 min/epoch |
| RTX 4060 Ti | 16GB | 8 | ~0.8 min/epoch |
| A100 | 40GB | 16 | ~0.4 min/epoch |

---

## Installation

### Option 1: Automated Setup (Recommended)

**Windows (PowerShell):**

```powershell
# Navigate to project directory
cd "D:\Ubuntu\TECHNOTAU (2)\Project_management_and_training_NOV_11_2025\training"

# Run setup script
.\setup_unetplusplus.ps1
```

The script will:
1. Check Python version (3.8+ required)
2. Verify CUDA/GPU availability
3. Optionally create virtual environment
4. Upgrade pip, setuptools, wheel
5. Install PyTorch with CUDA support
6. Install all dependencies
7. Verify installation

### Option 2: Manual Installation

**Step 1: Create Virtual Environment (Recommended)**

```bash
# Create virtual environment
python -m venv unetplusplus_env

# Activate (Windows)
.\unetplusplus_env\Scripts\Activate.ps1

# Activate (Linux/Mac)
source unetplusplus_env/bin/activate
```

**Step 2: Upgrade pip**

```bash
python -m pip install --upgrade pip setuptools wheel
```

**Step 3: Install PyTorch with CUDA**

Visit [PyTorch Official Website](https://pytorch.org/get-started/locally/) and select your configuration.

**For CUDA 11.8 (Most Compatible):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1 (Latest):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU Only (Not Recommended):**

```bash
pip install torch torchvision torchaudio
```

**Step 4: Install Dependencies**

```bash
pip install -r requirements_unetplusplus.txt
```

**Step 5: Verify Installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import segmentation_models_pytorch as smp; print(f'SMP Version: {smp.__version__}')"
```

Expected output:
```
PyTorch: 2.0.0+cu118
CUDA Available: True
SMP Version: 0.3.3
```

---

## Dataset Preparation

### Directory Structure

Ensure your dataset follows this structure:

```
training/
├── data/
│   ├── images/          # Input images (JPG/PNG)
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── masks/           # Binary masks (PNG)
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
```

### Dataset Requirements

✅ **Image Format**: JPG or PNG  
✅ **Mask Format**: PNG (grayscale)  
✅ **Mask Values**: 0 (background), 255 (fence)  
✅ **Resolution**: Any (will be resized to 512×512 during training)  
✅ **Naming**: Image and mask filenames must match (e.g., `fence_001.jpg` → `fence_001.png`)  

### Current Dataset Status

```
Total Images: 804
Train Split: 683 images (85%)
Val Split: 121 images (15%)
Scene Types: 27 unique (clean fences only)
```

⚠️ **Known Limitation**: Current dataset lacks scene diversity (no trees, pets, wooden distractors, occlusions). Model performance on complex scenes may be limited (40-70% IoU). See [Future Enhancements](#future-enhancements) for improvement plan.

### Dataset Validation

Run validation before training:

```bash
python validate_dataset.py
```

This will check:
- Image-mask pairing
- Mask value ranges
- File integrity
- Resolution statistics

---

## Configuration

All hyperparameters are configured in the `Config` class within `train_UNetPlusPlus.py`.

### Key Configuration Options

#### Model Architecture

```python
ENCODER_NAME = "efficientnet-b7"          # Backbone encoder
DECODER_CHANNELS = (512, 256, 128, 64, 32) # Decoder channel progression
DECODER_ATTENTION_TYPE = "scse"           # Attention mechanism
USE_DEEP_SUPERVISION = True               # Enable auxiliary losses
```

**Available Encoders:**
- `efficientnet-b0` to `efficientnet-b7` (b7 = best accuracy, b0 = fastest)
- `resnet50`, `resnet101`, `resnet152`
- `resnext50_32x4d`, `resnext101_32x8d`
- `densenet121`, `densenet169`, `densenet201`
- `mobilenet_v2` (lightweight)
- `timm-efficientnet-b5`, `timm-regnety-064` (alternatives)

#### Training Hyperparameters

```python
TRAIN_SIZE = 512              # Input resolution (higher = better detail)
BATCH_SIZE = 3                # Per-GPU batch size (reduce if OOM)
ACCUMULATION_STEPS = 4        # Effective batch = 3 × 4 = 12
EPOCHS = 250                  # Total training epochs
LEARNING_RATE = 3e-4          # Initial learning rate
ENCODER_LR_MULTIPLIER = 0.1   # Encoder LR = LR × 0.1
WEIGHT_DECAY = 1e-4           # L2 regularization
WARMUP_EPOCHS = 20            # LR warmup period (first 8% of training)
```

#### Loss Function Weights

```python
LOSS_WEIGHTS = {
    'focal_loss': 2.5,      # Class imbalance handling
    'dice_loss': 2.0,       # Overlap optimization
    'boundary_loss': 1.8,   # Edge precision (critical for fences!)
    'lovasz_loss': 1.5,     # IoU optimization
    'tversky_loss': 1.2,    # FP/FN control
    'ssim_loss': 0.8,       # Structural similarity
}
```

**Loss Function Descriptions:**

- **Focal Loss**: Handles class imbalance by focusing on hard examples
- **Dice Loss**: Optimizes overlap between prediction and ground truth
- **Tversky Loss**: Controls trade-off between false positives and false negatives
- **Boundary Loss**: Heavily weights boundary pixels for sharp edges
- **Lovász-Hinge Loss**: Directly optimizes IoU metric
- **SSIM Loss**: Preserves structural similarity (texture, patterns)

#### Augmentation Settings

```python
USE_ADVANCED_AUGMENTATION = True
AUGMENTATION_PROB = 0.85      # 85% chance of applying augmentations
```

**Augmentation Pipeline Includes:**

1. **Geometric**: Flip, Rotate, Scale, Shift, Elastic, Perspective, Grid/Optical Distortion
2. **Color**: ColorJitter, HSV, RGBShift, Brightness/Contrast, Gamma, CLAHE
3. **Weather**: Rain, Fog, Sun Flare, Shadows
4. **Noise**: Gaussian, Multiplicative, ISO
5. **Blur**: Gaussian, Motion, Median, Defocus
6. **Quality**: Compression, Downscale
7. **Cutout**: Random patches removed

#### Memory Optimization

```python
USE_AMP = True                # Mixed precision (critical for 6GB GPU)
NUM_WORKERS = 4               # Data loading workers
PIN_MEMORY = True             # Faster GPU transfer
PREFETCH_FACTOR = 2           # Prefetch batches
GRADIENT_CHECKPOINTING = False # Enable if OOM (slower but less memory)
```

#### Early Stopping

```python
EARLY_STOPPING = False        # Disabled by default (train full 250 epochs)
PATIENCE = 60                 # Stop if no improvement for 60 epochs
MIN_DELTA = 1e-5              # Minimum improvement threshold
```

⚠️ **Important**: Early stopping is **DISABLED** by default to allow full training convergence.

---

## Training

### Start Training

```bash
# Ensure you're in the training directory
cd "D:\Ubuntu\TECHNOTAU (2)\Project_management_and_training_NOV_11_2025\training"

# Start training
python train_UNetPlusPlus.py
```

### Expected Training Timeline

**Hardware**: 6GB GPU (RTX 2060 / GTX 1660 Ti)

| Checkpoint | Epoch | Time | Expected IoU | Status |
|------------|-------|------|--------------|--------|
| Warmup Complete | 20 | 30 min | 0.55-0.65 | 🟡 Learning started |
| Early Progress | 50 | 75 min | 0.75-0.80 | 🟢 Good progress |
| Mid Training | 100 | 150 min | 0.85-0.88 | 🟢 Main learning |
| Late Training | 200 | 300 min | 0.90-0.92 | 🟢 Fine-tuning |
| **Training Complete** | **250** | **~375 min (~6.25h)** | **0.92-0.95** | ✅ **Converged** |

### Training Output Example

```
==========================================
UNET++ TRAINING v3.0 - ULTRA ENTERPRISE EDITION
==========================================
Device: cuda
CUDA Available: True
GPU: NVIDIA GeForce RTX 2060
GPU Memory: 6.00 GB
Encoder: efficientnet-b7
Input Size: 512x512
Batch Size: 3
Accumulation Steps: 4
Effective Batch: 12
Epochs: 250
Learning Rate: 0.0003
Deep Supervision: True
Mixed Precision: True
EMA: True

[1/6] Loading dataset...
Found 804 images and 804 masks
Matched 804 image-mask pairs
Train set: 683 samples
Val set: 121 samples

[2/6] Creating DataLoaders...
Train batches: 227, Val batches: 40

[3/6] Loading UNet++ model...
Model created successfully
Total parameters: 66,123,456
Trainable parameters: 66,123,456
Encoder: efficientnet-b7
Decoder channels: (512, 256, 128, 64, 32)
Attention type: scse
Deep supervision: True

[4/6] Setting up training...
Optimizer: AdamW
Encoder LR: 3.00e-05
Decoder LR: 3.00e-04

[5/6] Starting training...

Epoch 1/250 [Train]: 100%|██████████| 227/227 [01:45<00:00, 2.15batch/s, loss=0.7234, iou=0.3421]
Epoch 1/250 [Val]: 100%|██████████| 40/40 [00:12<00:00, 3.25batch/s]

Epoch 1/250 Summary (117.3s):
  Train - Loss: 0.7234 | IoU: 0.3421 | Dice: 0.4123 | F1: 0.4056
  Val   - Loss: 0.6812 | IoU: 0.3892

✓ Saved best model (IoU: 0.3892)
✓ Saved checkpoint: checkpoints/unetplusplus/checkpoint_epoch_5.pth
```

### Checkpoint Management

Checkpoints are automatically saved to `checkpoints/unetplusplus/`:

- **`best_model.pth`** - Best validation IoU (use this for inference!)
- **`last_model.pth`** - Most recent epoch (for resuming training)
- **`checkpoint_epoch_5.pth`** - Saved every 5 epochs
- **`checkpoint_epoch_10.pth`** - ...
- **`checkpoint_epoch_15.pth`** - ...

### Resuming Training

If training is interrupted, it will **automatically resume** from `last_model.pth`:

```bash
# Just run the script again
python train_UNetPlusPlus.py
```

The script will:
1. Detect `checkpoints/unetplusplus/last_model.pth`
2. Load model weights, optimizer state, scheduler state
3. Resume from the saved epoch
4. Continue training seamlessly

---

## Monitoring

### TensorBoard (Real-Time Visualization)

**Start TensorBoard:**

```bash
tensorboard --logdir=logs/unetplusplus
```

**Access in Browser:**

Open [http://localhost:6006](http://localhost:6006)

**Available Metrics:**

- **SCALARS**
  - Train/Loss, Val/Loss
  - Train/IoU, Val/IoU
  - Train/Dice, Val/Dice
  - Train/F1, Val/F1
  - Train/Boundary_F1, Val/Boundary_F1
  - Learning rates (encoder, decoder)
  - Loss components (focal, dice, boundary, lovász, tversky, ssim)
  
- **IMAGES**
  - Prediction visualizations (every 5 epochs)
  
- **GRAPHS**
  - Model architecture visualization

### Training Visualizations

Prediction samples are saved every 5 epochs to:

```
training_visualizations/unetplusplus/
├── epoch_5_predictions.png
├── epoch_10_predictions.png
├── epoch_15_predictions.png
└── ...
```

Each visualization shows:
- **Left**: Input image
- **Middle**: Ground truth mask
- **Right**: Model prediction

### Log Files

Detailed logs are saved to `logs/unetplusplus/`:

- **`training_YYYYMMDD_HHMMSS.log`** - Training progress log
- **`config_YYYYMMDD_HHMMSS.json`** - Full configuration
- **`training_summary_YYYYMMDD_HHMMSS.json`** - Final training summary

### GPU Memory Monitoring

Monitor GPU usage during training:

```bash
# Windows
nvidia-smi -l 1

# Linux
watch -n 1 nvidia-smi
```

Expected memory usage:
- **Training**: ~5.5-5.8 GB (batch_size=3)
- **Validation**: ~5.0 GB
- **Peak**: ~5.9 GB (during optimizer step)

---

## Evaluation

### Metrics Explained

| Metric | Description | Target Value |
|--------|-------------|--------------|
| **IoU (Intersection over Union)** | Overlap ratio between prediction and ground truth | > 0.92 |
| **Dice Coefficient** | 2× IoU / (1 + IoU), emphasizes overlap | > 0.96 |
| **Precision** | True positives / (True positives + False positives) | > 0.95 |
| **Recall** | True positives / (True positives + False negatives) | > 0.94 |
| **F1 Score** | Harmonic mean of precision and recall | > 0.94 |
| **Boundary F1** | F1 score computed only on boundary pixels | > 0.85 |

### Best Model Selection

The **best model** is selected based on **validation IoU**. The script automatically saves `best_model.pth` whenever validation IoU improves.

### Inference Example

```python
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Load model
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b7",
    encoder_weights=None,  # Don't load ImageNet weights
    in_channels=3,
    classes=1,
    activation=None,
    decoder_attention_type="scse"
)

# Load trained weights
checkpoint = torch.load('checkpoints/unetplusplus/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.cuda()

# Preprocessing
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Load image
image = cv2.imread('test_fence.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_size = image.shape[:2]

# Preprocess
transformed = transform(image=image)
input_tensor = transformed['image'].unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output)
    prediction = prediction.cpu().numpy()[0, 0]

# Post-process
prediction_binary = (prediction > 0.5).astype(np.uint8) * 255
prediction_resized = cv2.resize(prediction_binary, (original_size[1], original_size[0]))

# Save result
cv2.imwrite('fence_mask.png', prediction_resized)
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

1. **Reduce Batch Size** (in `train_UNetPlusPlus.py`):
   ```python
   BATCH_SIZE = 2  # Reduce from 3 to 2
   ```

2. **Reduce Input Size** (loses detail):
   ```python
   TRAIN_SIZE = 384  # Reduce from 512 to 384
   ```

3. **Enable Gradient Checkpointing** (slower but less memory):
   ```python
   GRADIENT_CHECKPOINTING = True
   ```

4. **Reduce Workers**:
   ```python
   NUM_WORKERS = 2  # Reduce from 4 to 2
   ```

5. **Use Smaller Encoder**:
   ```python
   ENCODER_NAME = "efficientnet-b5"  # Reduce from b7 to b5
   ```

### Slow Training

**Symptoms:**
- Training taking > 3 min/epoch on 6GB GPU

**Solutions:**

1. **Enable AMP** (should be enabled by default):
   ```python
   USE_AMP = True
   ```

2. **Enable CUDNN Benchmark**:
   ```python
   DETERMINISTIC = False  # Enables cudnn.benchmark
   ```

3. **Increase Workers**:
   ```python
   NUM_WORKERS = 6  # Increase from 4
   ```

4. **Disable Advanced Augmentation** (loses quality):
   ```python
   USE_ADVANCED_AUGMENTATION = False
   ```

### Loss Not Decreasing

**Symptoms:**
- Loss stuck at high value (> 0.5) after 20+ epochs
- IoU not improving (< 0.6)

**Solutions:**

1. **Check Dataset**:
   ```bash
   python validate_dataset.py
   ```
   Ensure masks are binary (0 and 255).

2. **Reduce Learning Rate**:
   ```python
   LEARNING_RATE = 1e-4  # Reduce from 3e-4
   MAX_LR = 1e-3  # Reduce from 3e-3
   ```

3. **Increase Warmup**:
   ```python
   WARMUP_EPOCHS = 30  # Increase from 20
   ```

4. **Check Loss Weights**:
   Ensure weights are balanced (should be fine by default).

### Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'segmentation_models_pytorch'
```

**Solutions:**

```bash
# Reinstall dependencies
pip install --upgrade -r requirements_unetplusplus.txt

# Verify installation
python -c "import segmentation_models_pytorch as smp; print(smp.__version__)"
```

### CUDA Errors

**Symptoms:**
```
RuntimeError: CUDA error: device-side assert triggered
```

**Solutions:**

1. **Check Mask Values**:
   Masks must be in range [0, 1]. The script automatically converts masks > 127 to 1, but verify:
   ```python
   import cv2
   import numpy as np
   mask = cv2.imread('data/masks/sample.png', cv2.IMREAD_GRAYSCALE)
   print(f"Mask unique values: {np.unique(mask)}")
   # Should print: [0, 255]
   ```

2. **Update CUDA/Drivers**:
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Update PyTorch if needed
   pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu118
   ```

---

## Advanced Features

### Deep Supervision

Deep supervision uses **auxiliary losses** from intermediate decoder levels. This helps earlier layers learn useful features faster.

**Enable/Disable:**
```python
USE_DEEP_SUPERVISION = True  # Recommended
DEEP_SUPERVISION_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]  # Level weights
```

**How It Works:**
- Main output (decoder level 0): weight = 1.0
- Auxiliary output 1 (level 1): weight = 0.8
- Auxiliary output 2 (level 2): weight = 0.6
- Auxiliary output 3 (level 3): weight = 0.4
- Auxiliary output 4 (level 4): weight = 0.2

Total loss = Weighted average of all level losses.

### Exponential Moving Average (EMA)

EMA maintains a "shadow" copy of model weights that is a moving average of past weights. This produces more stable predictions.

**Configure EMA:**
```python
USE_EMA = True                # Enable EMA
EMA_DECAY = 0.9998            # Decay rate (higher = more averaging)
EMA_START_EPOCH = 10          # Start EMA after 10 epochs
```

**How It Works:**
- During training: Original weights are updated normally
- During validation: EMA shadow weights are used
- At inference: Use EMA weights for best results

### Test-Time Augmentation (TTA)

Apply augmentations during inference and average predictions for better accuracy.

**Enable TTA:**
```python
USE_TTA = True
TTA_TRANSFORMS = ['original', 'hflip', 'vflip', 'rotate90']
TTA_MERGE = 'mean'
```

**Inference with TTA:**
```python
# In inference code, apply multiple transforms and average
predictions = []
for transform in tta_transforms:
    # Apply transform, predict, inverse transform, add to list
    ...
final_prediction = np.mean(predictions, axis=0)
```

### Multi-Scale Testing

Evaluate at multiple resolutions and merge predictions.

**Enable Multi-Scale:**
```python
USE_MULTISCALE_TEST = True
TEST_SCALES = [0.75, 1.0, 1.25]  # 75%, 100%, 125% of training size
```

### Attention Mechanisms

**Available Attention Types:**

```python
DECODER_ATTENTION_TYPE = "scse"  # Spatial & Channel Squeeze-Excitation (recommended)
# Options: None, "scse"
```

**SCSE Attention:**
- **Spatial**: Focuses on "where" (spatial locations)
- **Channel**: Focuses on "what" (feature channels)
- Combined: Best of both worlds

---

## Performance Optimization

### Training Speed Tips

1. **Use Mixed Precision** (enabled by default):
   ```python
   USE_AMP = True
   ```
   Speedup: ~30-40% faster

2. **Optimize DataLoader**:
   ```python
   NUM_WORKERS = 6              # Match CPU cores
   PIN_MEMORY = True            # Faster GPU transfer
   PREFETCH_FACTOR = 3          # Prefetch more batches
   PERSISTENT_WORKERS = True    # Reuse workers
   ```

3. **Use Faster Encoder** (loses accuracy):
   ```python
   ENCODER_NAME = "efficientnet-b3"  # b3 is 2× faster than b7
   ```

4. **Reduce Augmentation** (loses robustness):
   ```python
   AUGMENTATION_PROB = 0.5  # Reduce from 0.85
   ```

### Memory Optimization Tips

1. **Gradient Accumulation** (already enabled):
   ```python
   ACCUMULATION_STEPS = 4  # Effective batch = 12 without extra memory
   ```

2. **Reduce Resolution** (loses detail):
   ```python
   TRAIN_SIZE = 384  # Reduce from 512
   ```

3. **Use Smaller Encoder**:
   ```python
   ENCODER_NAME = "efficientnet-b5"  # 30M params vs 66M for b7
   ```

4. **Disable Deep Supervision** (loses convergence speed):
   ```python
   USE_DEEP_SUPERVISION = False
   ```

### Inference Speed Tips

1. **Export to ONNX** (2-3× faster):
   ```python
   import torch.onnx
   
   dummy_input = torch.randn(1, 3, 512, 512).cuda()
   torch.onnx.export(model, dummy_input, "fence_unetplusplus.onnx")
   ```

2. **Use TensorRT** (5-10× faster):
   Requires NVIDIA GPU and TensorRT installation.

3. **Quantization** (INT8, 4× faster):
   ```python
   model_quantized = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

---

## Comparison with Mask2Former

### Architecture Comparison

| Feature | UNet++ | Mask2Former |
|---------|--------|-------------|
| **Type** | CNN-based | Transformer-based |
| **Encoder** | EfficientNet-B7 (66M) | SegFormer-B5 (98.6M) |
| **Resolution** | 512×512 | 384×384 |
| **Batch Size** | 3×4 = 12 | 2×4 = 8 |
| **Epochs** | 250 | 200 |
| **Loss Types** | 6 | 5 |
| **Training Speed** | ~1.5 min/epoch | ~2 min/epoch |
| **Total Training Time** | ~6.25 hours | ~6.7 hours |
| **Attention** | SCSE gates | Masked attention |
| **Skip Connections** | Dense nested | Standard FPN |

### Performance Comparison (Expected)

| Metric | UNet++ | Mask2Former |
|--------|--------|-------------|
| **Clean Scenes IoU** | 92-95% | 92-95% |
| **Complex Scenes IoU** | 70-85% | 75-88% |
| **Edge Quality** | Excellent | Excellent |
| **Inference Speed** | Fast (50ms) | Medium (80ms) |
| **Memory (Training)** | 5.5 GB | 5.8 GB |
| **Memory (Inference)** | 1.8 GB | 2.2 GB |

### When to Use UNet++

✅ **Use UNet++ When:**
- Speed is critical (inference < 50ms)
- Higher resolution needed (512×512)
- Limited GPU memory (< 8GB)
- Prefer CNN architectures
- Need faster training (~1.5 min/epoch)

### When to Use Mask2Former

✅ **Use Mask2Former When:**
- Best accuracy is critical
- Complex scenes (trees, occlusions)
- Transformer architectures preferred
- More GPU memory available (> 8GB)
- Instance segmentation needed (future)

### Recommendation

**For Fence Detection:**
- **Start with UNet++**: Faster training, higher resolution, excellent edge detection
- **Train Both**: Compare results on your specific data
- **Ensemble**: Average predictions from both models for best results

---

## Future Enhancements

### Dataset Improvements (High Priority)

⚠️ **Current Limitation**: Dataset has only 27 clean fence scene types.

**Recommended Enhancements:**

1. **Add Diverse Scenes** (~200 images):
   - 50-60 images: Trees behind/near fences
   - 40-50 images: Wooden distractors (decks, sheds, furniture)
   - 30-40 images: Vegetation on/near fences
   - 20-30 images: Pets/animals near fences
   - 20-30 images: Humans in scene
   - 30-40 images: Complex backgrounds

2. **Fine-Tune on Enhanced Data**:
   ```bash
   # After adding new images, resume training:
   python train_UNetPlusPlus.py
   ```
   The script will automatically include new images.

3. **Expected Improvement**:
   - Complex scenes IoU: 40-70% → 85-92%
   - Overall robustness: Significantly improved

### Model Improvements

1. **Ensemble with Mask2Former**:
   ```python
   # Average predictions from both models
   pred_unet = unet_model(image)
   pred_mask2former = mask2former_model(image)
   final_pred = (pred_unet + pred_mask2former) / 2
   ```

2. **Post-Processing with CRF**:
   ```python
   USE_CRF = True  # Enable in config
   ```

3. **Custom Loss for Fence Characteristics**:
   Add loss term that penalizes non-vertical edges (fences are usually vertical).

---

## FAQ

### Q: How long does training take?

**A:** ~6.25 hours on a 6GB GPU (RTX 2060 / GTX 1660 Ti). Faster GPUs will be quicker (~4 hours on RTX 3070).

### Q: Can I stop training and resume later?

**A:** Yes! Training automatically resumes from `last_model.pth`. Just run `python train_UNetPlusPlus.py` again.

### Q: Which model should I use for inference?

**A:** Use `checkpoints/unetplusplus/best_model.pth` - it has the best validation IoU.

### Q: Can I train on CPU?

**A:** Technically yes, but it will be **extremely slow** (100× slower). Training 250 epochs could take weeks. GPU is required for practical use.

### Q: How do I reduce GPU memory usage?

**A:** Reduce `BATCH_SIZE` from 3 to 2 or 1. See [Troubleshooting](#out-of-memory-oom-errors).

### Q: Why is early stopping disabled?

**A:** Previous training stopped prematurely at epoch 19 due to early stopping during warmup. Early stopping is now disabled to ensure full training convergence (250 epochs).

### Q: How do I know if training is working?

**A:** Check validation IoU:
- Epoch 20: Should be > 0.55
- Epoch 50: Should be > 0.75
- Epoch 100: Should be > 0.85
- Epoch 250: Should be > 0.92

### Q: What if loss is not decreasing?

**A:** See [Troubleshooting - Loss Not Decreasing](#loss-not-decreasing).

### Q: Can I use this for other segmentation tasks?

**A:** Yes! Just replace the dataset and adjust:
- `NUM_CLASSES` (1 for binary, 2+ for multi-class)
- Loss weights (may need tuning)
- Augmentation pipeline (task-specific)

---

## Support & Contact

### Documentation

- **This Guide**: `UNETPLUSPLUS_TRAINING_GUIDE.md`
- **Requirements**: `requirements_unetplusplus.txt`
- **Setup Script**: `setup_unetplusplus.ps1`
- **Training Script**: `train_UNetPlusPlus.py`

### Logs & Debugging

- **Training Logs**: `logs/unetplusplus/training_*.log`
- **TensorBoard**: `tensorboard --logdir=logs/unetplusplus`
- **Visualizations**: `training_visualizations/unetplusplus/`

### Resources

- **Segmentation Models PyTorch**: https://github.com/qubvel/segmentation_models.pytorch
- **Albumentations Docs**: https://albumentations.ai/docs/
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **TensorBoard Guide**: https://www.tensorflow.org/tensorboard

---

## Appendix

### Complete File Structure

```
training/
├── train_UNetPlusPlus.py                      # Main training script
├── requirements_unetplusplus.txt              # Dependencies
├── setup_unetplusplus.ps1                     # Setup script
├── UNETPLUSPLUS_TRAINING_GUIDE.md             # This guide
├── data/
│   ├── images/                                # Input images
│   └── masks/                                 # Binary masks
├── checkpoints/
│   └── unetplusplus/
│       ├── best_model.pth                     # Best model (use this!)
│       ├── last_model.pth                     # Latest checkpoint
│       └── checkpoint_epoch_*.pth             # Periodic saves
├── logs/
│   └── unetplusplus/
│       ├── training_*.log                     # Training logs
│       ├── config_*.json                      # Configuration
│       ├── training_summary_*.json            # Training summary
│       └── tensorboard_*/                     # TensorBoard logs
└── training_visualizations/
    └── unetplusplus/
        └── epoch_*_predictions.png            # Prediction samples
```

### Training Checklist

- [ ] Python 3.8+ installed
- [ ] NVIDIA GPU with 6GB+ VRAM
- [ ] CUDA 11.8+ installed
- [ ] Environment setup complete (`setup_unetplusplus.ps1`)
- [ ] Dataset in `data/images/` and `data/masks/`
- [ ] Dataset validated (`validate_dataset.py`)
- [ ] Configuration reviewed (`train_UNetPlusPlus.py`)
- [ ] TensorBoard ready (`tensorboard --logdir=logs/unetplusplus`)
- [ ] Start training (`python train_UNetPlusPlus.py`)
- [ ] Monitor first 20 epochs (warmup phase)
- [ ] Verify no early stopping messages
- [ ] Check IoU improving steadily
- [ ] Wait for convergence (~250 epochs)
- [ ] Evaluate best model
- [ ] Test on complex scenes
- [ ] Plan dataset enhancements if needed

---

**Good luck with your UNet++ training! 🚀**

**Questions? Check [Troubleshooting](#troubleshooting) or review logs in `logs/unetplusplus/`.**
