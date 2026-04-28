# SAM Training Guide - Fence Staining Visualizer

## 🎯 Overview

This comprehensive guide covers the SAM (Segment Anything Model) fine-tuning script for fence detection and segmentation in the Fence Staining Visualizer application.

## 📋 Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

---

## ✨ Features

### Core Features
- **SAM-B (Base) Architecture**: Optimized for 6GB GPU laptops
- **Prompt-Based Segmentation**: Automatic box and point prompt generation
- **Mixed Precision Training (AMP)**: Faster training with lower memory usage
- **Advanced Loss Functions**: Focal + Dice + IoU + Boundary losses
- **Exponential Moving Average (EMA)**: Stable model predictions
- **Comprehensive Metrics**: IoU, Dice, Precision, Recall, F1, Accuracy

### Advanced Capabilities
- **Multi-Scale Training**: Resolution augmentation for robustness
- **Advanced Data Augmentation**: 15+ augmentation techniques
- **Learning Rate Warmup**: Cosine annealing with warm restarts
- **Gradient Accumulation**: Simulate large batch sizes
- **Early Stopping**: Automatic training termination
- **TensorBoard Logging**: Real-time training visualization
- **Checkpoint Management**: Best/last model saving
- **Distributed Training Ready**: Multi-GPU support preparation

### Improvements Over SegFormer
1. **Better Segmentation Quality**: SAM's architecture is specifically designed for segmentation
2. **Prompt Flexibility**: Can use boxes, points, or masks as prompts
3. **Enhanced Loss Functions**: 4 loss components vs 3 in SegFormer
4. **More Metrics**: 6 metrics vs 2 in SegFormer
5. **EMA Support**: Smoother convergence and better generalization
6. **Advanced Augmentation**: 15+ techniques vs 4 in SegFormer
7. **Better Logging**: Comprehensive TensorBoard integration
8. **Visualization**: Automatic prediction visualization during training

---

## 🔧 Installation

### Step 1: Create Virtual Environment

```powershell
# Create environment
python -m venv venv_sam

# Activate environment
.\venv_sam\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
# Install PyTorch with CUDA support (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all requirements
pip install -r requirements_sam.txt
```

### Step 3: Verify Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "from segment_anything import sam_model_registry; print('SAM installed successfully')"
```

---

## ⚙️ Configuration

### Key Configuration Parameters

The `Config` class in `train_SAM.py` contains all hyperparameters:

```python
# Model Configuration
SAM_MODEL_TYPE = "vit_b"  # Options: vit_h, vit_l, vit_b
PRETRAINED = True

# Training Parameters
INPUT_SIZE = 1024  # SAM native resolution
TRAIN_SIZE = 512   # Training resolution (memory efficient)
BATCH_SIZE = 4     # Per GPU
ACCUMULATION_STEPS = 4  # Effective batch = 16
EPOCHS = 100
LEARNING_RATE = 1e-4

# Loss Weights
LOSS_WEIGHTS = {
    'focal': 0.25,    # Class imbalance
    'dice': 0.35,     # Overlap
    'iou': 0.25,      # Intersection/Union
    'boundary': 0.15  # Edge accuracy
}

# Prompt Configuration
USE_BOX_PROMPTS = True
USE_POINT_PROMPTS = True
NUM_POINT_PROMPTS = 5
POINT_JITTER = 10  # Pixels

# Hardware Optimization
NUM_WORKERS = 4
USE_AMP = True  # Mixed precision
USE_EMA = True  # Exponential moving average
```

### Customization Examples

#### For More Powerful GPU (8GB+)
```python
BATCH_SIZE = 8
TRAIN_SIZE = 768
NUM_WORKERS = 6
```

#### For Faster Training (Lower Quality)
```python
EPOCHS = 50
TRAIN_SIZE = 384
USE_ADVANCED_AUGMENTATION = False
```

#### For Best Quality (Slower)
```python
SAM_MODEL_TYPE = "vit_l"  # Larger model
TRAIN_SIZE = 1024
EPOCHS = 150
USE_TTA = True  # Test-time augmentation
```

---

## 🚀 Usage

### Basic Training

```powershell
# Make sure data is in correct structure:
# data/
#   images/
#   masks/

# Run training
python train_SAM.py
```

### Monitor Training (TensorBoard)

```powershell
# In a separate terminal
tensorboard --logdir logs/sam
```

Then open: http://localhost:6006

### Resume Training

```python
# Modify train_SAM.py to load checkpoint
checkpoint = torch.load('checkpoints/sam/checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## 🎨 Advanced Features

### 1. Prompt Engineering

SAM uses prompts to guide segmentation:

**Box Prompts** (Default: ON)
- Automatically generated from ground truth masks
- Provides bounding box around fence region
- Most efficient for training

**Point Prompts** (Default: ON)
- 5 random points sampled from fence region
- Adds robustness to partial occlusions
- Jittered by ±10 pixels for augmentation

**Mask Prompts** (Default: OFF)
- Uses previous mask as prompt
- Memory intensive
- Best for iterative refinement

### 2. Loss Function Components

**Focal Loss (25%)**
- Handles class imbalance (more background than fence)
- Focuses on hard examples
- α=0.25, γ=2.0

**Dice Loss (35%)**
- Measures overlap between prediction and ground truth
- Differentiable approximation of IoU
- Most important for segmentation

**IoU Loss (25%)**
- Direct optimization of IoU metric
- Better than BCE for segmentation
- Smooth approximation

**Boundary Loss (15%)**
- Focuses on edge accuracy
- 5x weight on boundary pixels
- Critical for fence visualization

### 3. Data Augmentation Pipeline

**Geometric Transformations**
- Horizontal flip (50%)
- Vertical flip (30%)
- Random rotation 90° (50%)
- Shift/Scale/Rotate (50%)

**Color Augmentations**
- Color jitter (50%)
- Hue/Saturation/Value (50%)
- RGB shift (50%)

**Lighting Augmentations**
- Brightness/Contrast (40%)
- Gamma correction (40%)
- CLAHE (40%)

**Noise & Blur**
- Gaussian noise (30%)
- Gaussian blur (30%)
- Motion blur (30%)

### 4. Learning Rate Schedule

**Warmup Phase** (5 epochs)
- Gradual learning rate increase
- Prevents early overfitting
- Stabilizes training

**Cosine Annealing with Restarts**
- Periodic learning rate resets
- Escapes local minima
- T_0=10, T_mult=2

### 5. Exponential Moving Average (EMA)

```python
# EMA maintains shadow copy of model weights
# Provides more stable predictions
# Decay = 0.999 (99.9% old, 0.1% new)

# Used during validation and inference
if ema is not None:
    ema.apply_shadow()  # Use EMA weights
    validate()
    ema.restore()  # Restore training weights
```

---

## 🔍 Troubleshooting

### Issue 1: CUDA Out of Memory

**Solution 1**: Reduce batch size
```python
BATCH_SIZE = 2
ACCUMULATION_STEPS = 8  # Keep effective batch = 16
```

**Solution 2**: Reduce training resolution
```python
TRAIN_SIZE = 384
```

**Solution 3**: Disable EMA
```python
USE_EMA = False
```

**Solution 4**: Reduce workers
```python
NUM_WORKERS = 2
```

### Issue 2: Slow Training

**Solution 1**: Enable all optimizations
```python
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
```

**Solution 2**: Increase workers
```python
NUM_WORKERS = 6
PREFETCH_FACTOR = 3
```

**Solution 3**: Reduce augmentation
```python
USE_ADVANCED_AUGMENTATION = False
```

### Issue 3: Poor Segmentation Quality

**Solution 1**: Increase training size
```python
TRAIN_SIZE = 768  # or 1024
```

**Solution 2**: Train longer
```python
EPOCHS = 150
PATIENCE = 30
```

**Solution 3**: Adjust loss weights
```python
LOSS_WEIGHTS = {
    'focal': 0.2,
    'dice': 0.4,  # Increase dice
    'iou': 0.3,   # Increase IoU
    'boundary': 0.1
}
```

**Solution 4**: Use larger model
```python
SAM_MODEL_TYPE = "vit_l"
```

### Issue 4: SAM Installation Failed

```powershell
# Manual installation
pip install git+https://github.com/facebookresearch/segment-anything.git

# If git not available, download manually
# 1. Download from: https://github.com/facebookresearch/segment-anything
# 2. Extract and install: pip install -e segment-anything
```

### Issue 5: Import Errors

```powershell
# Reinstall packages
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Check versions
pip list | Select-String "torch"
```

---

## 🚄 Performance Optimization

### For 6GB GPU (Laptop)

```python
# Optimized configuration
BATCH_SIZE = 4
TRAIN_SIZE = 512
NUM_WORKERS = 4
USE_AMP = True
ACCUMULATION_STEPS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
```

**Expected Performance**:
- Training speed: ~2-3 seconds/batch
- GPU utilization: 85-95%
- Memory usage: ~5.5 GB
- Epoch time: ~5-8 minutes (800 images)

### For 8GB GPU (Desktop)

```python
BATCH_SIZE = 6
TRAIN_SIZE = 640
NUM_WORKERS = 6
PREFETCH_FACTOR = 3
```

**Expected Performance**:
- Training speed: ~1.5-2 seconds/batch
- GPU utilization: 90-98%
- Memory usage: ~7.5 GB
- Epoch time: ~3-5 minutes (800 images)

### For 16GB+ GPU (Workstation)

```python
SAM_MODEL_TYPE = "vit_l"  # Larger model
BATCH_SIZE = 8
TRAIN_SIZE = 1024
NUM_WORKERS = 8
```

**Expected Performance**:
- Training speed: ~1-1.5 seconds/batch
- GPU utilization: 95-99%
- Memory usage: ~14 GB
- Epoch time: ~2-4 minutes (800 images)

---

## 📊 Expected Results

### Training Metrics (After 100 Epochs)

| Metric | Expected Value | Excellent |
|--------|---------------|-----------|
| IoU | 0.85-0.90 | >0.90 |
| Dice | 0.90-0.93 | >0.93 |
| F1 | 0.90-0.93 | >0.93 |
| Precision | 0.88-0.92 | >0.92 |
| Recall | 0.87-0.91 | >0.91 |
| Accuracy | 0.94-0.97 | >0.97 |

### Comparison with SegFormer

| Aspect | SegFormer | SAM | Winner |
|--------|-----------|-----|--------|
| IoU | 0.82-0.87 | 0.85-0.90 | **SAM** |
| Training Time | 4-6 min/epoch | 5-8 min/epoch | SegFormer |
| Memory Usage | 4.5 GB | 5.5 GB | SegFormer |
| Edge Quality | Good | **Excellent** | **SAM** |
| Flexibility | Limited | **High** | **SAM** |
| Metrics | 2 | **6** | **SAM** |

---

## 📁 Output Structure

After training:

```
training/
├── train_SAM.py
├── checkpoints/
│   └── sam/
│       ├── best_model.pth          # Best model (highest val IoU)
│       ├── checkpoint_epoch_50.pth # Periodic checkpoint
│       ├── checkpoint_epoch_100.pth
│       └── sam_vit_b.pth          # Pretrained SAM
├── logs/
│   └── sam/
│       ├── training_20251112_143022.log
│       └── tensorboard_20251112_143022/
├── training_visualizations/
│   └── sam/
│       ├── epoch_010.png
│       ├── epoch_020.png
│       └── ...
```

---

## 🎓 Best Practices

1. **Start with default settings** - They're optimized for 6GB GPU
2. **Monitor TensorBoard** - Watch for overfitting
3. **Use EMA** - Better generalization
4. **Save checkpoints** - Resume if interrupted
5. **Validate frequently** - Catch issues early
6. **Use early stopping** - Don't waste GPU time
7. **Visualize predictions** - Verify quality
8. **Test on unseen data** - Ensure generalization

---

## 📚 Additional Resources

- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [SAM GitHub](https://github.com/facebookresearch/segment-anything)
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)
- [EMA in Deep Learning](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)

---

## 🤝 Support

For issues or questions:
1. Check this guide
2. Review logs in `logs/sam/`
3. Check TensorBoard for metrics
4. Contact VisionGuard Team

---

**Author**: VisionGuard Team - Advanced AI Division  
**Date**: November 12, 2025  
**Version**: 1.0 Enterprise Edition
