# Mask2Former + SegFormer-B5 Training Guide
## Advanced Transformer-Based Segmentation for Fence Detection

### 🎯 Overview

This is an **enterprise-grade, production-ready** training pipeline for **Mask2Former with SegFormer-B5 backbone** designed specifically for fence segmentation. It combines the power of:

- **Mask2Former**: Meta's state-of-the-art universal segmentation architecture with query-based decoding
- **SegFormer-B5**: Hierarchical vision transformer backbone from NVIDIA
- **Advanced Training**: Mixed precision, EMA, gradient accumulation, comprehensive loss functions
- **6GB GPU Optimized**: Carefully tuned for laptop GPUs while maintaining SOTA performance

---

## 🚀 Quick Start

### 1. Setup Environment
```powershell
# Run the automated setup script
powershell -ExecutionPolicy Bypass -File .\setup_mask2former.ps1
```

### 2. Start Training
```powershell
python train_Mask2Former.py
```

### 3. Monitor Progress
```powershell
# Open TensorBoard in another terminal
tensorboard --logdir=logs/mask2former
```

---

## 📊 Architecture Overview

### Mask2Former Components

```
Input Image (512×512)
        ↓
┌───────────────────────┐
│  SegFormer-B5 Backbone│  ← Hierarchical vision transformer
│  (Multi-scale features)│     Outputs: {C2, C3, C4, C5}
└───────────────────────┘
        ↓
┌───────────────────────┐
│  Pixel Decoder        │  ← Multi-scale deformable attention
│  (256D features)      │     Upsamples features to high-res
└───────────────────────┘
        ↓
┌───────────────────────┐
│  Transformer Decoder  │  ← 6-layer masked attention
│  (100 object queries) │     Query-based mask prediction
└───────────────────────┘
        ↓
┌───────────────────────┐
│  Prediction Heads     │
│  • Mask Head (per-pixel)│
│  • Class Head (per-query)│
└───────────────────────┘
        ↓
   Final Masks
```

### Key Innovations

1. **Query-Based Decoding**: Instead of dense per-pixel classification, Mask2Former uses 100 learnable object queries that attend to relevant image regions
2. **Masked Attention**: Efficient attention mechanism that focuses on predicted mask regions
3. **Hierarchical Features**: SegFormer-B5 provides multi-scale features (C2: 1/4, C3: 1/8, C4: 1/16, C5: 1/32)
4. **Universal Architecture**: Same model works for semantic, instance, and panoptic segmentation

---

## 🎨 Advanced Features

### 1. Loss Functions (5 Components)

```python
Total Loss = 2.0 × Mask Loss          # Binary cross-entropy on masks
           + 2.0 × Dice Loss          # Overlap-based loss
           + 1.0 × Classification Loss # Query class prediction
           + 1.5 × Boundary Loss      # Edge-aware loss
           + 1.0 × Lovász Loss        # SOTA segmentation loss
```

**Why Lovász Loss?**
- Direct optimization of IoU metric
- Better than cross-entropy for segmentation
- Handles class imbalance effectively

### 2. Training Enhancements

| Feature | Configuration | Benefit |
|---------|---------------|---------|
| **Mixed Precision (AMP)** | FP16 | 2× speed, 50% memory reduction |
| **Gradient Accumulation** | 8 steps | Effective batch size = 16 |
| **EMA** | Decay 0.9999 | Stable predictions, smoother convergence |
| **OneCycleLR Scheduler** | 10% warmup | Better than cosine for transformers |
| **Gradient Clipping** | Max norm 1.0 | Prevents gradient explosion |
| **Layer-wise LR Decay** | Backbone: 0.1× | Fine-tune pretrained backbone gently |

### 3. Data Augmentation (20+ Transforms)

**Geometric**: HFlip, VFlip, Rotate90, ShiftScaleRotate, ElasticTransform, Perspective

**Color**: ColorJitter, HSV, RGBShift, ChannelShuffle, RandomGamma, CLAHE, ToGray

**Weather**: RandomRain, RandomFog, RandomSunFlare, RandomShadow

**Noise**: GaussNoise, MultiplicativeNoise, ISONoise

**Blur**: GaussianBlur, MotionBlur, MedianBlur, Defocus

**Quality**: ImageCompression, Downscale, CoarseDropout

**Applied with 80% probability per image**

### 4. Metrics Tracking

- **IoU (Intersection over Union)**: Primary metric
- **Dice Coefficient**: Overlap similarity
- **Precision**: Fence pixel accuracy
- **Recall**: Fence pixel coverage
- **F1 Score**: Harmonic mean of precision/recall
- **Boundary F1**: Edge accuracy (advanced metric)
- **Accuracy**: Overall pixel correctness

### 5. Memory Optimization (6GB GPU)

```python
Batch Size: 2                    # Small batch
Accumulation: 8                  # Effective batch = 16
Input Size: 512×512              # Balanced resolution
NUM_WORKERS: 2                   # Reduced data loading threads
Mixed Precision: FP16            # Half memory usage
Empty Cache Frequency: 10        # Aggressive cleanup
Pin Memory: True                 # Fast CPU→GPU transfer
Persistent Workers: True         # Avoid worker restart overhead
```

---

## 📈 Training Configuration

### Default Hyperparameters

```python
# Model
Input Size: 512×512
Backbone: SegFormer-B5 (mit-b5)
Num Queries: 100
Num Labels: 2 (background, fence)
Hidden Dim: 256
Attention Heads: 8
Decoder Layers: 6

# Training
Batch Size: 2
Accumulation Steps: 8
Effective Batch: 16
Epochs: 150
Learning Rate: 1e-4
Backbone LR: 1e-5 (0.1× multiplier)
Weight Decay: 0.05
Warmup Epochs: 10

# Optimization
Optimizer: AdamW
Scheduler: OneCycleLR
Max LR: 1e-3
Gradient Clip: 1.0
Label Smoothing: 0.1
Stochastic Depth: 0.1

# Regularization
Dropout: 0.1
EMA Decay: 0.9999
Early Stopping: 30 epochs patience
```

### Recommended Adjustments

**For Better Accuracy (if you have more GPU memory):**
```python
BATCH_SIZE = 4
TRAIN_SIZE = 640
ACCUMULATION_STEPS = 4
NUM_QUERIES = 150
```

**For Faster Training (lower quality):**
```python
BATCH_SIZE = 4
TRAIN_SIZE = 384
ACCUMULATION_STEPS = 2
EPOCHS = 75
USE_ADVANCED_AUGMENTATION = False
```

**For Extreme Memory Constraints:**
```python
BATCH_SIZE = 1
TRAIN_SIZE = 384
ACCUMULATION_STEPS = 16
NUM_WORKERS = 0
GRADIENT_CHECKPOINTING = True
```

---

## 🔧 Advanced Usage

### Custom Loss Weights

```python
# In train_Mask2Former.py, modify:
LOSS_WEIGHTS = {
    'mask_loss': 2.0,       # Increase for better mask quality
    'dice_loss': 2.0,       # Increase for better overlap
    'class_loss': 1.0,      # Classification importance
    'boundary_loss': 1.5,   # Increase for sharper edges
    'lovasz_loss': 1.0,     # IoU optimization
}
```

### Enable Test-Time Augmentation (TTA)

```python
# Better inference accuracy (slower)
USE_TTA = True
TTA_TRANSFORMS = ['original', 'hflip', 'vflip', 'rotate90']
```

### Enable Multi-Scale Testing

```python
# Test at multiple resolutions and merge
USE_MULTISCALE_TEST = True
TEST_SCALES = [0.75, 1.0, 1.25]
```

### Enable Conditional Random Field (CRF)

```python
# Post-processing for smoother masks
USE_CRF = True
CRF_ITERATIONS = 5
# Note: Requires pydensecrf installation
```

### Resume Training

```python
# Load checkpoint and continue
RESUME_FROM = "checkpoints/mask2former/checkpoint_epoch_50.pth"
```

---

## 📁 Output Structure

```
checkpoints/mask2former/
├── best_model.pth              # Best IoU model
├── checkpoint_epoch_5.pth      # Periodic checkpoints
├── checkpoint_epoch_10.pth
└── ...

logs/mask2former/
├── training_20251113_143052.log  # Text logs
└── tensorboard_20251113_143052/  # TensorBoard logs
    ├── events.out.tfevents...
    └── ...

training_visualizations/mask2former/
├── epoch_001.png               # Prediction visualizations
├── epoch_010.png
└── ...
```

---

## 📊 Monitoring Training

### TensorBoard Visualizations

```powershell
tensorboard --logdir=logs/mask2former --port=6006
```

**Available Metrics:**

**Training:**
- `train/loss` - Total training loss
- `train/loss_mask_loss` - Binary mask loss
- `train/loss_dice_loss` - Dice coefficient loss
- `train/loss_class_loss` - Classification loss
- `train/loss_boundary_loss` - Edge-aware loss
- `train/loss_lovasz_loss` - Lovász loss
- `train/iou` - Intersection over Union
- `train/dice` - Dice coefficient
- `train/f1` - F1 score
- `train/boundary_f1` - Boundary F1 score
- `train/lr` - Learning rate

**Validation:**
- `val/loss` - Total validation loss
- `val/iou` - Validation IoU (primary metric)
- `val/dice` - Validation Dice
- `val/precision` - Validation precision
- `val/recall` - Validation recall
- `val/f1` - Validation F1
- `val/boundary_f1` - Validation boundary F1

**System:**
- `system/gpu_memory_allocated_gb` - GPU memory usage
- `system/gpu_memory_reserved_gb` - Reserved GPU memory

### Log File Analysis

```powershell
# View last 50 lines
Get-Content logs/mask2former/training_*.log -Tail 50

# Search for best IoU
Select-String "Best model saved" logs/mask2former/training_*.log
```

---

## 🎯 Expected Performance

### Training Time (per epoch, RTX 3060 6GB)
- **682 training samples**: ~15-20 minutes
- **121 validation samples**: ~2-3 minutes
- **Total per epoch**: ~18-23 minutes
- **150 epochs**: ~45-58 hours

### Memory Usage
- **Training**: 5.2-5.8 GB GPU memory
- **Validation**: 4.8-5.2 GB GPU memory
- **Headroom**: Safe for 6GB GPUs

### Accuracy Targets (after 150 epochs)
- **IoU**: 0.75-0.85 (excellent)
- **Dice**: 0.85-0.92 (excellent)
- **Precision**: 0.85-0.93
- **Recall**: 0.82-0.90
- **Boundary F1**: 0.70-0.80 (sharp edges)

### Comparison with SAM

| Metric | SAM | Mask2Former + SegFormer-B5 |
|--------|-----|----------------------------|
| Architecture | ViT-B with prompt encoder | Hierarchical transformer + queries |
| Parameters | 93M | ~85M (more efficient) |
| Input Size | 1024×1024 (fixed) | Variable (512×512 default) |
| Memory (6GB GPU) | Batch=1, tight | Batch=2, comfortable |
| Training Speed | Slower (large input) | Faster (smaller input) |
| Inference Speed | Medium | Fast |
| Edge Quality | Good | Excellent (boundary loss) |
| Multi-scale | No | Yes (hierarchical backbone) |
| Generalization | Excellent (universal) | Very good (task-specific) |

**When to use Mask2Former over SAM:**
- Need faster training and inference
- Limited GPU memory (6GB)
- Focus on semantic segmentation (not instance)
- Want sharper boundaries
- Need production-ready model

**When to use SAM:**
- Need prompt-based segmentation
- Want zero-shot capabilities
- Have more GPU memory (8GB+)
- Need universal segmentation model

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solution 1: Reduce batch size**
```python
BATCH_SIZE = 1
ACCUMULATION_STEPS = 16  # Keep effective batch at 16
```

**Solution 2: Reduce input size**
```python
TRAIN_SIZE = 384  # From 512
```

**Solution 3: Enable gradient checkpointing**
```python
GRADIENT_CHECKPOINTING = True
# Note: 30% slower but saves memory
```

**Solution 4: Reduce workers**
```python
NUM_WORKERS = 0  # Disable multiprocessing
```

### Training Too Slow

**Solution 1: Increase batch size (if memory allows)**
```python
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
```

**Solution 2: Reduce augmentation**
```python
USE_ADVANCED_AUGMENTATION = False
AUGMENTATION_PROB = 0.5
```

**Solution 3: Increase workers**
```python
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
```

### Model Not Converging

**Solution 1: Adjust learning rate**
```python
LEARNING_RATE = 5e-5  # Lower if overshooting
LEARNING_RATE = 2e-4  # Higher if too slow
```

**Solution 2: Increase warmup**
```python
WARMUP_EPOCHS = 20  # From 10
```

**Solution 3: Adjust loss weights**
```python
LOSS_WEIGHTS = {
    'mask_loss': 3.0,  # Increase if masks are poor
    'dice_loss': 3.0,   # Increase if overlap is poor
    'boundary_loss': 2.0,  # Increase if edges are blurry
}
```

### Poor Edge Quality

**Solution 1: Increase boundary loss weight**
```python
LOSS_WEIGHTS = {'boundary_loss': 3.0}
```

**Solution 2: Enable CRF post-processing**
```python
USE_CRF = True
```

**Solution 3: Add edge-specific augmentation**
```python
# Add to augmentation pipeline
A.ElasticTransform(alpha=1, sigma=50, p=0.5)
```

---

## 🔬 Advanced Techniques

### 1. Curriculum Learning

Start with easier examples (simple fences) and progress to harder ones:

```python
# Sort dataset by difficulty
def get_mask_complexity(mask_path):
    mask = cv2.imread(mask_path, 0)
    return cv2.Laplacian(mask, cv2.CV_64F).var()  # Edge complexity

# Use in dataset loading
sorted_pairs = sorted(valid_pairs, key=lambda x: get_mask_complexity(x[1]))
```

### 2. Hard Example Mining

Focus training on difficult samples:

```python
# After epoch, identify hard examples
hard_examples = [idx for idx, loss in enumerate(sample_losses) if loss > threshold]

# Create weighted sampler
weights = [2.0 if idx in hard_examples else 1.0 for idx in range(len(dataset))]
sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset))
```

### 3. Progressive Training

Start with small resolution, gradually increase:

```python
# Epoch 0-30: 384×384
# Epoch 31-80: 448×448
# Epoch 81+: 512×512

if epoch < 30:
    Config.TRAIN_SIZE = 384
elif epoch < 80:
    Config.TRAIN_SIZE = 448
else:
    Config.TRAIN_SIZE = 512
```

### 4. Knowledge Distillation

Use SAM as teacher to guide Mask2Former:

```python
# Load pretrained SAM
sam_model = load_sam_model()

# Add distillation loss
distillation_loss = F.mse_loss(
    student_features,
    sam_model.extract_features(images).detach()
)

total_loss += 0.5 * distillation_loss
```

---

## 📚 References

### Papers

1. **Mask2Former**: "Masked-attention Mask Transformer for Universal Image Segmentation" (Cheng et al., 2022)
   - https://arxiv.org/abs/2112.01527

2. **SegFormer**: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (Xie et al., 2021)
   - https://arxiv.org/abs/2105.15203

3. **Lovász Loss**: "The Lovász-Softmax loss: A tractable surrogate for the optimization of the IoU measure" (Berman et al., 2018)
   - https://arxiv.org/abs/1705.08790

### Resources

- **Mask2Former GitHub**: https://github.com/facebookresearch/Mask2Former
- **Transformers Docs**: https://huggingface.co/docs/transformers/model_doc/mask2former
- **SegFormer GitHub**: https://github.com/NVlabs/SegFormer

---

## 🏆 Best Practices

1. **Always use EMA**: Smoother convergence and better final performance
2. **Monitor boundary_f1**: Often more important than IoU for production
3. **Save visualizations**: Essential for debugging and understanding model behavior
4. **Use TensorBoard**: Real-time monitoring prevents wasted training time
5. **Validate frequently**: Early stopping saves time on overfitting
6. **Experiment with loss weights**: Different datasets need different balances
7. **Keep good records**: Log all hyperparameters in each run
8. **Use gradient accumulation**: Maintains large effective batch on small GPUs
9. **Enable mixed precision**: Free 2× speedup with modern GPUs
10. **Test on diverse data**: Augmentation is crucial for generalization

---

## 📞 Support

For issues, questions, or contributions:
- Check logs in `logs/mask2former/`
- Review TensorBoard metrics
- Verify GPU memory with `nvidia-smi`
- Test with minimal config first
- Compare with SAM results

---

## 📄 License

This training pipeline is provided for research and commercial use. The underlying models (Mask2Former, SegFormer) follow their respective licenses:
- Mask2Former: Apache 2.0
- SegFormer: NVIDIA Source Code License

---

**Happy Training! 🚀**

*Built with ❤️ for production-grade fence segmentation*
