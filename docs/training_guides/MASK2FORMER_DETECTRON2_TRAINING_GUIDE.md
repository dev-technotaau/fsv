# Mask2Former + Detectron2 + SegFormer-B5 Training Guide

## 🎯 Overview

This is an **ultra-enterprise level** training script that combines three powerful technologies:

1. **Detectron2** - Facebook's production-ready object detection framework
2. **Mask2Former** - State-of-the-art universal image segmentation model
3. **SegFormer-B5** - Hierarchical Vision Transformer backbone (82M parameters)

### Architecture Highlights

```
Input Image (512x512)
    ↓
SegFormer-B5 Backbone (82M params)
    ├─ Multi-scale features: res2, res3, res4, res5
    ├─ Efficient self-attention
    └─ Hierarchical feature extraction
    ↓
Mask2Former Pixel Decoder
    ├─ Multi-scale deformable attention
    ├─ Feature pyramid network
    └─ Mask feature generation (256 dims)
    ↓
Transformer Decoder (9 layers)
    ├─ 100 object queries
    ├─ Masked attention
    ├─ Cross-attention with pixel features
    └─ Self-attention between queries
    ↓
Prediction Heads
    ├─ Mask predictions (per query)
    ├─ Class predictions (per query)
    └─ Deep supervision (all layers)
    ↓
Output: Instance Segmentation Masks
```

## 🚀 Key Features

### Production-Ready Infrastructure
✅ Detectron2's battle-tested training pipeline  
✅ COCO-format dataset support  
✅ Multi-GPU distributed training (DDP)  
✅ Advanced checkpoint management  
✅ Comprehensive evaluation metrics  
✅ Model zoo integration  

### Advanced Architecture
✅ SegFormer-B5 backbone (82M parameters)  
✅ Hierarchical Vision Transformer  
✅ Multi-scale deformable attention  
✅ Query-based mask prediction  
✅ Masked attention mechanisms  
✅ Deep supervision (all decoder layers)  

### Optimization & Acceleration
✅ Mixed precision training (AMP)  
✅ Gradient accumulation  
✅ GPU memory optimization  
✅ Efficient data loading with caching  
✅ Synchronized batch normalization  
✅ Gradient clipping  

### Data Augmentation
✅ Detectron2 augmentations (LSJ, flips, resizing)  
✅ Albumentations++ (weather, blur, noise, geometric)  
✅ Test-time augmentation (TTA)  
✅ Multi-scale testing  

### Loss Functions
✅ Mask loss (BCE + Dice)  
✅ Focal loss (class imbalance)  
✅ Boundary loss (edge detection)  
✅ Lovász-Softmax loss  
✅ Classification loss  
✅ Deep supervision  

### Tracking & Monitoring
✅ TensorBoard logging  
✅ Weights & Biases integration  
✅ GPU memory monitoring  
✅ System resource tracking  
✅ Live training visualizations  
✅ Comprehensive metrics logging  

### Evaluation
✅ COCO evaluation (mAP, mAR)  
✅ Panoptic quality (PQ, SQ, RQ)  
✅ IoU, Dice, F1 scores  
✅ Boundary F1 score  
✅ Per-class metrics  

## 📋 Requirements

### Hardware Requirements

**Minimum (Budget Training):**
- GPU: NVIDIA RTX 3060 (6GB VRAM)
- RAM: 16GB
- Storage: 50GB free
- Training time: ~24-36 hours

**Recommended (Fast Training):**
- GPU: NVIDIA RTX 3090/4090 (24GB VRAM)
- RAM: 32GB
- Storage: 100GB free (SSD)
- Training time: ~8-12 hours

**Optimal (Production):**
- GPU: 4x NVIDIA A100 (40GB VRAM each)
- RAM: 128GB
- Storage: 500GB NVMe SSD
- Training time: ~2-4 hours

### Software Requirements

- **OS:** Windows 10/11, Linux (Ubuntu 20.04+), macOS (limited)
- **Python:** 3.8, 3.9, 3.10, or 3.11
- **CUDA:** 11.8 or 12.1 (match with PyTorch)
- **cuDNN:** 8.x (comes with CUDA)
- **Build Tools:** 
  - Windows: Visual Studio 2019/2022 with C++ tools
  - Linux: GCC 7.5+

## 🔧 Installation

### Step 1: Install CUDA Toolkit

**Windows:**
1. Download from https://developer.nvidia.com/cuda-downloads
2. Install CUDA 11.8 or 12.1
3. Verify: `nvcc --version`

**Linux:**
```bash
# Ubuntu 20.04/22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Step 2: Install Build Tools

**Windows:**
1. Download Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
2. Install with C++ build tools, Windows SDK, and CMake

**Linux:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake git ninja-build
```

### Step 3: Run Setup Script

**Windows PowerShell:**
```powershell
# Set execution policy (run as Administrator)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run setup script
.\setup_mask2former_detectron2.ps1
```

**Linux/Mac:**
```bash
# Create and activate virtual environment
python3 -m venv venv_detectron2
source venv_detectron2/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install Mask2Former
git clone https://github.com/facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../..

# Install remaining requirements
pip install -r requirements_mask2former_detectron2.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import detectron2; print(f'Detectron2: {detectron2.__version__}')"
python -c "from mask2former import add_maskformer2_config; print('Mask2Former: OK')"
python -c "from transformers import SegformerModel; print('SegFormer: OK')"
```

## 📊 Dataset Preparation

### Directory Structure

```
training/
├── data/
│   ├── images/          # PNG images
│   ├── masks/           # PNG masks (binary)
│   ├── annotations_train.json  # Auto-generated
│   └── annotations_val.json    # Auto-generated
├── checkpoints/
│   └── mask2former_detectron2/
├── logs/
│   └── mask2former_detectron2/
└── training_visualizations/
    └── mask2former_detectron2/
```

### COCO Format Annotations

The script **automatically generates** COCO format annotations from your image-mask pairs:

```python
# Automatically called in main()
create_coco_annotations()  # Creates annotations_train.json and annotations_val.json
```

**Manual Generation (if needed):**
```python
python -c "from train_Mask2Former_Detectron2 import create_coco_annotations; create_coco_annotations()"
```

### Dataset Statistics

Your dataset: **803 image-mask pairs**
- **Class imbalance:** 2.65% fence pixels, 97.35% background
- **Solution:** Heavy class weighting (background: 0.05, fence: 20.0)
- **Train/Val split:** 85%/15% (682 train, 121 val)

## ⚙️ Configuration

All settings are in the `Config` class. Key parameters:

### Model Configuration
```python
BACKBONE_NAME = "segformer_b5"           # SegFormer-B5 (82M params)
NUM_QUERIES = 100                        # Object queries
DEC_LAYERS = 9                           # Transformer decoder layers
HIDDEN_DIM = 256                         # Hidden dimension
```

### Training Hyperparameters
```python
INPUT_SIZE = 512                         # Input image size
BATCH_SIZE = 2                           # Per GPU (6GB)
ACCUMULATION_STEPS = 8                   # Effective batch = 16
MAX_ITER = 40000                         # Training iterations
BASE_LR = 0.0001                         # Base learning rate
BACKBONE_LR_MULTIPLIER = 0.1             # Lower LR for backbone
```

### Loss Weights (Optimized for Imbalance)
```python
LOSS_WEIGHTS = {
    'mask_loss': 5.0,
    'dice_loss': 3.0,
    'class_loss': 2.0,
    'boundary_loss': 2.0,
    'lovasz_loss': 2.0,
    'focal_loss': 3.0,
}
CLASS_WEIGHT = [0.05, 20.0]              # [background, fence]
```

### GPU Memory Optimization
```python
USE_AMP = True                           # Mixed precision
EMPTY_CACHE_PERIOD = 100                 # Clear cache every N iters
GRADIENT_CHECKPOINTING = False           # Not supported by Mask2Former
```

### Augmentation
```python
USE_DETECTRON2_AUG = True                # Detectron2 augmentations
USE_ALBUMENTATIONS = True                # Albumentations++
AUGMENTATION_PROB = 0.8                  # Apply aug probability
USE_LSJ = True                           # Large Scale Jittering
```

## 🏃 Training

### Basic Training

```bash
# Activate environment
.\venv_detectron2\Scripts\Activate.ps1   # Windows
source venv_detectron2/bin/activate      # Linux/Mac

# Start training
python train_Mask2Former_Detectron2.py
```

### Multi-GPU Training

```bash
# 2 GPUs
python train_Mask2Former_Detectron2.py --num-gpus 2

# 4 GPUs
python train_Mask2Former_Detectron2.py --num-gpus 4
```

### Resume Training

```bash
# Auto-resume from last checkpoint
python train_Mask2Former_Detectron2.py --resume
```

### Custom Configuration

Modify `Config` class in `train_Mask2Former_Detectron2.py`:

```python
# For 12GB GPU
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4

# For 24GB GPU
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2

# Faster convergence (if time-limited)
MAX_ITER = 20000
BASE_LR = 0.0002
```

## 📈 Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=logs/mask2former_detectron2

# Open browser: http://localhost:6006
```

**Metrics logged:**
- Loss components (mask, dice, boundary, focal, etc.)
- Learning rates (backbone, decoder)
- GPU memory usage
- Training speed (iterations/sec)
- Validation metrics (mAP, IoU, Dice)

### Weights & Biases

Enable in Config:
```python
USE_WANDB = True
WANDB_PROJECT = "fence-staining-mask2former-detectron2"
```

Then:
```bash
wandb login
python train_Mask2Former_Detectron2.py
```

### Real-time GPU Monitoring

```bash
# Continuous GPU monitoring
watch -n 1 nvidia-smi

# Or Python
python -c "from train_Mask2Former_Detectron2 import get_gpu_memory_info; import json; print(json.dumps(get_gpu_memory_info(), indent=2))"
```

## 📊 Evaluation

### During Training

Automatic evaluation every `EVAL_PERIOD` iterations (default: 1000):
- COCO mAP (mask)
- IoU, Dice scores
- Panoptic quality (PQ, SQ, RQ)
- Boundary F1

### After Training

```bash
# Evaluate best model
python -c "
from detectron2.config import get_cfg
from train_Mask2Former_Detectron2 import setup_cfg, Mask2FormerTrainer

cfg = setup_cfg()
cfg.MODEL.WEIGHTS = 'checkpoints/mask2former_detectron2/model_best.pth'
trainer = Mask2FormerTrainer(cfg)
trainer.test(cfg, trainer.model)
"
```

### Custom Inference

```python
import torch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from train_Mask2Former_Detectron2 import setup_cfg

# Setup
cfg = setup_cfg()
cfg.MODEL.WEIGHTS = 'checkpoints/mask2former_detectron2/model_best.pth'
model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()

# Inference
import cv2
image = cv2.imread('test_image.png')
height, width = image.shape[:2]
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

with torch.no_grad():
    predictions = model([{"image": image, "height": height, "width": width}])

# Extract masks
masks = predictions[0]["instances"].pred_masks.cpu().numpy()
```

## 🎯 Expected Results

### Training Time (6GB GPU)
- **Total time:** ~24-30 hours
- **Iterations:** 40,000
- **Checkpoints:** Every 2,000 iterations
- **Evaluations:** Every 1,000 iterations

### Expected Metrics
- **mAP@50:** 85-92%
- **mAP@50-95:** 70-80%
- **IoU:** 85-90%
- **Dice:** 88-93%
- **Boundary F1:** 80-88%

### Model Size
- **Parameters:** ~120M (82M backbone + 38M decoder)
- **Checkpoint:** ~480MB
- **ONNX export:** ~500MB
- **Inference speed:** 15-25 FPS (512x512, RTX 3060)

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `BATCH_SIZE = 1`
2. Increase gradient accumulation: `ACCUMULATION_STEPS = 16`
3. Reduce input size: `INPUT_SIZE = 384`
4. Enable CPU offloading (slower)

### Detectron2 Install Fails

**Windows:**
- Install Visual Studio Build Tools
- Ensure CUDA toolkit is installed
- Use pre-built wheels: `pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html`

**Linux:**
- Install build essentials: `sudo apt-get install build-essential`
- Install ninja: `pip install ninja`

### Mask2Former CUDA Ops Fail

**Windows:**
- Use WSL (Windows Subsystem for Linux)
- Or compile manually with Visual Studio

**Linux:**
```bash
cd Mask2Former/mask2former/modeling/pixel_decoder/ops
python setup.py build install
```

### Slow Training

**Solutions:**
1. Enable benchmark mode (default): `torch.backends.cudnn.benchmark = True`
2. Increase workers: `NUM_WORKERS = 8`
3. Use SSD for dataset storage
4. Reduce evaluation frequency: `EVAL_PERIOD = 2000`
5. Disable visualization: `VIS_PERIOD = None`

### COCO API Issues (Windows)

```bash
pip uninstall pycocotools
pip install pycocotools-windows
```

## 📦 Model Export

### ONNX Export

```python
from detectron2.export import TracingAdapter
from detectron2.modeling import build_model
from train_Mask2Former_Detectron2 import setup_cfg

cfg = setup_cfg()
model = build_model(cfg)

# Export
adapter = TracingAdapter(model, inputs=[{"image": torch.randn(3, 512, 512)}])
torch.onnx.export(
    adapter,
    (torch.randn(1, 3, 512, 512),),
    "mask2former.onnx",
    opset_version=14,
    input_names=["image"],
    output_names=["masks", "scores"],
    dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}}
)
```

### TorchScript Export

```python
traced_model = torch.jit.trace(model, example_inputs)
torch.jit.save(traced_model, "mask2former.pt")
```

## 📝 Advanced Tips

### Hyperparameter Tuning

For best results on your dataset:
1. Start with default config
2. Monitor validation metrics
3. Adjust learning rate (try 0.00005 - 0.0002)
4. Tune class weights based on validation performance
5. Experiment with augmentation strength

### Multi-Scale Training

Enable for better accuracy:
```python
AUG_MIN_SIZE_TRAIN = (384, 448, 512, 576, 640)
USE_MULTISCALE = True
```

### Test-Time Augmentation

For best inference accuracy:
```python
USE_TTA = True
TTA_FLIP_HORIZONTAL = True
TTA_FLIP_VERTICAL = True
TTA_SCALES = [0.75, 1.0, 1.25]
```

### Early Stopping

```python
EARLY_STOPPING = True
PATIENCE = 20          # Stop if no improvement for 20 eval periods
MIN_DELTA = 0.001     # Minimum improvement threshold
```

## 🎓 References

1. **Mask2Former:** [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)
2. **Detectron2:** [Facebook Research Detectron2](https://github.com/facebookresearch/detectron2)
3. **SegFormer:** [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## 📧 Support

For issues:
1. Check logs in `logs/mask2former_detectron2/`
2. Review TensorBoard for training curves
3. Check GPU memory with `nvidia-smi`
4. Verify dataset with COCO API

## 🚀 Production Deployment

After training:
1. Export to ONNX for deployment
2. Use TensorRT for GPU inference optimization
3. Deploy with Triton Inference Server
4. Monitor with Prometheus + Grafana

---

**Happy Training! 🎉**

For fence staining visualization, this model will provide **state-of-the-art** segmentation accuracy, enabling precise fence detection and realistic color visualization.
