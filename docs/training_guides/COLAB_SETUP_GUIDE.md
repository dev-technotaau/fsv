# SegFormer-B5 Google Colab Training Setup Guide
## Complete Setup Instructions for T4 GPU Training

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step-by-Step Setup](#step-by-step-setup)
3. [Training Configuration](#training-configuration)
4. [Expected Performance](#expected-performance)
5. [Troubleshooting](#troubleshooting)

---

## 🎯 Prerequisites

### Required Files
- `train_SegFormerB5_PREMIUM_googleColab.py` - Training script
- `requirements_segformerb5_colab.txt` - Dependencies
- Your fence detection dataset (images + masks)

### Google Colab Requirements
- Google Account
- Google Colab (free tier works, Pro recommended for longer sessions)
- T4 GPU runtime (free tier provides ~12 hours)

---

## 🚀 Step-by-Step Setup

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → New Notebook**
3. Name it: `SegFormer_B5_Fence_Training`

### Step 2: Enable GPU Runtime
```python
# Run this first to check GPU
!nvidia-smi
```

If no GPU is shown:
1. Click **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU**
3. Click **Save**
4. Runtime will restart

### Step 3: Upload Training Script and Requirements

**Option A: Upload Files**
```python
from google.colab import files

# Upload training script
print("Upload train_SegFormerB5_PREMIUM_googleColab.py:")
uploaded = files.upload()

# Upload requirements
print("\nUpload requirements_segformerb5_colab.txt:")
uploaded = files.upload()
```

**Option B: Clone from GitHub** (if you have the repo)
```bash
!git clone https://github.com/your-username/your-repo.git
%cd your-repo/training
```

### Step 4: Install Dependencies
```bash
# Install all required packages
!pip install -q -r requirements_segformerb5_colab.txt

# Verify installation
import torch
import transformers
import albumentations
import cv2

print(f"✅ PyTorch Version: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ Albumentations: {albumentations.__version__}")
print(f"✅ OpenCV: {cv2.__version__}")
```

**Expected Output:**
```
✅ PyTorch Version: 2.1.0+cu121
✅ CUDA Available: True
✅ GPU: Tesla T4
✅ GPU Memory: 15.90 GB
✅ Transformers: 4.35.0
✅ Albumentations: 1.3.1
✅ OpenCV: 4.8.1
```

### Step 5: Upload Dataset

**Option A: Upload ZIP File**
```python
from google.colab import files
import zipfile

# Upload dataset ZIP
print("Upload your dataset.zip (containing data/images and data/masks):")
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')

# Verify
!ls -la data/images | head -10
!ls -la data/masks | head -10
```

**Option B: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset from Drive
!mkdir -p data/images data/masks
!cp -r /content/drive/MyDrive/fence_dataset/images/* ./data/images/
!cp -r /content/drive/MyDrive/fence_dataset/masks/* ./data/masks/

# Verify
!echo "Images: $(ls data/images | wc -l)"
!echo "Masks: $(ls data/masks | wc -l)"
```

**Option C: Download from URL**
```bash
!wget -q https://your-storage-url/fence_dataset.zip
!unzip -q fence_dataset.zip -d .
!ls -la data/
```

### Step 6: Create Required Directories
```bash
!mkdir -p checkpoints/segformerb5_colab
!mkdir -p data/images data/masks
```

### Step 7: Verify Dataset Structure
```python
import os

# Check dataset
images = len([f for f in os.listdir('data/images') if f.endswith(('.jpg', '.png', '.jpeg'))])
masks = len([f for f in os.listdir('data/masks') if f.endswith('.png')])

print(f"📊 Dataset Statistics:")
print(f"   Total Images: {images}")
print(f"   Total Masks: {masks}")
print(f"   Match: {'✅ YES' if images == masks else '❌ NO'}")

if images != masks:
    print("\n⚠️  Warning: Image and mask counts don't match!")
```

### Step 8: Start Training
```bash
# Run training script
!python train_SegFormerB5_PREMIUM_googleColab.py
```

### Step 9: Monitor Training (Optional)
Open a new code cell and run:
```python
# Real-time log monitoring
!tail -f checkpoints/segformerb5_colab/training.log
```

Press `Ctrl+C` in the cell to stop monitoring.

---

## ⚙️ Training Configuration

### Optimized for T4 16GB GPU

```python
# From Config class in train_SegFormerB5_PREMIUM_googleColab.py

INPUT_SIZE = 640          # Full native resolution
BATCH_SIZE = 4            # Optimized for 16GB VRAM
ACCUMULATION_STEPS = 4    # Effective batch = 16
EPOCHS = 100              # Full training run
LEARNING_RATE = 8e-5      # Tuned for T4
NUM_WORKERS = 4           # Colab has good CPUs
```

### Dataset Split
- **Training**: 80% of images
- **Validation**: 20% of images

### Memory Usage
- **Model Size**: 84.6M parameters (~330 MB)
- **Expected VRAM**: 12-14 GB during training
- **Available VRAM**: 15.9 GB on T4

---

## 📈 Expected Performance

### Training Speed on T4
- **Time per Epoch**: ~15-20 seconds (depends on dataset size)
- **Total Training Time**: ~2-3 hours for 100 epochs
- **Batch Processing**: ~0.8-1.2 seconds per batch

### Memory Footprint
- **Peak VRAM Usage**: ~13-14 GB
- **RAM Usage**: ~4-6 GB
- **Disk Usage**: ~2-3 GB (checkpoints)

### Target Metrics
- **Best IoU**: >0.90 (target)
- **Dice Score**: >0.92
- **Precision**: >0.88
- **Recall**: >0.85

### Checkpoint Saving
- **Frequency**: Every 10 epochs
- **Kept Checkpoints**: Last 3 checkpoints
- **Best Model**: Always saved when IoU improves

---

## 💾 Download Trained Model

After training completes, download your trained model:

```python
from google.colab import files
import shutil

# Zip the best model
shutil.make_archive(
    'segformer_b5_best_model', 
    'zip', 
    './checkpoints/segformerb5_colab/best_model'
)

# Download
files.download('segformer_b5_best_model.zip')
```

---

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

**Option A: Reduce Batch Size**
```python
# In train_SegFormerB5_PREMIUM_googleColab.py, change:
BATCH_SIZE = 2              # Reduce from 4
ACCUMULATION_STEPS = 8      # Increase to maintain effective batch
```

**Option B: Reduce Input Size**
```python
INPUT_SIZE = 512            # Reduce from 640
```

**Option C: Clear Cache**
```python
import torch
torch.cuda.empty_cache()
```

---

### Issue 2: Runtime Disconnected

**Symptoms:**
- Training interrupted after ~90 minutes
- "Runtime disconnected" message

**Solutions:**

**Option A: Keep Browser Active**
```javascript
// Run this in browser console (F12) to prevent disconnection
function KeepAlive() {
    console.log("Keeping Colab alive...");
}
setInterval(KeepAlive, 60000);  // Every minute
```

**Option B: Use Colab Pro**
- Longer runtime (24 hours)
- Background execution
- Faster GPUs (A100, V100)

**Option C: Regular Checkpoint Downloads**
```python
# Add this to your notebook to auto-download checkpoints
from google.colab import files
import shutil
import os

def backup_checkpoint(epoch):
    if epoch % 20 == 0:  # Every 20 epochs
        checkpoint_path = f'./checkpoints/segformerb5_colab/checkpoint_epoch_{epoch}'
        if os.path.exists(checkpoint_path):
            shutil.make_archive(f'checkpoint_epoch_{epoch}', 'zip', checkpoint_path)
            files.download(f'checkpoint_epoch_{epoch}.zip')
```

---

### Issue 3: Slow Data Loading

**Symptoms:**
- Training speed <1 it/s
- CPU at 100%

**Solutions:**

**Option A: Reduce Workers**
```python
NUM_WORKERS = 2             # Reduce from 4
```

**Option B: Reduce Prefetch**
```python
PREFETCH_FACTOR = 2         # Reduce from 3
```

**Option C: Disable Augmentation** (temporary testing)
```python
# In get_augmentation(), temporarily return:
return A.Compose([
    A.Resize(Config.INPUT_SIZE, Config.INPUT_SIZE),
])
```

---

### Issue 4: Disk Space Full

**Error:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**

**Option A: Keep Fewer Checkpoints**
```python
KEEP_LAST_N_CHECKPOINTS = 1  # Reduce from 3
```

**Option B: Clean Up**
```bash
# Remove old checkpoints manually
!rm -rf checkpoints/segformerb5_colab/checkpoint_epoch_*

# Keep only best model
```

**Option C: Download and Delete**
```python
# After each checkpoint save, download and delete
from google.colab import files
import shutil

# Zip and download
shutil.make_archive('checkpoint', 'zip', './checkpoints/segformerb5_colab/')
files.download('checkpoint.zip')

# Clean up
!rm -rf checkpoints/segformerb5_colab/checkpoint_epoch_*
```

---

### Issue 5: Dataset Upload Failed

**Symptoms:**
- Upload stuck at 99%
- Upload timeout

**Solutions:**

**Option A: Use Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive instead of uploading
!cp -r /content/drive/MyDrive/fence_dataset/* ./data/
```

**Option B: Split Upload**
```python
# Upload in parts
# images_part1.zip, images_part2.zip, etc.
```

**Option C: Download from Cloud**
```bash
# Use wget, gdown, or cloud storage
!pip install -q gdown
!gdown --id YOUR_GOOGLE_DRIVE_FILE_ID
```

---

## 📊 Monitor Training Progress

### View Training Logs
```python
# In a separate cell
!tail -n 50 checkpoints/segformerb5_colab/training.log
```

### Check GPU Usage
```bash
# Real-time GPU monitoring
!watch -n 1 nvidia-smi
```

### Plot Training Curves (After Training)
```python
import json
import matplotlib.pyplot as plt

# Load checkpoint info
checkpoints = []
for i in range(10, 101, 10):
    path = f'checkpoints/segformerb5_colab/checkpoint_epoch_{i}/checkpoint_info.json'
    try:
        with open(path, 'r') as f:
            checkpoints.append(json.load(f))
    except:
        pass

# Extract metrics
epochs = [c['epoch'] for c in checkpoints]
train_iou = [c['train_iou'] for c in checkpoints]
val_iou = [c['val_iou'] for c in checkpoints]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_iou, label='Train IoU', marker='o')
plt.plot(epochs, val_iou, label='Val IoU', marker='s')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('SegFormer-B5 Training Progress')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🎯 Next Steps After Training

### 1. Download Model
```python
from google.colab import files
files.download('segformer_b5_best_model.zip')
```

### 2. Test Inference (in Colab)
```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
from PIL import Image
import numpy as np

# Load model
model = SegformerForSemanticSegmentation.from_pretrained(
    './checkpoints/segformerb5_colab/best_model'
)
processor = SegformerImageProcessor.from_pretrained(
    './checkpoints/segformerb5_colab/best_model'
)
model.eval()

# Test on sample image
image = Image.open('data/images/sample.jpg')
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get prediction
pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
print(f"Prediction shape: {pred.shape}")
print(f"Unique values: {np.unique(pred)}")
```

### 3. Export to ONNX
```bash
!python export_segformer_to_onnx.py --checkpoint ./checkpoints/segformerb5_colab/best_model
```

### 4. Deploy to Production
- Use ONNX Runtime for inference
- Deploy to web server
- Integrate with your application

---

## 💡 Tips for Best Results

### Data Quality
- ✅ Use high-quality images (min 640x640)
- ✅ Ensure accurate mask annotations
- ✅ Balance dataset (similar # of fence/background pixels)
- ✅ Include diverse fence types and conditions

### Training Strategy
- ✅ Start with default config
- ✅ Monitor validation IoU (should plateau after 60-80 epochs)
- ✅ If overfitting, increase LABEL_SMOOTHING or WEIGHT_DECAY
- ✅ If underfitting, train for more epochs or increase LEARNING_RATE

### GPU Optimization
- ✅ Use full 640x640 resolution on T4
- ✅ Keep batch size at 4 for optimal speed
- ✅ Enable AMP for 2-3x speedup
- ✅ Monitor GPU temperature (should stay <80°C)

---

## 📞 Support

If you encounter issues not covered here:
1. Check the error message carefully
2. Review Colab's runtime logs
3. Verify dataset structure
4. Try with a smaller test dataset first
5. Check GPU memory with `!nvidia-smi`

---

**Happy Training! 🚀**
