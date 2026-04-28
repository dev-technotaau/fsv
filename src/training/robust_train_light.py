import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os

# Import FenceDataset from src/datasets/dataset.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'datasets'))
try:
    from vision_guard.training.dataset import FenceDataset
except ImportError:
    from dataset import FenceDataset

# --- LOW MEMORY CONFIGURATION (For 16GB CPU) ---
IMAGES_DIR = 'data/images'
MASKS_DIR = 'data/masks'
# Fallback for running from different directories
if not os.path.exists(IMAGES_DIR):
     IMAGES_DIR = 'vision_guard/data/images'
     MASKS_DIR = 'vision_guard/data/masks'

MODEL_SAVE_PATH = 'models/pytorch/best_fence_model_light.pth'
INPUT_SIZE = 384      # Reduced from 512 to save RAM
BATCH_SIZE = 2        # Reduced from 4 to save RAM
EPOCHS = 50
DEVICE = 'cpu'        # Forcing CPU since CUDA failed previously
NUM_WORKERS = 0       # 0 workers uses less RAM (no graphical forks)

# --- 1. LIGHTER ROBUST MODEL ---
# EfficientNet-B0 is much smaller than B4, better for CPU/Low-RAM
print("Initializing lightweight U-Net++ (EfficientNet-B0)...")
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b0", 
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)

# --- 2. LOSS ---
dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss = nn.BCEWithLogitsLoss()

def robust_criterion(pred, target):
    return 0.5 * bce_loss(pred, target.float()) + 0.5 * dice_loss(pred, target)

# --- SETUP ---
model.to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

print(f"Loading dataset from {IMAGES_DIR}...")
train_ds = FenceDataset(IMAGES_DIR, MASKS_DIR, INPUT_SIZE, train=True)
val_ds = FenceDataset(IMAGES_DIR, MASKS_DIR, INPUT_SIZE, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

best_loss = float('inf')

print(f"Starting LIGHTWEIGHT training on {DEVICE}...")
print(f"Config: Input={INPUT_SIZE}px | Batch={BATCH_SIZE} | Workers={NUM_WORKERS}")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    # Tqdm with less frequent updates to save a tiny bit of CPU
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", mininterval=2.0)
    
    for i, (images, masks) in enumerate(pbar):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).unsqueeze(1)
        
        # No AMP on CPU (it's often slower and can use more RAM sometimes)
        outputs = model(images)
        loss = robust_criterion(outputs, masks)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Memory cleanup every few batches
        if i % 10 == 0:
             torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).unsqueeze(1)
            outputs = model(images)
            loss = robust_criterion(outputs, masks)
            val_loss += loss.item()
            
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✓ Saved new best model (val_loss: {best_loss:.4f})")