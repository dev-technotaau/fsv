import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os

# Reuse your existing dataset loader (src/datasets/dataset.py)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'datasets'))
from dataset import FenceDataset

# --- CONFIG ---
IMAGES_DIR = 'data/images'
MASKS_DIR = 'data/masks'
MODEL_SAVE_PATH = 'models/pytorch/best_fence_model.pth'
INPUT_SIZE = 512 
BATCH_SIZE = 4
EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. IMPROVED MODEL: U-Net++ with EfficientNet encoder ---
# U-Net++ is often better at capturing thin, complex structures like fences than DeepLab.
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4", # Stronger encoder than standard ResNet
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,                      # Binary segmentation (fence vs background)
    activation=None                 # We'll use BCEWithLogitsLoss later
)

# --- 2. ROBUST LOSS FUNCTION (Crucial for thin objects) ---
# Dice Loss handles extreme class imbalance (fences are <5% of most images)
dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss = nn.BCEWithLogitsLoss()

def robust_criterion(pred, target):
    # Combine losses: BCE for overall pixel accuracy, Dice for fence shape overlap
    return 0.5 * bce_loss(pred, target.float()) + 0.5 * dice_loss(pred, target)

# --- SETUP ---
model.to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler() # For mixed precision training

train_ds = FenceDataset(IMAGES_DIR, MASKS_DIR, INPUT_SIZE, train=True)
val_ds = FenceDataset(IMAGES_DIR, MASKS_DIR, INPUT_SIZE, train=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

best_loss = float('inf')

print(f"Starting robust training on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE).unsqueeze(1) # Needs [B, 1, H, W] shape
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = robust_criterion(outputs, masks)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE).unsqueeze(1)
            outputs = model(images)
            loss = robust_criterion(outputs, masks)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✓ Saved new best model")