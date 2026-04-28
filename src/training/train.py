import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import FenceDataset from src/datasets/dataset.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'datasets'))
from dataset import FenceDataset

# ============================================================================
# CONFIGURATIONS
# ============================================================================
IMAGES_DIR = 'data/images'
MASKS_DIR = 'data/masks'
OUTPUT_DIR = 'export'
BATCH_SIZE = 2
NUM_EPOCHS = 50
LR = 1e-4
STEP_SIZE = 10
GAMMA = 0.1
PATIENCE = 5
TRAIN_VAL_SPLIT = 0.85  # 85% training, 15% validation
INPUT_SIZE = 256
NUM_WORKERS = 3  # Optimized for 16-core CPU
DEVICE = torch.device('cpu')  # Force CPU

# ============================================================================
# SETUP
# ============================================================================
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")
print(f"Number of data loader workers: {NUM_WORKERS}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, 'logs'))

# Validate data directories
if not os.path.exists(IMAGES_DIR) or not os.listdir(IMAGES_DIR):
    raise ValueError(f"No images found in {IMAGES_DIR}")
if not os.path.exists(MASKS_DIR) or not os.listdir(MASKS_DIR):
    raise ValueError(f"No masks found in {MASKS_DIR}")

# ============================================================================
# DATASET AND DATALOADER
# ============================================================================
print("\n" + "="*70)
print("PREPARING DATASETS")
print("="*70)

# Create separate dataset instances for train and validation
# This allows different augmentation settings for each
train_dataset = FenceDataset(IMAGES_DIR, MASKS_DIR, INPUT_SIZE, train=True)
val_dataset = FenceDataset(IMAGES_DIR, MASKS_DIR, INPUT_SIZE, train=False)

# Create train/val split indices
total_samples = len(train_dataset)
train_size = int(TRAIN_VAL_SPLIT * total_samples)
val_size = total_samples - train_size

# Use fixed seed for reproducibility
torch.manual_seed(42)
indices = torch.randperm(total_samples).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

print(f"Total samples: {total_samples}")
print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")

# Create subsets
train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,  # CPU training
    persistent_workers=True  # Keep workers alive between epochs
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True
)

print(f"Training batches per epoch: {len(train_loader)}")
print(f"Validation batches per epoch: {len(val_loader)}")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
print("\n" + "="*70)
print("INITIALIZING MODEL")
print("="*70)

model = deeplabv3_resnet50(pretrained=True)

# Modify output layers for binary segmentation (2 classes: background, fence)
model.classifier[-1] = nn.Conv2d(256, 2, kernel_size=1)
model.aux_classifier[-1] = nn.Conv2d(256, 2, kernel_size=1)

model = model.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
criterion = nn.CrossEntropyLoss()

best_loss = float('inf')
patience_counter = 0

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

for epoch in range(1, NUM_EPOCHS + 1):
    # ========================================================================
    # TRAINING PHASE
    # ========================================================================
    model.train()
    epoch_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)['out']  # DeepLabV3 returns dict with 'out' key
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} - Training Loss: {avg_train_loss:.4f}")
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    
    # ========================================================================
    # VALIDATION PHASE
    # ========================================================================
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]")
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    
    # ========================================================================
    # LEARNING RATE SCHEDULING
    # ========================================================================
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning_Rate', current_lr, epoch)
    print(f"Learning Rate: {current_lr:.6f}")
    
    # ========================================================================
    # MODEL CHECKPOINTING
    # ========================================================================
    # Save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, best_model_path)
        patience_counter = 0
        print(f"✓ Saved new best model (loss: {best_loss:.4f})")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{PATIENCE})")
    
    # Save latest checkpoint
    latest_checkpoint_path = os.path.join(OUTPUT_DIR, 'latest_checkpoint.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
    }, latest_checkpoint_path)
    
    # ========================================================================
    # EARLY STOPPING
    # ========================================================================
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered after {epoch} epochs")
        break
    
    print("-" * 70)

# ============================================================================
# TRAINING COMPLETE
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best validation loss: {best_loss:.4f}")
print(f"Model saved to: {OUTPUT_DIR}")

writer.close()
