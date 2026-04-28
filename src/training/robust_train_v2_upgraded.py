"""
Upgraded Robust Fence Detection Training Script v2.0
=====================================================
Improvements:
- Focal Loss for hard example mining
- Enhanced data augmentation (color jitter, blur, brightness)
- Boundary refinement loss for sharper edges
- Learning rate scheduling with warmup
- IoU metric tracking
- Better validation and early stopping
- Gradient accumulation for effective larger batch size
- Checkpoint saving with full state

Author: VisionGuard Team
Date: November 10, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
import json
from scipy.ndimage import binary_erosion

# Local imports — dataset_v2 lives in src/datasets/
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'datasets'))
try:
    from vision_guard.training.dataset_v2 import FenceDatasetV2
except ImportError:
    from dataset_v2 import FenceDatasetV2


# ========== CONFIGURATION ==========
class TrainingConfig:
    """Centralized configuration for training"""
    # Paths
    IMAGES_DIR = 'data/images'
    MASKS_DIR = 'data/masks'
    CHECKPOINT_DIR = './checkpoints'
    VIS_DIR = './training_visualizations'

    # Fallback paths
    if not os.path.exists(IMAGES_DIR):
        IMAGES_DIR = 'vision_guard/data/images'
        MASKS_DIR = 'vision_guard/data/masks'
    
    # Model Configuration
    ENCODER = "efficientnet-b1"  # Upgraded from b0 for better features
    INPUT_SIZE = 512  # Increased from 384 for better detail
    
    # Training Hyperparameters
    BATCH_SIZE = 2  # Keep low for CPU/RAM constraints
    ACCUMULATION_STEPS = 2  # Effective batch size = 2 * 2 = 4
    EPOCHS = 80  # Increased for better convergence
    INITIAL_LR = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0 if DEVICE == 'cpu' else 2
    
    # Loss weights
    FOCAL_WEIGHT = 0.4
    DICE_WEIGHT = 0.4
    BCE_WEIGHT = 0.1
    BOUNDARY_WEIGHT = 0.1
    
    # Focal Loss parameters
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Training behavior
    EARLY_STOP_PATIENCE = 15
    VAL_SPLIT = 0.15  # 15% for validation
    SAVE_VIS_EVERY = 5  # Save visualizations every N epochs
    
    # Model save path
    BEST_MODEL_PATH = 'models/pytorch/best_fence_model_v2.pth'
    LAST_MODEL_PATH = 'models/pytorch/last_fence_model_v2.pth'
    
    @classmethod
    def save_config(cls, path):
        """Save configuration to JSON - FIXED VERSION"""
        config_dict = {
            'IMAGES_DIR': cls.IMAGES_DIR,
            'MASKS_DIR': cls.MASKS_DIR,
            'CHECKPOINT_DIR': cls.CHECKPOINT_DIR,
            'VIS_DIR': cls.VIS_DIR,
            'ENCODER': cls.ENCODER,
            'INPUT_SIZE': cls.INPUT_SIZE,
            'BATCH_SIZE': cls.BATCH_SIZE,
            'ACCUMULATION_STEPS': cls.ACCUMULATION_STEPS,
            'EPOCHS': cls.EPOCHS,
            'INITIAL_LR': cls.INITIAL_LR,
            'MIN_LR': cls.MIN_LR,
            'WEIGHT_DECAY': cls.WEIGHT_DECAY,
            'DEVICE': str(cls.DEVICE),
            'NUM_WORKERS': cls.NUM_WORKERS,
            'FOCAL_WEIGHT': cls.FOCAL_WEIGHT,
            'DICE_WEIGHT': cls.DICE_WEIGHT,
            'BCE_WEIGHT': cls.BCE_WEIGHT,
            'BOUNDARY_WEIGHT': cls.BOUNDARY_WEIGHT,
            'FOCAL_ALPHA': cls.FOCAL_ALPHA,
            'FOCAL_GAMMA': cls.FOCAL_GAMMA,
            'EARLY_STOP_PATIENCE': cls.EARLY_STOP_PATIENCE,
            'VAL_SPLIT': cls.VAL_SPLIT,
            'SAVE_VIS_EVERY': cls.SAVE_VIS_EVERY,
            'BEST_MODEL_PATH': cls.BEST_MODEL_PATH,
            'LAST_MODEL_PATH': cls.LAST_MODEL_PATH,
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {path}")


# ========== UTILITY FUNCTIONS ==========
def calculate_iou(pred, target, num_classes=2):
    """
    Calculate Intersection over Union (IoU) metric.
    
    Args:
        pred: Predicted mask tensor (N, 1, H, W) or (N, H, W)
        target: Ground truth mask tensor (N, 1, H, W) or (N, H, W)
        num_classes: Number of segmentation classes
    
    Returns:
        Mean IoU across all classes
    """
    # Ensure same shape
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    # Flatten
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    
    # Convert to binary
    pred = (pred > 0.5).long()
    target = (target > 0.5).long()
    
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))  # Ignore if class not present
        else:
            ious.append((intersection / union).item())
    
    # Return mean IoU, ignoring NaN values
    ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(ious) if ious else 0.0


# ========== LOSS FUNCTIONS ==========
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    Focuses training on hard-to-classify pixels (like thin fence posts).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)  # Probability of correct classification
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class BoundaryLoss(nn.Module):
    """
    Boundary refinement loss to improve edge detection.
    Focuses on pixels near object boundaries.
    """
    def __init__(self, theta=3):
        super().__init__()
        self.theta = theta
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W)
        """
        # Compute boundary using morphological operations
        target_np = target.cpu().numpy()
        
        # Simple boundary detection using erosion
        boundaries = []
        for i in range(target.shape[0]):
            mask = target_np[i, 0] > 0.5
            
            # Skip if mask is empty
            if not mask.any():
                boundaries.append(np.zeros_like(mask))
                continue
            
            try:
                eroded = binary_erosion(mask, iterations=2)
                boundary = mask & ~eroded
                boundaries.append(boundary)
            except:
                # Fallback if erosion fails
                boundaries.append(np.zeros_like(mask))
        
        boundaries = torch.from_numpy(np.array(boundaries)).unsqueeze(1).float().to(pred.device)
        
        # Weight boundary pixels more heavily
        weights = 1.0 + boundaries * self.theta
        
        # BCE on boundary regions
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        weighted_bce = (bce * weights).mean()
        
        return weighted_bce


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components for robust training.
    """
    def __init__(self, config):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.boundary_loss = BoundaryLoss(theta=3)
        
        self.focal_weight = config.FOCAL_WEIGHT
        self.dice_weight = config.DICE_WEIGHT
        self.bce_weight = config.BCE_WEIGHT
        self.boundary_weight = config.BOUNDARY_WEIGHT
        
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total_loss = (
            self.focal_weight * focal +
            self.dice_weight * dice +
            self.bce_weight * bce +
            self.boundary_weight * boundary
        )
        
        return total_loss, {
            'focal': focal.item(),
            'dice': dice.item(),
            'bce': bce.item(),
            'boundary': boundary.item(),
            'total': total_loss.item()
        }


# ========== LEARNING RATE SCHEDULER ==========
class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, initial_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


# ========== TRAINING FUNCTIONS ==========
def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps, epoch):
    """Train for one epoch with gradient accumulation."""
    model.train()
    epoch_loss = 0
    epoch_metrics = {'focal': 0, 'dice': 0, 'bce': 0, 'boundary': 0}
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    
    optimizer.zero_grad()
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss, loss_dict = criterion(outputs, masks)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Track metrics
        epoch_loss += loss_dict['total']
        for key in epoch_metrics:
            epoch_metrics[key] += loss_dict[key]
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'dice': f"{loss_dict['dice']:.4f}"
        })
    
    # Average metrics
    n_batches = len(loader)
    avg_loss = epoch_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
    
    return avg_loss, avg_metrics


def validate_epoch(model, loader, criterion, device, epoch):
    """Validate for one epoch and compute IoU."""
    model.eval()
    epoch_loss = 0
    epoch_metrics = {'focal': 0, 'dice': 0, 'bce': 0, 'boundary': 0}
    iou_scores = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss, loss_dict = criterion(outputs, masks)
            
            # Track metrics
            epoch_loss += loss_dict['total']
            for key in epoch_metrics:
                epoch_metrics[key] += loss_dict[key]
            
            # Compute IoU
            pred_masks = torch.sigmoid(outputs) > 0.5
            iou = calculate_iou(pred_masks, masks, num_classes=2)
            iou_scores.append(iou)
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'iou': f"{iou:.4f}"
            })
    
    # Average metrics
    n_batches = len(loader)
    avg_loss = epoch_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
    avg_iou = np.mean(iou_scores)
    
    return avg_loss, avg_metrics, avg_iou


def save_checkpoint(model, optimizer, epoch, loss, iou, path):
    """Save training checkpoint with full state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'iou': iou,
    }
    torch.save(checkpoint, path)


# ========== MAIN TRAINING LOOP ==========
def main():
    config = TrainingConfig()
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.VIS_DIR, exist_ok=True)
    
    # Save configuration
    config.save_config(os.path.join(config.CHECKPOINT_DIR, 'config.json'))
    
    print("=" * 60)
    print("UPGRADED FENCE DETECTION TRAINING v2.0")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Encoder: {config.ENCODER}")
    print(f"Input Size: {config.INPUT_SIZE}px")
    print(f"Batch Size: {config.BATCH_SIZE} (effective: {config.BATCH_SIZE * config.ACCUMULATION_STEPS})")
    print(f"Epochs: {config.EPOCHS}")
    print("=" * 60)
    
    # ========== MODEL ==========
    print("\n[1/6] Initializing model...")
    model = smp.UnetPlusPlus(
        encoder_name=config.ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None  # We use sigmoid in loss
    )
    model.to(config.DEVICE)
    print(f"✓ Model initialized: UNet++ with {config.ENCODER}")
    
    # ========== DATASET ==========
    print("\n[2/6] Loading dataset...")
    full_dataset = FenceDatasetV2(
        config.IMAGES_DIR, 
        config.MASKS_DIR, 
        config.INPUT_SIZE, 
        train=True
    )
    
    # Split into train and validation
    val_size = int(len(full_dataset) * config.VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS
    )
    
    print(f"✓ DataLoaders created")
    
    # ========== LOSS & OPTIMIZER ==========
    print("\n[3/6] Setting up loss and optimizer...")
    criterion = CombinedLoss(config)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.INITIAL_LR, 
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=5, 
        total_epochs=config.EPOCHS,
        min_lr=config.MIN_LR,
        initial_lr=config.INITIAL_LR
    )
    print("✓ Loss: Focal + Dice + BCE + Boundary")
    print("✓ Optimizer: AdamW with warmup + cosine annealing")
    
    # ========== TRAINING LOOP ==========
    print("\n[4/6] Starting training...")
    print("=" * 60)
    
    best_iou = 0.0
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'learning_rate': []
    }
    
    for epoch in range(config.EPOCHS):
        # Learning rate scheduling
        current_lr = scheduler.step()
        history['learning_rate'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{config.EPOCHS} | LR: {current_lr:.2e}")
        print("-" * 60)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, 
            config.DEVICE, config.ACCUMULATION_STEPS, epoch
        )
        
        # Validate
        val_loss, val_metrics, val_iou = validate_epoch(
            model, val_loader, criterion, config.DEVICE, epoch
        )
        
        # Log results
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val IoU:    {val_iou:.4f}")
        
        # Save best model based on IoU
        if val_iou > best_iou:
            best_iou = val_iou
            best_loss = val_loss
            epochs_without_improvement = 0
            
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_iou,
                config.BEST_MODEL_PATH
            )
            print(f"  ✓ New best model saved! (IoU: {best_iou:.4f})")
        else:
            epochs_without_improvement += 1
        
        # Save last model
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_iou,
            config.LAST_MODEL_PATH
        )
        
        # Early stopping
        if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            print(f"  No improvement for {config.EARLY_STOP_PATIENCE} epochs")
            break
    
    # ========== TRAINING COMPLETE ==========
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation IoU: {best_iou:.4f}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Model saved to: {config.BEST_MODEL_PATH}")
    
    # Save training history
    history_path = os.path.join(config.CHECKPOINT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        history_serializable = {
            k: [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in vals]
            for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
