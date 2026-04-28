"""
Utility functions for training and evaluation.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_prediction_visualization(image, mask, prediction, save_path):
    """
    Save a visualization comparing input image, ground truth mask, and prediction.
    
    Args:
        image: Input image tensor (C, H, W)
        mask: Ground truth mask tensor (H, W)
        prediction: Model prediction tensor (H, W)
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Display images
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


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


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state
        device: Device to load model to
    
    Returns:
        Tuple of (epoch, loss, iou)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    iou = checkpoint.get('iou', 0.0)
    
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"  Loss: {loss:.4f}, IoU: {iou:.4f}")
    
    return epoch, loss, iou
