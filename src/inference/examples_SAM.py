"""
SAM Training - Quick Usage Examples
====================================
Practical examples for common SAM training scenarios.

Author: VisionGuard Team
Date: November 12, 2025
"""

# ============================================================================
# EXAMPLE 1: Basic Training
# ============================================================================

def example_basic_training():
    """
    Most common use case: Train SAM with default settings.
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Training")
    print("=" * 60)
    
    print("""
# 1. Ensure data structure:
data/
  images/  # Your fence images (.jpg, .png)
  masks/   # Binary masks (.png)

# 2. Run training:
python train_SAM.py

# 3. Monitor with TensorBoard:
tensorboard --logdir logs/sam

# Output:
- Best model: checkpoints/sam/best_model.pth
- Logs: logs/sam/training_*.log
- Visualizations: training_visualizations/sam/
    """)


# ============================================================================
# EXAMPLE 2: Custom Configuration
# ============================================================================

def example_custom_config():
    """
    Modify training parameters for specific requirements.
    """
    print("=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)
    
    print("""
# Modify Config class in train_SAM.py:

# For faster training (lower quality):
class Config:
    BATCH_SIZE = 8          # Increase if GPU allows
    TRAIN_SIZE = 384        # Smaller resolution
    EPOCHS = 50             # Fewer epochs
    USE_ADVANCED_AUGMENTATION = False

# For best quality (slower):
class Config:
    SAM_MODEL_TYPE = "vit_l"  # Larger model
    TRAIN_SIZE = 1024       # Higher resolution
    EPOCHS = 150            # More epochs
    USE_EMA = True          # Enable EMA
    
    LOSS_WEIGHTS = {
        'focal': 0.2,
        'dice': 0.4,        # Increase dice
        'iou': 0.3,
        'boundary': 0.1
    }

# For limited GPU (4GB):
class Config:
    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 8  # Effective batch = 16
    TRAIN_SIZE = 384
    NUM_WORKERS = 2
    USE_EMA = False         # Disable to save memory
    """)


# ============================================================================
# EXAMPLE 3: Resume Training
# ============================================================================

def example_resume_training():
    """
    Resume interrupted training from checkpoint.
    """
    print("=" * 60)
    print("EXAMPLE 3: Resume Training")
    print("=" * 60)
    
    print("""
# Add this to train_SAM.py before training loop:

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_iou = checkpoint['val_iou']
    
    if 'ema_shadow' in checkpoint and ema is not None:
        ema.shadow = checkpoint['ema_shadow']
    
    return start_epoch, best_val_iou

# Load checkpoint
if os.path.exists('checkpoints/sam/checkpoint_epoch_50.pth'):
    start_epoch, best_val_iou = load_checkpoint(
        'checkpoints/sam/checkpoint_epoch_50.pth'
    )
else:
    start_epoch = 0
    best_val_iou = 0.0

# Modify training loop:
for epoch in range(start_epoch, Config.EPOCHS):
    # ... training code ...
    """)


# ============================================================================
# EXAMPLE 4: Inference on Single Image
# ============================================================================

def example_single_image_inference():
    """
    Run inference on a single image.
    """
    print("=" * 60)
    print("EXAMPLE 4: Single Image Inference")
    print("=" * 60)
    
    print("""
# Command line:
python inference_SAM.py \\
    --checkpoint checkpoints/sam/best_model.pth \\
    --input path/to/image.jpg \\
    --output results/

# Python script:
from inference_SAM import SAMInference
import cv2

# Load model
predictor = SAMInference(
    checkpoint_path='checkpoints/sam/best_model.pth',
    device='cuda'
)

# Load image
image = cv2.imread('fence_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict
mask = predictor.predict(image_rgb, threshold=0.5)

# Save result
cv2.imwrite('fence_mask.png', mask * 255)

# Get probability map
prob_map = predictor.predict(image_rgb, return_prob=True)
cv2.imwrite('fence_probability.png', (prob_map * 255).astype('uint8'))
    """)


# ============================================================================
# EXAMPLE 5: Batch Processing
# ============================================================================

def example_batch_processing():
    """
    Process multiple images efficiently.
    """
    print("=" * 60)
    print("EXAMPLE 5: Batch Processing")
    print("=" * 60)
    
    print("""
# Command line (process directory):
python inference_SAM.py \\
    --checkpoint checkpoints/sam/best_model.pth \\
    --input input_images/ \\
    --output results/ \\
    --threshold 0.5

# This will create:
results/
  masks/              # Binary masks
  visualizations/     # Overlay visualizations
  results.json        # Statistics

# Python script:
from inference_SAM import SAMInference
import cv2
from pathlib import Path

# Load model
predictor = SAMInference(
    checkpoint_path='checkpoints/sam/best_model.pth',
    device='cuda'
)

# Get all images
image_dir = Path('input_images')
image_files = list(image_dir.glob('*.jpg'))

# Process batch
for img_file in image_files:
    # Load
    image = cv2.imread(str(img_file))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Predict
    mask = predictor.predict(image_rgb)
    
    # Save
    output_path = f'results/{img_file.stem}_mask.png'
    cv2.imwrite(output_path, mask * 255)
    
    print(f'Processed: {img_file.name}')
    """)


# ============================================================================
# EXAMPLE 6: Test-Time Augmentation (TTA)
# ============================================================================

def example_tta():
    """
    Use test-time augmentation for better accuracy.
    """
    print("=" * 60)
    print("EXAMPLE 6: Test-Time Augmentation")
    print("=" * 60)
    
    print("""
# TTA averages predictions from multiple augmented versions

from inference_SAM import SAMInference
import cv2
import numpy as np

predictor = SAMInference(
    checkpoint_path='checkpoints/sam/best_model.pth',
    device='cuda'
)

def predict_with_tta(image, predictor):
    '''Predict with horizontal flip augmentation.'''
    
    # Original prediction
    pred1 = predictor.predict(image, return_prob=True)
    
    # Horizontal flip
    image_flip = cv2.flip(image, 1)
    pred2 = predictor.predict(image_flip, return_prob=True)
    pred2 = cv2.flip(pred2, 1)
    
    # Average probabilities
    avg_prob = (pred1 + pred2) / 2
    
    # Threshold
    mask = (avg_prob > 0.5).astype(np.uint8)
    
    return mask

# Usage
image = cv2.imread('fence.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = predict_with_tta(image_rgb, predictor)
cv2.imwrite('fence_mask_tta.png', mask * 255)

# TTA typically improves IoU by 1-2%
    """)


# ============================================================================
# EXAMPLE 7: Hyperparameter Tuning
# ============================================================================

def example_hyperparameter_tuning():
    """
    Optimize hyperparameters for best results.
    """
    print("=" * 60)
    print("EXAMPLE 7: Hyperparameter Tuning")
    print("=" * 60)
    
    print("""
# Test different configurations:

# 1. Learning Rate Grid Search
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

for lr in learning_rates:
    Config.LEARNING_RATE = lr
    Config.CHECKPOINT_DIR = f'checkpoints/sam_lr_{lr}'
    train_sam()  # Run training
    # Compare val IoU

# 2. Loss Weight Optimization
loss_configs = [
    {'focal': 0.25, 'dice': 0.35, 'iou': 0.25, 'boundary': 0.15},
    {'focal': 0.20, 'dice': 0.40, 'iou': 0.30, 'boundary': 0.10},
    {'focal': 0.30, 'dice': 0.30, 'iou': 0.30, 'boundary': 0.10},
]

for weights in loss_configs:
    Config.LOSS_WEIGHTS = weights
    train_sam()
    # Compare results

# 3. Batch Size vs Accumulation
configs = [
    (2, 8),   # batch=2, accum=8, effective=16
    (4, 4),   # batch=4, accum=4, effective=16
    (8, 2),   # batch=8, accum=2, effective=16
]

for batch, accum in configs:
    Config.BATCH_SIZE = batch
    Config.ACCUMULATION_STEPS = accum
    train_sam()
    # Compare speed and quality

# 4. Resolution Testing
resolutions = [384, 512, 640, 768]

for size in resolutions:
    Config.TRAIN_SIZE = size
    train_sam()
    # Higher resolution = better quality but slower
    """)


# ============================================================================
# EXAMPLE 8: Monitoring Training
# ============================================================================

def example_monitoring():
    """
    Monitor training progress effectively.
    """
    print("=" * 60)
    print("EXAMPLE 8: Monitoring Training")
    print("=" * 60)
    
    print("""
# 1. TensorBoard (Real-time)
tensorboard --logdir logs/sam --port 6006
# Open: http://localhost:6006

# View:
- Train/Val loss curves
- IoU, Dice, F1 metrics
- Learning rate schedule
- Prediction visualizations

# 2. Log File Analysis
tail -f logs/sam/training_*.log

# Check for:
- Steady loss decrease
- No overfitting (train << val loss)
- Metrics improving
- No errors/warnings

# 3. Visualization Inspection
# Check: training_visualizations/sam/epoch_*.png
# Look for:
- Clean mask predictions
- Good edge detection
- No false positives
- Consistency across epochs

# 4. GPU Monitoring (separate terminal)
nvidia-smi -l 1

# Check:
- GPU utilization: 85-95% (good)
- Memory usage: <90% of available
- Temperature: <80°C

# 5. Python Script for Analysis
import json
import matplotlib.pyplot as plt

# Load training history
with open('logs/sam/metrics.json', 'r') as f:
    history = json.load(f)

# Plot IoU over epochs
plt.figure(figsize=(10, 6))
plt.plot(history['train_iou'], label='Train IoU')
plt.plot(history['val_iou'], label='Val IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.savefig('iou_curve.png')
    """)


# ============================================================================
# EXAMPLE 9: Transfer Learning
# ============================================================================

def example_transfer_learning():
    """
    Fine-tune on new data using trained model.
    """
    print("=" * 60)
    print("EXAMPLE 9: Transfer Learning")
    print("=" * 60)
    
    print("""
# Scenario: You have a trained model on Dataset A,
# want to fine-tune on Dataset B

# 1. Load pretrained model
checkpoint = torch.load('checkpoints/sam/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 2. Freeze encoder (optional, faster training)
for param in model.sam.image_encoder.parameters():
    param.requires_grad = False

# 3. Train with lower learning rate
Config.LEARNING_RATE = 1e-5  # 10× lower
Config.EPOCHS = 20  # Fewer epochs

# 4. Fine-tune on new data
# Place new data in data/images and data/masks
train_sam()

# Benefits:
- Faster training (20 epochs vs 100)
- Better initial performance
- Less data required
- Preserves learned features
    """)


# ============================================================================
# EXAMPLE 10: Model Ensemble
# ============================================================================

def example_model_ensemble():
    """
    Combine multiple models for better accuracy.
    """
    print("=" * 60)
    print("EXAMPLE 10: Model Ensemble")
    print("=" * 60)
    
    print("""
# Ensemble different models for best accuracy

from inference_SAM import SAMInference
import cv2
import numpy as np

# Load multiple models
model1 = SAMInference('checkpoints/sam/best_model.pth')
model2 = SAMInference('checkpoints/sam/checkpoint_epoch_90.pth')
model3 = SAMInference('checkpoints/sam/checkpoint_epoch_95.pth')

def ensemble_predict(image, models):
    '''Average predictions from multiple models.'''
    
    predictions = []
    for model in models:
        prob = model.predict(image, return_prob=True)
        predictions.append(prob)
    
    # Average probabilities
    avg_prob = np.mean(predictions, axis=0)
    
    # Threshold
    mask = (avg_prob > 0.5).astype(np.uint8)
    
    return mask

# Usage
image = cv2.imread('fence.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = ensemble_predict(image_rgb, [model1, model2, model3])
cv2.imwrite('fence_ensemble.png', mask * 255)

# Ensemble typically improves IoU by 2-3%
# Trade-off: 3× slower inference
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\\n" + "=" * 60)
    print("SAM TRAINING - USAGE EXAMPLES")
    print("=" * 60 + "\\n")
    
    examples = [
        ("Basic Training", example_basic_training),
        ("Custom Configuration", example_custom_config),
        ("Resume Training", example_resume_training),
        ("Single Image Inference", example_single_image_inference),
        ("Batch Processing", example_batch_processing),
        ("Test-Time Augmentation", example_tta),
        ("Hyperparameter Tuning", example_hyperparameter_tuning),
        ("Monitoring Training", example_monitoring),
        ("Transfer Learning", example_transfer_learning),
        ("Model Ensemble", example_model_ensemble)
    ]
    
    print("Available Examples:\\n")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\\nRunning all examples...\\n")
    
    for name, example_func in examples:
        try:
            example_func()
            print()
        except Exception as e:
            print(f"Error in {name}: {e}")
            print()
    
    print("=" * 60)
    print("For more details, see:")
    print("  - SAM_TRAINING_GUIDE.md")
    print("  - SAM_vs_SegFormer_Comparison.md")
    print("=" * 60)
