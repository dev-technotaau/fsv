import os
import sys

# --- FIX: Set headless backend before importing pyplot ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---------------------------------------------------------

import torch
import numpy as np
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================
# Point to your new lightweight model
MODEL_PATH = './models/pytorch/best_fence_model_light.pth'
# Match the input size used in robust_train_light.py
INPUT_SIZE = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(model_path):
    """
    Load the trained U-Net++ model from checkpoint.
    """
    print(f"Loading model from {model_path} on {DEVICE}...")
    
    # MUST match the architecture used in robust_train_light.py
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",
        encoder_weights=None, # No need to download pretrained weights for inference
        in_channels=3,
        classes=1,
        activation='sigmoid'  # Use sigmoid for inference to get 0-1 probabilities
    )
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle both full checkpoint dict and just state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded checkpoint with metadata")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded model state dictionary")
            
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        sys.exit(1)
    
    model.to(DEVICE)
    model.eval()
    return model

# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================
def create_prediction_visualization(image_path, pred_mask, save_path):
    """
    Create a 3-panel visualization: Input Image | Predicted Fence Mask | Overlay Detection
    """
    # Load original image for visualization (keep original aspect ratio for display if preferred, 
    # but for simplicity here we resize to match model input)
    img = Image.open(image_path).convert('RGB').resize((INPUT_SIZE, INPUT_SIZE))
    img_np = np.array(img)
    
    # Ensure mask is binary (0 or 1)
    mask = np.array(pred_mask).astype(np.uint8)
    
    # Create overlay: red mask over detected fence regions
    overlay = img_np.copy()
    red_color = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.5
    # Create boolean mask for indexing
    fence_pixels = (mask == 1)
    overlay[fence_pixels] = (alpha * red_color + (1 - alpha) * overlay[fence_pixels]).astype(np.uint8)
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Input Image
    axes[0].imshow(img_np)
    axes[0].set_title('Input Image', fontsize=16, fontweight='bold', pad=15)
    axes[0].axis('off')
    
    # Panel 2: Predicted Fence Mask
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Predicted Fence Mask', fontsize=16, fontweight='bold', pad=15)
    axes[1].axis('off')
    
    # Panel 3: Overlay Detection
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay Detection', fontsize=16, fontweight='bold', pad=15)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_image(model, image_path, output_dir='./export'):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    print(f"\nProcessing: {os.path.basename(image_path)}")
    
    # Load and preprocess image
    original_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = original_img.size
    
    preprocess = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    
    input_tensor = preprocess(original_img).unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        # Output is already [0,1] because of 'sigmoid' activation in model def
        # Threshold at 0.5 to get binary mask
        pred_mask_small = (output > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    # Optional: Resize mask back to original image size for better quality result
    # pred_mask = cv2.resize(pred_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    # For now, we'll just use the small one for visualization consistency
    pred_mask = pred_mask_small
    
    # Calculate statistics
    fence_pixels = np.sum(pred_mask == 1)
    total_pixels = pred_mask.size
    fence_percentage = (fence_pixels / total_pixels) * 100
    print(f"  -> Fence coverage: {fence_percentage:.2f}%")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save visualization
    viz_path = os.path.join(output_dir, f'pred_{base_name}.jpg')
    create_prediction_visualization(image_path, pred_mask, viz_path)
    print(f"  -> Saved visualization: {viz_path}")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("FENCE DETECTION INFERENCE (Lightweight U-Net++)")
    print("="*50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    # Get images from command line arguments
    if len(sys.argv) > 1:
        test_images = sys.argv[1:]
    else:
        print("Usage: python predict.py <path_to_image1> [path_to_image2 ...]")
        sys.exit(1)
        
    # Load model once
    model = load_model(MODEL_PATH)
    
    # Process images
    for img_path in test_images:
        predict_image(model, img_path)
        
    print("\nDone!")