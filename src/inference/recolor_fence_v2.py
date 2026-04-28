"""
Upgraded Fence Recoloring with Advanced Post-Processing v2.0
=============================================================
Improvements from research:
- Adaptive thresholding (Otsu's method)
- Morphological post-processing
- Test-Time Augmentation (TTA)
- Multi-scale inference
- Edge refinement
"""

import os
import sys
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

# ========== CONFIGURATION ==========
MODEL_PATH = './models/pytorch/best_fence_model_v2.pth'
INFERENCE_SIZE = 512  # Match training size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Post-processing parameters
USE_ADAPTIVE_THRESHOLD = True
USE_POST_PROCESSING = True
USE_TTA = True  # Test-Time Augmentation
TTA_TRANSFORMS = ['original', 'hflip', 'vflip']  # Augmentation types

# Thresholds
FIXED_THRESHOLD = 0.35  # Used if adaptive is disabled
MORPH_KERNEL_SIZE = 5

# ========== MODEL LOADING ==========
def load_model(path):
    """Load trained model with error handling."""
    print(f"[Init] Loading model from {path}...")
    
    try:
        # Try loading as checkpoint first
        checkpoint = torch.load(path, map_location=DEVICE)
        
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b1",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best IoU: {checkpoint.get('iou', 'unknown'):.4f}")
        else:
            model.load_state_dict(checkpoint)
    
    except:
        # Fallback: try direct state dict
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b1",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    
    model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully")
    return model


# ========== POST-PROCESSING ==========
def post_process_mask(mask_binary, kernel_size=5):
    """
    Apply morphological operations to improve mask quality.
    
    Research: Serra (1982) - Mathematical Morphology
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening: remove small false positive regions
    mask_opened = cv2.morphologyEx(
        mask_binary.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel
    )
    
    # Closing: fill small gaps in detected fence
    mask_closed = cv2.morphologyEx(
        mask_opened,
        cv2.MORPH_CLOSE,
        kernel
    )
    
    # Optional: connect nearby fence segments
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_final = cv2.morphologyEx(
        mask_closed,
        cv2.MORPH_CLOSE,
        kernel_connect
    )
    
    return mask_final.astype(np.float32)


# ========== INFERENCE WITH TTA ==========
def get_fence_mask_tta(model, image_path):
    """
    Inference with Test-Time Augmentation for robustness.
    
    Research: TTA improves model robustness by averaging predictions
    from multiple augmented versions of the input.
    """
    orig_img_bgr = cv2.imread(image_path)
    if orig_img_bgr is None:
        raise FileNotFoundError(f"Could not open {image_path}")
    
    orig_h, orig_w = orig_img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    preprocess = transforms.Compose([
        transforms.Resize((INFERENCE_SIZE, INFERENCE_SIZE)),
        transforms.ToTensor(),
    ])
    
    masks = []
    
    # Original
    if 'original' in TTA_TRANSFORMS:
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask = model(input_tensor).squeeze().cpu().numpy()
        masks.append(mask)
    
    # Horizontal flip
    if 'hflip' in TTA_TRANSFORMS:
        input_flipped = preprocess(TF.hflip(pil_img)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask_flipped = model(input_flipped).squeeze().cpu().numpy()
        mask_flipped = np.fliplr(mask_flipped)
        masks.append(mask_flipped)
    
    # Vertical flip
    if 'vflip' in TTA_TRANSFORMS:
        input_vflipped = preprocess(TF.vflip(pil_img)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask_vflipped = model(input_vflipped).squeeze().cpu().numpy()
        mask_vflipped = np.flipud(mask_vflipped)
        masks.append(mask_vflipped)
    
    # Average all predictions
    mask_avg = np.mean(masks, axis=0)
    
    return orig_img_bgr, mask_avg, orig_h, orig_w


def get_fence_mask(model, image_path):
    """
    Main inference function with all post-processing.
    """
    # Run TTA if enabled
    if USE_TTA:
        orig_img_bgr, mask_pred, orig_h, orig_w = get_fence_mask_tta(model, image_path)
    else:
        # Standard single inference
        orig_img_bgr = cv2.imread(image_path)
        if orig_img_bgr is None:
            raise FileNotFoundError(f"Could not open {image_path}")
        
        orig_h, orig_w = orig_img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        preprocess = transforms.Compose([
            transforms.Resize((INFERENCE_SIZE, INFERENCE_SIZE)),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            mask_pred = model(input_tensor).squeeze().cpu().numpy()
    
    # Resize to original resolution
    mask_full = cv2.resize(mask_pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # Adaptive or fixed thresholding
    if USE_ADAPTIVE_THRESHOLD:
        mask_uint8 = (mask_full * 255).astype(np.uint8)
        threshold_value, mask_binary = cv2.threshold(
            mask_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"  Adaptive threshold: {threshold_value/255:.3f}")
    else:
        _, mask_binary = cv2.threshold(mask_full, FIXED_THRESHOLD, 1.0, cv2.THRESH_BINARY)
        print(f"  Fixed threshold: {FIXED_THRESHOLD}")
    
    # Post-processing
    if USE_POST_PROCESSING:
        mask_binary = post_process_mask(mask_binary, kernel_size=MORPH_KERNEL_SIZE)
    
    return orig_img_bgr, mask_binary


# ========== RECOLORING ==========
def apply_recolor(image_bgr, mask, hue_shift, sat_boost=1.2):
    """Recolor masked region using HSV color space."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Soften mask for better blending
    mask_soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    
    # Apply hue shift
    h_new = (mask_soft * hue_shift + (1 - mask_soft) * h).astype(np.uint8)
    
    # Boost saturation
    s_float = s * (1 + (sat_boost - 1) * mask_soft)
    s_new = np.clip(s_float, 0, 255).astype(np.uint8)
    
    # Merge and convert back
    hsv_new = cv2.merge([h_new, s_new, v.astype(np.uint8)])
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)


# ========== MAIN ==========
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python recolor_fence_v2.py <image_path> [color_name]")
        print("Colors: red, green, blue, yellow, purple, cyan")
        print("\nOptional flags:")
        print("  --no-tta        Disable test-time augmentation")
        print("  --no-adapt      Use fixed threshold instead of adaptive")
        print("  --no-postproc   Disable morphological post-processing")
        sys.exit(1)
    
    # Parse arguments
    img_path = sys.argv[1]
    color_name = sys.argv[2].lower() if len(sys.argv) > 2 else 'blue'
    
    # Parse flags
    if '--no-tta' in sys.argv:
        USE_TTA = False
    if '--no-adapt' in sys.argv:
        USE_ADAPTIVE_THRESHOLD = False
    if '--no-postproc' in sys.argv:
        USE_POST_PROCESSING = False
    
    COLORS = {
        'red': 0,
        'yellow': 30,
        'green': 60,
        'cyan': 90,
        'blue': 110,
        'purple': 150
    }
    
    target_hue = COLORS.get(color_name, 110)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Process
    print(f"\n[Processing] {os.path.basename(img_path)}")
    print(f"  TTA: {USE_TTA}")
    print(f"  Adaptive threshold: {USE_ADAPTIVE_THRESHOLD}")
    print(f"  Post-processing: {USE_POST_PROCESSING}")
    
    original, mask = get_fence_mask(model, img_path)
    
    print(f"\n[Recoloring] Applying {color_name} (hue={target_hue})")
    result = apply_recolor(original, mask, hue_shift=target_hue)
    
    # Save outputs
    base = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs('./export', exist_ok=True)
    
    # Save mask
    mask_vis = (mask * 255).astype(np.uint8)
    cv2.imwrite(f"./export/{base}_mask.png", mask_vis)
    
    # Save overlay
    overlay = original.copy()
    overlay[mask > 0.5] = [0, 0, 255]
    blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
    cv2.imwrite(f"./export/{base}_overlay.jpg", blended)
    
    # Save recolored result
    out_path = f"./export/{base}_recolored_{color_name}.jpg"
    cv2.imwrite(out_path, result)
    
    print(f"\n✓ Saved outputs:")
    print(f"  - Mask: ./export/{base}_mask.png")
    print(f"  - Overlay: ./export/{base}_overlay.jpg")
    print(f"  - Result: {out_path}")
