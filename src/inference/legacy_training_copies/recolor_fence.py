import os
import sys
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
from torchvision import transforms

# --- CONFIGURATION ---
MODEL_PATH = './models/pytorch/best_fence_model_light.pth'
# Try 640 for inference (higher than training's 384 for better detail). 
# If it crashes RAM, lower to 512.
INFERENCE_SIZE = 640  
DEVICE = torch.device('cpu')

def load_model(path):
    print(f"[Init] Loading model from {path}...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_fence_mask(model, image_path):
    """Runs model inference to get a high-res binary mask."""
    # 1. Load and Preprocess
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

    # 2. Inference
    with torch.no_grad():
        output_mask = model(input_tensor).squeeze().cpu().numpy()

    # 3. Post-process mask
    # Resize back to original high resolution
    mask_full = cv2.resize(output_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    # Threshold to get binary mask (adjust 0.5 if needed to catch more/less fence)
    _, mask_binary = cv2.threshold(mask_full, 0.5, 1.0, cv2.THRESH_BINARY)
    
    return orig_img_bgr, mask_binary

def apply_recolor(image_bgr, mask, hue_shift, sat_boost=1.2):
    """
    Recolors the masked region using HSV color space to preserve texture.
    hue_shift: Target hue [0-179] in OpenCV. (e.g., Yellow~30, Green~60, Blue~120, Red~0/179)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Soften mask for better blending at edges
    mask_soft = cv2.GaussianBlur(mask, (5, 5), 0)

    # -- Apply color changes --
    # 1. Set new Hue where mask is present
    # We use blended assignment: new_h = mask*target_h + (1-mask)*old_h
    h_new = (mask_soft * hue_shift + (1 - mask_soft) * h).astype(np.uint8)

    # 2. Boost Saturation slightly on fence to make color 'pop'
    s_float = s.astype(np.float32) * (1 + (sat_boost - 1) * mask_soft)
    s_new = np.clip(s_float, 0, 255).astype(np.uint8)

    # 3. Value (brightness/texture) is kept largely EXACT same to preserve realism
    # Optional: slight gamma correction ONLY on fence if it's too dark to take color
    # v_new = cv2.addWeighted(v, 1.0, (mask_soft * 20).astype(np.uint8), 1.0, 0)

    # Merge back
    hsv_new = cv2.merge([h_new, s_new, v])
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python recolor.py <image_path> [color_name]")
        print("Colors: red, green, blue, yellow, purple, cyan")
        sys.exit(1)

    img_path = sys.argv[1]
    color_name = sys.argv[2].lower() if len(sys.argv) > 2 else 'blue'

    # Color mapping for OpenCV HSV (Hue is 0-179)
    COLORS = {
        'red': 0,
        'yellow': 30,
        'green': 60,
        'cyan': 90,
        'blue': 110,
        'purple': 150
    }
    target_hue = COLORS.get(color_name, 110)

    # Run Pipeline
    model = load_model(MODEL_PATH)
    print(f"Processing {os.path.basename(img_path)}...")
    original, mask = get_fence_mask(model, img_path)
    
    print(f"Applying {color_name} (hue={target_hue})...")
    result = apply_recolor(original, mask, hue_shift=target_hue)

    # Save output
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = f"./export/{base}_recolored_{color_name}.jpg"
    os.makedirs('./export', exist_ok=True)
    cv2.imwrite(out_path, result)
    
    print(f"✓ Saved to {out_path}")