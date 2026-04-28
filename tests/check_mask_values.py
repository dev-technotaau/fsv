"""
Check actual mask pixel values to confirm inversion
"""
from pathlib import Path
import numpy as np
from PIL import Image

images_dir = Path("data/images")
masks_dir = Path("data/masks")

# Get first 5 images for inspection
jpg_images = sorted([f for f in images_dir.glob("*.jpg")])[:5]

print("Checking first 5 masks for pixel value distribution:\n")
print("=" * 80)

for img_file in jpg_images:
    mask_file = img_file.name.replace('.jpg', '.png')
    mask_path = masks_dir / mask_file
    
    if mask_path.exists():
        mask = np.array(Image.open(mask_path).convert('L'))
        unique_values = np.unique(mask)
        value_counts = {val: np.sum(mask == val) for val in unique_values}
        total_pixels = mask.size
        
        fence_percentage = (np.sum(mask == 255) / total_pixels) * 100
        
        print(f"\n{img_file.name}")
        print(f"  Unique pixel values: {unique_values.tolist()}")
        print(f"  Pixel distribution:")
        for val, count in sorted(value_counts.items()):
            pct = (count / total_pixels) * 100
            print(f"    Value {val:3d}: {count:8d} pixels ({pct:6.2f}%)")
        print(f"  Fence coverage (pixels=255): {fence_percentage:.2f}%")
        
        # Check if mask is binary
        if len(unique_values) == 2 and set(unique_values) == {0, 255}:
            print(f"  ✓ Binary mask (0, 255)")
        elif len(unique_values) == 2:
            print(f"  ⚠ Binary but not (0, 255): {unique_values}")
        else:
            print(f"  ✗ NOT BINARY - has {len(unique_values)} unique values")

print("\n" + "=" * 80)
print("\nConclusion:")
print("If fence coverage is high (>40%), masks are INVERTED:")
print("  Current: fence=0 (black), background=255 (white)")
print("  Expected: fence=255 (white), background=0 (black)")
print("\nFix required: Invert all masks (255 - pixel_value)")
