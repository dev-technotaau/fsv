"""Quick dataset validation"""
import cv2
import numpy as np
from pathlib import Path
import random

images_dir = Path('data/images')
masks_dir = Path('data/masks')

# Get JPG images only (exclude JSON)
image_files = sorted([f for f in images_dir.glob('*.jpg')])

print(f"Total JPG images: {len(image_files)}")
print(f"Checking 30 random samples...\n")

random.seed(42)
samples = random.sample(image_files, min(30, len(image_files)))

issues = []
fence_pcts = []

for img_file in samples:
    mask_file = masks_dir / img_file.with_suffix('.png').name
    
    if not mask_file.exists():
        issues.append(f"Missing mask: {img_file.name}")
        continue
    
    try:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            issues.append(f"Failed to load: {mask_file.name}")
            continue
        
        unique_vals = np.unique(mask)
        
        # Check for valid binary values
        if not all(v in [0, 255] for v in unique_vals):
            issues.append(f"{img_file.name}: Invalid values {unique_vals}")
            continue
        
        # Calculate fence percentage
        fence_pixels = np.sum(mask == 255)
        total_pixels = mask.size
        fence_pct = (fence_pixels / total_pixels) * 100
        fence_pcts.append(fence_pct)
        
        print(f"{img_file.name:60s} → {fence_pct:6.2f}% fence")
        
    except Exception as e:
        issues.append(f"{img_file.name}: {str(e)}")

print(f"\n{'='*80}")
print(f"Results:")
print(f"  Samples checked: {len(samples)}")
print(f"  Valid: {len(fence_pcts)}")
print(f"  Issues: {len(issues)}")

if issues:
    print(f"\n  Issues found:")
    for issue in issues[:10]:
        print(f"    - {issue}")

if fence_pcts:
    print(f"\n  Fence Coverage Stats:")
    print(f"    Mean:   {np.mean(fence_pcts):.2f}%")
    print(f"    Median: {np.median(fence_pcts):.2f}%")
    print(f"    Min:    {np.min(fence_pcts):.2f}%")
    print(f"    Max:    {np.max(fence_pcts):.2f}%")
    
    # Check for concerning patterns
    zero_fence = sum(1 for p in fence_pcts if p < 0.1)
    high_fence = sum(1 for p in fence_pcts if p > 50)
    
    print(f"    Samples with <0.1% fence: {zero_fence}")
    print(f"    Samples with >50% fence: {high_fence}")
    
    if zero_fence > len(fence_pcts) * 0.3:
        print(f"\n❌ WARNING: {zero_fence}/{len(fence_pcts)} samples have almost no fence!")
    
    if np.mean(fence_pcts) < 1:
        print(f"✅ Good: Average fence coverage ({np.mean(fence_pcts):.2f}%) is reasonable for imbalanced data")
    elif np.mean(fence_pcts) > 50:
        print(f"⚠️  WARNING: Average fence coverage is {np.mean(fence_pcts):.2f}% - masks might be inverted!")
    else:
        print(f"✅ Fence coverage looks normal")

print(f"{'='*80}")
