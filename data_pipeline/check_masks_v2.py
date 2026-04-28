"""
Visual Mask Inspection Tool v2.0
=================================
Generates overlay visualizations for manual quality review.
"""

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGES_DIR = 'data/images'
MASKS_DIR = 'data/masks'
OUTPUT_DIR = 'data/mask_overlays'

OVERLAY_COLOR = [0, 0, 255]  # Red in BGR
OVERLAY_ALPHA = 0.4  # Transparency (0=transparent, 1=opaque)

# Validation thresholds
MIN_FENCE_PERCENTAGE = 0.5
MAX_FENCE_PERCENTAGE = 80.0

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def create_overlay(img_path, mask_path, output_path):
    """
    Create and save mask overlay visualization.
    
    Returns:
        dict: Statistics and status
    """
    try:
        # Read image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            return {
                'filename': os.path.basename(img_path),
                'success': False,
                'percentage': 0,
                'status': 'read_error'
            }
        
        # Calculate statistics
        fence_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        percentage = (fence_pixels / total_pixels) * 100
        
        # Determine status
        if percentage < MIN_FENCE_PERCENTAGE:
            status = 'sparse'
        elif percentage > MAX_FENCE_PERCENTAGE:
            status = 'excessive'
        else:
            status = 'ok'
        
        # Create overlay
        overlay = img.copy()
        overlay[mask > 127] = OVERLAY_COLOR
        
        # Blend
        result = cv2.addWeighted(img, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)
        
        # Add text annotation
        text = f"{percentage:.1f}% | {status.upper()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            result,
            (10, 10),
            (20 + text_width, 20 + text_height + baseline),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            result,
            text,
            (15, 15 + text_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        # Save
        cv2.imwrite(output_path, result)
        
        return {
            'filename': os.path.basename(img_path),
            'success': True,
            'percentage': percentage,
            'status': status
        }
    
    except Exception as e:
        return {
            'filename': os.path.basename(img_path),
            'success': False,
            'percentage': 0,
            'status': f'error: {str(e)}'
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all masks
    mask_files = glob(os.path.join(MASKS_DIR, '*.png'))
    
    if not mask_files:
        print(f"❌ No masks found in {MASKS_DIR}")
        return
    
    print("=" * 70)
    print("VISUAL MASK INSPECTION v2.0")
    print("=" * 70)
    print(f"Found {len(mask_files)} masks to inspect")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Prepare tasks
    tasks = []
    for mask_path in mask_files:
        base_name = os.path.basename(mask_path)
        
        # Find corresponding image
        img_name = base_name.replace('.png', '.jpg')
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        if not os.path.exists(img_path):
            img_name = base_name.replace('.png', '.jpeg')
            img_path = os.path.join(IMAGES_DIR, img_name)
        
        if not os.path.exists(img_path):
            continue
        
        output_path = os.path.join(OUTPUT_DIR, base_name)
        tasks.append((img_path, mask_path, output_path))
    
    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(create_overlay, *task) for task in tasks]
        
        with tqdm(total=len(futures), desc="Creating overlays") as pbar:
            for future in futures:
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    # Analyze results
    success_count = sum(1 for r in results if r['success'])
    sparse_count = sum(1 for r in results if r['status'] == 'sparse')
    excessive_count = sum(1 for r in results if r['status'] == 'excessive')
    ok_count = sum(1 for r in results if r['status'] == 'ok')
    
    # Print summary
    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)
    print(f"✓ Processed: {success_count}/{len(results)}")
    print(f"✓ Good quality: {ok_count}")
    print(f"⚠ Sparse masks: {sparse_count}")
    print(f"⚠ Excessive coverage: {excessive_count}")
    print("=" * 70)
    
    if sparse_count > 0:
        print("\n⚠️  Sparse masks (review recommended):")
        sparse_files = [r for r in results if r['status'] == 'sparse']
        for r in sorted(sparse_files, key=lambda x: x['percentage'])[:10]:
            print(f"  - {r['filename']} ({r['percentage']:.2f}%)")
    
    if excessive_count > 0:
        print("\n⚠️  Excessive coverage (review recommended):")
        excessive_files = [r for r in results if r['status'] == 'excessive']
        for r in sorted(excessive_files, key=lambda x: -x['percentage'])[:10]:
            print(f"  - {r['filename']} ({r['percentage']:.2f}%)")
    
    print(f"\n✓ Visual overlays saved to: {OUTPUT_DIR}")
    print("  Open these files to manually verify mask quality")


if __name__ == '__main__':
    main()
