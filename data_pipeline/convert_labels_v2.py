"""
Enhanced Label-to-Mask Converter v2.0
======================================
Upgrades:
- Anti-aliased edge smoothing
- Morphological dilation for better coverage
- Multi-scale rendering for quality
- Comprehensive validation and logging
- Parallel processing support

Research basis:
- Anti-aliasing: Supersampling with Gaussian filtering
- Morphological operations: Serra (1982)
"""

import json
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGES_DIR = 'data/images'
MASKS_DIR = 'data/masks'
OUTPUT_REPORT = 'mask_conversion_report.txt'

# Mask generation parameters
SMOOTH_EDGES = True  # Enable anti-aliasing
DILATION_SIZE = 2  # Slight dilation for better coverage
SUPERSAMPLE_FACTOR = 4  # For anti-aliasing (higher = smoother, slower)
GAUSSIAN_KERNEL_SIZE = 5  # For edge smoothing

# Validation thresholds
MIN_FENCE_PERCENTAGE = 0.5  # Warn if less than 0.5% of image
MAX_FENCE_PERCENTAGE = 80.0  # Warn if more than 80% of image

# ============================================================================
# MASK GENERATION FUNCTIONS
# ============================================================================

def create_smooth_mask(data, smooth_edges=True, dilation_size=2):
    """
    Enhanced mask creation with optional edge smoothing and dilation.
    
    Args:
        data: LabelMe JSON data
        smooth_edges: If True, apply anti-aliasing
        dilation_size: Kernel size for morphological dilation
    
    Returns:
        numpy.ndarray: Binary mask (0=background, 255=fence)
    """
    height = data['imageHeight']
    width = data['imageWidth']
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for shape in data['shapes']:
        if shape['label'].lower() != 'fence':
            continue
        
        points = np.array(shape['points'], dtype=np.float32)
        
        if len(points) < 3:
            continue  # Skip invalid polygons
        
        # Convert to integer coordinates
        points_int = np.round(points).astype(np.int32)
        
        if smooth_edges and SUPERSAMPLE_FACTOR > 1:
            # Create high-resolution mask for anti-aliasing
            scale = SUPERSAMPLE_FACTOR
            temp_height = height * scale
            temp_width = width * scale
            temp_mask = np.zeros((temp_height, temp_width), dtype=np.uint8)
            
            # Scale points
            scaled_points = (points * scale).astype(np.int32)
            
            # Fill polygon in high-res
            cv2.fillPoly(temp_mask, [scaled_points], 255)
            
            # Apply Gaussian blur for smooth edges
            if GAUSSIAN_KERNEL_SIZE > 0:
                kernel_size = GAUSSIAN_KERNEL_SIZE * scale
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Must be odd
                temp_mask = cv2.GaussianBlur(
                    temp_mask, 
                    (kernel_size, kernel_size), 
                    0
                )
            
            # Downscale back to original resolution
            smooth_mask = cv2.resize(
                temp_mask,
                (width, height),
                interpolation=cv2.INTER_AREA
            )
            
            # Threshold to binary
            _, smooth_mask = cv2.threshold(
                smooth_mask, 
                127, 
                255, 
                cv2.THRESH_BINARY
            )
            
            # Combine with main mask
            mask = cv2.bitwise_or(mask, smooth_mask)
        else:
            # Simple fill without anti-aliasing
            cv2.fillPoly(mask, [points_int], 255)
    
    # Apply morphological dilation for better coverage
    if dilation_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (dilation_size, dilation_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def validate_mask(mask, filename):
    """
    Validate mask quality and return warnings.
    
    Args:
        mask: Binary mask array
        filename: Name of the mask file
    
    Returns:
        tuple: (fence_percentage, warnings_list)
    """
    fence_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    percentage = (fence_pixels / total_pixels) * 100
    
    warnings = []
    
    if percentage < MIN_FENCE_PERCENTAGE:
        warnings.append(f"Very sparse mask ({percentage:.2f}% < {MIN_FENCE_PERCENTAGE}%)")
    
    if percentage > MAX_FENCE_PERCENTAGE:
        warnings.append(f"Excessive coverage ({percentage:.2f}% > {MAX_FENCE_PERCENTAGE}%)")
    
    if fence_pixels == 0:
        warnings.append("Empty mask (no fence pixels)")
    
    # Check if mask is mostly border (possible annotation error)
    border_thickness = 5
    border_mask = np.zeros_like(mask)
    border_mask[:border_thickness, :] = 1
    border_mask[-border_thickness:, :] = 1
    border_mask[:, :border_thickness] = 1
    border_mask[:, -border_thickness:] = 1
    
    border_pixels = np.count_nonzero(mask & border_mask)
    if fence_pixels > 0 and (border_pixels / fence_pixels) > 0.5:
        warnings.append("Mask concentrated on image borders (possible error)")
    
    return percentage, warnings


def process_single_json(json_path):
    """
    Process a single JSON file and create mask.
    
    Returns:
        dict: Processing results and statistics
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Create enhanced mask
        mask = create_smooth_mask(
            data, 
            smooth_edges=SMOOTH_EDGES, 
            dilation_size=DILATION_SIZE
        )
        
        # Save mask
        base = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(MASKS_DIR, f'{base}.png')
        cv2.imwrite(output_path, mask)
        
        # Validate
        percentage, warnings = validate_mask(mask, base)
        
        return {
            'filename': base,
            'success': True,
            'percentage': percentage,
            'warnings': warnings,
            'error': None
        }
    
    except Exception as e:
        return {
            'filename': os.path.basename(json_path),
            'success': False,
            'percentage': 0,
            'warnings': [],
            'error': str(e)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Create masks directory
    os.makedirs(MASKS_DIR, exist_ok=True)
    
    # Get all JSON files
    json_files = glob(os.path.join(IMAGES_DIR, '*.json'))
    
    if not json_files:
        print(f"❌ No JSON files found in {IMAGES_DIR}")
        return
    
    print("=" * 70)
    print("ENHANCED MASK CONVERSION v2.0")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Anti-aliasing: {SMOOTH_EDGES}")
    print(f"  - Dilation size: {DILATION_SIZE}px")
    print(f"  - Supersample factor: {SUPERSAMPLE_FACTOR}x")
    print(f"  - Found {len(json_files)} JSON files")
    print("=" * 70)
    
    # Process files
    results = []
    
    # Use ProcessPoolExecutor for CPU-intensive image processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_json, f) for f in json_files]
        
        with tqdm(total=len(json_files), desc="Converting masks") as pbar:
            for future in futures:
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    # Generate report
    success_count = sum(1 for r in results if r['success'])
    error_count = len(results) - success_count
    warning_count = sum(1 for r in results if r['warnings'])
    
    # Calculate statistics
    percentages = [r['percentage'] for r in results if r['success']]
    avg_percentage = np.mean(percentages) if percentages else 0
    
    # Print summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"✓ Successful: {success_count}/{len(results)}")
    print(f"✗ Errors: {error_count}")
    print(f"⚠ Warnings: {warning_count}")
    print(f"📊 Average fence coverage: {avg_percentage:.2f}%")
    print("=" * 70)
    
    # Save detailed report
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MASK CONVERSION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total files: {len(results)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Errors: {error_count}\n")
        f.write(f"With warnings: {warning_count}\n")
        f.write(f"Average fence coverage: {avg_percentage:.2f}%\n\n")
        
        # Errors
        if error_count > 0:
            f.write("ERRORS\n")
            f.write("-" * 70 + "\n")
            for r in results:
                if not r['success']:
                    f.write(f"{r['filename']}: {r['error']}\n")
            f.write("\n")
        
        # Warnings
        if warning_count > 0:
            f.write("WARNINGS\n")
            f.write("-" * 70 + "\n")
            for r in results:
                if r['warnings']:
                    f.write(f"{r['filename']} ({r['percentage']:.2f}%):\n")
                    for warn in r['warnings']:
                        f.write(f"  - {warn}\n")
            f.write("\n")
        
        # All files
        f.write("ALL FILES\n")
        f.write("-" * 70 + "\n")
        for r in sorted(results, key=lambda x: x['filename']):
            status = "✓" if r['success'] else "✗"
            f.write(f"{status} {r['filename']}: {r['percentage']:.2f}%\n")
    
    print(f"\n✓ Detailed report saved to: {OUTPUT_REPORT}")
    
    # Show problematic files
    if warning_count > 0:
        print("\n⚠️  Files with warnings (review recommended):")
        for r in results[:10]:  # Show first 10
            if r['warnings']:
                print(f"  - {r['filename']} ({r['percentage']:.2f}%)")
        
        if warning_count > 10:
            print(f"  ... and {warning_count - 10} more (see report)")


if __name__ == '__main__':
    main()
