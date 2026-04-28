"""
Comprehensive Dataset Validation Script
Checks:
1. Image-mask pairs existence
2. Mask pixel values (should be 0 and 255 only)
3. Fence coverage percentage
4. Image and mask dimensions
5. Data quality issues
"""

import cv2
import numpy as np
from pathlib import Path
import random
import json

def validate_dataset():
    print("="*80)
    print("COMPREHENSIVE DATASET VALIDATION")
    print("="*80)
    
    images_dir = Path('data/images')
    masks_dir = Path('data/masks')
    
    # Get all images (exclude JSON files!)
    image_files = sorted([f for f in images_dir.glob('*') 
                         if f.suffix.lower() in ['.png', '.jpg', '.jpeg'] and f.is_file()])
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Images directory: {images_dir}")
    print(f"  Masks directory: {masks_dir}")
    print(f"  Total image files: {len(image_files)}")
    
    # File type breakdown
    jpg_count = sum(1 for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg'])
    png_count = sum(1 for f in image_files if f.suffix.lower() == '.png')
    print(f"  File types: {jpg_count} JPG, {png_count} PNG")
    
    # Validation checks
    issues = {
        'missing_masks': [],
        'load_errors': [],
        'invalid_values': [],
        'dimension_mismatch': [],
        'zero_fence': [],
        'all_fence': []
    }
    
    fence_percentages = []
    valid_pairs = 0
    
    print(f"\n🔍 Validating pairs...")
    
    for img_file in image_files:
        # Find corresponding mask
        mask_file = masks_dir / img_file.name
        
        # Check if mask exists
        if not mask_file.exists():
            issues['missing_masks'].append(str(img_file.name))
            continue
        
        try:
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                issues['load_errors'].append(f"Image: {img_file.name}")
                continue
            
            # Load mask
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                issues['load_errors'].append(f"Mask: {mask_file.name}")
                continue
            
            # Check dimensions match
            if image.shape[:2] != mask.shape:
                issues['dimension_mismatch'].append(
                    f"{img_file.name}: Image {image.shape[:2]} vs Mask {mask.shape}"
                )
            
            # Check mask values
            unique_vals = np.unique(mask)
            
            # Should only have 0 and/or 255
            if not all(v in [0, 255] for v in unique_vals):
                issues['invalid_values'].append(
                    f"{img_file.name}: Unique values {unique_vals}"
                )
                continue
            
            # Calculate fence percentage
            fence_pixels = np.sum(mask == 255)
            total_pixels = mask.shape[0] * mask.shape[1]
            fence_pct = (fence_pixels / total_pixels) * 100
            fence_percentages.append(fence_pct)
            
            # Track extreme cases
            if fence_pct == 0:
                issues['zero_fence'].append(img_file.name)
            elif fence_pct > 99.9:
                issues['all_fence'].append(f"{img_file.name}: {fence_pct:.2f}%")
            
            valid_pairs += 1
            
        except Exception as e:
            issues['load_errors'].append(f"{img_file.name}: {str(e)}")
    
    # Results
    print(f"\n✅ Valid Pairs: {valid_pairs}/{len(image_files)}")
    
    # Report issues
    print(f"\n⚠️  Issues Found:")
    total_issues = sum(len(v) for v in issues.values())
    
    if total_issues == 0:
        print("  ✓ No issues found! Dataset is clean.")
    else:
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"\n  {issue_type.replace('_', ' ').title()}: {len(issue_list)}")
                for item in issue_list[:5]:  # Show first 5
                    print(f"    - {item}")
                if len(issue_list) > 5:
                    print(f"    ... and {len(issue_list) - 5} more")
    
    # Fence coverage statistics
    if fence_percentages:
        print(f"\n📈 Fence Coverage Statistics (from {len(fence_percentages)} valid samples):")
        print(f"  Mean:   {np.mean(fence_percentages):.2f}%")
        print(f"  Median: {np.median(fence_percentages):.2f}%")
        print(f"  Std Dev: {np.std(fence_percentages):.2f}%")
        print(f"  Min:    {np.min(fence_percentages):.2f}%")
        print(f"  Max:    {np.max(fence_percentages):.2f}%")
        
        # Distribution
        ranges = [
            (0, 1, "0-1%"),
            (1, 5, "1-5%"),
            (5, 10, "5-10%"),
            (10, 20, "10-20%"),
            (20, 50, "20-50%"),
            (50, 100, "50-100%")
        ]
        
        print(f"\n  Distribution:")
        for low, high, label in ranges:
            count = sum(1 for p in fence_percentages if low <= p < high)
            pct = (count / len(fence_percentages)) * 100
            print(f"    {label:12s}: {count:4d} samples ({pct:5.1f}%)")
    
    # Sample validation
    if valid_pairs > 0:
        print(f"\n🔬 Random Sample Validation (10 samples):")
        random.seed(42)
        sample_files = random.sample([f for f in image_files 
                                     if masks_dir / f.name in [masks_dir / img.name for img in image_files]], 
                                    min(10, valid_pairs))
        
        for img_file in sample_files:
            mask_file = masks_dir / img_file.name
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    fence_pct = (np.sum(mask == 255) / mask.size) * 100
                    print(f"  {img_file.name:50s} → {fence_pct:6.2f}% fence")
    
    # Critical warnings
    print(f"\n🚨 Critical Checks:")
    
    if len(issues['missing_masks']) > len(image_files) * 0.1:
        print(f"  ❌ CRITICAL: {len(issues['missing_masks'])} missing masks (>10%)")
    else:
        print(f"  ✓ Missing masks: {len(issues['missing_masks'])} (<10%)")
    
    if len(issues['invalid_values']) > 0:
        print(f"  ❌ CRITICAL: {len(issues['invalid_values'])} masks with invalid pixel values")
    else:
        print(f"  ✓ All masks have valid binary values (0, 255)")
    
    if fence_percentages:
        avg_fence = np.mean(fence_percentages)
        if avg_fence < 0.5:
            print(f"  ❌ WARNING: Average fence coverage is very low ({avg_fence:.2f}%)")
        elif avg_fence > 50:
            print(f"  ⚠️  WARNING: Average fence coverage is high ({avg_fence:.2f}%) - check for inverted masks")
        else:
            print(f"  ✓ Average fence coverage is reasonable ({avg_fence:.2f}%)")
    
    # Save report
    report = {
        'total_images': len(image_files),
        'valid_pairs': valid_pairs,
        'issues': {k: len(v) for k, v in issues.items()},
        'fence_stats': {
            'mean': float(np.mean(fence_percentages)) if fence_percentages else 0,
            'median': float(np.median(fence_percentages)) if fence_percentages else 0,
            'std': float(np.std(fence_percentages)) if fence_percentages else 0,
            'min': float(np.min(fence_percentages)) if fence_percentages else 0,
            'max': float(np.max(fence_percentages)) if fence_percentages else 0,
        }
    }
    
    with open('tests/reports/dataset_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report saved to: tests/reports/dataset_validation_report.json")
    print("="*80)
    
    # Final verdict
    if total_issues == 0 and valid_pairs == len(image_files):
        print("\n✅ DATASET IS READY FOR TRAINING!")
        return True
    else:
        print("\n⚠️  DATASET HAS ISSUES - Please review and fix before training")
        return False

if __name__ == "__main__":
    validate_dataset()
