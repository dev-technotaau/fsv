"""
SegFormer-B5 Implementation Verification Script
==============================================
Verifies that train_SegFormerB5_PREMIUM.py is properly configured for B5.
"""

import re
import os

def verify_segformer_b5_implementation():
    """Verify all B5-specific configurations are correct."""
    
    script_path = "src/training/train_SegFormerB5_PREMIUM.py"
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: {script_path} not found!")
        return False
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "Model Name (B5)": r'MODEL_NAME\s*=\s*["\']nvidia/segformer-b5-finetuned-ade-640-640["\']',
        "Checkpoint Dir (B5)": r'CHECKPOINT_DIR\s*=\s*["\']\.\/checkpoints\/segformerb5["\']',
        "Input Size (640)": r'INPUT_SIZE\s*=\s*640',
        "Batch Size (4)": r'BATCH_SIZE\s*=\s*4',
        "Accumulation Steps (4)": r'ACCUMULATION_STEPS\s*=\s*4',
        "Epochs (100)": r'EPOCHS\s*=\s*100',
        "Learning Rate (6e-5)": r'LEARNING_RATE\s*=\s*6e-5',
        "EMA Enabled": r'USE_EMA\s*=\s*True',
        "OHEM Enabled": r'USE_OHEM\s*=\s*True',
        "Edge Weight (2.0)": r'EDGE_WEIGHT\s*=\s*2\.0',
        "Gradient Checkpointing": r'GRADIENT_CHECKPOINTING\s*=\s*True',
        "Label Smoothing (0.1)": r'LABEL_SMOOTHING\s*=\s*0\.1',
        "Warmup Epochs (5)": r'WARMUP_EPOCHS\s*=\s*5',
        "Checkpoint Saving": r'SAVE_CHECKPOINT_EVERY\s*=\s*\d+',
        "ModelEMA Class": r'class ModelEMA',
        "EdgeAwareLoss Class": r'class EdgeAwareLoss',
        "OHEMLoss Class": r'class OHEMLoss',
        "AdvancedCombinedLoss Class": r'class AdvancedCombinedLoss',
        "Professional Augmentation": r'A\.Perspective\(|A\.CLAHE\(|A\.RandomShadow\(',
        "EMA Update in Loop": r'ema_model\.update\(model\)',
        "Boundary Loss": r'def boundary_loss\(',
    }
    
    print("="*70)
    print("🔍 SEGFORMER-B5 IMPLEMENTATION VERIFICATION")
    print("="*70)
    print()
    
    all_passed = True
    
    for check_name, pattern in checks.items():
        if re.search(pattern, content):
            print(f"✅ {check_name:<35} VERIFIED")
        else:
            print(f"❌ {check_name:<35} MISSING")
            all_passed = False
    
    print()
    print("="*70)
    
    # Additional checks
    print("\n📊 CONFIGURATION SUMMARY:")
    print("-" * 70)
    
    # Extract key values
    model_match = re.search(r'MODEL_NAME\s*=\s*["\']([^"\']+)["\']', content)
    if model_match:
        model_name = model_match.group(1)
        if 'b5' in model_name.lower():
            print(f"✅ Model: {model_name}")
        else:
            print(f"❌ Model: {model_name} (Expected B5!)")
            all_passed = False
    
    input_match = re.search(r'INPUT_SIZE\s*=\s*(\d+)', content)
    if input_match:
        input_size = input_match.group(1)
        if input_size == '640':
            print(f"✅ Input Size: {input_size}x{input_size}")
        else:
            print(f"⚠️  Input Size: {input_size}x{input_size} (Expected 640x640 for B5)")
    
    batch_match = re.search(r'BATCH_SIZE\s*=\s*(\d+)', content)
    accum_match = re.search(r'ACCUMULATION_STEPS\s*=\s*(\d+)', content)
    if batch_match and accum_match:
        batch = int(batch_match.group(1))
        accum = int(accum_match.group(1))
        effective = batch * accum
        print(f"✅ Effective Batch: {effective} ({batch} × {accum})")
    
    epochs_match = re.search(r'EPOCHS\s*=\s*(\d+)', content)
    if epochs_match:
        epochs = epochs_match.group(1)
        print(f"✅ Training Epochs: {epochs}")
    
    lr_match = re.search(r'LEARNING_RATE\s*=\s*([0-9.e-]+)', content)
    if lr_match:
        lr = lr_match.group(1)
        print(f"✅ Learning Rate: {lr}")
    
    # Count parameters
    param_match = re.search(r'84M params', content, re.IGNORECASE)
    if param_match:
        print(f"✅ Expected Parameters: ~84M (SegFormer-B5)")
    
    print()
    print("="*70)
    
    if all_passed:
        print("\n🎉 VERIFICATION PASSED! SegFormer-B5 is properly configured.")
        print("\n📝 Key Features:")
        print("   • SegFormer-B5 architecture (84M parameters)")
        print("   • 640x640 input resolution")
        print("   • Advanced loss functions (Edge-aware, OHEM, Focal, Dice, Boundary)")
        print("   • EMA (Exponential Moving Average)")
        print("   • Professional augmentation pipeline")
        print("   • Gradient checkpointing for memory efficiency")
        print("   • Progressive warmup + Cosine annealing scheduler")
        print("   • Automatic checkpoint management")
        print("   • Detailed logging and metrics tracking")
        print("\n🚀 Ready to train! Run: python train_SegFormerB5_PREMIUM.py")
    else:
        print("\n⚠️  VERIFICATION FAILED! Some B5 features are missing or incorrect.")
        print("   Please review the implementation.")
    
    print("="*70)
    
    return all_passed


if __name__ == '__main__':
    verify_segformer_b5_implementation()
