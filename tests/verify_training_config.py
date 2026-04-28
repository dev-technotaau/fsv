"""
Verify Training Configuration Before Starting
Checks early stopping, epochs, and other critical settings
"""
import sys
sys.path.insert(0, 'src/training')

# Import config from training script
from train_Mask2Former import Config

print("=" * 80)
print("TRAINING CONFIGURATION VERIFICATION")
print("=" * 80)

print(f"\n✅ CRITICAL SETTINGS:")
print(f"   EPOCHS: {Config.EPOCHS}")
print(f"   EARLY_STOPPING: {Config.EARLY_STOPPING}")
print(f"   PATIENCE: {Config.PATIENCE}")
print(f"   MIN_DELTA: {Config.MIN_DELTA}")

print(f"\n✅ TRAINING PARAMETERS:")
print(f"   BATCH_SIZE: {Config.BATCH_SIZE}")
print(f"   ACCUMULATION_STEPS: {Config.ACCUMULATION_STEPS}")
print(f"   EFFECTIVE_BATCH: {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS}")
print(f"   LEARNING_RATE: {Config.LEARNING_RATE}")
print(f"   WARMUP_EPOCHS: {Config.WARMUP_EPOCHS}")
print(f"   PCT_START: {Config.PCT_START}")
print(f"   ACTUAL_WARMUP_EPOCHS: {int(Config.EPOCHS * Config.PCT_START)}")

print(f"\n✅ LOSS CONFIGURATION:")
print(f"   LOSS_WEIGHTS: {Config.LOSS_WEIGHTS}")
print(f"   CLASS_WEIGHT: {Config.CLASS_WEIGHT}")

print(f"\n✅ HARDWARE:")
print(f"   DEVICE: {Config.DEVICE}")
print(f"   USE_AMP: {Config.USE_AMP}")
print(f"   NUM_WORKERS: {Config.NUM_WORKERS}")

print(f"\n" + "=" * 80)

# Validation
issues = []

if Config.EARLY_STOPPING:
    issues.append("❌ EARLY_STOPPING is ENABLED - training may stop prematurely!")

if Config.EPOCHS < 100:
    issues.append(f"⚠️  EPOCHS = {Config.EPOCHS} is too low (recommended: 200)")

if Config.BATCH_SIZE * Config.ACCUMULATION_STEPS < 4:
    issues.append(f"⚠️  Effective batch size = {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS} is very small")

if issues:
    print("ISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("✅ ALL CHECKS PASSED - READY TO TRAIN!")
    print(f"\n📊 Expected Training Timeline:")
    warmup_epochs = int(Config.EPOCHS * Config.PCT_START)
    print(f"   Epochs 1-{warmup_epochs}: Warmup Phase (LR increases)")
    print(f"   Epochs {warmup_epochs+1}-{Config.EPOCHS//2}: Main Learning (high LR)")
    print(f"   Epochs {Config.EPOCHS//2+1}-{Config.EPOCHS}: Fine-tuning (LR decay)")
    print(f"\n⏱️  Estimated Time: ~{Config.EPOCHS * 2} minutes (2 min/epoch)")

print("=" * 80)
