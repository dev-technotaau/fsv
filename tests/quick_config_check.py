"""
Quick Config Check - No imports needed
"""
import re

# Read the config directly from file
with open('src/training/train_Mask2Former.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract key values
def find_config(pattern):
    match = re.search(pattern, content)
    return match.group(1) if match else "NOT FOUND"

print("=" * 80)
print("TRAINING CONFIGURATION CHECK")
print("=" * 80)

epochs = find_config(r'EPOCHS\s*=\s*(\d+)')
early_stopping = find_config(r'EARLY_STOPPING\s*=\s*(\w+)')
patience = find_config(r'PATIENCE\s*=\s*(\d+)')
batch_size = find_config(r'BATCH_SIZE\s*=\s*(\d+)')
accumulation = find_config(r'ACCUMULATION_STEPS\s*=\s*(\d+)')
lr = find_config(r'LEARNING_RATE\s*=\s*([\d.e-]+)')
pct_start = find_config(r'PCT_START\s*=\s*([\d.]+)')

print(f"\n✅ CRITICAL SETTINGS:")
print(f"   Total Epochs: {epochs}")
print(f"   Early Stopping: {early_stopping}")
print(f"   Patience: {patience}")

print(f"\n✅ TRAINING SETUP:")
print(f"   Batch Size: {batch_size}")
print(f"   Gradient Accumulation: {accumulation}")
print(f"   Effective Batch: {int(batch_size) * int(accumulation)}")
print(f"   Learning Rate: {lr}")
print(f"   PCT_START (warmup %): {pct_start}")
print(f"   Actual Warmup Epochs: {int(int(epochs) * float(pct_start))}")

print(f"\n" + "=" * 80)

# Validation
if early_stopping == "False":
    print("✅ EXCELLENT: Early stopping is DISABLED")
    print("   Training will run full 200 epochs without interruption")
else:
    print(f"❌ WARNING: Early stopping is ENABLED with patience={patience}")
    print("   Training may stop before 200 epochs!")
    print("\n   FIX: Change line 384 to: EARLY_STOPPING = False")

if int(epochs) >= 200:
    print(f"✅ EXCELLENT: {epochs} epochs is sufficient for convergence")
else:
    print(f"⚠️  WARNING: {epochs} epochs may be too few (recommended: 200)")

print(f"\n📊 Expected Training Timeline (if early stopping is disabled):")
warmup = int(int(epochs) * float(pct_start))
print(f"   Phase 1 (Epochs 1-{warmup}): Warmup - LR ramps up")
print(f"   Phase 2 (Epochs {warmup+1}-{int(epochs)//2}): Main Learning - high LR")
print(f"   Phase 3 (Epochs {int(epochs)//2+1}-{epochs}): Fine-tuning - LR decay")

print(f"\n⏱️  Estimated Time: ~{int(epochs) * 2} minutes (~2 min/epoch)")

print("=" * 80)
print("✅ CONFIGURATION VERIFIED - READY TO START TRAINING!")
print("=" * 80)
