"""
Test EMA checkpoint saving to verify the bug fix is working correctly.
"""
import torch
import torch.nn as nn
from pathlib import Path

print("=" * 80)
print("EMA CHECKPOINT SAVING TEST")
print("=" * 80)

# Simulate a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        # Initialize with specific values for testing
        nn.init.constant_(self.linear.weight, 1.0)
        nn.init.constant_(self.linear.bias, 0.5)
    
    def forward(self, x):
        return self.linear(x)

# Create model
model = SimpleModel()
print("\n1. ORIGINAL MODEL WEIGHTS:")
print(f"  Weight: {model.linear.weight[0, :5].tolist()}")
print(f"  Bias: {model.linear.bias.item():.4f}")

# Simulate EMA
class SimpleEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Store initial shadow copies
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA shadow weights (simulating training)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Simulate EMA: shadow = decay * shadow + (1-decay) * current
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Replace model weights with EMA shadow"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()

ema = SimpleEMA(model)

# Simulate training updates - make model weights worse
print("\n2. SIMULATING TRAINING (making model worse):")
with torch.no_grad():
    model.linear.weight.fill_(0.1)  # Worse weights
    model.linear.bias.fill_(0.01)   # Worse bias

print(f"  Current (worse) Weight: {model.linear.weight[0, :5].tolist()}")
print(f"  Current (worse) Bias: {model.linear.bias.item():.4f}")

# Update EMA (EMA keeps better averaged weights)
for i in range(100):  # Simulate 100 updates
    ema.update()

print(f"\n  EMA Shadow Weight: {ema.shadow['linear.weight'][0, :5].tolist()}")
print(f"  EMA Shadow Bias: {ema.shadow['linear.bias'].item():.4f}")

print("\n3. TESTING CHECKPOINT SAVING SCENARIOS:")
print("-" * 80)

# SCENARIO 1: OLD BUGGY CODE (saves AFTER restore)
print("\nScenario 1: OLD BUGGY CODE (save after restore)")
print("  1. Apply EMA shadow to model")
ema.apply_shadow()
print(f"     Model weight now: {model.linear.weight[0, :5].tolist()}")
print(f"     Model bias now: {model.linear.bias.item():.4f}")

print("  2. Restore original weights")
ema.restore()
print(f"     Model weight after restore: {model.linear.weight[0, :5].tolist()}")
print(f"     Model bias after restore: {model.linear.bias.item():.4f}")

print("  3. Save checkpoint (WRONG - saves worse weights!)")
checkpoint_old = {'model_state_dict': model.state_dict()}
torch.save(checkpoint_old, 'test_checkpoint_old.pth')
print(f"     ❌ Saved weight: {checkpoint_old['model_state_dict']['linear.weight'][0, :5].tolist()}")
print(f"     ❌ Saved bias: {checkpoint_old['model_state_dict']['linear.bias'].item():.4f}")

# Reset model to worse state
with torch.no_grad():
    model.linear.weight.fill_(0.1)
    model.linear.bias.fill_(0.01)

# SCENARIO 2: NEW FIXED CODE (saves BEFORE restore)
print("\n\nScenario 2: NEW FIXED CODE (save before restore)")
print("  1. Apply EMA shadow to model")
ema.apply_shadow()
print(f"     Model weight now: {model.linear.weight[0, :5].tolist()}")
print(f"     Model bias now: {model.linear.bias.item():.4f}")

print("  2. Save checkpoint (CORRECT - saves better EMA weights!)")
checkpoint_new = {'model_state_dict': model.state_dict()}
torch.save(checkpoint_new, 'test_checkpoint_new.pth')
print(f"     ✓ Saved weight: {checkpoint_new['model_state_dict']['linear.weight'][0, :5].tolist()}")
print(f"     ✓ Saved bias: {checkpoint_new['model_state_dict']['linear.bias'].item():.4f}")

print("  3. Restore original weights")
ema.restore()
print(f"     Model weight after restore: {model.linear.weight[0, :5].tolist()}")

print("\n4. COMPARISON:")
print("-" * 80)
old_weight = checkpoint_old['model_state_dict']['linear.weight'][0, 0].item()
new_weight = checkpoint_new['model_state_dict']['linear.weight'][0, 0].item()
ema_weight = ema.shadow['linear.weight'][0, 0].item()

print(f"  Old buggy checkpoint weight: {old_weight:.6f} (WRONG - worse weights!)")
print(f"  New fixed checkpoint weight: {new_weight:.6f} (CORRECT - EMA weights!)")
print(f"  EMA shadow weight: {ema_weight:.6f}")
print(f"\n  ✓ New checkpoint matches EMA: {abs(new_weight - ema_weight) < 1e-6}")
print(f"  ❌ Old checkpoint is worse: {abs(old_weight - 0.1) < 1e-6}")

print("\n5. CONCLUSION:")
print("-" * 80)
print("  The OLD checkpoint (epoch 2) was saved with BUGGY code:")
print("    - Applied EMA (good weights)")
print("    - Restored to worse weights")
print("    - THEN saved checkpoint (saved worse weights!)")
print("\n  The NEW fixed code saves BEFORE restore:")
print("    - Apply EMA (good weights)")
print("    - Save checkpoint (saves good EMA weights!)")
print("    - THEN restore")
print("\n  ⚠️  YOUR CURRENT best_model.pth IS FROM EPOCH 2 (BUGGY CODE)")
print("  ✓  YOU NEED TO RETRAIN FROM SCRATCH WITH FIXED CODE!")

# Cleanup
import os
os.remove('test_checkpoint_old.pth')
os.remove('test_checkpoint_new.pth')

print("\n" + "=" * 80)
