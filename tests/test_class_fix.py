"""Test fixing the class predictor properly"""
import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

config = Mask2FormerConfig.from_pretrained('facebook/mask2former-swin-base-coco-panoptic')
print(f"Original config num_labels: {config.num_labels}")
print(f"Original config num_classes: {config.num_classes if hasattr(config, 'num_classes') else 'Not set'}")

# Set to 2 classes
config.num_labels = 2
config.num_classes = 2

model = Mask2FormerForUniversalSegmentation(config)

print(f"\n=== BEFORE FIX ===")
print(f"class_predictor output features: {model.class_predictor.out_features}")

dummy_input = torch.randn(1, 3, 384, 384)
output = model(pixel_values=dummy_input)
print(f"Class queries logits shape: {output.class_queries_logits.shape}")

# Now try to fix it
print(f"\n=== APPLYING FIX ===")
hidden_dim = config.hidden_dim  # 256
num_classes = 2

# Try different paths
try:
    model.class_predictor = nn.Linear(hidden_dim, num_classes + 1)
    print(f"✓ Replaced model.class_predictor")
except Exception as e:
    print(f"✗ model.class_predictor failed: {e}")

print(f"\n=== AFTER FIX ===")
print(f"class_predictor output features: {model.class_predictor.out_features}")

output = model(pixel_values=dummy_input)
print(f"Class queries logits shape: {output.class_queries_logits.shape}")
print(f"Expected: torch.Size([1, 100, 3]) for 2 classes + no-object")
