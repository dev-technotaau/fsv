"""Quick script to find the class predictor path in Mask2Former"""
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

config = Mask2FormerConfig.from_pretrained('facebook/mask2former-swin-base-coco-panoptic')
model = Mask2FormerForUniversalSegmentation(config)

print("\n=== SEARCHING FOR CLASS PREDICTOR ===\n")
for name, module in model.named_modules():
    if 'class' in name.lower() and isinstance(module, torch.nn.Linear):
        print(f"Found: {name}")
        print(f"  Type: {type(module).__name__}")
        print(f"  Input: {module.in_features}, Output: {module.out_features}")
        print()

print("\n=== TESTING FORWARD PASS ===\n")
dummy_input = torch.randn(1, 3, 384, 384)
output = model(pixel_values=dummy_input)
print(f"Class queries logits shape: {output.class_queries_logits.shape}")
print(f"Expected: [1, 100, num_classes]")
print(f"Actual num_classes in output: {output.class_queries_logits.shape[-1]}")
