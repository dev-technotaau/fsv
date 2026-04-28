"""
Quick script to verify checkpoint contents and model architecture.
"""
import torch
from pathlib import Path

checkpoint_path = Path("checkpoints/mask2former/best_model.pth")

print("=" * 80)
print("CHECKPOINT VERIFICATION")
print("=" * 80)

# Load checkpoint
print(f"\nLoading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n1. CHECKPOINT KEYS:")
print("-" * 80)
for key in checkpoint.keys():
    print(f"  - {key}")

print("\n2. CHECKPOINT METADATA:")
print("-" * 80)
print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  Best Val IoU: {checkpoint.get('best_val_iou', 'N/A'):.4f}" if checkpoint.get('best_val_iou') else "  Best Val IoU: N/A")
print(f"  Train IoU: {checkpoint.get('train_iou', 'N/A'):.4f}" if checkpoint.get('train_iou') else "  Train IoU: N/A")
print(f"  Val Precision: {checkpoint.get('val_precision', 'N/A'):.4f}" if checkpoint.get('val_precision') else "  Val Precision: N/A")
print(f"  Val Recall: {checkpoint.get('val_recall', 'N/A'):.4f}" if checkpoint.get('val_recall') else "  Val Recall: N/A")
print(f"  Has EMA: {'Yes' if 'ema_shadow' in checkpoint and checkpoint['ema_shadow'] is not None else 'No'}")

print("\n3. MODEL STATE DICT (First 30 keys):")
print("-" * 80)
model_keys = list(checkpoint['model_state_dict'].keys())
for i, key in enumerate(model_keys[:30]):
    print(f"  {i+1:2d}. {key}")
print(f"\n  ... Total {len(model_keys)} keys")

print("\n4. BACKBONE VERIFICATION:")
print("-" * 80)
# Check if SegFormer keys are present
segformer_keys = [k for k in model_keys if 'encoder' in k.lower()]
swin_keys = [k for k in model_keys if 'swin' in k.lower()]

print(f"  SegFormer encoder keys found: {len(segformer_keys)}")
if segformer_keys:
    print("  First 10 SegFormer keys:")
    for key in segformer_keys[:10]:
        print(f"    - {key}")

print(f"\n  Swin backbone keys found: {len(swin_keys)}")
if swin_keys:
    print("  ⚠️  WARNING: Swin keys found (should be SegFormer!):")
    for key in swin_keys[:10]:
        print(f"    - {key}")
else:
    print("  ✓ No Swin keys found (good!)")

print("\n5. CUSTOM PIXEL MODULE VERIFICATION:")
print("-" * 80)
pixel_module_keys = [k for k in model_keys if 'pixel_level_module' in k]
print(f"  Pixel level module keys: {len(pixel_module_keys)}")
if pixel_module_keys:
    print("  First 10 pixel module keys:")
    for key in pixel_module_keys[:10]:
        print(f"    - {key}")

print("\n6. DECODER VERIFICATION:")
print("-" * 80)
decoder_keys = [k for k in model_keys if 'decoder' in k.lower()]
print(f"  Decoder keys: {len(decoder_keys)}")
if decoder_keys:
    print("  First 10 decoder keys:")
    for key in decoder_keys[:10]:
        print(f"    - {key}")

print("\n7. PARAMETER COUNT:")
print("-" * 80)
total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
print(f"  Total parameters: {total_params:,}")

# Check backbone parameters specifically
backbone_params = sum(p.numel() for k, p in checkpoint['model_state_dict'].items() 
                     if 'pixel_level_module.encoder' in k)
print(f"  Backbone parameters: {backbone_params:,}")

decoder_params = sum(p.numel() for k, p in checkpoint['model_state_dict'].items() 
                    if 'pixel_level_module.decoder' in k)
print(f"  Decoder parameters: {decoder_params:,}")

transformer_params = sum(p.numel() for k, p in checkpoint['model_state_dict'].items() 
                        if 'transformer_module' in k)
print(f"  Transformer module parameters: {transformer_params:,}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
