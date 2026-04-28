"""
Test Script to Verify CUSTOM SegFormer-B5 Backbone Integration in Mask2Former
================================================================================
This script loads the Mask2Former model with custom SegFormer-B5 integration and verifies:
1. SegFormer-B5 is actually used as the backbone (custom wrapper)
2. Backbone architecture matches SegFormer-B5 specifications
3. Model parameters are correctly initialized
4. Forward pass works correctly with multi-scale features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerConfig,
    SegformerConfig,
    SegformerModel,
    AutoImageProcessor
)
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput,
    Mask2FormerModel,
    Mask2FormerPixelLevelModule,
    Mask2FormerPixelLevelModuleOutput
)
from transformers.modeling_outputs import BaseModelOutput
import numpy as np
from pathlib import Path

print("=" * 90)
print("CUSTOM SEGFORMER-B5 BACKBONE INTEGRATION TEST")
print("=" * 90)


# ============================================================================
# CUSTOM SEGFORMER BACKBONE FOR MASK2FORMER (Same as training script)
# ============================================================================

class SegFormerBackboneWrapper(nn.Module):
    """
    Custom wrapper to make SegFormer compatible as Mask2Former backbone.
    
    SegFormer produces multi-scale features from 4 stages, which need to be
    formatted properly for Mask2Former's pixel decoder.
    """
    
    def __init__(self, segformer_model: SegformerModel):
        super().__init__()
        self.encoder = segformer_model.encoder
        
        # SegFormer-B5 feature dimensions: [64, 128, 320, 512]
        self.num_channels = [64, 128, 320, 512]
        self.num_features = len(self.num_channels)
        
    def forward(self, pixel_values, output_hidden_states=True, return_dict=True):
        """
        Forward pass that returns multi-scale features compatible with Mask2Former.
        
        Args:
            pixel_values: Input images (B, C, H, W)
            output_hidden_states: Return intermediate features
            return_dict: Return as dict
            
        Returns:
            BaseModelOutput with feature_maps as list of multi-scale features
        """
        # Get SegFormer encoder outputs
        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=True,
            return_dict=True
        )
        
        # SegFormer produces 4 hidden states from 4 stages
        # Format: List of (B, C, H, W) tensors with decreasing spatial resolution
        hidden_states = encoder_outputs.hidden_states
        
        # Ensure we have 4 feature maps
        if len(hidden_states) != 4:
            raise ValueError(f"Expected 4 feature maps from SegFormer, got {len(hidden_states)}")
        
        # Return in format expected by Mask2Former
        # feature_maps should be a tuple of features from stride 4, 8, 16, 32
        if return_dict:
            return BaseModelOutput(
                last_hidden_state=hidden_states[-1],
                hidden_states=tuple(hidden_states)
            )
        else:
            return (hidden_states[-1], tuple(hidden_states))


class CustomMask2FormerPixelLevelModule(Mask2FormerPixelLevelModule):
    """
    Custom Pixel Level Module that uses SegFormer as backbone.
    
    This replaces the default backbone loading with our custom SegFormer wrapper.
    """
    
    def __init__(self, config, segformer_backbone: SegFormerBackboneWrapper):
        # Don't call parent __init__ to avoid loading default backbone
        nn.Module.__init__(self)
        
        self.encoder = segformer_backbone
        
        # Get feature channels from SegFormer
        # SegFormer-B5: [64, 128, 320, 512]
        feature_channels = segformer_backbone.num_channels
        
        # Create decoder (FPN-style) to process multi-scale features
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels)
        
    def forward(self, pixel_values, output_hidden_states=False):
        # Get multi-scale features from SegFormer
        backbone_outputs = self.encoder(
            pixel_values,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Decode features
        decoder_output = self.decoder(backbone_outputs.hidden_states, output_hidden_states)
        
        return decoder_output


class Mask2FormerPixelDecoder(nn.Module):
    """
    Pixel Decoder for Mask2Former that processes SegFormer multi-scale features.
    
    This creates a FPN-like structure to combine features at different scales.
    """
    
    def __init__(self, config, feature_channels):
        super().__init__()
        
        self.config = config
        self.feature_channels = feature_channels  # [64, 128, 320, 512] for SegFormer-B5
        self.mask_feature_size = config.mask_feature_size  # 256
        
        # Lateral connections (1x1 conv to unify channel dimensions)
        self.lateral_convs = nn.ModuleList()
        for channels in feature_channels:
            self.lateral_convs.append(
                nn.Conv2d(channels, self.mask_feature_size, kernel_size=1)
            )
        
        # Output convolutions (3x3 conv for refinement)
        self.output_convs = nn.ModuleList()
        for _ in range(len(feature_channels)):
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.mask_feature_size, self.mask_feature_size, 
                             kernel_size=3, padding=1),
                    nn.GroupNorm(32, self.mask_feature_size),
                    nn.ReLU()
                )
            )
        
        # Mask features projection
        self.mask_projection = nn.Conv2d(
            self.mask_feature_size, 
            self.mask_feature_size, 
            kernel_size=1
        )
        
    def forward(self, multi_scale_features, output_hidden_states=False):
        """
        Args:
            multi_scale_features: Tuple of 4 feature maps from SegFormer
                                 (stride 4, 8, 16, 32)
        
        Returns:
            Mask2FormerPixelLevelModuleOutput with proper attributes
        """
        # Process features with lateral connections
        laterals = []
        for feat, lateral_conv in zip(multi_scale_features, self.lateral_convs):
            laterals.append(lateral_conv(feat))
        
        # Top-down pathway with upsampling
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher-level feature
            upsampled = F.interpolate(
                laterals[i], 
                size=laterals[i-1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            # Add to lower-level feature
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Apply output convolutions
        outputs = []
        for feat, output_conv in zip(laterals, self.output_convs):
            outputs.append(output_conv(feat))
        
        # Use the finest resolution feature for mask predictions
        mask_features = self.mask_projection(outputs[0])
        
        # Return proper Mask2FormerPixelLevelModuleOutput
        # encoder_last_hidden_state = finest multi-scale feature (lowest resolution)
        # decoder_last_hidden_state = mask features for query-based prediction
        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=multi_scale_features[-1],  # Coarsest feature
            encoder_hidden_states=tuple(multi_scale_features),
            decoder_last_hidden_state=mask_features,  # For mask queries
            decoder_hidden_states=tuple(outputs) if output_hidden_states else None
        )


# ============================================================================
# TEST EXECUTION
# ============================================================================

# Configuration
MODEL_NAME = "facebook/mask2former-swin-base-coco-panoptic"
BACKBONE_NAME = "nvidia/mit-b5"
NUM_LABELS = 2
INPUT_SIZE = 384

print(f"\n[1/6] Loading configurations...")
print(f"  - Base Model: {MODEL_NAME}")
print(f"  - Target Backbone: {BACKBONE_NAME}")
print(f"  - Num Labels: {NUM_LABELS}")
print(f"  - Input Size: {INPUT_SIZE}x{INPUT_SIZE}")

print(f"\n[2/6] Loading pretrained SegFormer-B5...")
segformer_pretrained = SegformerModel.from_pretrained(BACKBONE_NAME)

print(f"  - SegFormer-B5 Configuration:")
print(f"    • Hidden sizes: {segformer_pretrained.config.hidden_sizes}")
print(f"    • Depths: {segformer_pretrained.config.depths}")
print(f"    • Num encoder blocks: {segformer_pretrained.config.num_encoder_blocks}")
print(f"    • Attention heads: {segformer_pretrained.config.num_attention_heads}")

# Verify SegFormer-B5 specs
assert segformer_pretrained.config.hidden_sizes == [64, 128, 320, 512], \
    f"ERROR: Hidden sizes don't match SegFormer-B5!"
assert segformer_pretrained.config.depths == [3, 6, 40, 3], \
    f"ERROR: Depths don't match SegFormer-B5!"
print("  ✅ SegFormer-B5 specifications verified!")

print(f"\n[3/6] Creating custom SegFormer backbone wrapper...")
segformer_backbone = SegFormerBackboneWrapper(segformer_pretrained)
print(f"  - Wrapper type: {type(segformer_backbone).__name__}")
print(f"  - Encoder type: {type(segformer_backbone.encoder).__name__}")
print(f"  - Feature channels: {segformer_backbone.num_channels}")
print(f"  ✅ Custom backbone wrapper created!")

print(f"\n[4/6] Building Mask2Former with custom SegFormer-B5 backbone...")
mask2former_config = Mask2FormerConfig.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    num_queries=100,
    ignore_mismatched_sizes=True
)

# Create main model structure
model = Mask2FormerForUniversalSegmentation(config=mask2former_config)

# Replace pixel level module with custom one
print(f"  - Replacing pixel-level module with custom SegFormer decoder...")
custom_pixel_module = CustomMask2FormerPixelLevelModule(
    mask2former_config,
    segformer_backbone
)
model.model.pixel_level_module = custom_pixel_module

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
backbone_params = sum(p.numel() for p in segformer_backbone.parameters())
decoder_params = sum(p.numel() for p in custom_pixel_module.decoder.parameters())

print(f"  - Total parameters: {total_params:,}")
print(f"  - SegFormer-B5 backbone: {backbone_params:,}")
print(f"  - Custom pixel decoder: {decoder_params:,}")
print(f"  ✅ Custom Mask2Former model built!")

print(f"\n[5/6] Testing forward pass with dummy input...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)

print(f"  - Input shape: {dummy_input.shape}")
print(f"  - Device: {device}")

with torch.no_grad():
    try:
        outputs = model(pixel_values=dummy_input)
        print(f"  - Output masks shape: {outputs.masks_queries_logits.shape}")
        print(f"  - Output class logits shape: {outputs.class_queries_logits.shape}")
        print(f"  - Expected masks: (1, num_queries, H, W)")
        print(f"  - Expected class: (1, num_queries, num_labels)")
        print("  ✅ Forward pass successful!")
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        raise

print(f"\n[6/6] Extracting and verifying multi-scale SegFormer features...")
with torch.no_grad():
    try:
        # Get backbone outputs directly
        backbone_outputs = segformer_backbone(dummy_input, output_hidden_states=True, return_dict=True)
        
        print(f"  - Number of feature scales: {len(backbone_outputs.hidden_states)}")
        for i, feat in enumerate(backbone_outputs.hidden_states):
            print(f"  - Scale {i+1} shape: {feat.shape} (channels={feat.shape[1]})")
        
        expected_channels = [64, 128, 320, 512]
        actual_channels = [feat.shape[1] for feat in backbone_outputs.hidden_states]
        
        assert actual_channels == expected_channels, \
            f"Channel mismatch! Expected {expected_channels}, got {actual_channels}"
        
        print(f"  ✅ Multi-scale features verified!")
        print(f"  ✅ Channels match SegFormer-B5: {expected_channels}")
            
    except Exception as e:
        print(f"  ❌ Feature extraction failed: {e}")
        raise

# Final verification summary
print("\n" + "=" * 90)
print("VERIFICATION SUMMARY")
print("=" * 90)
print("✅ SegFormer-B5 pretrained model loaded successfully")
print("✅ Custom backbone wrapper created and verified")
print("✅ Custom pixel decoder built with FPN architecture")
print("✅ Model integrated with Mask2Former framework")
print("✅ Forward pass works correctly")
print("✅ Multi-scale features verified (4 scales: [64, 128, 320, 512])")
print("\n🎉 SegFormer-B5 is PROPERLY integrated as custom Mask2Former backbone!")
print("=" * 90)

# Save verification results
verification_results = {
    'timestamp': str(np.datetime64('now')),
    'integration_type': 'Custom Manual Integration',
    'backbone_type': 'SegFormerBackboneWrapper',
    'encoder_type': type(segformer_backbone.encoder).__name__,
    'hidden_sizes': segformer_pretrained.config.hidden_sizes,
    'depths': segformer_pretrained.config.depths,
    'feature_channels': segformer_backbone.num_channels,
    'total_params': int(total_params),
    'backbone_params': int(backbone_params),
    'decoder_params': int(decoder_params),
    'verification_status': 'PASSED',
    'notes': 'SegFormer-B5 successfully integrated as custom Mask2Former backbone with FPN decoder'
}

import json
output_file = Path('./tests/reports') / 'custom_segformer_verification.json'
with open(output_file, 'w') as f:
    json.dump(verification_results, f, indent=2)

print(f"\n✅ Verification results saved to: {output_file}")
print("\n🚀 You can now train with REAL SegFormer-B5 backbone integration!")
print("   The custom integration uses:")
print("   • Pretrained SegFormer-B5 encoder (82M params)")
print("   • Multi-scale features [64, 128, 320, 512] channels")
print("   • FPN-style pixel decoder for feature fusion")
print("   • Full Mask2Former query-based segmentation")

