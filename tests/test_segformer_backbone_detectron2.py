"""
SegFormer-B5 Backbone Verification for Detectron2 Mask2Former
==============================================================
This script verifies that SegFormer-B5 is properly integrated as the backbone
in the Detectron2 Mask2Former training script.

Tests:
1. SegFormer-B5 model loads correctly
2. Custom backbone wrapper creates multi-scale features
3. Features have correct shapes and channels
4. Integration with Detectron2 format works
5. Forward pass succeeds

Expected Output:
- SegFormer-B5 with [64, 128, 320, 512] channels
- Multi-scale features at strides [4, 8, 16, 32]
- Detectron2 compatible output format (res2, res3, res4, res5)
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from transformers import SegformerModel, SegformerConfig
from transformers.modeling_outputs import BaseModelOutput

# ============================================================================
# CUSTOM SEGFORMER BACKBONE INTEGRATION (Same as training script)
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


class SegFormerBackbone(nn.Module):
    """
    SegFormer-B5 backbone for Detectron2 Mask2Former.
    Uses custom wrapper for proper multi-scale feature extraction.
    """
    
    def __init__(self, pretrained_weights="nvidia/segformer-b5-finetuned-ade-640-640"):
        super().__init__()
        
        # Load pretrained SegFormer-B5
        print(f"Loading pretrained SegFormer-B5 from {pretrained_weights}...")
        segformer_pretrained = SegformerModel.from_pretrained(pretrained_weights)
        
        # Wrap in custom backbone wrapper
        print("Creating custom SegFormer backbone wrapper for Detectron2...")
        self.segformer = SegFormerBackboneWrapper(segformer_pretrained)
        
        # Feature dimensions: [64, 128, 320, 512] for B5
        self.num_channels = self.segformer.num_channels
        
        # Output feature strides: [4, 8, 16, 32]
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": self.num_channels[0],
            "res3": self.num_channels[1],
            "res4": self.num_channels[2],
            "res5": self.num_channels[3]
        }
        
        print(f"✓ Custom SegFormer-B5 Backbone (Detectron2):")
        print(f"  - Feature channels: {self.num_channels}")
        print(f"  - Feature strides: [4, 8, 16, 32]")
        print(f"  - Encoder type: {type(self.segformer.encoder).__name__}")
    
    def forward(self, x):
        """
        Forward pass compatible with Detectron2.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dict with multi-scale features in Detectron2 format
        """
        # Get multi-scale features from custom SegFormer wrapper
        outputs = self.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract 4-scale features from hidden_states
        features = outputs.hidden_states  # Tuple of 4 features
        
        # Return in Detectron2 format (dict with res2, res3, res4, res5)
        return {
            "res2": features[0],  # 64 channels, stride 4
            "res3": features[1],  # 128 channels, stride 8
            "res4": features[2],  # 320 channels, stride 16
            "res5": features[3]   # 512 channels, stride 32
        }
    
    def output_shape(self):
        """Return output shape info for Detectron2."""
        return {
            name: {"channels": self._out_feature_channels[name], "stride": self._out_feature_strides[name]}
            for name in ["res2", "res3", "res4", "res5"]
        }


# ============================================================================
# VERIFICATION TESTS
# ============================================================================

def verify_segformer_detectron2():
    """Run comprehensive verification of SegFormer-B5 integration with Detectron2."""
    
    print("\n" + "="*80)
    print("SEGFORMER-B5 BACKBONE VERIFICATION FOR DETECTRON2 MASK2FORMER")
    print("="*80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    results = {
        "device": str(device),
        "tests_passed": [],
        "tests_failed": [],
        "model_info": {},
        "feature_info": {}
    }
    
    try:
        # Test 1: Load SegFormer-B5 pretrained model
        print("[1/6] Loading pretrained SegFormer-B5 model...")
        segformer_pretrained = SegformerModel.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        
        # Verify SegFormer-B5 configuration
        config = segformer_pretrained.config
        print(f"  - Hidden sizes: {config.hidden_sizes}")
        print(f"  - Depths: {config.depths}")
        
        expected_hidden_sizes = [64, 128, 320, 512]
        if config.hidden_sizes == expected_hidden_sizes:
            print("  ✅ SegFormer-B5 specifications verified!")
            results["tests_passed"].append("SegFormer-B5 pretrained model loaded")
            results["model_info"]["hidden_sizes"] = config.hidden_sizes
            results["model_info"]["depths"] = config.depths
        else:
            print(f"  ❌ Unexpected hidden sizes: {config.hidden_sizes}")
            results["tests_failed"].append("SegFormer-B5 config mismatch")
        
        # Test 2: Create custom backbone wrapper
        print("\n[2/6] Creating custom SegFormer backbone wrapper...")
        wrapper = SegFormerBackboneWrapper(segformer_pretrained)
        print(f"  - Wrapper type: {type(wrapper).__name__}")
        print(f"  - Encoder type: {type(wrapper.encoder).__name__}")
        print(f"  - Num channels: {wrapper.num_channels}")
        
        if wrapper.num_channels == [64, 128, 320, 512]:
            print("  ✅ Custom backbone wrapper created and verified!")
            results["tests_passed"].append("Custom backbone wrapper created")
        else:
            print(f"  ❌ Unexpected num_channels: {wrapper.num_channels}")
            results["tests_failed"].append("Wrapper channel mismatch")
        
        # Test 3: Create Detectron2 compatible backbone
        print("\n[3/6] Building Detectron2 SegFormer-B5 backbone...")
        backbone = SegFormerBackbone("nvidia/segformer-b5-finetuned-ade-640-640")
        backbone.to(device)
        backbone.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        results["model_info"]["total_parameters"] = total_params
        results["model_info"]["trainable_parameters"] = trainable_params
        results["model_info"]["backbone_type"] = type(backbone).__name__
        results["model_info"]["wrapper_type"] = type(backbone.segformer).__name__
        
        print("  ✅ Detectron2 backbone built successfully!")
        results["tests_passed"].append("Detectron2 backbone created")
        
        # Test 4: Verify output_shape method
        print("\n[4/6] Verifying output_shape method...")
        output_shape = backbone.output_shape()
        print("  Output shape info:")
        for name, info in output_shape.items():
            print(f"    - {name}: channels={info['channels']}, stride={info['stride']}")
        
        expected_shapes = {
            "res2": {"channels": 64, "stride": 4},
            "res3": {"channels": 128, "stride": 8},
            "res4": {"channels": 320, "stride": 16},
            "res5": {"channels": 512, "stride": 32}
        }
        
        if output_shape == expected_shapes:
            print("  ✅ Output shape info correct!")
            results["tests_passed"].append("Output shape verification")
            results["feature_info"]["output_shape"] = output_shape
        else:
            print("  ❌ Output shape mismatch!")
            results["tests_failed"].append("Output shape mismatch")
        
        # Test 5: Forward pass test
        print("\n[5/6] Testing forward pass...")
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        print(f"  - Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            features = backbone(dummy_input)
        
        print(f"  - Output type: {type(features)}")
        print(f"  - Number of feature levels: {len(features)}")
        
        # Verify each feature level
        print("  - Feature shapes:")
        for name, feat in features.items():
            print(f"    - {name}: {feat.shape}")
        
        # Expected shapes for 512x512 input
        expected_feat_shapes = {
            "res2": (1, 64, 128, 128),   # stride 4: 512/4 = 128
            "res3": (1, 128, 64, 64),    # stride 8: 512/8 = 64
            "res4": (1, 320, 32, 32),    # stride 16: 512/16 = 32
            "res5": (1, 512, 16, 16)     # stride 32: 512/32 = 16
        }
        
        shapes_match = all(
            features[name].shape == expected_feat_shapes[name]
            for name in ["res2", "res3", "res4", "res5"]
        )
        
        if shapes_match:
            print("  ✅ Forward pass successful with correct shapes!")
            results["tests_passed"].append("Forward pass")
            results["feature_info"]["feature_shapes"] = {
                name: list(feat.shape) for name, feat in features.items()
            }
        else:
            print("  ❌ Feature shapes don't match expected!")
            results["tests_failed"].append("Feature shape mismatch")
        
        # Test 6: Multi-scale feature extraction
        print("\n[6/6] Extracting and verifying multi-scale features...")
        
        feature_stats = {}
        for name, feat in features.items():
            stats = {
                "shape": list(feat.shape),
                "channels": feat.shape[1],
                "spatial": (feat.shape[2], feat.shape[3]),
                "mean": float(feat.mean()),
                "std": float(feat.std()),
                "min": float(feat.min()),
                "max": float(feat.max())
            }
            feature_stats[name] = stats
            print(f"  - {name}: shape={stats['shape']}, channels={stats['channels']}, "
                  f"spatial={stats['spatial']}")
        
        results["feature_info"]["feature_stats"] = feature_stats
        
        # Verify channel counts match SegFormer-B5 spec
        channel_counts = [features[name].shape[1] for name in ["res2", "res3", "res4", "res5"]]
        expected_channels = [64, 128, 320, 512]
        
        if channel_counts == expected_channels:
            print(f"  ✅ Multi-scale features verified! Channels: {channel_counts}")
            results["tests_passed"].append("Multi-scale feature extraction")
        else:
            print(f"  ❌ Channel mismatch! Got {channel_counts}, expected {expected_channels}")
            results["tests_failed"].append("Channel count mismatch")
        
    except Exception as e:
        print(f"\n❌ ERROR during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        results["tests_failed"].append(f"Exception: {str(e)}")
        return results
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    total_tests = len(results["tests_passed"]) + len(results["tests_failed"])
    passed_tests = len(results["tests_passed"])
    
    print(f"\nTests Passed: {passed_tests}/{total_tests}")
    
    if results["tests_passed"]:
        print("\n✅ Passed:")
        for test in results["tests_passed"]:
            print(f"  - {test}")
    
    if results["tests_failed"]:
        print("\n❌ Failed:")
        for test in results["tests_failed"]:
            print(f"  - {test}")
    
    # Final verdict
    print("\n" + "="*80)
    if len(results["tests_failed"]) == 0:
        print("🎉 ALL TESTS PASSED!")
        print("\n✓ SegFormer-B5 is PROPERLY integrated as Detectron2 Mask2Former backbone!")
        print(f"✓ Total parameters: {results['model_info']['total_parameters']:,}")
        print(f"✓ Multi-scale features: {expected_channels} channels")
        print(f"✓ Feature strides: [4, 8, 16, 32]")
        print(f"✓ Detectron2 format: res2, res3, res4, res5")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Please review the errors above.")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    # Run verification
    results = verify_segformer_detectron2()
    
    # Save results
    output_file = Path("tests/reports/custom_segformer_detectron2_verification.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Verification results saved to: {output_file}")
