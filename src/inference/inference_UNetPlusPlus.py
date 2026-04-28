"""
UNet++ Inference Script - Quick Start
======================================
Production-ready inference for fence detection.

Usage:
    python inference_UNetPlusPlus.py --image path/to/image.jpg
    python inference_UNetPlusPlus.py --folder path/to/images/ --output results/
    python inference_UNetPlusPlus.py --image test.jpg --tta  # Use TTA for higher accuracy
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt

# Import from training script
import sys
sys.path.append(str(Path(__file__).parent))


class UNetPlusPlusInference:
    """Simple inference wrapper for UNet++."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        use_amp: bool = True,
        use_tta: bool = False
    ):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            use_amp: Use mixed precision
            use_tta: Use test-time augmentation
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp
        self.use_tta = use_tta
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            print("Installing segmentation-models-pytorch...")
            import os
            os.system("pip install segmentation-models-pytorch")
            import segmentation_models_pytorch as smp
        
        # Create model
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights=None,  # Will load from checkpoint
            in_channels=3,
            classes=1,
            activation=None,
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"   Best validation IoU: {checkpoint.get('best_val_iou', 'unknown'):.4f}")
        else:
            self.model.load_state_dict(checkpoint)
            print("✅ Loaded checkpoint")
        
        self.model.eval()
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {use_amp}")
        print(f"Test-Time Augmentation: {use_tta}")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model."""
        # Resize to 512x512
        h, w = image.shape[:2]
        self.original_size = (h, w)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # HWC -> CHW -> NCHW
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        return_probs: bool = False
    ) -> np.ndarray:
        """
        Run inference on image.
        
        Args:
            image: Input image (HxWxC numpy array, BGR or RGB)
            threshold: Binarization threshold
            return_probs: Return probability map instead of binary mask
            
        Returns:
            Binary mask or probability map (HxW)
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Predict
        if self.use_tta:
            # Test-time augmentation
            predictions = []
            
            # Original
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(input_tensor)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
            predictions.append(pred)
            
            # Horizontal flip
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(torch.flip(input_tensor, dims=[3]))
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
            predictions.append(torch.flip(pred, dims=[3]))
            
            # Vertical flip
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(torch.flip(input_tensor, dims=[2]))
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
            predictions.append(torch.flip(pred, dims=[2]))
            
            # Average
            output = torch.stack(predictions).mean(dim=0)
        else:
            # Standard inference
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(input_tensor)
                if isinstance(output, (list, tuple)):
                    output = output[0]
        
        # Post-process
        prob_map = torch.sigmoid(output[0, 0]).cpu().numpy()
        
        # Resize back to original size
        prob_map = cv2.resize(prob_map, (self.original_size[1], self.original_size[0]))
        
        if return_probs:
            return prob_map
        
        binary_mask = (prob_map > threshold).astype(np.uint8) * 255
        return binary_mask
    
    def visualize(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        save_path: str = None,
        alpha: float = 0.5
    ):
        """
        Visualize prediction overlay.
        
        Args:
            image: Original image
            mask: Predicted binary mask
            save_path: Path to save visualization
            alpha: Overlay transparency
        """
        # Create colored overlay
        overlay = image.copy()
        mask_color = np.array([0, 255, 0], dtype=np.uint8)  # Green
        
        # Apply mask
        mask_bool = mask > 127
        overlay[mask_bool] = cv2.addWeighted(
            overlay[mask_bool],
            1 - alpha,
            np.full_like(overlay[mask_bool], mask_color),
            alpha,
            0
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Convert BGR to RGB for display
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            overlay_rgb = overlay
        
        axes[0].imshow(image_rgb)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        axes[2].imshow(overlay_rgb)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='UNet++ Fence Detection Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/unetplusplus/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--output', type=str, default='inference_results',
                        help='Output folder for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold (0.0-1.0)')
    parser.add_argument('--tta', action='store_true',
                        help='Use test-time augmentation (slower but more accurate)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization images')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference engine
    engine = UNetPlusPlusInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_amp=True,
        use_tta=args.tta
    )
    
    # Process single image
    if args.image:
        print(f"\n{'='*80}")
        print(f"Processing: {args.image}")
        print(f"{'='*80}")
        
        image = cv2.imread(args.image)
        if image is None:
            print(f"❌ Failed to load image: {args.image}")
            return
        
        # Predict
        mask = engine.predict(image, threshold=args.threshold)
        
        # Save mask
        mask_path = output_dir / f"{Path(args.image).stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"✅ Saved mask to {mask_path}")
        
        # Visualize
        if args.visualize:
            vis_path = output_dir / f"{Path(args.image).stem}_visualization.png"
            engine.visualize(image, mask, save_path=str(vis_path))
        
        # Calculate coverage
        coverage = (mask > 127).sum() / mask.size * 100
        print(f"Fence coverage: {coverage:.2f}%")
    
    # Process folder
    elif args.folder:
        folder_path = Path(args.folder)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        print(f"\n{'='*80}")
        print(f"Processing {len(image_files)} images from {args.folder}")
        print(f"{'='*80}\n")
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
            
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  ❌ Failed to load image")
                continue
            
            # Predict
            mask = engine.predict(image, threshold=args.threshold)
            
            # Save mask
            mask_path = output_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)
            
            # Visualize
            if args.visualize:
                vis_path = output_dir / f"{img_path.stem}_visualization.png"
                engine.visualize(image, mask, save_path=str(vis_path))
            
            # Calculate coverage
            coverage = (mask > 127).sum() / mask.size * 100
            print(f"  ✅ Saved | Fence coverage: {coverage:.2f}%")
        
        print(f"\n✅ Processed {len(image_files)} images")
        print(f"Results saved to: {output_dir}")
    
    else:
        print("❌ Please specify --image or --folder")
        parser.print_help()


if __name__ == '__main__':
    main()
