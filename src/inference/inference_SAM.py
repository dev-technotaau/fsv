"""
SAM Inference Script for Fence Detection
=========================================
Fast inference script for trained SAM model on new images.
Supports batch processing and visualization.

Author: VisionGuard Team
Date: November 12, 2025
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# SAM imports
try:
    from segment_anything import sam_model_registry
    from segment_anything.modeling import Sam
except ImportError:
    print("ERROR: segment-anything not installed")
    print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)


class SAMInference:
    """SAM model wrapper for inference."""
    
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_b",
        device: str = "cuda",
        input_size: int = 512
    ):
        """
        Initialize SAM inference.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            model_type: SAM model type (vit_b, vit_l, vit_h)
            device: Device to run inference on
            input_size: Input image size
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model_type = model_type
        
        print(f"Loading SAM model from: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            if 'SAM_MODEL_TYPE' in config:
                self.model_type = config['SAM_MODEL_TYPE']
        
        # Initialize SAM model
        # First load pretrained SAM
        sam_checkpoint = self._get_sam_checkpoint(self.model_type)
        sam = sam_model_registry[self.model_type](checkpoint=sam_checkpoint)
        
        # Wrap in our training wrapper
        from train_SAM import SAMForFinetuning
        self.model = SAMForFinetuning(sam)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        
        # Setup preprocessing
        self.transform = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_sam_checkpoint(self, model_type: str) -> str:
        """Get path to pretrained SAM checkpoint."""
        checkpoint_dir = Path("checkpoints/sam")
        checkpoint_path = checkpoint_dir / f"sam_{model_type}.pth"
        
        if checkpoint_path.exists():
            return str(checkpoint_path)
        
        # Download if not exists
        checkpoint_urls = {
            'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
        }
        
        print(f"Downloading pretrained SAM checkpoint...")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        import urllib.request
        urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
        
        return str(checkpoint_path)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        return image_tensor.unsqueeze(0)
    
    def generate_box_prompt(self, image_shape: tuple) -> torch.Tensor:
        """Generate default box prompt (whole image)."""
        h, w = image_shape[:2]
        # Box covering entire image with small margin
        margin = 10
        box = torch.tensor([[margin, margin, w - margin, h - margin]], dtype=torch.float32)
        return box
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        use_box: bool = True,
        threshold: float = 0.5,
        return_prob: bool = False
    ) -> np.ndarray:
        """
        Predict fence mask for image.
        
        Args:
            image: Input image (H, W, 3) in RGB
            use_box: Use box prompt (whole image)
            threshold: Probability threshold for binary mask
            return_prob: Return probability map instead of binary mask
            
        Returns:
            Predicted mask (H, W) or probability map
        """
        original_shape = image.shape[:2]
        
        # Preprocess
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Generate prompts
        box = None
        if use_box:
            box = self.generate_box_prompt(original_shape)
            box = box.to(self.device)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.model(image_tensor, boxes=box)
        
        # Post-process
        outputs = outputs.squeeze(0).squeeze(0)  # Remove batch and channel dims
        
        # Resize to original size
        outputs = F.interpolate(
            outputs.unsqueeze(0).unsqueeze(0),
            size=original_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Get probability
        prob = torch.sigmoid(outputs).cpu().numpy()
        
        if return_prob:
            return prob
        
        # Threshold to binary mask
        mask = (prob > threshold).astype(np.uint8)
        
        return mask
    
    def predict_batch(
        self,
        images: list,
        use_box: bool = True,
        threshold: float = 0.5,
        show_progress: bool = True
    ) -> list:
        """
        Predict masks for batch of images.
        
        Args:
            images: List of images (H, W, 3)
            use_box: Use box prompts
            threshold: Probability threshold
            show_progress: Show progress bar
            
        Returns:
            List of predicted masks
        """
        masks = []
        
        iterator = tqdm(images, desc="Processing") if show_progress else images
        
        for image in iterator:
            mask = self.predict(image, use_box=use_box, threshold=threshold)
            masks.append(mask)
        
        return masks


def visualize_prediction(
    image: np.ndarray,
    mask: np.ndarray,
    save_path: str = None,
    alpha: float = 0.5,
    color: tuple = (0, 255, 0)
):
    """
    Visualize prediction overlay on image.
    
    Args:
        image: Original image (H, W, 3)
        mask: Predicted mask (H, W)
        save_path: Path to save visualization
        alpha: Overlay transparency
        color: Overlay color (B, G, R)
    """
    import matplotlib.pyplot as plt
    
    # Create overlay
    overlay = image.copy()
    overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + np.array(color) * alpha
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def process_directory(
    input_dir: str,
    output_dir: str,
    checkpoint_path: str,
    model_type: str = "vit_b",
    device: str = "cuda",
    threshold: float = 0.5,
    visualize: bool = True
):
    """
    Process all images in directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        checkpoint_path: Path to model checkpoint
        model_type: SAM model type
        device: Device to run on
        threshold: Prediction threshold
        visualize: Save visualizations
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    masks_dir = output_path / "masks"
    vis_dir = output_path / "visualizations"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Initialize model
    predictor = SAMInference(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device
    )
    
    # Process images
    results = []
    
    for img_file in tqdm(image_files, desc="Processing images"):
        # Load image
        image = cv2.imread(str(img_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        mask = predictor.predict(image_rgb, threshold=threshold)
        
        # Save mask
        mask_file = masks_dir / f"{img_file.stem}_mask.png"
        cv2.imwrite(str(mask_file), mask * 255)
        
        # Save visualization
        if visualize:
            vis_file = vis_dir / f"{img_file.stem}_vis.png"
            visualize_prediction(image, mask, save_path=str(vis_file))
        
        # Record results
        fence_pixels = mask.sum()
        total_pixels = mask.size
        fence_percentage = (fence_pixels / total_pixels) * 100
        
        results.append({
            'image': img_file.name,
            'fence_pixels': int(fence_pixels),
            'total_pixels': int(total_pixels),
            'fence_percentage': float(fence_percentage)
        })
    
    # Save results JSON
    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Processing complete!")
    print(f"  Masks saved to: {masks_dir}")
    if visualize:
        print(f"  Visualizations saved to: {vis_dir}")
    print(f"  Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="SAM Inference for Fence Detection")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input image or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='vit_b',
        choices=['vit_b', 'vit_l', 'vit_h'],
        help='SAM model type'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Prediction threshold (0-1)'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization'
    )
    
    args = parser.parse_args()
    
    # Check input
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"ERROR: Input path not found: {args.input}")
        return
    
    # Process
    if input_path.is_dir():
        # Process directory
        process_directory(
            input_dir=args.input,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            device=args.device,
            threshold=args.threshold,
            visualize=not args.no_visualize
        )
    else:
        # Process single image
        print(f"Processing single image: {args.input}")
        
        # Initialize model
        predictor = SAMInference(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            device=args.device
        )
        
        # Load and predict
        image = cv2.imread(str(input_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = predictor.predict(image_rgb, threshold=args.threshold)
        
        # Save outputs
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        mask_file = output_path / f"{input_path.stem}_mask.png"
        cv2.imwrite(str(mask_file), mask * 255)
        print(f"✓ Mask saved to: {mask_file}")
        
        if not args.no_visualize:
            vis_file = output_path / f"{input_path.stem}_vis.png"
            visualize_prediction(image, mask, save_path=str(vis_file))


if __name__ == '__main__':
    main()
