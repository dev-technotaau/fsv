"""
Enhanced Fence Dataset with improved data augmentation.
Adds color jitter, blur, brightness adjustments for robustness.
"""

import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class FenceDatasetV2(Dataset):
    """Enhanced dataset for fence segmentation with advanced augmentation."""
    
    def __init__(self, images_dir, masks_dir, input_size=512, train=True):
        """
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing mask images
            input_size: Size to resize images to
            train: If True, applies data augmentation
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.input_size = input_size
        self.train = train
        
        # Get all valid image files
        self.ids = [f for f in sorted(os.listdir(images_dir))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Verify matching masks exist
        self.pairs = []
        for img_file in self.ids:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}.png"
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                self.pairs.append((img_file, mask_file))
        
        print(f"Found {len(self.pairs)} image-mask pairs.")
    
    def __len__(self):
        return len(self.pairs)
    
    def apply_color_augmentation(self, image):
        """Apply color-based augmentations."""
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        
        # Random contrast
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
        
        # Random saturation
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)
        
        return image
    
    def apply_blur(self, image):
        """Apply random Gaussian blur."""
        if random.random() > 0.7:  # 30% chance
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius))
        return image
    
    def __getitem__(self, idx):
        img_file, mask_file = self.pairs[idx]
        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, mask_file)
        
        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # 1. Resize to target size
        image = TF.resize(image, [self.input_size, self.input_size])
        mask = TF.resize(mask, [self.input_size, self.input_size],
                        interpolation=Image.NEAREST)
        
        # 2. Data augmentation (only if training)
        if self.train:
            # Geometric augmentations (applied to both image and mask)
            
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Random rotation (0, 90, 180, 270 degrees)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            
            # Small random rotation (-15 to +15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            
            # Random affine transformations
            if random.random() > 0.5:
                # Slight scale and translation
                scale = random.uniform(0.9, 1.1)
                translate = (random.randint(-20, 20), random.randint(-20, 20))
                image = TF.affine(image, angle=0, translate=translate, 
                                 scale=scale, shear=0)
                mask = TF.affine(mask, angle=0, translate=translate,
                                scale=scale, shear=0)
            
            # Color augmentations (applied only to image)
            image = self.apply_color_augmentation(image)
            
            # Blur augmentation (applied only to image)
            image = self.apply_blur(image)
        
        # 3. Convert to tensors
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # 4. Binarize mask
        mask = (mask > 0.5).squeeze().long()
        
        return image, mask
