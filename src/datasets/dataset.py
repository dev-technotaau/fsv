import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class FenceDataset(Dataset):
    """Dataset for fence segmentation with data augmentation support."""
    
    def __init__(self, images_dir, masks_dir, input_size=512, train=True):
        """
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing mask images
            input_size: Size to resize images to (default: 512)
            train: If True, applies data augmentation (default: True)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.input_size = input_size
        self.train = train  # Flag to enable/disable augmentation
        
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
    
    def __getitem__(self, idx):
        img_file, mask_file = self.pairs[idx]
        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, mask_file)
        
        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # 1. Resize (Always apply same resize to both)
        image = TF.resize(image, [self.input_size, self.input_size])
        mask = TF.resize(mask, [self.input_size, self.input_size], 
                        interpolation=Image.NEAREST)
        
        # 2. Data augmentation (only if training)
        if self.train:
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
        
        # 3. Convert to tensors
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # 4. Binarize mask
        mask = (mask > 0.5).squeeze().long()
        
        return image, mask
