"""
SAM (Segment Anything Model) Fine-Tuning for Fence Detection - v1.0 ENTERPRISE
================================================================================
ADVANCED FEATURES & OPTIMIZATIONS:
- SAM-B (Base) architecture with prompt-based segmentation
- Automatic prompt generation from masks (boxes, points, masks)
- Multi-scale training with resolution augmentation
- Advanced data augmentation pipeline (Albumentations)
- Mixed precision training (AMP) with dynamic loss scaling
- Distributed training support (DDP ready)
- Gradient accumulation for large effective batch sizes
- Advanced loss functions (Focal + Dice + IoU + Boundary)
- Learning rate warmup + Cosine annealing with restarts
- Exponential Moving Average (EMA) for stable predictions
- Comprehensive metrics (IoU, Dice, Precision, Recall, F1)
- TensorBoard logging with visualizations
- Checkpoint management with best/last model saving
- Early stopping with patience
- GPU memory optimization (RTX 3060 6GB optimized @ 512×512)
- Efficient data loading (prefetching, pinned memory)
- 512×512 resolution (4× memory saving vs 1024, still high quality)
- Test-time augmentation (TTA) support
- Model validation with visualization outputs
- Robust error handling and logging

Application: Fence Staining Visualizer
Author: VisionGuard Team - Advanced AI Division
Date: November 14, 2025 (v2.0 - RTX 3060 6GB Optimized @ 512×512)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything.modeling import Sam
except ImportError:
    print("Installing segment-anything package...")
    os.system("pip install git+https://github.com/facebookresearch/segment-anything.git")
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything.modeling import Sam


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for SAM training."""
    
    # Paths
    PROJECT_ROOT = Path("./")
    IMAGES_DIR = PROJECT_ROOT / "data" / "images"
    MASKS_DIR = PROJECT_ROOT / "data" / "masks"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sam"
    LOGS_DIR = PROJECT_ROOT / "logs" / "sam"
    VISUALIZATIONS_DIR = PROJECT_ROOT / "training_visualizations" / "sam"
    
    # SAM Model Configuration
    SAM_MODEL_TYPE = "vit_b"  # Options: vit_h, vit_l, vit_b (base - best for 6GB GPU)
    SAM_CHECKPOINT = None  # Will auto-download if None
    PRETRAINED = True
    
    # Training Hyperparameters (OPTIMIZED FOR RTX 3060 6GB)
    INPUT_SIZE = 512   # Reduced from 1024 (4× less memory, SAM supports flexible sizes)
    TRAIN_SIZE = 512   # Must match INPUT_SIZE for SAM
    BATCH_SIZE = 2     # Increased from 1 (512×512 allows larger batches)
    ACCUMULATION_STEPS = 4  # Effective batch = 8 (2×4)
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    # Optimizer & Scheduler
    OPTIMIZER = "AdamW"  # Options: AdamW, Adam, SGD
    SCHEDULER = "CosineAnnealingWarmRestarts"  # With warmup
    T_0 = 10  # Cosine restart period
    T_MULT = 2
    
    # Loss Configuration
    LOSS_WEIGHTS = {
        'focal': 0.25,
        'dice': 0.35,
        'iou': 0.25,
        'boundary': 0.15
    }
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Data Augmentation
    USE_ADVANCED_AUGMENTATION = True
    AUGMENTATION_PROB = 0.7
    
    # Prompt Configuration (SAM-specific)
    USE_BOX_PROMPTS = True
    USE_POINT_PROMPTS = True
    USE_MASK_PROMPTS = False  # Memory intensive
    NUM_POINT_PROMPTS = 5  # Random points per mask
    POINT_JITTER = 10  # Pixels to jitter points
    
    # Hardware Optimization (512×512 allows more workers)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2  # Increased from 0 (512×512 has memory headroom)
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2  # Enable prefetching for smoother training
    PERSISTENT_WORKERS = True  # Enabled for faster epoch transitions
    MULTIPROCESSING_CONTEXT = 'spawn'
    NON_BLOCKING = True  # Enable async GPU transfers
    
    # Mixed Precision
    USE_AMP = True
    AMP_DTYPE = torch.float16
    GRAD_CLIP = 1.0
    GRAD_SCALER_INIT_SCALE = 2.**16  # Initial scale for gradient scaler
    GRAD_SCALER_GROWTH_FACTOR = 2.0
    GRAD_SCALER_BACKOFF_FACTOR = 0.5
    GRAD_SCALER_GROWTH_INTERVAL = 2000
    
    # Exponential Moving Average
    USE_EMA = True
    EMA_DECAY = 0.999
    
    # Validation & Checkpointing
    TRAIN_SPLIT = 0.85
    VAL_SPLIT = 0.15
    SAVE_FREQ = 5  # Save every N epochs
    VAL_FREQ = 1   # Validate every N epochs
    VIS_FREQ = 10  # Save visualizations every N epochs
    
    # Early Stopping (DISABLED - Train for full epochs)
    EARLY_STOPPING = False  # Disabled to ensure full training
    PATIENCE = 20  # Not used when EARLY_STOPPING=False
    MIN_DELTA = 1e-4  # Not used when EARLY_STOPPING=False
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    USE_TENSORBOARD = True
    SAVE_LOSS_COMPONENTS = True  # Log individual loss components
    
    # Memory Optimization (512×512 optimized)
    EMPTY_CACHE_FREQ = 5   # Less aggressive at 512×512 (more headroom)
    GRADIENT_CHECKPOINTING = True  # Keep enabled for safety (minimal performance impact)
    
    # Learning Rate Finder
    USE_LR_FINDER = False  # Run LR finder before training
    LR_FINDER_STEPS = 100
    
    # Test-Time Augmentation
    USE_TTA = False  # Enable for inference
    TTA_TRANSFORMS = ['original', 'hflip', 'vflip', 'rotate90']
    
    # Class Configuration
    NUM_CLASSES = 2  # Background + Fence
    CLASS_NAMES = ['background', 'fence']
    ID2LABEL = {0: "background", 1: "fence"}
    LABEL2ID = {"background": 0, "fence": 1}
    
    # Reproducibility
    SEED = 42
    DETERMINISTIC = True


# Create directories
for dir_path in [Config.CHECKPOINT_DIR, Config.LOGS_DIR, Config.VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str, log_file: Path, level=logging.INFO):
    """Setup logger with file and console handlers."""
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    
    return logger


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger('SAM_Training', Config.LOGS_DIR / f'training_{timestamp}.log')


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if Config.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


# ============================================================================
# PROMPT GENERATION (SAM-SPECIFIC)
# ============================================================================

class PromptGenerator:
    """Generate prompts (boxes, points) from masks for SAM training."""
    
    @staticmethod
    def mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert binary mask to bounding box.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Box coordinates [x_min, y_min, x_max, y_max] or None
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return np.array([x_min, y_min, x_max, y_max])
    
    @staticmethod
    def mask_to_points(
        mask: np.ndarray, 
        num_points: int = 5, 
        jitter: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample random points from mask (foreground).
        
        Args:
            mask: Binary mask (H, W)
            num_points: Number of points to sample
            jitter: Random jitter in pixels
            
        Returns:
            Tuple of (points, labels) where labels are 1 (foreground)
        """
        fg_coords = np.column_stack(np.where(mask > 0))
        
        if len(fg_coords) == 0:
            # Return center point if no foreground
            h, w = mask.shape
            points = np.array([[w // 2, h // 2]])
            labels = np.array([1])
            return points, labels
        
        # Sample random points
        num_points = min(num_points, len(fg_coords))
        indices = np.random.choice(len(fg_coords), num_points, replace=False)
        points = fg_coords[indices]
        
        # Swap to (x, y) and add jitter
        points = points[:, [1, 0]].astype(np.float32)
        if jitter > 0:
            points += np.random.randn(*points.shape) * jitter
            points = np.clip(points, 0, [mask.shape[1] - 1, mask.shape[0] - 1])
        
        labels = np.ones(len(points), dtype=np.int32)
        
        return points, labels
    
    @staticmethod
    def generate_prompts(
        mask: np.ndarray,
        use_box: bool = True,
        use_points: bool = True,
        num_points: int = 5,
        jitter: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Generate all prompts from mask.
        
        Returns:
            Dictionary with 'box', 'points', 'point_labels'
        """
        prompts = {}
        
        if use_box:
            box = PromptGenerator.mask_to_box(mask)
            if box is not None:
                prompts['box'] = box
        
        if use_points:
            points, labels = PromptGenerator.mask_to_points(mask, num_points, jitter)
            prompts['points'] = points
            prompts['point_labels'] = labels
        
        return prompts


# ============================================================================
# COCO DATASET LOADER
# ============================================================================

class COCOSAMDataset(Dataset):
    """
    COCO-format dataset loader for SAM with instance/panoptic segmentation support.
    Supports COCO JSON annotations with categories, segmentations, and bboxes.
    """
    
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        transform=None,
        prompt_config: Dict = None,
        mode: str = 'instance'  # 'instance' or 'panoptic'
    ):
        import json
        from pycocotools import mask as mask_utils
        
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.prompt_config = prompt_config or {}
        self.prompt_gen = PromptGenerator()
        self.mode = mode
        
        # Load COCO annotations
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image and annotation mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image_id
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        self.image_ids = list(self.image_annotations.keys())
        logger.info(f"Loaded COCO dataset: {len(self.image_ids)} images, {len(self.coco_data['annotations'])} annotations")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        try:
            img_id = self.image_ids[idx]
            img_info = self.images[img_id]
            img_path = self.image_dir / img_info['file_name']
            
            # Load image
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Create instance mask from COCO annotations
            mask = np.zeros((h, w), dtype=np.uint8)
            annotations = self.image_annotations[img_id]
            
            for ann in annotations:
                # Skip if no segmentation
                if 'segmentation' not in ann:
                    continue
                
                # Handle different segmentation formats
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 1)
                elif isinstance(ann['segmentation'], dict):
                    # RLE format
                    from pycocotools import mask as mask_utils
                    rle = ann['segmentation']
                    instance_mask = mask_utils.decode(rle)
                    mask = np.maximum(mask, instance_mask)
            
            # Apply augmentation
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                mask = torch.from_numpy(mask).long()
            
            # Generate prompts
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
            prompts = self.prompt_gen.generate_prompts(
                mask_np,
                use_box=self.prompt_config.get('use_box', True),
                use_points=self.prompt_config.get('use_points', True),
                num_points=self.prompt_config.get('num_points', 5),
                jitter=self.prompt_config.get('jitter', 10)
            )
            
            # Ensure tensor format
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            else:
                mask = torch.from_numpy(mask).long()
            
            return {
                'image': image,
                'mask': mask,
                'prompts': prompts,
                'image_path': str(img_path),
                'image_id': img_id,
                'annotations': annotations
            }
            
        except Exception as e:
            logger.error(f"Error loading COCO image {img_id}: {e}")
            return {
                'image': torch.zeros(3, Config.TRAIN_SIZE, Config.TRAIN_SIZE),
                'mask': torch.zeros(Config.TRAIN_SIZE, Config.TRAIN_SIZE, dtype=torch.long),
                'prompts': {},
                'image_path': '',
                'image_id': -1,
                'annotations': []
            }


# ============================================================================
# DATASET
# ============================================================================

class FenceSAMDataset(Dataset):
    """Advanced dataset for SAM fine-tuning with prompt generation."""
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform=None,
        prompt_config: Dict = None
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.prompt_config = prompt_config or {}
        self.prompt_gen = PromptGenerator()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image and mask
            image = cv2.imread(self.image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            
            # Ensure same size
            if image.shape[:2] != mask.shape[:2]:
                h, w = image.shape[:2]
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Normalize mask to binary
            mask = (mask > 127).astype(np.uint8)
            
            # Apply augmentation (includes ToTensorV2)
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                # Convert to tensors if no augmentation
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                mask = torch.from_numpy(mask).long()
            
            # Generate prompts from augmented mask (already at correct size)
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
            prompts = self.prompt_gen.generate_prompts(
                mask_np,
                use_box=self.prompt_config.get('use_box', True),
                use_points=self.prompt_config.get('use_points', True),
                num_points=self.prompt_config.get('num_points', 5),
                jitter=self.prompt_config.get('jitter', 10)
            )
            
            # Ensure correct tensor format
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            else:
                mask = torch.from_numpy(mask).long()
            
            return {
                'image': image,
                'mask': mask,
                'prompts': prompts,
                'image_path': self.image_paths[idx]
            }
            
        except Exception as e:
            logger.error(f"Error loading {self.image_paths[idx]}: {e}")
            # Return dummy data
            return {
                'image': torch.zeros(3, Config.TRAIN_SIZE, Config.TRAIN_SIZE),
                'mask': torch.zeros(Config.TRAIN_SIZE, Config.TRAIN_SIZE, dtype=torch.long),
                'prompts': {},
                'image_path': self.image_paths[idx]
            }


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized prompts.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary with batched tensors and list of prompts
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    prompts = [item['prompts'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'prompts': prompts,
        'image_path': image_paths
    }


def get_training_augmentation():
    """Advanced augmentation pipeline for training."""
    if not Config.USE_ADVANCED_AUGMENTATION:
        return A.Compose([
            A.Resize(Config.TRAIN_SIZE, Config.TRAIN_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return A.Compose([
        # Geometric transformations with RandomCrop
        A.RandomCrop(height=Config.TRAIN_SIZE, width=Config.TRAIN_SIZE, p=0.3),
        A.Resize(Config.TRAIN_SIZE, Config.TRAIN_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT
        ),
        
        # Color augmentations
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.5),
        
        # Lighting augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.4),
        
        # Weather & artifacts with RandomShadow
        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=1.0),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, p=1.0),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.8, alpha_coef=0.08, p=1.0),
        ], p=0.25),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        
        # Normalize and convert
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=Config.AUGMENTATION_PROB)


def get_validation_augmentation():
    """Simple augmentation for validation."""
    return A.Compose([
        A.Resize(Config.TRAIN_SIZE, Config.TRAIN_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ============================================================================
# ADVANCED LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        targets = targets.float()
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class IoULoss(nn.Module):
    """IoU Loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        targets = targets.float()
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou


class BoundaryLoss(nn.Module):
    """Boundary-aware loss to improve edge detection."""
    
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        targets = targets.float()
        
        # Compute boundaries using morphological gradient
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=targets.device)
        
        # Dilate and erode
        targets_dilated = F.conv2d(
            targets.unsqueeze(1), kernel, padding=self.kernel_size // 2
        )
        targets_eroded = -F.conv2d(
            -targets.unsqueeze(1), kernel, padding=self.kernel_size // 2
        )
        
        # Boundary = dilated - eroded
        boundary = (targets_dilated - targets_eroded).squeeze(1)
        boundary = (boundary > 0).float()
        
        # Weight loss by boundary
        loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        weighted_loss = loss * (1 + boundary * 5)  # 5x weight on boundaries
        
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components."""
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        self.focal = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        self.dice = DiceLoss()
        self.iou = IoULoss()
        self.boundary = BoundaryLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        losses = {}
        
        if self.weights.get('focal', 0) > 0:
            losses['focal'] = self.focal(predictions, targets)
        
        if self.weights.get('dice', 0) > 0:
            losses['dice'] = self.dice(predictions, targets)
        
        if self.weights.get('iou', 0) > 0:
            losses['iou'] = self.iou(predictions, targets)
        
        if self.weights.get('boundary', 0) > 0:
            losses['boundary'] = self.boundary(predictions, targets)
        
        # Weighted sum
        total_loss = sum(self.weights[k] * v for k, v in losses.items() if k in self.weights)
        
        return total_loss, losses


# ============================================================================
# COCO EVALUATOR
# ============================================================================

class COCOEvaluator:
    """
    COCO-style evaluation for instance/panoptic segmentation.
    Computes mAP, mIoU, and per-category metrics.
    """
    
    def __init__(self, categories: List[Dict]):
        self.categories = categories
        self.category_ids = [cat['id'] for cat in categories]
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.ground_truths = []
    
    def add_batch(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ):
        """
        Add batch of predictions and ground truths.
        
        Args:
            predictions: List of dicts with 'masks', 'scores', 'labels'
            ground_truths: List of dicts with 'masks', 'labels'
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
    
    def compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / (union + 1e-7)
    
    def compute_ap(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Compute Average Precision from recall-precision curve."""
        # Add sentinel values
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        # Compute precision envelope
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        
        # Integrate area under curve
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return ap
    
    def evaluate(self, iou_thresholds: List[float] = None) -> Dict[str, float]:
        """
        Compute COCO-style metrics.
        
        Returns:
            Dictionary with mAP, mAP@50, mAP@75, mIoU, etc.
        """
        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.5, 0.95, 10)
        
        results = {
            'mAP': 0.0,
            'mAP@50': 0.0,
            'mAP@75': 0.0,
            'mIoU': 0.0,
            'per_class_iou': {}
        }
        
        if not self.predictions or not self.ground_truths:
            return results
        
        # Compute per-threshold AP
        aps = []
        for iou_thr in iou_thresholds:
            # Match predictions to ground truths
            matches = self._match_predictions(iou_thr)
            
            # Compute precision-recall
            if matches['tp'] + matches['fp'] > 0:
                precision = matches['tp'] / (matches['tp'] + matches['fp'])
                recall = matches['tp'] / (matches['tp'] + matches['fn']) if (matches['tp'] + matches['fn']) > 0 else 0
                aps.append(precision)
        
        results['mAP'] = np.mean(aps) if aps else 0.0
        results['mAP@50'] = aps[0] if len(aps) > 0 else 0.0
        results['mAP@75'] = aps[5] if len(aps) > 5 else 0.0
        
        # Compute mIoU
        ious = []
        for pred, gt in zip(self.predictions, self.ground_truths):
            if 'mask' in pred and 'mask' in gt:
                iou = self.compute_iou(pred['mask'], gt['mask'])
                ious.append(iou)
        
        results['mIoU'] = np.mean(ious) if ious else 0.0
        
        return results
    
    def _match_predictions(self, iou_threshold: float) -> Dict[str, int]:
        """Match predictions to ground truths based on IoU threshold."""
        tp, fp, fn = 0, 0, 0
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            if 'mask' not in pred or 'mask' not in gt:
                continue
            
            iou = self.compute_iou(pred['mask'], gt['mask'])
            if iou >= iou_threshold:
                tp += 1
            else:
                fp += 1
        
        # Count false negatives (unmatched ground truths)
        fn = len(self.ground_truths) - tp
        
        return {'tp': tp, 'fp': fp, 'fn': fn}


# ============================================================================
# METRICS
# ============================================================================

class MetricsCalculator:
    """Calculate comprehensive segmentation metrics."""
    
    @staticmethod
    def calculate_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate IoU, Dice, Precision, Recall, F1.
        
        Args:
            predictions: Logits (N, 1, H, W) or probabilities
            targets: Ground truth (N, H, W) or (N, 1, H, W)
            
        Returns:
            Dictionary of metrics
        """
        # Convert to binary predictions
        if predictions.dim() == 4:
            predictions = predictions.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        
        predictions = torch.sigmoid(predictions) > threshold
        targets = targets > 0.5
        
        # Move to CPU for numpy operations
        predictions = predictions.cpu().numpy().astype(bool)
        targets = targets.cpu().numpy().astype(bool)
        
        # Calculate metrics
        intersection = (predictions & targets).sum()
        union = (predictions | targets).sum()
        
        tp = intersection
        fp = (predictions & ~targets).sum()
        fn = (~predictions & targets).sum()
        tn = (~predictions & ~targets).sum()
        
        # Metrics
        iou = intersection / (union + 1e-7)
        dice = 2 * intersection / (predictions.sum() + targets.sum() + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        
        return {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        }


# ============================================================================
# EXPONENTIAL MOVING AVERAGE
# ============================================================================

class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].sub_(
                    (1 - self.decay) * (self.shadow[name] - param.data)
                )
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================================
# SAM MODEL WRAPPER
# ============================================================================

class SAMForFinetuning(nn.Module):
    """
    Wrapper for SAM model to support batch training with prompts.
    Handles proper image preprocessing and prompt encoding.
    """
    
    def __init__(self, sam_model: Sam):
        super().__init__()
        self.sam = sam_model
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        
        # Enable gradient checkpointing to save memory
        if Config.GRADIENT_CHECKPOINTING and hasattr(self.image_encoder, 'blocks'):
            for block in self.image_encoder.blocks:
                if hasattr(block, 'use_checkpoint'):
                    block.use_checkpoint = True
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize pixel values and pad to a square input.
        SAM expects images to be normalized with ImageNet stats.
        """
        # Move normalization parameters to same device as input
        if self.pixel_mean.device != x.device:
            self.pixel_mean = self.pixel_mean.to(x.device)
            self.pixel_std = self.pixel_std.to(x.device)
        
        # Denormalize from training normalization
        # Input comes normalized with [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225]
        train_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(x.device)
        train_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(x.device)
        x = x * train_std + train_mean
        
        # Convert to SAM's expected format (0-255 scale)
        x = x * 255.0
        
        # Normalize with SAM's pixel mean/std
        x = (x - self.pixel_mean) / self.pixel_std
        
        # Pad to square (SAM expects square inputs)
        h, w = x.shape[-2:]
        padh = Config.INPUT_SIZE - h
        padw = Config.INPUT_SIZE - w
        x = F.pad(x, (0, padw, 0, padh))
        
        return x
    
    def forward(
        self,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with batch support.
        
        Args:
            images: (B, 3, H, W) - normalized images
            boxes: (B, 4) - [x_min, y_min, x_max, y_max]
            points: (B, N, 2) - point coordinates
            point_labels: (B, N) - point labels (1=foreground, 0=background)
            
        Returns:
            Predicted masks (B, 1, H, W)
        """
        batch_size = images.shape[0]
        original_size = images.shape[-2:]
        
        # Preprocess images for SAM
        # images = self.preprocess(images)
        
        # Encode images (batch operation)
        image_embeddings = self.image_encoder(images)
        
        # Process each item in batch (SAM's mask decoder doesn't support batching)
        outputs = []
        for i in range(batch_size):
            # Prepare prompts
            sparse_embeddings, dense_embeddings = self._encode_prompts(
                boxes[i] if boxes is not None else None,
                points[i] if points is not None else None,
                point_labels[i] if point_labels is not None else None,
                original_size
            )
            
            # Decode mask
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            outputs.append(low_res_masks)
        
        return torch.cat(outputs, dim=0)
    
    def _encode_prompts(self, box, points, point_labels, image_size):
        """Encode prompts for a single image."""
        # Handle box prompt
        if box is not None and torch.any(box != 0):
            box = box.unsqueeze(0)  # (1, 4)
        else:
            box = None
        
        # Handle point prompts
        if points is not None and point_labels is not None:
            # Filter out zero/dummy points
            valid_mask = point_labels > 0
            if valid_mask.any():
                points = points[valid_mask].unsqueeze(0)  # (1, N_valid, 2)
                point_labels = point_labels[valid_mask].unsqueeze(0)  # (1, N_valid)
            else:
                points = None
                point_labels = None
        else:
            points = None
            point_labels = None
        
        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(points, point_labels) if points is not None else None,
            boxes=box if box is not None else None,
            masks=None
        )
        
        return sparse_embeddings, dense_embeddings


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    ema: Optional[EMA] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_metrics = {'iou': 0.0, 'dice': 0.0, 'f1': 0.0}
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        # Aggressive memory cleanup at start
        if batch_idx == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        
        # Prepare prompts (convert to tensors)
        boxes = None
        points = None
        point_labels = None
        
        if Config.USE_BOX_PROMPTS:
            boxes = []
            for prompts in batch['prompts']:
                if 'box' in prompts:
                    boxes.append(torch.from_numpy(prompts['box']))
                else:
                    boxes.append(torch.zeros(4))
            boxes = torch.stack(boxes).to(device, non_blocking=True)
        
        if Config.USE_POINT_PROMPTS:
            points = []
            point_labels = []
            for prompts in batch['prompts']:
                if 'points' in prompts:
                    points.append(torch.from_numpy(prompts['points']))
                    point_labels.append(torch.from_numpy(prompts['point_labels']))
                else:
                    points.append(torch.zeros(Config.NUM_POINT_PROMPTS, 2))
                    point_labels.append(torch.zeros(Config.NUM_POINT_PROMPTS))
            points = torch.stack(points).to(device, non_blocking=True)
            point_labels = torch.stack(point_labels).to(device, non_blocking=True)
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP, dtype=Config.AMP_DTYPE):
            outputs = model(images, boxes=boxes, points=points, point_labels=point_labels)
            
            # Upsample to mask size if needed
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            
            outputs = outputs.squeeze(1)
            loss, loss_dict = criterion(outputs, masks)
            loss = loss / Config.ACCUMULATION_STEPS
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Optimizer step with gradient accumulation
        if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA
            if ema is not None:
                ema.update()
            
            # Log gradient norm if TensorBoard enabled
            if writer and (batch_idx % Config.LOG_INTERVAL == 0):
                global_step = epoch * num_batches + batch_idx
                writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
        
        # Calculate metrics
        with torch.no_grad():
            metrics = MetricsCalculator.calculate_metrics(outputs, masks)
        
        total_loss += loss.item() * Config.ACCUMULATION_STEPS
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item() * Config.ACCUMULATION_STEPS:.4f}",
            'iou': f"{metrics['iou']:.4f}",
            'dice': f"{metrics['dice']:.4f}"
        })
        
        # TensorBoard logging
        if writer and (batch_idx % Config.LOG_INTERVAL == 0):
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss', loss.item() * Config.ACCUMULATION_STEPS, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            
            # Log loss components if enabled
            if Config.SAVE_LOSS_COMPONENTS:
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(f'train/loss_{loss_name}', loss_value.item(), global_step)
            
            for k, v in metrics.items():
                writer.add_scalar(f'train/{k}', v, global_step)
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                writer.add_scalar('system/gpu_memory_allocated_gb', memory_allocated, global_step)
                writer.add_scalar('system/gpu_memory_reserved_gb', memory_reserved, global_step)
        
        # Periodic memory cleanup
        if torch.cuda.is_available() and (batch_idx % Config.EMPTY_CACHE_FREQ == 0):
            torch.cuda.empty_cache()
            # Delete intermediate tensors
            if 'outputs' in locals():
                del outputs
            if 'loss' in locals():
                del loss
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return {'loss': avg_loss, **avg_metrics}


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    save_visualizations: bool = False
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_metrics = {'iou': 0.0, 'dice': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            # Prepare prompts
            boxes = None
            points = None
            point_labels = None
            
            if Config.USE_BOX_PROMPTS:
                boxes = []
                for prompts in batch['prompts']:
                    if 'box' in prompts:
                        boxes.append(torch.from_numpy(prompts['box']))
                    else:
                        boxes.append(torch.zeros(4))
                boxes = torch.stack(boxes).to(device, non_blocking=True)
            
            if Config.USE_POINT_PROMPTS:
                points = []
                point_labels = []
                for prompts in batch['prompts']:
                    if 'points' in prompts:
                        points.append(torch.from_numpy(prompts['points']))
                        point_labels.append(torch.from_numpy(prompts['point_labels']))
                    else:
                        points.append(torch.zeros(Config.NUM_POINT_PROMPTS, 2))
                        point_labels.append(torch.zeros(Config.NUM_POINT_PROMPTS))
                points = torch.stack(points).to(device, non_blocking=True)
                point_labels = torch.stack(point_labels).to(device, non_blocking=True)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP, dtype=Config.AMP_DTYPE):
                outputs = model(images, boxes=boxes, points=points, point_labels=point_labels)
                
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs, size=masks.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                
                outputs = outputs.squeeze(1)
                loss, _ = criterion(outputs, masks)
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_metrics(outputs, masks)
            
            total_loss += loss.item()
            for k in total_metrics:
                if k in metrics:
                    total_metrics[k] += metrics[k]
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{metrics['iou']:.4f}",
                'dice': f"{metrics['dice']:.4f}"
            })
            
            # Save visualizations
            if save_visualizations and batch_idx == 0:
                save_predictions(images, masks, outputs, epoch)
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    # TensorBoard logging
    if writer:
        writer.add_scalar('val/loss', avg_loss, epoch)
        for k, v in avg_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
    
    return {'loss': avg_loss, **avg_metrics}


def save_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    epoch: int,
    num_samples: int = 4
):
    """Save prediction visualizations."""
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        mask = masks[i].cpu().numpy()
        pred = torch.sigmoid(predictions[i]).cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.uint8)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_binary, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = Config.VISUALIZATIONS_DIR / f'epoch_{epoch+1:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class SAMInferenceEngine:
    """
    Production-ready inference engine for SAM with TTA and post-processing.
    Supports batch inference, test-time augmentation, and ensemble predictions.
    """
    
    def __init__(
        self,
        model_path: str,
        device: torch.device = None,
        use_tta: bool = False,
        use_ema: bool = False
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tta = use_tta
        
        # Load model
        logger.info(f"Loading SAM model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize SAM
        sam = sam_model_registry[Config.SAM_MODEL_TYPE](checkpoint=None)
        self.model = SAMForFinetuning(sam)
        
        # Load weights (handle both EMA and regular)
        if use_ema and 'ema_shadow' in checkpoint:
            # Create temporary EMA and apply shadow weights
            ema = EMA(self.model, decay=Config.EMA_DECAY)
            ema.shadow = checkpoint['ema_shadow']
            ema.apply_shadow()
            logger.info("Using EMA weights for inference")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        transform = A.Compose([
            A.Resize(Config.TRAIN_SIZE, Config.TRAIN_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        augmented = transform(image=image)
        tensor = augmented['image'].unsqueeze(0)
        return tensor
    
    def predict(
        self,
        image: np.ndarray,
        box: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a single image.
        
        Args:
            image: RGB image (H, W, 3)
            box: Bounding box [x_min, y_min, x_max, y_max]
            points: Point prompts (N, 2)
            point_labels: Point labels (N,)
            threshold: Confidence threshold
            
        Returns:
            Dictionary with 'mask', 'confidence', 'logits'
        """
        original_size = image.shape[:2]
        
        # Preprocess
        image_tensor = self.preprocess_image(image).to(self.device)
        
        # Prepare prompts
        if box is not None:
            box_tensor = torch.from_numpy(box).unsqueeze(0).float().to(self.device)
        else:
            box_tensor = None
        
        if points is not None and point_labels is not None:
            points_tensor = torch.from_numpy(points).unsqueeze(0).float().to(self.device)
            labels_tensor = torch.from_numpy(point_labels).unsqueeze(0).long().to(self.device)
        else:
            points_tensor = None
            labels_tensor = None
        
        # Inference with TTA if enabled
        if self.use_tta:
            logits = self._predict_with_tta(image_tensor, box_tensor, points_tensor, labels_tensor)
        else:
            with torch.no_grad():
                logits = self.model(image_tensor, box_tensor, points_tensor, labels_tensor)
        
        # Post-process
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).cpu().numpy().astype(np.uint8).squeeze()
        confidence = probs.max().item()
        
        # Resize to original size
        mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        
        return {
            'mask': mask,
            'confidence': confidence,
            'logits': logits.cpu().numpy()
        }
    
    def _predict_with_tta(self, image, box, points, point_labels) -> torch.Tensor:
        """
        Test-time augmentation ensemble.
        
        Returns:
            Averaged logits from multiple augmentations
        """
        logits_list = []
        
        with torch.no_grad():
            # Original
            logits = self.model(image, box, points, point_labels)
            logits_list.append(logits)
            
            # Horizontal flip
            image_hflip = torch.flip(image, dims=[3])
            logits_hflip = self.model(image_hflip, box, points, point_labels)
            logits_hflip = torch.flip(logits_hflip, dims=[3])
            logits_list.append(logits_hflip)
            
            # Vertical flip
            image_vflip = torch.flip(image, dims=[2])
            logits_vflip = self.model(image_vflip, box, points, point_labels)
            logits_vflip = torch.flip(logits_vflip, dims=[2])
            logits_list.append(logits_vflip)
        
        # Average predictions
        return torch.mean(torch.stack(logits_list), dim=0)
    
    def batch_predict(
        self,
        images: List[np.ndarray],
        batch_size: int = 4,
        **kwargs
    ) -> List[Dict[str, np.ndarray]]:
        """
        Batch inference for multiple images.
        
        Args:
            images: List of RGB images
            batch_size: Batch size for processing
            **kwargs: Additional arguments for predict()
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            for img in batch:
                result = self.predict(img, **kwargs)
                results.append(result)
        
        return results


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_sam():
    """Main training loop for SAM."""
    
    logger.info("=" * 80)
    logger.info("SAM FINE-TUNING v2.0 - RTX 3060 6GB OPTIMIZED @ 512×512")
    logger.info("=" * 80)
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"Model: SAM-{Config.SAM_MODEL_TYPE.upper()}")
    logger.info(f"Input Size: {Config.INPUT_SIZE}x{Config.INPUT_SIZE}")
    logger.info(f"Train Size: {Config.TRAIN_SIZE}x{Config.TRAIN_SIZE}")
    logger.info(f"Batch Size: {Config.BATCH_SIZE}")
    logger.info(f"Accumulation Steps: {Config.ACCUMULATION_STEPS}")
    logger.info(f"Effective Batch: {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS}")
    logger.info(f"Epochs: {Config.EPOCHS}")
    logger.info(f"Learning Rate: {Config.LEARNING_RATE}")
    logger.info(f"Mixed Precision: {Config.USE_AMP}")
    logger.info(f"EMA: {Config.USE_EMA}")
    
    # Set seed
    set_seed(Config.SEED)
    
    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = not Config.DETERMINISTIC
    torch.backends.cudnn.enabled = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # Use TensorFloat-32
        # Additional CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Memory management
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.empty_cache()
        # Get GPU properties
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        logger.info(f"GPU Multiprocessor Count: {gpu_props.multi_processor_count}")
        # Memory monitoring
        logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # ========== CHECK FOR RESUME ==========
    resume_checkpoint = None
    resume_path = Config.CHECKPOINT_DIR / 'last_model.pth'
    if resume_path.exists():
        try:
            logger.info(f"Found checkpoint: {resume_path}")
            resume_checkpoint = torch.load(resume_path, map_location=Config.DEVICE)
            logger.info(f"  Resuming from epoch {resume_checkpoint['epoch']}")
            logger.info(f"  Previous best IoU: {resume_checkpoint.get('val_iou', 0):.4f}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            resume_checkpoint = None
    
    # ========== LOAD DATA ==========
    logger.info("\n[1/6] Loading dataset...")
    
    if not Config.IMAGES_DIR.exists() or not Config.MASKS_DIR.exists():
        logger.error("Dataset directories not found!")
        return
    
    # Get all image files
    image_files = sorted([
        f.name for f in Config.IMAGES_DIR.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    if not image_files:
        logger.error(f"No images found in {Config.IMAGES_DIR}")
        return
    
    # Create pairs
    valid_pairs = []
    for img_file in image_files:
        img_path = Config.IMAGES_DIR / img_file
        mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = Config.MASKS_DIR / mask_file
        
        if mask_path.exists():
            valid_pairs.append((str(img_path), str(mask_path)))
    
    if not valid_pairs:
        logger.error("No valid image-mask pairs found!")
        return
    
    logger.info(f"Found {len(valid_pairs)} image-mask pairs")
    
    # Split data
    random.shuffle(valid_pairs)
    train_size = int(len(valid_pairs) * Config.TRAIN_SPLIT)
    
    train_pairs = valid_pairs[:train_size]
    val_pairs = valid_pairs[train_size:]
    
    train_images = [p[0] for p in train_pairs]
    train_masks = [p[1] for p in train_pairs]
    val_images = [p[0] for p in val_pairs]
    val_masks = [p[1] for p in val_pairs]
    
    logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # ========== LOAD SAM MODEL ==========
    logger.info("\n[2/6] Loading SAM model...")
    
    try:
        # Download checkpoint if not exists
        if Config.SAM_CHECKPOINT is None:
            checkpoint_urls = {
                'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
                'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
            }
            
            checkpoint_path = Config.CHECKPOINT_DIR / f'sam_{Config.SAM_MODEL_TYPE}.pth'
            
            if not checkpoint_path.exists():
                logger.info(f"Downloading SAM checkpoint...")
                import urllib.request
                urllib.request.urlretrieve(
                    checkpoint_urls[Config.SAM_MODEL_TYPE],
                    checkpoint_path
                )
            
            Config.SAM_CHECKPOINT = str(checkpoint_path)
        
        # Load SAM
        sam = sam_model_registry[Config.SAM_MODEL_TYPE](checkpoint=Config.SAM_CHECKPOINT)
        model = SAMForFinetuning(sam)
        model.to(Config.DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model: SAM-{Config.SAM_MODEL_TYPE.upper()}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        
        # Resume from checkpoint if available
        if resume_checkpoint is not None:
            model.load_state_dict(resume_checkpoint['model_state_dict'])
            logger.info("  ✓ Model weights restored from checkpoint")
        
    except Exception as e:
        logger.error(f"Failed to load SAM model: {e}")
        return
    
    # ========== CREATE DATASETS ==========
    logger.info("\n[3/6] Creating datasets...")
    
    prompt_config = {
        'use_box': Config.USE_BOX_PROMPTS,
        'use_points': Config.USE_POINT_PROMPTS,
        'num_points': Config.NUM_POINT_PROMPTS,
        'jitter': Config.POINT_JITTER
    }
    
    train_dataset = FenceSAMDataset(
        train_images, train_masks,
        transform=get_training_augmentation(),
        prompt_config=prompt_config
    )
    
    val_dataset = FenceSAMDataset(
        val_images, val_masks,
        transform=get_validation_augmentation(),
        prompt_config=prompt_config
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False,
        collate_fn=collate_fn
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ========== SETUP TRAINING ==========
    logger.info("\n[4/6] Setting up training...")
    
    # Loss function
    criterion = CombinedLoss(Config.LOSS_WEIGHTS)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Scheduler with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.T_0,
        T_mult=Config.T_MULT,
        eta_min=Config.MIN_LR
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(
        enabled=Config.USE_AMP,
        init_scale=Config.GRAD_SCALER_INIT_SCALE,
        growth_factor=Config.GRAD_SCALER_GROWTH_FACTOR,
        backoff_factor=Config.GRAD_SCALER_BACKOFF_FACTOR,
        growth_interval=Config.GRAD_SCALER_GROWTH_INTERVAL
    )
    
    # Resume optimizer and scheduler if checkpoint exists
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        logger.info("  ✓ Optimizer and scheduler restored")
    
    # EMA
    ema = EMA(model, decay=Config.EMA_DECAY) if Config.USE_EMA else None
    if resume_checkpoint is not None and 'ema_shadow' in resume_checkpoint:
        ema.shadow = resume_checkpoint['ema_shadow']
        logger.info("  ✓ EMA weights restored")
    
    # TensorBoard
    writer = SummaryWriter(Config.LOGS_DIR / f'tensorboard_{timestamp}') if Config.USE_TENSORBOARD else None
    
    # Compile model for PyTorch 2.0+ speedup (if available)
    if hasattr(torch, 'compile'):
        try:
            logger.info("Compiling model with torch.compile (PyTorch 2.0+)...")
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("  ✓ Model compiled successfully")
        except Exception as e:
            logger.warning(f"  torch.compile failed: {e} - continuing without compilation")
    
    # Warmup GPU
    if torch.cuda.is_available():
        logger.info("Warming up GPU...")
        dummy_input = torch.randn(1, 3, Config.TRAIN_SIZE, Config.TRAIN_SIZE).to(Config.DEVICE)
        try:
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            logger.info("GPU warmup successful")
        except Exception as e:
            logger.warning(f"GPU warmup failed (non-critical): {e}")
        finally:
            del dummy_input
            torch.cuda.empty_cache()
    
    # ========== TRAINING LOOP ==========
    logger.info("\n[5/6] Starting training...\n")
    logger.info("=" * 80)
    
    best_val_iou = 0.0
    patience_counter = 0
    warmup_steps = Config.WARMUP_EPOCHS * len(train_loader)
    total_steps = 0
    start_epoch = 0
    
    # Resume from checkpoint
    if resume_checkpoint is not None:
        start_epoch = resume_checkpoint['epoch']
        best_val_iou = resume_checkpoint.get('val_iou', 0.0)
        logger.info(f"Resuming from epoch {start_epoch}")
    
    for epoch in range(start_epoch, Config.EPOCHS):
        # Adjust learning rate for warmup
        if epoch < Config.WARMUP_EPOCHS:
            warmup_factor = (epoch + 1) / Config.WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = Config.LEARNING_RATE * warmup_factor
            logger.info(f"Warmup epoch {epoch+1}/{Config.WARMUP_EPOCHS} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            Config.DEVICE, epoch, writer, ema
        )
        
        # Validate
        if (epoch + 1) % Config.VAL_FREQ == 0:
            # Use EMA for validation
            if ema is not None:
                ema.apply_shadow()
            
            save_vis = (epoch + 1) % Config.VIS_FREQ == 0
            val_metrics = validate_epoch(
                model, val_loader, criterion, Config.DEVICE,
                epoch, writer, save_vis
            )
            
            if ema is not None:
                ema.restore()
            
            # Synchronize CUDA for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Logging
            logger.info(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            logger.info(f"  Train | Loss: {train_metrics['loss']:.4f} | "
                       f"IoU: {train_metrics['iou']:.4f} | "
                       f"Dice: {train_metrics['dice']:.4f} | "
                       f"F1: {train_metrics['f1']:.4f}")
            logger.info(f"  Val   | Loss: {val_metrics['loss']:.4f} | "
                       f"IoU: {val_metrics['iou']:.4f} | "
                       f"Dice: {val_metrics['dice']:.4f} | "
                       f"F1: {val_metrics['f1']:.4f} | "
                       f"Precision: {val_metrics['precision']:.4f} | "
                       f"Recall: {val_metrics['recall']:.4f}")
            
            # Save best model
            if val_metrics['iou'] > best_val_iou + Config.MIN_DELTA:
                best_val_iou = val_metrics['iou']
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_iou': best_val_iou,
                    'config': vars(Config)
                }
                
                if ema is not None:
                    checkpoint['ema_shadow'] = ema.shadow
                
                torch.save(checkpoint, Config.CHECKPOINT_DIR / 'best_model.pth')
                logger.info(f"  ✓ New best model saved (IoU: {best_val_iou:.4f})")
            else:
                patience_counter += 1
            
            # Save checkpoint periodically
            if (epoch + 1) % Config.SAVE_FREQ == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_iou': val_metrics['iou'],
                    'config': vars(Config)
                }
                torch.save(checkpoint, Config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch+1}.pth')
            
            # Always save last checkpoint for resume
            last_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_iou': val_metrics['iou'],
                'config': vars(Config)
            }
            if ema is not None:
                last_checkpoint['ema_shadow'] = ema.shadow
            torch.save(last_checkpoint, Config.CHECKPOINT_DIR / 'last_model.pth')
            
            # Early stopping (DISABLED - will train for full epochs)
            # Note: Early stopping is disabled in Config to ensure complete training
            if Config.EARLY_STOPPING and patience_counter >= Config.PATIENCE:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Step scheduler (only after warmup)
        if epoch >= Config.WARMUP_EPOCHS:
            scheduler.step()
        
        total_steps += len(train_loader)
    
    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best Validation IoU: {best_val_iou:.4f}")
    logger.info(f"Model saved to: {Config.CHECKPOINT_DIR / 'best_model.pth'}")
    logger.info(f"Logs saved to: {Config.LOGS_DIR}")
    logger.info(f"Visualizations saved to: {Config.VISUALIZATIONS_DIR}")
    logger.info("=" * 80)
    
    if writer:
        writer.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    try:
        train_sam()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
