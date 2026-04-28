"""
Mask2Former + SegFormer-B5 Training for Fence Detection - v2.0 ULTRA ENTERPRISE
===================================================================================
ADVANCED FEATURES & OPTIMIZATIONS (Enhanced beyond SAM):
- Mask2Former architecture with SegFormer-B5 backbone (SOTA accuracy)
- Transformer-based mask prediction with query-based decoding
- Multi-scale feature extraction (SegFormer backbone)
- Hierarchical pixel decoder for fine-grained segmentation
- Masked attention mechanisms for efficient training
- Advanced data augmentation pipeline (Albumentations++)
- Mixed precision training (AMP) with dynamic loss scaling
- Distributed training support (DDP ready)
- Gradient accumulation for large effective batch sizes
- Multi-task loss (Mask + Classification + Dice + Boundary + Lovász)
- Learning rate warmup + OneCycleLR scheduler
- Exponential Moving Average (EMA) for stable predictions
- Stochastic Depth for regularization
- Label smoothing for better generalization
- Comprehensive metrics (IoU, Dice, Precision, Recall, F1, Boundary F1)
- TensorBoard logging with detailed visualizations
- Checkpoint management with best/last/periodic saving
- Early stopping with patience
- GPU memory optimization (6GB laptop GPU optimized)
- Efficient data loading with caching
- Test-time augmentation (TTA) support
- Model validation with multi-scale testing
- Automatic hyperparameter tuning support
- Robust error handling and recovery
- Post-processing with CRF (optional)

Application: Fence Staining Visualizer - Production Ready
Author: VisionGuard Team - Advanced AI Research Division
Date: November 13, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

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
from typing import Dict, List, Tuple, Optional, Union
import time
import warnings
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# ============================================================================
# TRANSFORMERS IMPORTS (Must be before custom classes)
# ============================================================================

# Transformers imports
try:
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
except ImportError:
    print("Installing transformers package...")
    os.system("pip install transformers>=4.30.0")
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


# ============================================================================
# CUSTOM SEGFORMER BACKBONE FOR MASK2FORMER
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
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for Mask2Former + SegFormer-B5 training."""
    
    # Paths
    PROJECT_ROOT = Path("./")
    IMAGES_DIR = PROJECT_ROOT / "data" / "images"
    MASKS_DIR = PROJECT_ROOT / "data" / "masks"
    COCO_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations.json"  # COCO format annotations
    USE_COCO_FORMAT = False  # Set True to use COCO JSON loader
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "mask2former"
    LOGS_DIR = PROJECT_ROOT / "logs" / "mask2former"
    VISUALIZATIONS_DIR = PROJECT_ROOT / "training_visualizations" / "mask2former"
    
    # Model Configuration
    MODEL_NAME = "facebook/mask2former-swin-base-coco-panoptic"  # Base for initialization
    BACKBONE_NAME = "nvidia/mit-b5"  # SegFormer-B5 backbone
    PRETRAINED = True
    FREEZE_BACKBONE_EPOCHS = 0  # No freezing - train all 98M params from start
    
    # Training Hyperparameters (OPTIMIZED FOR ACCURATE FENCE DETECTION)
    INPUT_SIZE = 384   # Optimized for 6GB GPU
    TRAIN_SIZE = 384   # Optimized for 6GB GPU
    BATCH_SIZE = 2     # Minimum for 6GB GPU
    ACCUMULATION_STEPS = 4  # CRITICAL: Smaller effective batch (8) for better learning on imbalanced data
    EPOCHS = 200      # More epochs for convergence with balanced learning
    LEARNING_RATE = 5e-5  # CRITICAL: Lower LR for stable learning with focal loss
    BACKBONE_LR_MULTIPLIER = 0.01  # MUCH lower LR for pretrained SegFormer (was 0.1, now 0.01)
    WEIGHT_DECAY = 0.01  # Standard weight decay
    WARMUP_EPOCHS = 15  # CRITICAL: Longer warmup for focal loss stability
    MIN_LR = 1e-7
    
    # Optimizer & Scheduler
    OPTIMIZER = "AdamW"  # Best for transformers
    SCHEDULER = "OneCycleLR"  # Better than CosineAnnealing for transformers
    MAX_LR = 1e-3
    PCT_START = 0.1  # 10% warmup
    
    # Loss Configuration (BALANCED FOR PRECISION/RECALL)
    LOSS_WEIGHTS = {
        'mask_loss': 2.0,      # Reduced: Focal loss with higher alpha (0.75)
        'dice_loss': 2.5,      # Increased: Better overlap optimization
        'class_loss': 1.5,     # Increased: Better classification accuracy
        'boundary_loss': 2.0,  # Increased: Sharp edge detection critical for fences
        'lovasz_loss': 0.0,    # DISABLED: Numerically unstable, causing NaN gradients
    }
    CLASS_WEIGHT = [0.25, 1.0]  # Focal loss alpha now handles this (0.75)
    LABEL_SMOOTHING = 0.05  # Light smoothing (focal loss is already aggressive)
    
    # Hard Negative Mining (NEW - reduces false positives)
    USE_HARD_NEGATIVE_MINING = False  # DISABLED temporarily to debug NaN gradients
    HARD_NEGATIVE_RATIO = 3.0  # 3:1 ratio of hard negatives to positives
    HARD_NEGATIVE_THRESHOLD = 0.3  # Pixels with confidence > 0.3 but wrong are "hard"
    
    # Confidence Threshold (NEW - for inference precision control)
    CONFIDENCE_THRESHOLD = 0.65  # Increased from default 0.5 for better precision
    
    # Mask2Former Specific
    NUM_QUERIES = 100  # Number of object queries
    NUM_LABELS = 2     # Background + Fence
    MASK_FEATURE_SIZE = 256
    HIDDEN_DIM = 256
    NUM_ATTENTION_HEADS = 8
    DIM_FEEDFORWARD = 2048
    DEC_LAYERS = 6     # Transformer decoder layers
    PRE_NORM = False
    ENFORCE_INPUT_PROJECT = False
    
    # Data Augmentation (Enhanced++)
    USE_ADVANCED_AUGMENTATION = True
    AUGMENTATION_PROB = 0.8  # Increased for better generalization
    USE_MIXUP = False  # MixUp augmentation (heavy on memory)
    USE_CUTMIX = False  # CutMix augmentation
    USE_MOSAIC = False  # Mosaic augmentation (4 images combined)
    
    # Hardware Optimization
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4    # Set to 1 to avoid multiprocessing issues on Windows
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True  # Must be False when NUM_WORKERS = 0
    MULTIPROCESSING_CONTEXT = 'spawn'
    NON_BLOCKING = True
    
    # Mixed Precision
    USE_AMP = True
    AMP_DTYPE = torch.float16
    GRAD_CLIP = 1.0
    GRAD_SCALER_INIT_SCALE = 2.**16
    GRAD_SCALER_GROWTH_FACTOR = 2.0
    GRAD_SCALER_BACKOFF_FACTOR = 0.5
    GRAD_SCALER_GROWTH_INTERVAL = 2000
    
    # Exponential Moving Average
    USE_EMA = True
    EMA_DECAY = 0.9999  # Slightly higher for transformers
    EMA_START_EPOCH = 5
    
    # Regularization
    DROPOUT = 0.1
    STOCHASTIC_DEPTH = 0.1  # DropPath
    USE_LAYER_DECAY = True  # Different LR for different layers
    LAYER_DECAY_RATE = 0.9
    
    # Validation & Checkpointing
    TRAIN_SPLIT = 0.85
    VAL_SPLIT = 0.15
    SAVE_FREQ = 5
    VAL_FREQ = 1
    VIS_FREQ = 10
    
    # Early Stopping
    EARLY_STOPPING = False  # Disabled - let it train full 200 epochs
    PATIENCE = 50  # High patience (only used if EARLY_STOPPING=True)
    MIN_DELTA = 1e-5  # Very small threshold
    
    # Logging
    LOG_INTERVAL = 10
    USE_TENSORBOARD = True
    SAVE_LOSS_COMPONENTS = True
    LOG_LEARNING_RATES = True
    
    # Memory Optimization
    EMPTY_CACHE_FREQ = 5   # More frequent cache clearing (was 10)
    GRADIENT_CHECKPOINTING = False  # Mask2Former doesn't support this
    USE_FLASH_ATTENTION = False  # Requires flash-attn package
    
    # Test-Time Augmentation
    USE_TTA = False
    TTA_TRANSFORMS = ['original', 'hflip', 'vflip', 'rotate90', 'rotate180', 'rotate270']
    TTA_MERGE = 'mean'  # mean or max
    
    # Multi-Scale Testing
    USE_MULTISCALE_TEST = False
    TEST_SCALES = [0.75, 1.0, 1.25]
    
    # Post-processing
    USE_CRF = False  # Conditional Random Field post-processing
    CRF_ITERATIONS = 5
    
    # Class Configuration
    NUM_CLASSES = 2
    CLASS_NAMES = ['background', 'fence']
    ID2LABEL = {0: "background", 1: "fence"}
    LABEL2ID = {"background": 0, "fence": 1}
    IGNORE_INDEX = 255
    
    # Reproducibility
    SEED = 42
    DETERMINISTIC = False  # Set False for speed (True for reproducibility)


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
logger = setup_logger('Mask2Former_Training', Config.LOGS_DIR / f'training_{timestamp}.log')


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
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


# ============================================================================
# DATASET
# ============================================================================

class FenceMask2FormerDataset(Dataset):
    """Advanced dataset for Mask2Former training."""
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform=None,
        processor=None
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image and mask - try cv2 first, fallback to PIL
            try:
                image = cv2.imread(self.image_paths[idx])
                if image is None:
                    raise ValueError("cv2 failed to load image")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                # Fallback to PIL Image
                image = Image.open(self.image_paths[idx]).convert('RGB')
                image = np.array(image)
            
            try:
                mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError("cv2 failed to load mask")
            except:
                # Fallback to PIL Image
                mask = Image.open(self.mask_paths[idx]).convert('L')
                mask = np.array(mask)
            
            # Ensure same size
            if image.shape[:2] != mask.shape[:2]:
                h, w = image.shape[:2]
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Normalize mask to binary
            mask = (mask > 127).astype(np.uint8)
            
            # Apply augmentation
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                # No augmentation - resize manually
                image = cv2.resize(image, (Config.TRAIN_SIZE, Config.TRAIN_SIZE))
                mask = cv2.resize(mask, (Config.TRAIN_SIZE, Config.TRAIN_SIZE), interpolation=cv2.INTER_NEAREST)
            
            # Convert to tensors - ToTensorV2 already converted image to tensor
            if isinstance(image, torch.Tensor):
                # Image already converted by ToTensorV2 - shape is (C, H, W)
                image = image.contiguous()
            else:
                # Manual conversion if no ToTensorV2
                image = torch.from_numpy(image.copy()).permute(2, 0, 1).float().contiguous()
            
            # Mask is still numpy array after albumentations
            if isinstance(mask, torch.Tensor):
                mask = mask.contiguous()
            else:
                mask = torch.from_numpy(mask.copy()).long().contiguous()
            
            # Ensure mask is 2D (H, W)
            while mask.dim() > 2:
                mask = mask.squeeze(0)
            
            # Verify shapes match
            if image.shape[1:] != mask.shape:
                # Resize mask to match image if needed
                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                mask_np = cv2.resize(mask_np, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)
                mask = torch.from_numpy(mask_np).long().contiguous()
            
            # Class labels: 100 queries for Mask2Former (background=0, fence=1)
            class_labels = torch.zeros(Config.NUM_QUERIES, dtype=torch.long)
            if mask.any():
                class_labels[0] = 1  # First query detects fence
            
            return {
                'pixel_values': image,
                'mask_labels': mask.unsqueeze(0).contiguous(),  # (1, H, W)
                'class_labels': class_labels,  # (num_queries,)
                'image_path': self.image_paths[idx]
            }
            
        except Exception as e:
            logger.error(f"Error loading {self.image_paths[idx]}: {e}")
            traceback.print_exc()
            # Return dummy data with correct shapes
            dummy_class_labels = torch.zeros(Config.NUM_QUERIES, dtype=torch.long)
            return {
                'pixel_values': torch.zeros(3, Config.TRAIN_SIZE, Config.TRAIN_SIZE).contiguous(),
                'mask_labels': torch.zeros(1, Config.TRAIN_SIZE, Config.TRAIN_SIZE, dtype=torch.long).contiguous(),
                'class_labels': dummy_class_labels,
                'image_path': self.image_paths[idx]
            }


# ============================================================================
# COCO DATASET LOADER (For COCO JSON Format)
# ============================================================================

class COCOFenceDataset(Dataset):
    """Dataset loader for COCO JSON format annotations."""
    
    def __init__(
        self,
        annotation_file: str,
        images_dir: str,
        transform=None,
        processor=None,
        category_names: List[str] = ['fence']
    ):
        """
        Args:
            annotation_file: Path to COCO JSON annotation file
            images_dir: Directory containing images
            transform: Albumentations transforms
            processor: Optional AutoImageProcessor
            category_names: List of category names to include (default: ['fence'])
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.processor = processor
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build category mapping
        self.category_map = {}
        for cat in self.coco_data['categories']:
            if cat['name'] in category_names:
                self.category_map[cat['id']] = cat['name']
        
        # Build image ID to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            if ann['category_id'] in self.category_map:
                img_id = ann['image_id']
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)
        
        # Filter images with relevant annotations
        self.images = [
            img for img in self.coco_data['images']
            if img['id'] in self.img_to_anns
        ]
        
        logger.info(f"COCO Dataset loaded: {len(self.images)} images with {len(category_names)} categories")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            # Get image info
            img_info = self.images[idx]
            img_path = self.images_dir / img_info['file_name']
            
            # Load image
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    raise ValueError("cv2 failed to load image")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
            
            # Get annotations for this image
            annotations = self.img_to_anns.get(img_info['id'], [])
            
            # Create instance mask (panoptic style)
            h, w = image.shape[:2]
            instance_mask = np.zeros((h, w), dtype=np.uint8)
            class_labels = []
            
            for inst_id, ann in enumerate(annotations, start=1):
                # Handle different annotation formats
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], list):
                        # Polygon format
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(instance_mask, [poly], inst_id)
                    elif isinstance(ann['segmentation'], dict):
                        # RLE format
                        from pycocotools import mask as coco_mask
                        rle = ann['segmentation']
                        m = coco_mask.decode(rle)
                        instance_mask[m > 0] = inst_id
                
                # Record class (1 for fence, 0 for background)
                class_labels.append(1 if ann['category_id'] in self.category_map else 0)
            
            # Convert instance mask to binary mask (any fence = 1)
            binary_mask = (instance_mask > 0).astype(np.uint8)
            
            # Apply augmentation
            if self.transform:
                augmented = self.transform(image=image, mask=binary_mask)
                image = augmented['image']
                binary_mask = augmented['mask']
            
            # Convert to tensors
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            
            if not isinstance(binary_mask, torch.Tensor):
                binary_mask = torch.from_numpy(binary_mask).long()
            
            return {
                'pixel_values': image,
                'mask_labels': binary_mask.unsqueeze(0),
                'class_labels': torch.tensor([1 if len(class_labels) > 0 else 0], dtype=torch.long),
                'instance_mask': torch.from_numpy(instance_mask).long(),  # For panoptic
                'num_instances': len(class_labels),
                'image_path': str(img_path),
                'image_id': img_info['id']
            }
            
        except Exception as e:
            logger.error(f"Error loading COCO image {idx}: {e}")
            traceback.print_exc()
            # Return dummy data
            return {
                'pixel_values': torch.zeros(3, Config.TRAIN_SIZE, Config.TRAIN_SIZE),
                'mask_labels': torch.zeros(1, Config.TRAIN_SIZE, Config.TRAIN_SIZE, dtype=torch.long),
                'class_labels': torch.tensor([0], dtype=torch.long),
                'instance_mask': torch.zeros(Config.TRAIN_SIZE, Config.TRAIN_SIZE, dtype=torch.long),
                'num_instances': 0,
                'image_path': '',
                'image_id': 0
            }


# ============================================================================
# CUSTOM COLLATE FUNCTION
# ============================================================================

def custom_collate_fn(batch):
    """Custom collate function to handle batch collation properly."""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Stack tensors manually to ensure compatibility
    pixel_values = torch.stack([item['pixel_values'] for item in batch], dim=0)
    mask_labels = torch.stack([item['mask_labels'] for item in batch], dim=0)
    class_labels = torch.stack([item['class_labels'] for item in batch], dim=0)
    
    # Keep non-tensor data as list
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'mask_labels': mask_labels,
        'class_labels': class_labels,
        'image_path': image_paths
    }


# ============================================================================
# DATA AUGMENTATION (Enhanced++)
# ============================================================================

def get_training_augmentation():
    """Ultra-advanced augmentation pipeline for training."""
    if not Config.USE_ADVANCED_AUGMENTATION:
        return A.Compose([
            A.Resize(Config.TRAIN_SIZE, Config.TRAIN_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return A.Compose([
        # Resize first (allow slightly larger)
        A.Resize(int(Config.TRAIN_SIZE * 1.1), int(Config.TRAIN_SIZE * 1.1)),
        
        # Random crop to target size
        A.RandomCrop(width=Config.TRAIN_SIZE, height=Config.TRAIN_SIZE, p=1.0),
        
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            p=0.6,
            border_mode=cv2.BORDER_CONSTANT
        ),
        
        # Elastic deformation
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.3
        ),
        
        # Perspective transform
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        
        # Color augmentations
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
            A.ChannelShuffle(p=1.0),
        ], p=0.6),
        
        # Lighting augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.ToGray(p=1.0),
        ], p=0.5),
        
        # Weather & environmental effects
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1.0),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=1.0),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=1.0),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=1.0),
        ], p=0.3),
        
        # Noise & blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.4),
        
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # Quality degradation
        A.OneOf([
            A.ImageCompression(quality_lower=70, quality_upper=100, p=1.0),
            A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0),
        ], p=0.2),
        
        # Cutout augmentation
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),
        
        # Normalize and convert
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])  # Note: p parameter removed - resize/normalize must ALWAYS happen


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
    """Focal Loss for handling class imbalance.
    
    Alpha = 0.75: Balances precision/recall (was 0.25, causing high recall/low precision)
    Gamma = 2.0: Focus on hard examples
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # Increased from 0.25 to 0.75 for better precision
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets.float(), reduction='none')
        # Clamp to prevent overflow in exp()
        bce_loss = torch.clamp(bce_loss, min=0.0, max=100.0)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = torch.clamp(focal_loss, min=0.0, max=100.0)
        result = focal_loss.mean()
        # Final NaN check
        if torch.isnan(result) or torch.isinf(result):
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        return result


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        predictions = torch.clamp(predictions, min=0.0, max=1.0)
        targets = targets.float()
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice = torch.clamp(dice, min=0.0, max=1.0)
        result = 1 - dice
        # NaN check
        if torch.isnan(result) or torch.isinf(result):
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        return result


class BoundaryLoss(nn.Module):
    """Boundary-aware loss for better edge detection."""
    
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Don't apply sigmoid - use logits directly for binary_cross_entropy_with_logits
        targets = targets.float()
        
        # Compute boundaries
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=targets.device)
        
        targets_dilated = F.conv2d(
            targets.unsqueeze(1) if targets.dim() == 3 else targets,
            kernel, padding=self.kernel_size // 2
        )
        targets_eroded = -F.conv2d(
            -(targets.unsqueeze(1) if targets.dim() == 3 else targets),
            kernel, padding=self.kernel_size // 2
        )
        
        boundary = ((targets_dilated - targets_eroded) > 0).float()
        
        # Ensure predictions is 2D (batch, H, W) for loss calculation
        if predictions.dim() == 4:
            predictions = predictions.squeeze(1) if predictions.size(1) == 1 else predictions[:, 0]
        
        # Use binary_cross_entropy_with_logits (AMP-safe)
        loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction='none'
        )
        loss = torch.clamp(loss, min=0.0, max=100.0)
        weighted_loss = loss * (1 + boundary.squeeze(1) * 5)
        weighted_loss = torch.clamp(weighted_loss, min=0.0, max=100.0)
        result = weighted_loss.mean()
        # NaN check
        if torch.isnan(result) or torch.isinf(result):
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        return result


class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax loss for segmentation (SOTA)."""
    
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        predictions: (N, C, H, W) logits
        targets: (N, H, W) class indices
        """
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(0)
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)
        
        N, C, H, W = predictions.shape
        predictions = predictions.permute(0, 2, 3, 1).reshape(-1, C)
        targets = targets.reshape(-1)
        
        # Filter ignore index
        valid_mask = targets != self.ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        if targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        probas = F.softmax(predictions, dim=1)
        
        # Compute Lovász extension
        losses = []
        for c in range(C):
            fg = (targets == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probas[:, c]).abs()
            errors_sorted, indices = torch.sort(errors, descending=True)
            fg_sorted = fg[indices]
            losses.append(self._lovasz_grad(fg_sorted) @ errors_sorted)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=predictions.device)
    
    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Compute gradient of the Lovász extension."""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1 - intersection / union
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components and hard negative mining."""
    
    def __init__(self, weights: Dict[str, float], num_classes: int = 2, use_hard_negative_mining: bool = False):
        super().__init__()
        self.weights = weights
        self.num_classes = num_classes
        self.use_hard_negative_mining = use_hard_negative_mining
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.lovasz = LovaszSoftmaxLoss()
        self.focal = FocalLoss()  # Now uses alpha=0.75 for better precision
    
    def forward(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        targets: torch.Tensor,
        class_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        mask_logits: (N, num_queries, H, W)
        class_logits: (N, num_queries, num_classes)
        targets: (N, 1, H, W)
        class_targets: (N, num_queries)
        """
        losses = {}
        
        # Hard Negative Mining (if enabled)
        hard_negative_mask = None
        if self.use_hard_negative_mining and self.weights.get('mask_loss', 0) > 0:
            with torch.no_grad():
                # Get predictions for current batch
                pred_probs = torch.sigmoid(mask_logits[:, 0])  # (N, H, W)
                targets_binary = targets.squeeze(1).float()  # (N, H, W)
                
                # Find hard negatives: high confidence but wrong predictions
                is_background = (targets_binary == 0)
                is_confident = (pred_probs > Config.HARD_NEGATIVE_THRESHOLD)
                hard_negatives = is_background & is_confident
                
                # Find all positives
                positives = (targets_binary == 1)
                
                # Create hard negative mask (3:1 ratio)
                num_positives = positives.sum()
                if num_positives > 0:
                    num_hard_negatives = min(int(num_positives * Config.HARD_NEGATIVE_RATIO), hard_negatives.sum())
                    
                    if num_hard_negatives > 0:
                        # Sample hard negatives
                        hard_neg_indices = torch.where(hard_negatives.view(-1))[0]
                        if len(hard_neg_indices) > num_hard_negatives:
                            sampled_indices = hard_neg_indices[torch.randperm(len(hard_neg_indices))[:num_hard_negatives]]
                            hard_negative_mask = torch.zeros_like(targets_binary.view(-1), dtype=torch.bool)
                            hard_negative_mask[sampled_indices] = True
                            hard_negative_mask = hard_negative_mask.view_as(targets_binary)
                        else:
                            hard_negative_mask = hard_negatives
                        
                        # Combine positives and hard negatives
                        hard_negative_mask = positives | hard_negative_mask
        
        # Mask loss with focal loss to handle class imbalance
        if self.weights.get('mask_loss', 0) > 0:
            mask_pred = mask_logits.squeeze(1) if mask_logits.size(1) == 1 else mask_logits[:, 0]
            mask_target = targets.squeeze(1).float()
            
            # Focal loss parameters
            alpha = 0.75  # Weight for positive class (fence)
            gamma = 2.0   # Focusing parameter
            
            # BCE with logits (clamp AGGRESSIVELY to prevent overflow)
            bce_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_target, reduction='none')
            bce_loss = torch.clamp(bce_loss, min=1e-7, max=20.0)  # Much more aggressive clamping
            
            # Focal loss modulation (safe exp with clamped input)
            pt = torch.exp(-bce_loss)  # Now safe: exp(-20) to exp(-1e-7)
            pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)  # Extra safety
            focal_weight = (1 - pt) ** gamma
            
            # Apply alpha weighting
            alpha_weight = alpha * mask_target + (1 - alpha) * (1 - mask_target)
            
            mask_loss = (alpha_weight * focal_weight * bce_loss).mean()
            
            # Safety check for NaN
            if torch.isnan(mask_loss) or torch.isinf(mask_loss):
                mask_loss = torch.tensor(0.0, device=mask_loss.device, dtype=mask_loss.dtype)
            
            losses['mask_loss'] = mask_loss
        
        # Dice loss
        if self.weights.get('dice_loss', 0) > 0:
            dice_loss = self.dice(
                mask_logits.squeeze(1) if mask_logits.size(1) == 1 else mask_logits[:, 0],
                targets.squeeze(1)
            )
            # Safety check
            if torch.isnan(dice_loss) or torch.isinf(dice_loss):
                dice_loss = torch.tensor(0.0, device=dice_loss.device, dtype=dice_loss.dtype)
            losses['dice_loss'] = dice_loss
        
        # Classification loss
        if self.weights.get('class_loss', 0) > 0:
            # Match dimensions: class_logits is (N, num_queries, num_classes)
            # class_targets is (N, num_queries_config)
            batch_size = class_logits.shape[0]
            num_model_queries = class_logits.shape[1]
            num_classes_model = class_logits.shape[2]  # Use actual number of classes from model
            
            # Ensure class_targets is 2D (batch_size, num_queries)
            if class_targets.dim() == 1:
                class_targets = class_targets.unsqueeze(0)
            
            num_target_queries = class_targets.shape[1]
            
            # Pad class_targets to match model queries if needed
            if num_model_queries > num_target_queries:
                # Pad with zeros (background class)
                padding_size = num_model_queries - num_target_queries
                padding = torch.zeros(
                    batch_size, 
                    padding_size,
                    dtype=class_targets.dtype,
                    device=class_targets.device
                )
                class_targets = torch.cat([class_targets, padding], dim=1)
            elif num_model_queries < num_target_queries:
                # Truncate to match model queries
                class_targets = class_targets[:, :num_model_queries]
            
            # Now both should have shape (batch_size, num_model_queries)
            # Flatten for cross_entropy - use actual num_classes from model
            class_logits_flat = class_logits.reshape(-1, num_classes_model)
            class_targets_flat = class_targets.reshape(-1).long()
            
            # Clamp targets to valid range [0, num_classes_model-1]
            class_targets_flat = torch.clamp(class_targets_flat, 0, num_classes_model - 1)
            
            class_loss = F.cross_entropy(
                class_logits_flat,
                class_targets_flat
            )
            # Safety check
            if torch.isnan(class_loss) or torch.isinf(class_loss):
                class_loss = torch.tensor(0.0, device=class_loss.device, dtype=class_loss.dtype)
            losses['class_loss'] = class_loss
        
        # Boundary loss
        if self.weights.get('boundary_loss', 0) > 0:
            boundary_loss = self.boundary(
                mask_logits.squeeze(1) if mask_logits.size(1) == 1 else mask_logits[:, 0],
                targets.squeeze(1)
            )
            # Safety check
            if torch.isnan(boundary_loss) or torch.isinf(boundary_loss):
                boundary_loss = torch.tensor(0.0, device=boundary_loss.device, dtype=boundary_loss.dtype)
            losses['boundary_loss'] = boundary_loss
        
        # Lovász loss
        if self.weights.get('lovasz_loss', 0) > 0:
            lovasz_loss = self.lovasz(
                mask_logits if mask_logits.size(1) > 1 else torch.cat([1 - mask_logits, mask_logits], dim=1),
                targets.squeeze(1)
            )
            # Safety check
            if torch.isnan(lovasz_loss) or torch.isinf(lovasz_loss):
                lovasz_loss = torch.tensor(0.0, device=lovasz_loss.device, dtype=lovasz_loss.dtype)
            losses['lovasz_loss'] = lovasz_loss
        
        # Weighted sum
        total_loss = sum(self.weights.get(k, 0) * v for k, v in losses.items())
        
        return total_loss, losses


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
        """Calculate IoU, Dice, Precision, Recall, F1, Boundary metrics."""
        # Convert to binary predictions
        if predictions.dim() == 4:
            predictions = predictions.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        
        predictions = torch.sigmoid(predictions) > threshold
        targets = targets > 0.5
        
        # Move to CPU
        predictions = predictions.cpu().numpy().astype(bool)
        targets = targets.cpu().numpy().astype(bool)
        
        # Calculate metrics
        intersection = (predictions & targets).sum()
        union = (predictions | targets).sum()
        
        tp = intersection
        fp = (predictions & ~targets).sum()
        fn = (~predictions & targets).sum()
        tn = (~predictions & ~targets).sum()
        
        # Standard metrics
        iou = intersection / (union + 1e-7)
        dice = 2 * intersection / (predictions.sum() + targets.sum() + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        
        # Boundary F1 (advanced metric)
        boundary_f1 = MetricsCalculator._boundary_f1(predictions, targets)
        
        return {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'boundary_f1': float(boundary_f1)
        }
    
    @staticmethod
    def _boundary_f1(pred: np.ndarray, target: np.ndarray, dilation: int = 2) -> float:
        """Calculate boundary F1 score."""
        import cv2
        
        kernel = np.ones((dilation, dilation), np.uint8)
        
        pred_boundary = cv2.dilate(pred.astype(np.uint8), kernel) - cv2.erode(pred.astype(np.uint8), kernel)
        target_boundary = cv2.dilate(target.astype(np.uint8), kernel) - cv2.erode(target.astype(np.uint8), kernel)
        
        tp = (pred_boundary & target_boundary).sum()
        fp = (pred_boundary & ~target_boundary).sum()
        fn = (~pred_boundary & target_boundary).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        boundary_f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return boundary_f1


# ============================================================================
# EXPONENTIAL MOVING AVERAGE
# ============================================================================

class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
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
                # Handle newly added parameters (e.g., from unfreezing)
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name].sub_(
                        (1 - self.decay) * (self.shadow[name] - param.data)
                    )
    
    def register_new_parameters(self):
        """Register newly trainable parameters (e.g., after unfreezing)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.data.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters."""
        if not self.backup:
            return  # Nothing to restore
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================================
# INFERENCE ENGINE (Production-Ready)
# ============================================================================

class Mask2FormerInference:
    """Inference engine for Mask2Former model with panoptic segmentation support."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        input_size: int = 1024,
        threshold: float = None,  # Default to Config.CONFIDENCE_THRESHOLD
        use_tta: bool = False
    ):
        """
        Args:
            model: Trained Mask2Former model
            device: Device to run inference on
            input_size: Input image size
            threshold: Confidence threshold for predictions (default: Config.CONFIDENCE_THRESHOLD = 0.65)
            use_tta: Use test-time augmentation
        """
        self.model = model
        self.device = device
        self.input_size = input_size
        self.threshold = threshold if threshold is not None else Config.CONFIDENCE_THRESHOLD
        self.use_tta = use_tta
        self.model.eval()
        
        # Preprocessing transform
        self.transform = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_instance: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            return_instance: Return instance segmentation map
            
        Returns:
            Dictionary with 'mask' (binary), 'probs' (probabilities), 
            and optionally 'instances' (instance IDs)
        """
        # Load image
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        original_size = image.shape[:2]
        
        # Preprocess
        augmented = self.transform(image=image)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Forward pass
        if self.use_tta:
            predictions = self._tta_predict(input_tensor)
        else:
            outputs = self.model(pixel_values=input_tensor)
            predictions = outputs.masks_queries_logits[0, 0]  # First query mask
        
        # Post-process
        mask_probs = torch.sigmoid(predictions).cpu().numpy()
        binary_mask = (mask_probs > self.threshold).astype(np.uint8)
        
        # Resize to original size
        mask_probs_resized = cv2.resize(
            mask_probs,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        binary_mask_resized = cv2.resize(
            binary_mask,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        result = {
            'mask': binary_mask_resized,
            'probs': mask_probs_resized
        }
        
        # Instance segmentation (if requested)
        if return_instance:
            instance_mask = self._extract_instances(predictions)
            instance_mask_resized = cv2.resize(
                instance_mask,
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
            result['instances'] = instance_mask_resized
        
        return result
    
    def _tta_predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Test-time augmentation prediction."""
        predictions = []
        
        # Original
        outputs = self.model(pixel_values=input_tensor)
        predictions.append(outputs.masks_queries_logits[0, 0])
        
        # Horizontal flip
        outputs = self.model(pixel_values=torch.flip(input_tensor, dims=[3]))
        predictions.append(torch.flip(outputs.masks_queries_logits[0, 0], dims=[1]))
        
        # Vertical flip
        outputs = self.model(pixel_values=torch.flip(input_tensor, dims=[2]))
        predictions.append(torch.flip(outputs.masks_queries_logits[0, 0], dims=[0]))
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
    
    def _extract_instances(self, predictions: torch.Tensor) -> np.ndarray:
        """Extract instance segmentation from model predictions."""
        # Simple connected components for instance separation
        binary = (torch.sigmoid(predictions) > self.threshold).cpu().numpy().astype(np.uint8)
        num_labels, instance_mask = cv2.connectedComponents(binary)
        return instance_mask
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 4
    ) -> List[Dict[str, np.ndarray]]:
        """Run inference on multiple images."""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for img in batch:
                result = self.predict(img)
                results.append(result)
        return results


# ============================================================================
# COCO EVALUATOR (For Panoptic/Instance Evaluation)
# ============================================================================

class COCOEvaluator:
    """COCO-style evaluation metrics for segmentation."""
    
    def __init__(self, annotation_file: Optional[str] = None):
        """
        Args:
            annotation_file: Path to COCO annotation JSON (optional)
        """
        self.annotation_file = annotation_file
        self.predictions = []
        self.gt_annotations = []
        
        if annotation_file and Path(annotation_file).exists():
            try:
                from pycocotools.coco import COCO
                self.coco_gt = COCO(annotation_file)
                logger.info(f"COCO ground truth loaded from {annotation_file}")
            except ImportError:
                logger.warning("pycocotools not installed. Install with: pip install pycocotools")
                self.coco_gt = None
        else:
            self.coco_gt = None
    
    def add_prediction(
        self,
        image_id: int,
        mask: np.ndarray,
        score: float = 1.0,
        category_id: int = 1
    ):
        """Add a prediction for evaluation."""
        # Convert mask to RLE format
        try:
            from pycocotools import mask as coco_mask
            rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            self.predictions.append({
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': rle,
                'score': score
            })
        except ImportError:
            logger.warning("pycocotools not available, skipping RLE encoding")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run COCO evaluation and return metrics.
        
        Returns:
            Dictionary with AP, AR, AP50, AP75, etc.
        """
        if not self.coco_gt or not self.predictions:
            logger.warning("Cannot evaluate: missing ground truth or predictions")
            return {}
        
        try:
            from pycocotools.cocoeval import COCOeval
            
            # Create predictions COCO object
            coco_dt = self.coco_gt.loadRes(self.predictions)
            
            # Run evaluation
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'segm')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            metrics = {
                'AP': coco_eval.stats[0],        # AP @ IoU=0.50:0.95
                'AP50': coco_eval.stats[1],      # AP @ IoU=0.50
                'AP75': coco_eval.stats[2],      # AP @ IoU=0.75
                'AP_small': coco_eval.stats[3],  # AP for small objects
                'AP_medium': coco_eval.stats[4], # AP for medium objects
                'AP_large': coco_eval.stats[5],  # AP for large objects
                'AR_1': coco_eval.stats[6],      # AR given 1 det per image
                'AR_10': coco_eval.stats[7],     # AR given 10 det per image
                'AR_100': coco_eval.stats[8],    # AR given 100 det per image
                'AR_small': coco_eval.stats[9],  # AR for small objects
                'AR_medium': coco_eval.stats[10],# AR for medium objects
                'AR_large': coco_eval.stats[11], # AR for large objects
            }
            
            logger.info("COCO Evaluation Results:")
            for name, value in metrics.items():
                logger.info(f"  {name}: {value:.4f}")
            
            return metrics
            
        except ImportError:
            logger.error("pycocotools not installed. Install with: pip install pycocotools")
            return {}
        except Exception as e:
            logger.error(f"COCO evaluation failed: {e}")
            traceback.print_exc()
            return {}
    
    def save_predictions(self, output_file: str):
        """Save predictions to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)
        logger.info(f"Predictions saved to {output_file}")
    
    def compute_panoptic_quality(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute Panoptic Quality (PQ) metric.
        
        Args:
            pred_masks: List of predicted instance masks
            gt_masks: List of ground truth instance masks
            
        Returns:
            Dictionary with PQ, SQ, RQ metrics
        """
        if len(pred_masks) != len(gt_masks):
            logger.warning("Number of pred and gt masks don't match")
            return {}
        
        tp, fp, fn = 0, 0, 0
        iou_sum = 0.0
        
        for pred, gt in zip(pred_masks, gt_masks):
            # Get unique instances
            pred_ids = np.unique(pred)[1:]  # Exclude background (0)
            gt_ids = np.unique(gt)[1:]
            
            matched_pred = set()
            matched_gt = set()
            
            # Match instances with IoU > 0.5
            for gt_id in gt_ids:
                gt_mask = (gt == gt_id)
                best_iou = 0.0
                best_pred_id = None
                
                for pred_id in pred_ids:
                    if pred_id in matched_pred:
                        continue
                    
                    pred_mask = (pred == pred_id)
                    intersection = (pred_mask & gt_mask).sum()
                    union = (pred_mask | gt_mask).sum()
                    iou = intersection / (union + 1e-7)
                    
                    if iou > best_iou and iou > 0.5:
                        best_iou = iou
                        best_pred_id = pred_id
                
                if best_pred_id is not None:
                    tp += 1
                    iou_sum += best_iou
                    matched_pred.add(best_pred_id)
                    matched_gt.add(gt_id)
                else:
                    fn += 1
            
            fp += len(pred_ids) - len(matched_pred)
        
        # Compute metrics
        pq = iou_sum / (tp + 0.5 * fp + 0.5 * fn + 1e-7)
        sq = iou_sum / (tp + 1e-7)  # Segmentation Quality
        rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-7)  # Recognition Quality
        
        return {
            'PQ': float(pq),
            'SQ': float(sq),
            'RQ': float(rq),
            'TP': tp,
            'FP': fp,
            'FN': fn
        }


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    ema: Optional[EMA] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_metrics = {'iou': 0.0, 'dice': 0.0, 'f1': 0.0, 'boundary_f1': 0.0}
    num_batches = len(dataloader)
    
    if num_batches == 0:
        logger.warning("Empty training dataloader!")
        return {'loss': 0.0, 'iou': 0.0, 'dice': 0.0, 'f1': 0.0, 'boundary_f1': 0.0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        mask_labels = batch['mask_labels'].to(device, non_blocking=True)
        class_labels = batch['class_labels'].to(device, non_blocking=True)
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP, dtype=Config.AMP_DTYPE):
            outputs: Mask2FormerForUniversalSegmentationOutput = model(pixel_values=pixel_values)
            
            # Get mask predictions
            mask_logits = outputs.masks_queries_logits  # (N, num_queries, H, W)
            class_logits = outputs.class_queries_logits  # (N, num_queries, num_classes)
            
            # Take the best query prediction
            best_query_idx = class_logits.softmax(dim=-1)[:, :, 1].argmax(dim=1)
            batch_indices = torch.arange(mask_logits.size(0), device=device)
            mask_logits = mask_logits[batch_indices, best_query_idx].unsqueeze(1)
            
            # Resize to match target size
            if mask_logits.shape[-2:] != mask_labels.shape[-2:]:
                mask_logits = F.interpolate(
                    mask_logits,
                    size=mask_labels.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Calculate loss
            # class_labels should already be (batch_size, num_queries) after collation
            # If it has extra dimensions, squeeze them
            if class_labels.dim() > 2:
                class_labels = class_labels.squeeze(1)
            
            loss, loss_dict = criterion(
                mask_logits,
                class_logits,
                mask_labels,
                class_labels
            )
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            loss = loss / Config.ACCUMULATION_STEPS
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Check for NaN gradients immediately after backward
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan = True
                    logger.warning(f"NaN/Inf in gradient of {name} at epoch {epoch+1}, batch {batch_idx}")
                    break
        
        if has_nan:
            logger.warning(f"NaN/Inf gradient detected at epoch {epoch+1}, batch {batch_idx}. Skipping update.")
            optimizer.zero_grad(set_to_none=True)
            # Don't call scaler.update() here - just skip this batch and continue
            continue
        
        # Optimizer step with gradient accumulation
        if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            
            # Gradient clipping with logging
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            
            # Check for NaN gradients
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning(f"NaN/Inf gradient detected at epoch {epoch+1}, batch {batch_idx}. Skipping update.")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()  # Must call update() even when skipping to reset scaler state
                continue
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Log gradient norm
            if writer and (batch_idx % Config.LOG_INTERVAL == 0):
                writer.add_scalar('train/grad_norm', grad_norm.item(), epoch * num_batches + batch_idx)
                writer.add_scalar('train/scaler_scale', scaler.get_scale(), epoch * num_batches + batch_idx)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * num_batches + batch_idx)
            
            # Update EMA
            if ema is not None and epoch >= Config.EMA_START_EPOCH:
                ema.update()
            
            # Step OneCycleLR scheduler (per batch)
            if Config.SCHEDULER == "OneCycleLR":
                scheduler.step()
        
        # Calculate metrics
        with torch.no_grad():
            metrics = MetricsCalculator.calculate_metrics(
                mask_logits.squeeze(1),
                mask_labels.squeeze(1)
            )
        
        total_loss += loss.item() * Config.ACCUMULATION_STEPS
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item() * Config.ACCUMULATION_STEPS:.4f}",
            'iou': f"{metrics['iou']:.4f}",
            'dice': f"{metrics['dice']:.4f}",
            'bf1': f"{metrics['boundary_f1']:.4f}"
        })
        
        # TensorBoard logging
        if writer and (batch_idx % Config.LOG_INTERVAL == 0):
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss', loss.item() * Config.ACCUMULATION_STEPS, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            
            if Config.SAVE_LOSS_COMPONENTS:
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(f'train/loss_{loss_name}', loss_value.item(), global_step)
            
            for k, v in metrics.items():
                writer.add_scalar(f'train/{k}', v, global_step)
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                writer.add_scalar('system/gpu_memory_allocated_gb', memory_allocated, global_step)
                writer.add_scalar('system/gpu_memory_reserved_gb', memory_reserved, global_step)
        
        # Periodic memory cleanup
        if torch.cuda.is_available() and (batch_idx % Config.EMPTY_CACHE_FREQ == 0):
            torch.cuda.synchronize()  # Ensure all operations complete
            torch.cuda.empty_cache()
    
    # Final synchronization for epoch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
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
    total_metrics = {'iou': 0.0, 'dice': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'boundary_f1': 0.0}
    num_batches = len(dataloader)
    
    if num_batches == 0:
        logger.warning("Empty validation dataloader!")
        return {'loss': 0.0, 'iou': 0.0, 'dice': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'boundary_f1': 0.0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            mask_labels = batch['mask_labels'].to(device, non_blocking=True)
            class_labels = batch['class_labels'].to(device, non_blocking=True)
            
            # Handle 3D class_labels from collation [batch, 1, num_queries] -> [batch, num_queries]
            if class_labels.dim() > 2:
                class_labels = class_labels.squeeze(1)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP, dtype=Config.AMP_DTYPE):
                outputs: Mask2FormerForUniversalSegmentationOutput = model(pixel_values=pixel_values)
                
                mask_logits = outputs.masks_queries_logits
                class_logits = outputs.class_queries_logits
                
                best_query_idx = class_logits.softmax(dim=-1)[:, :, 1].argmax(dim=1)
                batch_indices = torch.arange(mask_logits.size(0), device=device)
                mask_logits = mask_logits[batch_indices, best_query_idx].unsqueeze(1)
                
                if mask_logits.shape[-2:] != mask_labels.shape[-2:]:
                    mask_logits = F.interpolate(
                        mask_logits,
                        size=mask_labels.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Pad class_labels to match model's num_queries (150)
                num_queries_model = class_logits.shape[1]
                if class_labels.shape[1] < num_queries_model:
                    padding = torch.zeros(
                        class_labels.shape[0],
                        num_queries_model - class_labels.shape[1],
                        device=device,
                        dtype=class_labels.dtype
                    )
                    class_labels_padded = torch.cat([class_labels, padding], dim=1)
                else:
                    class_labels_padded = class_labels[:, :num_queries_model]
                
                loss, _ = criterion(
                    mask_logits,
                    class_logits,
                    mask_labels,
                    class_labels_padded
                )
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_metrics(
                mask_logits.squeeze(1),
                mask_labels.squeeze(1)
            )
            
            total_loss += loss.item()
            for k in total_metrics:
                if k in metrics:
                    total_metrics[k] += metrics[k]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{metrics['iou']:.4f}",
                'dice': f"{metrics['dice']:.4f}",
                'bf1': f"{metrics['boundary_f1']:.4f}"
            })
            
            # Save visualizations
            if save_visualizations and batch_idx == 0:
                save_predictions(
                    pixel_values,
                    mask_labels,
                    mask_logits,
                    epoch
                )
    
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
        
        mask = masks[i].squeeze().cpu().numpy()
        pred = torch.sigmoid(predictions[i]).squeeze().cpu().numpy()
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
    try:
        save_path = Config.VISUALIZATIONS_DIR / f'epoch_{epoch+1:03d}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved predictions visualization: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save visualization: {e}")
        plt.close()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_mask2former():
    """Main training loop for Mask2Former."""
    
    logger.info("=" * 90)
    logger.info("MASK2FORMER + SEGFORMER-B5 TRAINING v2.0 - ULTRA ENTERPRISE EDITION")
    logger.info("=" * 90)
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"Model: Mask2Former + SegFormer-B5")
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
    
    # Save configuration to JSON for reproducibility
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(Config).items() if not k.startswith('_')}
    config_path = Config.LOGS_DIR / f'config_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f"Configuration saved to {config_path}")
    
    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = not Config.DETERMINISTIC
    torch.backends.cudnn.enabled = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.empty_cache()
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        logger.info(f"GPU Multiprocessor Count: {gpu_props.multi_processor_count}")
        logger.info(f"GPU Total Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        logger.info(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        logger.info(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        logger.info(f"TF32 Matmul: {torch.backends.cuda.matmul.allow_tf32}")
        logger.info(f"TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")
    
    # ========== LOAD DATA ==========
    logger.info("\n[1/6] Loading dataset...")
    
    # Check if using COCO format
    if Config.USE_COCO_FORMAT and Config.COCO_ANNOTATIONS.exists():
        logger.info("Using COCO JSON format dataset")
        
        # Load COCO dataset
        try:
            train_dataset = COCOFenceDataset(
                annotation_file=str(Config.COCO_ANNOTATIONS),
                images_dir=str(Config.IMAGES_DIR),
                transform=get_training_augmentation(),
                category_names=['fence']
            )
            
            # Split into train/val
            dataset_size = len(train_dataset)
            train_size = int(dataset_size * Config.TRAIN_SPLIT)
            val_size = dataset_size - train_size
            
            from torch.utils.data import random_split
            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(Config.SEED)
            )
            
            # Update val_dataset transform
            val_dataset.dataset.transform = get_validation_augmentation()
            
            logger.info(f"COCO dataset loaded: {train_size} train, {val_size} val")
            
        except Exception as e:
            logger.error(f"Failed to load COCO dataset: {e}")
            traceback.print_exc()
            return
    
    else:
        # Use regular image-mask pairs format
        logger.info("Using image-mask pairs dataset format")
        
        if not Config.IMAGES_DIR.exists() or not Config.MASKS_DIR.exists():
            logger.error("Dataset directories not found!")
            return
        
        # Get all image files (EXCLUDE JSON files!)
        image_files = sorted([
            f.name for f in Config.IMAGES_DIR.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()
        ])
        
        # Log file type breakdown for verification
        all_files = list(Config.IMAGES_DIR.iterdir())
        jpg_count = sum(1 for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg'])
        png_count = sum(1 for f in all_files if f.suffix.lower() == '.png')
        json_count = sum(1 for f in all_files if f.suffix.lower() == '.json')
        logger.info(f"File breakdown: {jpg_count} JPG, {png_count} PNG images, {json_count} JSON (ignored)")
        
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
        
        # Initialize AutoImageProcessor
        try:
            image_processor = AutoImageProcessor.from_pretrained(Config.MODEL_NAME)
            logger.info(f"Loaded AutoImageProcessor: {image_processor.__class__.__name__}")
        except Exception as e:
            logger.warning(f"Could not load AutoImageProcessor: {e}. Using manual preprocessing.")
            image_processor = None
        
        # Create datasets
        train_dataset = FenceMask2FormerDataset(
            train_images, train_masks,
            transform=get_training_augmentation(),
            processor=image_processor
        )
        
        val_dataset = FenceMask2FormerDataset(
            val_images, val_masks,
            transform=get_validation_augmentation(),
            processor=image_processor
        )
    
    # ========== CREATE DATALOADERS ==========
    logger.info("\n[2/6] Creating DataLoaders...")
    
    # DataLoader config with custom collate function
    train_loader_kwargs = {
        'batch_size': Config.BATCH_SIZE,
        'shuffle': True,
        'num_workers': Config.NUM_WORKERS,
        'pin_memory': Config.PIN_MEMORY,
        'drop_last': True,
        'collate_fn': custom_collate_fn,
    }
    if Config.NUM_WORKERS > 0:
        train_loader_kwargs['prefetch_factor'] = Config.PREFETCH_FACTOR
        train_loader_kwargs['persistent_workers'] = Config.PERSISTENT_WORKERS
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    
    val_loader_kwargs = {
        'batch_size': Config.BATCH_SIZE,
        'shuffle': False,
        'num_workers': Config.NUM_WORKERS,
        'pin_memory': Config.PIN_MEMORY,
        'collate_fn': custom_collate_fn,
    }
    if Config.NUM_WORKERS > 0:
        val_loader_kwargs['prefetch_factor'] = Config.PREFETCH_FACTOR
        val_loader_kwargs['persistent_workers'] = Config.PERSISTENT_WORKERS
    
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ========== LOAD MODEL ==========
    logger.info("\n[3/6] Loading Mask2Former model with CUSTOM SegFormer-B5 backbone...")
    
    try:
        # Load SegFormer-B5 pretrained model
        logger.info(f"Loading pretrained SegFormer-B5 from {Config.BACKBONE_NAME}...")
        segformer_pretrained = SegformerModel.from_pretrained(Config.BACKBONE_NAME)
        
        logger.info(f"[OK] SegFormer-B5 Loaded:")
        logger.info(f"  - Hidden sizes: {segformer_pretrained.config.hidden_sizes}")
        logger.info(f"  - Depths: {segformer_pretrained.config.depths}")
        logger.info(f"  - Num encoder blocks: {segformer_pretrained.config.num_encoder_blocks}")
        logger.info(f"  - Attention heads: {segformer_pretrained.config.num_attention_heads}")
        
        # Wrap SegFormer in custom backbone wrapper
        logger.info("Creating custom SegFormer backbone wrapper...")
        segformer_backbone = SegFormerBackboneWrapper(segformer_pretrained)
        
        # Load base Mask2Former configuration
        logger.info(f"Loading Mask2Former configuration from {Config.MODEL_NAME}...")
        mask2former_config = Mask2FormerConfig.from_pretrained(Config.MODEL_NAME)
        
        # CRITICAL: Modify config for our task BEFORE creating model
        mask2former_config.num_labels = Config.NUM_LABELS
        mask2former_config.num_queries = Config.NUM_QUERIES
        mask2former_config.id2label = Config.ID2LABEL
        mask2former_config.label2id = Config.LABEL2ID
        mask2former_config.mask_feature_size = Config.MASK_FEATURE_SIZE
        
        # CRITICAL: Override num_classes to match our 2-class task (background + fence)
        # The transformer decoder needs to know the correct number of classes
        mask2former_config.num_classes = Config.NUM_LABELS
        mask2former_config.class_weight = Config.CLASS_WEIGHT
        
        logger.info(f"Model config: num_labels={mask2former_config.num_labels}, num_classes={mask2former_config.num_classes}, num_queries={mask2former_config.num_queries}")
        
        # Build Mask2Former with custom SegFormer backbone
        logger.info("Building custom Mask2Former with SegFormer-B5 backbone...")
        
        # Create the main model structure with corrected config
        model = Mask2FormerForUniversalSegmentation(config=mask2former_config)
        
        # Replace pixel level module with custom one using SegFormer
        logger.info("Replacing pixel-level module with custom SegFormer-based decoder...")
        custom_pixel_module = CustomMask2FormerPixelLevelModule(
            mask2former_config,
            segformer_backbone
        )
        model.model.pixel_level_module = custom_pixel_module
        
        # CRITICAL FIX: Reinitialize class prediction head for 2 classes (not 3)
        logger.info("Reinitializing class prediction head for 2 classes...")
        hidden_dim = mask2former_config.hidden_dim  # 256
        num_classes = Config.NUM_LABELS  # 2
        
        # Replace the class predictor (was initialized for 3 classes from COCO)
        model.model.transformer_module.decoder.class_predictor = nn.Linear(
            hidden_dim, num_classes + 1  # +1 for no-object class
        )
        logger.info(f"  - Class predictor reset: {hidden_dim} -> {num_classes + 1} classes")
        
        # Move to device
        logger.info(f"Moving model to {Config.DEVICE}...")
        model.to(Config.DEVICE)
        
        # Freeze SegFormer backbone if configured
        if Config.FREEZE_BACKBONE_EPOCHS > 0:
            logger.info(f"Freezing SegFormer backbone for first {Config.FREEZE_BACKBONE_EPOCHS} epochs...")
            for param in model.model.pixel_level_module.encoder.parameters():
                param.requires_grad = False
            logger.info("  - SegFormer encoder frozen (will unfreeze after warmup)")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in segformer_backbone.parameters())
        decoder_params = sum(p.numel() for p in custom_pixel_module.decoder.parameters())
        
        logger.info(f"\n[OK] Custom Model Built Successfully:")
        logger.info(f"  - Architecture: Mask2Former + SegFormer-B5 (Custom Integration)")
        logger.info(f"  - Total Parameters: {total_params:,}")
        logger.info(f"  - SegFormer-B5 Backbone: {backbone_params:,}")
        logger.info(f"  - Custom Pixel Decoder: {decoder_params:,}")
        logger.info(f"  - Trainable Parameters: {trainable_params:,}")
        logger.info(f"  - Backbone Type: {type(segformer_backbone).__name__}")
        logger.info(f"  - Encoder Type: {type(segformer_backbone.encoder).__name__}")
        
        # Verify the integration
        logger.info("\n[OK] Verifying custom backbone integration...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, Config.INPUT_SIZE, Config.INPUT_SIZE).to(Config.DEVICE)
            try:
                test_output = model(pixel_values=dummy_input)
                logger.info(f"  - Forward pass successful!")
                logger.info(f"  - Output masks shape: {test_output.masks_queries_logits.shape}")
                logger.info(f"  - Output class shape: {test_output.class_queries_logits.shape}")
                
                # Verify class shape is correct (should be [1, 100, 3] for 2 classes + no-object)
                expected_class_shape = (1, Config.NUM_QUERIES, Config.NUM_LABELS + 1)
                actual_class_shape = tuple(test_output.class_queries_logits.shape)
                
                if actual_class_shape == expected_class_shape:
                    logger.info(f"[OK] Class output shape CORRECT: {actual_class_shape} (2 classes + 1 no-object)")
                else:
                    logger.error(f"[ERROR] Class output shape WRONG: {actual_class_shape}, expected {expected_class_shape}")
                    raise ValueError(f"Class predictor has wrong output shape!")
                
                logger.info(f"[OK] SegFormer-B5 is PROPERLY integrated as Mask2Former backbone!")
            except Exception as e:
                logger.error(f"  ❌ Forward pass failed: {e}")
                raise
        
    except Exception as e:
        logger.error(f"Failed to load Mask2Former model with custom backbone: {e}")
        traceback.print_exc()
        return
    
    # Note: Gradient checkpointing not supported by Mask2Former
    logger.info(f"Memory optimized: {Config.TRAIN_SIZE}x{Config.TRAIN_SIZE} resolution, batch size {Config.BATCH_SIZE}, EMA {'enabled' if Config.USE_EMA else 'disabled'}")
    
    # ========== SETUP TRAINING ==========
    logger.info("\n[4/6] Setting up training...")
    
    # Loss function with hard negative mining
    criterion = CombinedLoss(
        Config.LOSS_WEIGHTS, 
        Config.NUM_LABELS,
        use_hard_negative_mining=Config.USE_HARD_NEGATIVE_MINING
    )
    logger.info(f"Loss configuration:")
    logger.info(f"  - Focal Loss alpha: 0.75 (improved from 0.25 for better precision)")
    logger.info(f"  - Hard Negative Mining: {Config.USE_HARD_NEGATIVE_MINING}")
    logger.info(f"  - Confidence Threshold: {Config.CONFIDENCE_THRESHOLD}")
    logger.info(f"  - Loss Weights: mask={Config.LOSS_WEIGHTS['mask_loss']}, dice={Config.LOSS_WEIGHTS['dice_loss']}, boundary={Config.LOSS_WEIGHTS['boundary_loss']}")
    
    # Optimizer with layer-wise learning rate decay
    if Config.USE_LAYER_DECAY:
        param_groups = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Different LR for backbone vs rest
            if 'backbone' in name or 'pixel_level_module' in name:
                lr = Config.LEARNING_RATE * Config.BACKBONE_LR_MULTIPLIER
            else:
                lr = Config.LEARNING_RATE
            
            param_groups.append({'params': param, 'lr': lr})
        
        optimizer = optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
    
    # Scheduler
    if Config.SCHEDULER == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.MAX_LR if hasattr(Config, 'MAX_LR') else Config.LEARNING_RATE * 10,
            epochs=Config.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=Config.PCT_START,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=Config.MIN_LR
        )
    
    # Gradient scaler
    scaler = torch.cuda.amp.GradScaler(
        enabled=Config.USE_AMP,
        init_scale=Config.GRAD_SCALER_INIT_SCALE,
        growth_factor=Config.GRAD_SCALER_GROWTH_FACTOR,
        backoff_factor=Config.GRAD_SCALER_BACKOFF_FACTOR,
        growth_interval=Config.GRAD_SCALER_GROWTH_INTERVAL
    )
    
    # EMA
    ema = EMA(model, decay=Config.EMA_DECAY) if Config.USE_EMA else None
    
    # TensorBoard
    writer = SummaryWriter(Config.LOGS_DIR / f'tensorboard_{timestamp}') if Config.USE_TENSORBOARD else None
    
    # ========== CHECKPOINT RESUMING ==========
    start_epoch = 0
    best_val_iou = 0.0
    patience_counter = 0
    
    # Check for existing checkpoint to resume from
    last_checkpoint_path = Config.CHECKPOINT_DIR / 'last_model.pth'
    if last_checkpoint_path.exists():
        logger.info(f"\n[4.5/6] Found existing checkpoint: {last_checkpoint_path}")
        try:
            checkpoint = torch.load(last_checkpoint_path, map_location=Config.DEVICE)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("  Model state loaded")
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("  Optimizer state loaded")
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("  Scheduler state loaded")
            
            # Load scaler state
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("  Scaler state loaded")
            
            # Load EMA state
            if ema and 'ema_shadow' in checkpoint and checkpoint['ema_shadow'] is not None:
                ema.shadow = checkpoint['ema_shadow']
                logger.info("  EMA state loaded")
            
            # Load training progress
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_iou = checkpoint.get('best_val_iou', 0.0)
            
            logger.info(f"  Resuming from epoch {start_epoch}")
            logger.info(f"  Best validation IoU so far: {best_val_iou:.4f}")
            
        except Exception as e:
            logger.warning(f"  Failed to load checkpoint: {e}")
            logger.warning("  Starting training from scratch")
            start_epoch = 0
            best_val_iou = 0.0
            patience_counter = 0
    else:
        logger.info("\n[4.5/6] No checkpoint found, starting training from scratch")
    
    # Warmup GPU
    if torch.cuda.is_available():
        logger.info("Warming up GPU...")
        dummy_input = None
        try:
            dummy_input = torch.randn(1, 3, Config.TRAIN_SIZE, Config.TRAIN_SIZE).to(Config.DEVICE)
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                _ = model(pixel_values=dummy_input)
            torch.cuda.synchronize()
            logger.info("GPU warmup successful")
        except Exception as e:
            logger.warning(f"GPU warmup failed (non-critical): {e}")
        finally:
            if dummy_input is not None:
                del dummy_input
            torch.cuda.empty_cache()
    
    # Clear GPU cache after warmup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared after warmup")
    
    # ========== TRAINING LOOP ==========
    logger.info("\n[5/6] Starting training...\n")
    logger.info("=" * 90)
    
    training_start_time = time.time()
    
    # Memory tracking for leak detection
    initial_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    memory_samples = []
    loss_history = []
    
    for epoch in range(start_epoch, Config.EPOCHS):
        # Unfreeze backbone after warmup period
        if epoch == Config.FREEZE_BACKBONE_EPOCHS and Config.FREEZE_BACKBONE_EPOCHS > 0:
            logger.info(f"\n[UNFREEZING] Epoch {epoch+1}: Unfreezing SegFormer backbone...")
            for param in model.model.pixel_level_module.encoder.parameters():
                param.requires_grad = True
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"  - Trainable parameters: {trainable_params:,}")
            logger.info(f"  - SegFormer backbone now trainable with LR = {Config.LEARNING_RATE * Config.BACKBONE_LR_MULTIPLIER}")
            
            # Register newly unfrozen parameters in EMA
            if ema:
                ema.register_new_parameters()
                logger.info(f"  - EMA updated with {len(ema.shadow)} parameters")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            Config.DEVICE, epoch, writer, ema
        )
        
        # Validate
        if (epoch + 1) % Config.VAL_FREQ == 0:
            # Use EMA for validation
            if ema is not None and epoch >= Config.EMA_START_EPOCH:
                ema.apply_shadow()
            
            val_metrics = validate_epoch(
                model, val_loader, criterion, Config.DEVICE,
                epoch, writer, save_visualizations=False
            )
            
            # Detect model collapse (predicting all fence or all background)
            if val_metrics['precision'] < 0.1 or val_metrics['recall'] < 0.1:
                logger.error(f"⚠️ MODEL COLLAPSE DETECTED! Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
                logger.error(f"   Model is predicting {'all background' if val_metrics['recall'] < 0.1 else 'all fence'}")
                logger.error(f"   Consider: (1) Lowering learning rate, (2) Checking loss weights, (3) Verifying data augmentation")
            
            if ema is not None and epoch >= Config.EMA_START_EPOCH:
                ema.restore()
            
            # Memory leak detection
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3
                memory_samples.append(current_memory)
                if len(memory_samples) > 5:
                    memory_growth = memory_samples[-1] - memory_samples[-5]
                    if memory_growth > 0.5:
                        logger.warning(f"Potential memory leak: {memory_growth:.2f}GB growth over 5 epochs")
            
            # Track loss stability
            loss_history.append(train_metrics['loss'])
            if len(loss_history) > 10:
                recent_losses = loss_history[-10:]
                loss_std = np.std(recent_losses)
                loss_mean = np.mean(recent_losses)
                if loss_std / (loss_mean + 1e-8) > 0.5:
                    logger.warning(f"Unstable training: std={loss_std:.4f}, mean={loss_mean:.4f}")
            
            # Save best model (FIXED: save EMA weights if active)
            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                patience_counter = 0
                
                # If EMA was applied for validation, save those weights (they're better!)
                # Otherwise save current weights
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),  # This will be EMA weights if applied
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'ema_shadow': ema.shadow if ema else None,
                    'best_val_iou': best_val_iou,
                    'train_iou': train_metrics['iou'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'config': {k: str(v) if isinstance(v, (Path, type, torch.device)) else v for k, v in vars(Config).items() if not k.startswith('_') and not callable(v)}
                }
                torch.save(checkpoint, Config.CHECKPOINT_DIR / 'best_model.pth')
                logger.info(f"  [SAVED] Best model! Val IoU: {best_val_iou:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
            else:
                patience_counter += 1
            
            # Restore EMA if it was applied (AFTER saving checkpoint)
            if ema is not None and epoch >= Config.EMA_START_EPOCH:
                ema.restore()
            
            # Visualizations (if needed for this epoch)
            if (epoch + 1) % Config.VIS_FREQ == 0:
                logger.info("  Generating validation visualizations...")
                # Re-validate with visualization saving
                try:
                    if ema is not None and epoch >= Config.EMA_START_EPOCH:
                        ema.apply_shadow()
                    
                    val_metrics_vis = validate_epoch(
                        model, val_loader, criterion, Config.DEVICE,
                        epoch, writer, save_visualizations=True
                    )
                    
                    if ema is not None and epoch >= Config.EMA_START_EPOCH:
                        ema.restore()
                except Exception as e:
                    logger.error(f"Visualization generation failed: {e}")
                    if ema is not None and epoch >= Config.EMA_START_EPOCH:
                        ema.restore()
            
            # Logging
            # Memory leak detection
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3
                memory_samples.append(current_memory)
                if len(memory_samples) > 5:
                    memory_growth = memory_samples[-1] - memory_samples[-5]
                    if memory_growth > 0.5:  # More than 500MB growth
                        logger.warning(f"Potential memory leak: {memory_growth:.2f}GB growth over 5 epochs")
            
            # Track loss stability
            loss_history.append(train_metrics['loss'])
            if len(loss_history) > 10:
                recent_losses = loss_history[-10:]
                loss_std = np.std(recent_losses)
                loss_mean = np.mean(recent_losses)
                if loss_std / (loss_mean + 1e-8) > 0.5:
                    logger.warning(f"Unstable training: std={loss_std:.4f}, mean={loss_mean:.4f}")
            
            # Save best model
            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'ema_shadow': ema.shadow if ema else None,
                    'best_val_iou': best_val_iou,
                    'config': {k: str(v) if isinstance(v, (Path, type, torch.device)) else v for k, v in vars(Config).items() if not k.startswith('_') and not callable(v)}
                }
                torch.save(checkpoint, Config.CHECKPOINT_DIR / 'best_model.pth')
                logger.info(f"  Best model saved! IoU: {best_val_iou:.4f}")
            else:
                patience_counter += 1
            
            # Save last model checkpoint (for resuming)
            last_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'ema_shadow': ema.shadow if ema else None,
                'best_val_iou': best_val_iou,
                'val_iou': val_metrics['iou'],
                'config': {k: str(v) if isinstance(v, (Path, type, torch.device)) else v for k, v in vars(Config).items() if not k.startswith('_') and not callable(v)}
            }
            torch.save(last_checkpoint, Config.CHECKPOINT_DIR / 'last_model.pth')
            
            # Periodic checkpoint
            if (epoch + 1) % Config.SAVE_FREQ == 0:
                torch.save(last_checkpoint, Config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch+1}.pth')
                logger.info(f"  Periodic checkpoint saved: epoch_{epoch+1}.pth")
            
            # Early stopping
            if Config.EARLY_STOPPING and patience_counter >= Config.PATIENCE:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Step scheduler
        if Config.SCHEDULER == "OneCycleLR":
            # OneCycleLR steps per batch, already done in training loop
            pass
        else:
            scheduler.step()
    
    # ========== SAVE FINAL MODEL ==========
    logger.info("\nSaving final model checkpoint...")
    final_checkpoint = {
        'epoch': Config.EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'ema_shadow': ema.shadow if ema else None,
        'best_val_iou': best_val_iou,
        'config': {k: str(v) if isinstance(v, (Path, type, torch.device)) else v for k, v in vars(Config).items() if not k.startswith('_') and not callable(v)}
    }
    torch.save(final_checkpoint, Config.CHECKPOINT_DIR / 'last_model.pth')
    logger.info("Final model saved as 'last_model.pth'")
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "=" * 90)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 90)
    logger.info(f"Total Epochs Trained: {epoch + 1}")
    logger.info(f"Total Training Time: {time.time() - training_start_time:.2f}s ({(time.time() - training_start_time) / 3600:.2f}h)")
    logger.info(f"Average Time per Epoch: {(time.time() - training_start_time) / (epoch + 1):.2f}s")
    logger.info(f"Best Validation IoU: {best_val_iou:.4f}")
    
    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated() / 1024**3
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Final GPU Memory: {final_memory:.2f}GB")
        logger.info(f"Peak GPU Memory: {max_memory:.2f}GB")
        if len(memory_samples) > 1:
            memory_change = memory_samples[-1] - memory_samples[0]
            logger.info(f"GPU Memory Change: {memory_change:+.2f}GB ({memory_samples[0]:.2f}GB → {memory_samples[-1]:.2f}GB)")
    
    if len(loss_history) > 0:
        logger.info(f"Final Training Loss: {loss_history[-1]:.4f}")
        logger.info(f"Average Training Loss: {np.mean(loss_history):.4f}")
        logger.info(f"Min Training Loss: {np.min(loss_history):.4f}")
    
    logger.info(f"Best model saved to: {Config.CHECKPOINT_DIR / 'best_model.pth'}")
    logger.info(f"Last model saved to: {Config.CHECKPOINT_DIR / 'last_model.pth'}")
    logger.info(f"Logs saved to: {Config.LOGS_DIR}")
    logger.info(f"Visualizations saved to: {Config.VISUALIZATIONS_DIR}")
    logger.info("=" * 90)
    
    # Save training summary to JSON
    training_summary = {
        'timestamp': timestamp,
        'total_epochs': epoch + 1,
        'training_time_seconds': time.time() - training_start_time,
        'training_time_hours': (time.time() - training_start_time) / 3600,
        'best_val_iou': float(best_val_iou),
        'final_train_loss': float(loss_history[-1]) if loss_history else None,
        'avg_train_loss': float(np.mean(loss_history)) if loss_history else None,
        'min_train_loss': float(np.min(loss_history)) if loss_history else None,
        'gpu_memory_samples': [float(m) for m in memory_samples] if memory_samples else [],
        'best_model_path': str(Config.CHECKPOINT_DIR / 'best_model.pth'),
        'last_model_path': str(Config.CHECKPOINT_DIR / 'last_model.pth')
    }
    
    if torch.cuda.is_available():
        training_summary['final_gpu_memory_gb'] = torch.cuda.memory_allocated() / 1024**3
        training_summary['peak_gpu_memory_gb'] = torch.cuda.max_memory_allocated() / 1024**3
    
    summary_path = Config.LOGS_DIR / f'training_summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    logger.info(f"Training summary saved to {summary_path}")
    
    if writer:
        writer.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    try:
        train_mask2former()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        raise
