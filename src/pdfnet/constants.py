"""
Constants and default values for PDFNet.

This module centralizes all constants used throughout the codebase.
"""

# Model constants
MODEL_NAMES = ['PDFNet_swinB', 'PDFNet_swinL', 'PDFNet_swinT']
DEFAULT_MODEL = 'PDFNet_swinB'
DEFAULT_INPUT_SIZE = 1024

# Data normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training defaults
DEFAULT_BATCH_SIZE = 1
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_MOMENTUM = 0.9
DEFAULT_NUM_WORKERS = 8

# Optimizer settings
OPTIMIZER_CHOICES = ['sgd', 'adam', 'adamw']
DEFAULT_OPTIMIZER = 'adamw'
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8

# Scheduler settings
SCHEDULER_CHOICES = ['cosine', 'step', 'plateau', 'linear']
DEFAULT_SCHEDULER = 'cosine'
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_MIN_LR = 1e-5
DEFAULT_WARMUP_LR = 1e-6

# Augmentation defaults
DEFAULT_COLOR_JITTER = 0.4
DEFAULT_MIXUP_ALPHA = 0.8
DEFAULT_CUTMIX_ALPHA = 1.0
DEFAULT_MIXUP_PROB = 1.0
DEFAULT_MIXUP_SWITCH_PROB = 0.5

# Dataset information
DATASET_INFO = {
    'DIS': {
        'train': 'DIS-TR',
        'val': 'DIS-VD',
        'test': ['DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4'],
        'num_classes': 1,
    },
    'HRSOD': {
        'train': 'HRSOD-TR',
        'val': 'HRSOD-VD',
        'test': ['HRSOD-TE'],
        'num_classes': 1,
    },
    'UHRSD': {
        'train': 'UHRSD-TR',
        'val': 'UHRSD-VD',
        'test': ['UHRSD-TE'],
        'num_classes': 1,
    },
}

# File extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
MASK_EXTENSIONS = ('.png', '.bmp')

# Checkpoint patterns
CHECKPOINT_PATTERNS = {
    'best_f1': '*best_f1*.pth',
    'best_mae': '*best_mae*.pth',
    'last': 'LAST.pth',
}

# Metrics
METRICS = ['F1', 'MAE', 'Smeasure', 'Emeasure', 'wFmeasure']
DEFAULT_EVAL_METRIC = 'F1'

# Loss weights
LOSS_WEIGHTS = {
    'bce': 1.0,
    'iou': 0.5,
    'ssim': 0.5,
    'integrity': 0.3,
}

# Device settings
DEFAULT_DEVICE = 'cuda'
CUDA_VISIBLE_DEVICES = '0'

# Paths
DEFAULT_DATA_PATH = 'DATA/DIS-DATA'
DEFAULT_CHECKPOINT_DIR = 'checkpoints'
DEFAULT_OUTPUT_DIR = 'runs'
DEFAULT_LOG_DIR = 'logs'
DEFAULT_RESULT_DIR = 'results'
DEFAULT_CACHE_DIR = 'cache'

# MoGe settings
MOGE_MODEL_PATH = 'checkpoints/moge/moge-2-vitl-normal/model.pt'
MOGE_INPUT_SIZE = 518

# TTA settings
TTA_SCALES = [0.75, 1.0, 1.25]
TTA_FLIPS = ['horizontal']

# Logging
LOG_FREQUENCY = 10
TENSORBOARD_LOG_FREQ = 100
CHECKPOINT_SAVE_FREQ = 1
KEEP_CHECKPOINTS = 3

# Evaluation
EVAL_BATCH_SIZE = 1
EVAL_NUM_WORKERS = 8
EVAL_N_JOBS = 12  # For parallel metric computation

# Random seeds
DEFAULT_SEED = 0

# Gradient clipping
DEFAULT_CLIP_GRAD = None  # Set to float value to enable

# Model specific settings
SWIN_PRETRAINED_PATH = 'checkpoints/swin_base_patch4_window12_384_22k.pth'
SWIN_WINDOW_SIZE = 12
SWIN_PATCH_SIZE = 4
SWIN_EMBED_DIM = 128
SWIN_DEPTHS = [2, 2, 18, 2]
SWIN_NUM_HEADS = [4, 8, 16, 32]

# Training flags
USE_MIXED_PRECISION = True
USE_GRADIENT_ACCUMULATION = False
GRADIENT_ACCUMULATION_STEPS = 1

# Validation
VALIDATE_EVERY_N_EPOCHS = 1
EARLY_STOPPING_PATIENCE = 10

# Debug settings
DEBUG_MODE = False
VERBOSE = True

# Memory optimization
EMPTY_CACHE_FREQ = 10  # Empty CUDA cache every N batches
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Data loading
DROP_LAST = False
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False

# Image processing
INTERPOLATION_MODE = 'bilinear'
RESIZE_MODE = 'bilinear'

# Loss computation
USE_SIGMOID = True
REDUCTION = 'mean'  # 'mean' or 'sum'

# Multi-scale settings
MULTI_SCALE_TRAINING = False
MULTI_SCALE_SIZES = [512, 768, 1024, 1280]

# Test-time settings
TEST_FLIP = False
TEST_MULTI_SCALE = False

# Distributed training
DISTRIBUTED = False
WORLD_SIZE = 1
LOCAL_RANK = -1
DIST_BACKEND = 'nccl'
DIST_URL = 'env://'