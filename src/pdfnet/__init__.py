"""
PDFNet: Patch-Depth Fusion for Dichotomous Image Segmentation

This package implements PDFNet, a novel approach for high-precision dichotomous
image segmentation that combines patch fine-grained strategies with depth
integrity-prior from pseudo-depth maps.
"""

__version__ = "0.1.0"
__author__ = "Xianjie Liu, Keren Fu, Qijun Zhao"


# Training-related imports
from .models.PDFNet import build_model
from .dataloaders.dis_dataset import build_dataset, DISDataset

# Isolated inference imports (no training dependencies)
from .model_loader import (
    load_pdfnet_model,
    load_moge_depth_model,
    get_model_info,
)
from .inference import (
    # Preprocessing
    preprocess_image,
    preprocess_depth,
    postprocess_prediction,
    # Inference
    run_inference,
    run_inference_with_tta,
    run_batch_inference,
    process_directory,
    # Utilities
    generate_depth,
    create_placeholder_depth,
    save_prediction,
    # Class-based interface
    PDFNetInference,
)

__all__ = [
    # Types module
    # Training
    "build_model",
    "build_dataset",
    "DISDataset",
    # Isolated Inference - Model Loading
    "load_pdfnet_model",
    "load_moge_depth_model",
    "get_model_info",
    # Isolated Inference - Running
    "run_inference",
    "run_batch_inference",
    "run_inference_with_tta",
    "process_directory",
    "generate_depth",
    "create_placeholder_depth",
    "save_prediction",
    # Isolated Inference - Preprocessing
    "preprocess_image",
    "preprocess_depth",
    "postprocess_prediction",
    # Class-based Interface
    "PDFNetInference",
]
