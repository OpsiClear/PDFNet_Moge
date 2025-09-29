"""
PDFNet: Patch-Depth Fusion for Dichotomous Image Segmentation

This package implements PDFNet, a novel approach for high-precision dichotomous 
image segmentation that combines patch fine-grained strategies with depth 
integrity-prior from pseudo-depth maps.
"""

__version__ = "0.1.0"
__author__ = "Xianjie Liu, Keren Fu, Qijun Zhao"

from .models.PDFNet import build_model
from .dataloaders.Mydataset import build_dataset

__all__ = [
    "build_model",
    "build_dataset",
]
