"""
Isolated model loading functionality for PDFNet.

This module provides clean, isolated functions for loading PDFNet models
without dependencies on training infrastructure.
"""

from pathlib import Path
import warnings
import logging
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_pdfnet_model(
    checkpoint_path: Path | str,
    model_name: Literal["PDFNet_swinB", "PDFNet_swinL", "PDFNet_swinT"] = "PDFNet_swinB",
    device: torch.device | str = "auto",
    strict: bool = False
) -> nn.Module:
    """
    Load a PDFNet model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        model_name: Model architecture variant
        device: Device to load model on ("cuda", "cpu", or "auto")
        strict: Whether to strictly enforce state dict loading

    Returns:
        Loaded PDFNet model in eval mode

    Example:
        >>> model = load_pdfnet_model("checkpoints/PDFNet_Best.pth")
        >>> model = load_pdfnet_model("model.pth", device="cuda")
    """
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Build model architecture
    model = _build_model_architecture(model_name, device)

    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    if not strict and (missing_keys or unexpected_keys):
        if missing_keys:
            warnings.warn(f"Missing keys in checkpoint: {len(missing_keys)} keys")
        if unexpected_keys:
            warnings.warn(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded on {device}")
    return model


def _build_model_architecture(model_name: Literal["PDFNet_swinB", "PDFNet_swinL", "PDFNet_swinT"], device: torch.device) -> nn.Module:
    """
    Build PDFNet model architecture without checkpoint.

    Args:
        model_name: Model architecture variant
        device: Target device

    Returns:
        Uninitialized model architecture
    """
    from .models.PDFNet import PDFNet_process, PDF_decoder, PDF_depth_decoder
    from .models.swin_transformer import SwinB, SwinT

    # Model configuration mapping
    MODEL_CONFIGS = {
        "PDFNet_swinB": {
            "encoder": SwinB,
            "channels": (128, 256, 512, 1024)
        },
        "PDFNet_swinT": {
            "encoder": SwinT,
            "channels": (96, 192, 384, 768)
        },
        "PDFNet_swinL": {
            "encoder": SwinB,  # Use SwinB encoder for L variant
            "channels": (192, 384, 768, 1536)
        }
    }

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]

    # Create minimal args object
    class ModelArgs:
        def __init__(self):
            self.model = model_name
            self.back_bone = model_name
            self.device = device
            self.emb = 128
            self.drop_path = 0.1
            self.DEBUG = False
            # Unpack channels
            (self.back_bone_channels_stage1,
             self.back_bone_channels_stage2,
             self.back_bone_channels_stage3,
             self.back_bone_channels_stage4) = config["channels"]

    args = ModelArgs()

    # Build encoder and full model
    encoder = config["encoder"](args=args, in_chans=3, pretrained=False)

    model = PDFNet_process(
        encoder=encoder,
        decoder=PDF_decoder(args=args),
        depth_decoder=PDF_depth_decoder(args=args),
        device=device,
        args=args
    )

    return model


def load_moge_depth_model(
    checkpoint_path: Path | str | None = None,
    device: torch.device | str = "auto"
) -> nn.Module | None:
    """
    Load MoGe depth estimation model.

    Args:
        checkpoint_path: Path to MoGe checkpoint (optional)
        device: Device to load model on

    Returns:
        Loaded MoGe model or None if unavailable

    Example:
        >>> moge = load_moge_depth_model("checkpoints/moge/model.pt")
    """
    try:
        from moge.model.v2 import MoGeModel

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Load model
        if checkpoint_path is None:
            checkpoint_path = Path("checkpoints/moge/moge-2-vitl-normal/model.pt")

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            warnings.warn(f"MoGe checkpoint not found at {ckpt_path}")
            return None

        logger.info(f"Loading MoGe model from: {ckpt_path}")
        model = MoGeModel.from_pretrained(str(ckpt_path))
        model = model.to(device)
        model.eval()

        logger.info(f"MoGe model loaded on {device}")
        return model

    except ImportError:
        warnings.warn("MoGe not installed. Depth estimation unavailable.")
        return None
    except Exception as e:
        warnings.warn(f"Failed to load MoGe model: {e}")
        return None


def get_model_info(model: nn.Module) -> dict[str, int | float]:
    """
    Get information about a loaded model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model statistics

    Example:
        >>> model = load_pdfnet_model("model.pth")
        >>> info = get_model_info(model)
        >>> print(f"Parameters: {info['total_params']:,}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get model size in MB
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "size_mb": size_mb
    }