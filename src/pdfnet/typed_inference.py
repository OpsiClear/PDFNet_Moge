"""
Type-safe inference module for PDFNet using Python 3.12 type hints.

This module provides fully typed inference functionality for PDFNet,
including single image, batch processing, and test-time augmentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias
from pathlib import Path
from collections.abc import Sequence, Iterator
import warnings

import torch
import torch.nn as nn
import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

if TYPE_CHECKING:
    from .config_types import PDFNetConfig

# Type aliases for clarity
ImageArray: TypeAlias = npt.NDArray[np.uint8]
FloatArray: TypeAlias = npt.NDArray[np.float32 | np.float64]
Tensor: TypeAlias = torch.Tensor
PathLike: TypeAlias = Path | str


class PDFNetInference:
    """Type-safe inference engine for PDFNet."""

    model: nn.Module
    device: torch.device
    config: PDFNetConfig
    moge_model: nn.Module | None
    input_size: int

    def __init__(
        self,
        config: PDFNetConfig | None = None,
        checkpoint_path: PathLike | None = None
    ) -> None:
        """
        Initialize inference engine with type-safe configuration.

        Args:
            config: PDFNet configuration object
            checkpoint_path: Override path to model checkpoint
        """
        # Import here to avoid circular imports
        from .config_types import PDFNetConfig

        self.config = config or PDFNetConfig()
        self.device = self._setup_device()
        self.input_size = self.config.data.input_size
        self.model = self._load_model(checkpoint_path)
        self.moge_model = self._load_moge() if self.config.inference.use_moge else None

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device_str = self.config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_str)

    def _load_model(self, checkpoint_path: PathLike | None = None) -> nn.Module:
        """
        Load PDFNet model with checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Loaded model in eval mode
        """
        from .models.PDFNet import build_model
        from .args import get_args_parser
        import argparse

        # Build model
        parser = argparse.ArgumentParser(parents=[get_args_parser()])
        args = parser.parse_args(args=[])
        args.model = self.config.model.name

        model, _ = build_model(args)

        # Load checkpoint
        ckpt_path = Path(checkpoint_path) if checkpoint_path else self.config.inference.checkpoint_path
        if ckpt_path and ckpt_path.exists():
            print(f"Loading checkpoint: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        else:
            warnings.warn(f"No checkpoint found at {ckpt_path}")

        return model.to(self.device).eval()

    def _load_moge(self) -> nn.Module | None:
        """Load MoGe depth estimation model."""
        try:
            from moge.model import MoGeModel

            moge_path = self.config.inference.moge_model
            if moge_path and moge_path.exists():
                print("Loading MoGe model for depth estimation...")
                model = MoGeModel.from_pretrained(str(moge_path))
                return model.to(self.device).eval()
            else:
                warnings.warn(f"MoGe model not found at {moge_path}")
                return None

        except ImportError:
            warnings.warn("MoGe not available, using placeholder depth")
            return None

    @torch.no_grad()
    def generate_depth(self, image: ImageArray) -> ImageArray:
        """
        Generate depth map for an image.

        Args:
            image: Input image as numpy array (H, W, 3)

        Returns:
            Depth map as numpy array (H, W) with values in [0, 255]
        """
        if self.moge_model is not None:
            return self._generate_moge_depth(image)
        else:
            # Return placeholder depth
            h, w = image.shape[:2]
            return np.full((h, w), 128, dtype=np.uint8)

    def _generate_moge_depth(self, image: ImageArray) -> ImageArray:
        """Generate depth using MoGe model."""
        h, w = image.shape[:2]
        input_size = self.config.inference.moge_input_size

        # Prepare image
        img_resized = cv2.resize(image, (input_size, input_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Run inference
        output = self.moge_model.infer(img_tensor)
        depth = output["depth"]

        # Handle different depth tensor formats
        match depth.dim():
            case 4:
                depth = depth.squeeze(0).squeeze(0)
            case 3:
                depth = depth.squeeze(0)
            case _:
                pass

        # Resize to original size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze()

        # Convert and normalize
        depth_np = depth.cpu().numpy()
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=1.0, neginf=0.0)

        # Normalize to 0-255 range
        depth_min, depth_max = depth_np.min(), depth_np.max()
        if depth_min == depth_max:
            return np.full_like(depth_np, 128, dtype=np.uint8)

        depth_normalized = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
        return np.clip(depth_normalized, 0, 255).astype(np.uint8)

    def _prepare_inputs(
        self,
        image: ImageArray,
        depth: ImageArray
    ) -> tuple[Tensor, Tensor]:
        """
        Prepare image and depth tensors for model input.

        Args:
            image: RGB image array
            depth: Depth map array

        Returns:
            Tuple of (image_tensor, depth_tensor) ready for inference
        """
        # Resize inputs
        img_resized = cv2.resize(image, (self.input_size, self.input_size))
        depth_resized = cv2.resize(depth, (self.input_size, self.input_size))

        # Convert to tensors
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255

        # Apply normalization
        mean = torch.tensor(self.config.data.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.config.data.normalize_std).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0).unsqueeze(0).float() / 255
        depth_tensor = depth_tensor.to(self.device)

        return img_tensor, depth_tensor

    @torch.no_grad()
    def predict_single(
        self,
        image: ImageArray | PathLike,
        depth: ImageArray | None = None,
        return_depth: bool = False
    ) -> FloatArray | tuple[FloatArray, ImageArray]:
        """
        Predict segmentation for a single image.

        Args:
            image: Input image array or path
            depth: Optional pre-computed depth map
            return_depth: Whether to return the depth map

        Returns:
            Segmentation mask as float array [0, 1]
            If return_depth is True, returns (mask, depth)
        """
        # Load image if path provided
        if isinstance(image, (Path, str)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Could not load image: {image}")
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # Generate depth if not provided
        if depth is None:
            depth = self.generate_depth(image)

        # Prepare inputs
        img_tensor, depth_tensor = self._prepare_inputs(image, depth)

        # Run inference
        outputs = self.model.inference(img_tensor, depth_tensor)

        # Extract prediction
        if isinstance(outputs, (list, tuple)):
            pred = outputs[0]
        else:
            pred = outputs

        while isinstance(pred, (list, tuple)) and len(pred) > 0:
            pred = pred[0]

        # Convert to numpy and resize
        pred_np = pred.squeeze().cpu().numpy()
        pred_resized = cv2.resize(pred_np, (w, h))

        if return_depth:
            return pred_resized, depth
        return pred_resized

    @torch.no_grad()
    def predict_with_tta(
        self,
        image: ImageArray | PathLike,
        depth: ImageArray | None = None,
        scales: Sequence[float] | None = None,
        flips: Sequence[Literal["horizontal", "vertical", "both"]] | None = None
    ) -> FloatArray:
        """
        Predict with test-time augmentation.

        Args:
            image: Input image array or path
            depth: Optional pre-computed depth map
            scales: Scaling factors for TTA
            flips: Flip types for TTA

        Returns:
            Averaged segmentation mask
        """
        import ttach as tta

        # Use default TTA settings if not provided
        if scales is None:
            scales = self.config.inference.tta_scales
        if flips is None:
            flips = ["horizontal"]

        # Load image
        if isinstance(image, (Path, str)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Could not load image: {image}")
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # Generate depth
        if depth is None:
            depth = self.generate_depth(image)

        # Prepare base inputs
        img_tensor, depth_tensor = self._prepare_inputs(image, depth)

        # Create TTA transforms
        transforms = []
        if "horizontal" in flips or "both" in flips:
            transforms.append(tta.HorizontalFlip())
        if "vertical" in flips or "both" in flips:
            transforms.append(tta.VerticalFlip())
        if scales and len(scales) > 1:
            transforms.append(tta.Scale(scales=list(scales), interpolation="bilinear"))

        tta_transforms = tta.Compose(transforms) if transforms else tta.Compose([tta.HorizontalFlip()])

        # Collect predictions
        predictions: list[Tensor] = []

        for transformer in tta_transforms:
            # Apply augmentation
            aug_img = transformer.augment_image(img_tensor)
            aug_depth = transformer.augment_image(depth_tensor)

            # Get prediction
            outputs = self.model.inference(aug_img, aug_depth)
            if isinstance(outputs, (list, tuple)):
                pred = outputs[0]
            else:
                pred = outputs

            while isinstance(pred, (list, tuple)) and len(pred) > 0:
                pred = pred[0]

            # Reverse augmentation
            pred_deaug = transformer.deaugment_mask(pred)
            predictions.append(pred_deaug)

        # Average predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        pred_np = final_pred.squeeze().cpu().numpy()

        return cv2.resize(pred_np, (w, h))

    def predict_batch(
        self,
        images: Sequence[ImageArray | PathLike],
        batch_size: int = 1,
        use_tta: bool = False,
        progress: bool = True
    ) -> list[FloatArray]:
        """
        Predict segmentation for multiple images.

        Args:
            images: Sequence of images or paths
            batch_size: Batch size for processing
            use_tta: Use test-time augmentation
            progress: Show progress bar

        Returns:
            List of segmentation masks
        """
        results: list[FloatArray] = []

        # Create iterator with optional progress bar
        iterator: Iterator = tqdm(images, desc="Processing") if progress else images

        for img in iterator:
            if use_tta:
                pred = self.predict_with_tta(img)
            else:
                pred = self.predict_single(img)
            results.append(pred)

        return results

    def predict_directory(
        self,
        input_dir: PathLike,
        output_dir: PathLike,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
        use_tta: bool = False,
        save_depth: bool = False
    ) -> None:
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            extensions: Image file extensions to process
            use_tta: Use test-time augmentation
            save_depth: Save depth maps alongside predictions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files: list[Path] = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        if not image_files:
            warnings.warn(f"No images found in {input_dir}")
            return

        print(f"Found {len(image_files)} images to process")

        # Process images
        for img_path in tqdm(image_files, desc="Processing"):
            # Predict
            if use_tta:
                pred = self.predict_with_tta(img_path)
                depth = None
            else:
                result = self.predict_single(img_path, return_depth=save_depth)
                if save_depth:
                    pred, depth = result
                else:
                    pred = result
                    depth = None

            # Save prediction
            output_file = output_path / f"{img_path.stem}.png"
            pred_img = (pred * 255).astype(np.uint8)
            cv2.imwrite(str(output_file), pred_img)

            # Save depth if requested
            if save_depth and depth is not None:
                depth_file = output_path / f"{img_path.stem}_depth.png"
                cv2.imwrite(str(depth_file), depth)

        print(f"Results saved to {output_dir}")