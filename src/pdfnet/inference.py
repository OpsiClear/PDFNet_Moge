"""
Complete inference module for PDFNet.

This module provides both functional and class-based APIs for inference.
All inference code is isolated from training dependencies.
"""

from pathlib import Path
from collections.abc import Sequence
import warnings
import logging

import torch
import torch.nn as nn
import cv2
import numpy as np
import numpy.typing as npt

from .config import PDFNetConfig
from .model_loader import load_pdfnet_model, load_moge_depth_model

logger = logging.getLogger(__name__)


# ==============================================================================
# FUNCTIONAL API - Pure inference functions
# ==============================================================================


def preprocess_image(
    image: npt.NDArray[np.uint8], input_size: int = 1024, normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess image for PDFNet inference.

    Args:
        image: Input RGB image (H, W, 3) with values 0-255
        input_size: Target size for model input
        normalize: Whether to apply ImageNet normalization

    Returns:
        Preprocessed image tensor (1, 3, H, W)

    Example:
        >>> img = cv2.imread("image.jpg")
        >>> img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        >>> tensor = preprocess_image(img_rgb)
    """
    # Resize
    img_resized = cv2.resize(image, (input_size, input_size))

    # Convert to tensor and normalize to [0, 1]
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

    # Apply ImageNet normalization if requested
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

    # Add batch dimension
    return img_tensor.unsqueeze(0)


def preprocess_depth(
    depth: npt.NDArray[np.uint8], input_size: int = 1024
) -> torch.Tensor:
    """
    Preprocess depth map for PDFNet inference.

    Args:
        depth: Input depth map (H, W) with values 0-255
        input_size: Target size for model input

    Returns:
        Preprocessed depth tensor (1, 1, H, W)

    Example:
        >>> depth = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE)
        >>> depth_tensor = preprocess_depth(depth)
    """
    # Resize
    depth_resized = cv2.resize(depth, (input_size, input_size))

    # Convert to tensor and normalize to [0, 1]
    depth_tensor = (
        torch.from_numpy(depth_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    )

    return depth_tensor


def postprocess_prediction(
    prediction: torch.Tensor, original_size: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """
    Postprocess model prediction to original image size.

    Args:
        prediction: Model output tensor (1, 1, H, W) or (1, H, W)
        original_size: Target size (width, height)

    Returns:
        Prediction mask as numpy array (H, W) with values [0, 1]

    Example:
        >>> pred = postprocess_prediction(output, (1920, 1080))
    """
    # Remove batch and channel dimensions
    pred_np = prediction.squeeze().cpu().numpy()

    # Resize to original size (width, height)
    pred_resized = cv2.resize(pred_np, original_size)

    return pred_resized.astype(np.float32)


@torch.no_grad()
def generate_depth(
    depth_model: nn.Module,
    image: npt.NDArray[np.uint8],
    device: torch.device | str = "cuda",
    depth_input_size: int = 518,
) -> npt.NDArray[np.uint8]:
    """
    Generate depth map using MoGe model.

    Args:
        depth_model: Loaded MoGe model
        image: Input RGB image (H, W, 3)
        device: Device to run on
        depth_input_size: Input size for depth model

    Returns:
        Depth map as uint8 array (H, W) with values 0-255

    Example:
        >>> moge = load_moge_depth_model()
        >>> depth = generate_depth(moge, image)
    """
    h, w = image.shape[:2]

    # Resize for depth model
    img_resized = cv2.resize(image, (depth_input_size, depth_input_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Run depth estimation
    depth_model.eval()
    output = depth_model.infer(img_tensor)
    depth = output["depth"]

    # Handle different tensor shapes
    match depth.dim():
        case 4:
            depth = depth.squeeze(0).squeeze(0)
        case 3:
            depth = depth.squeeze(0)
        case 2:
            pass
        case _:
            depth = depth.view(depth_input_size, depth_input_size)

    # Resize to original size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Convert to numpy and normalize
    depth_np = depth.cpu().numpy()
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=1.0, neginf=0.0)

    # Normalize to 0-255
    depth_min, depth_max = depth_np.min(), depth_np.max()
    if depth_min == depth_max:
        return np.full((h, w), 128, dtype=np.uint8)

    depth_normalized = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
    return np.clip(depth_normalized, 0, 255).astype(np.uint8)


def create_placeholder_depth(height: int, width: int) -> npt.NDArray[np.uint8]:
    """
    Create placeholder depth map with radial gradient.

    Args:
        height: Image height
        width: Image width

    Returns:
        Placeholder depth map (H, W) as uint8

    Example:
        >>> depth = create_placeholder_depth(1024, 1024)
    """
    center_y, center_x = height // 2, width // 2
    y_coords, x_coords = np.ogrid[:height, :width]
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    # Normalize to 0-255
    max_distance = np.sqrt(center_x**2 + center_y**2)
    depth = (distances / max_distance * 255).astype(np.uint8)

    return depth


def save_prediction(
    prediction: npt.NDArray[np.float32],
    output_path: Path | str,
    as_binary: bool = False,
    threshold: float = 0.5,
) -> None:
    """
    Save prediction mask to file.

    Args:
        prediction: Prediction mask (H, W) with values [0, 1]
        output_path: Path to save output
        as_binary: Whether to save as binary mask
        threshold: Threshold for binary conversion

    Example:
        >>> save_prediction(mask, "output.png")
        >>> save_prediction(mask, "binary.png", as_binary=True)
    """
    if as_binary:
        pred_img = (prediction > threshold).astype(np.uint8) * 255
    else:
        pred_img = (prediction * 255).astype(np.uint8)

    cv2.imwrite(str(output_path), pred_img)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    image: npt.NDArray[np.uint8] | Path | str,
    depth: npt.NDArray[np.uint8] | None = None,
    depth_model: nn.Module | None = None,
    input_size: int = 1024,
    device: torch.device | str = "cuda",
) -> npt.NDArray[np.float32]:
    """
    Run inference on a single image.

    Args:
        model: Loaded PDFNet model
        image: Input image as numpy array or path
        depth: Optional depth map (will be generated if None and depth_model provided)
        depth_model: Optional depth estimation model (MoGe)
        input_size: Model input size
        device: Device to run inference on

    Returns:
        Segmentation mask as numpy array (H, W) with values [0, 1]

    Example:
        >>> from pdfnet import load_pdfnet_model, run_inference
        >>> model = load_pdfnet_model("checkpoints/PDFNet_Best.pth")
        >>> mask = run_inference(model, "image.jpg")
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image}")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]

    # Generate depth if not provided
    if depth is None:
        if depth_model is not None:
            depth = generate_depth(depth_model, image, device)
        else:
            # Create placeholder depth
            depth = create_placeholder_depth(h, w)

    # Preprocess inputs
    img_tensor = preprocess_image(image, input_size, normalize=True)
    depth_tensor = preprocess_depth(depth, input_size)

    # Move to device
    img_tensor = img_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    # Run inference
    model.eval()
    outputs = model.inference(img_tensor, depth_tensor)

    # Extract prediction
    if isinstance(outputs, (list, tuple)):
        pred = outputs[0]
    else:
        pred = outputs

    # Handle nested outputs
    while isinstance(pred, (list, tuple)) and len(pred) > 0:
        pred = pred[0]

    # Postprocess
    result = postprocess_prediction(pred, (w, h))

    return result


@torch.no_grad()
def run_inference_with_tta(
    model: nn.Module,
    image: npt.NDArray[np.uint8] | Path | str,
    depth_model: nn.Module | None = None,
    input_size: int = 1024,
    device: torch.device | str = "cuda",
    flips: list[str] = ["horizontal", "vertical"],
) -> npt.NDArray[np.float32]:
    """
    Run inference with test-time augmentation (TTA).

    Args:
        model: Loaded PDFNet model
        image: Input image or path
        depth_model: Optional depth model
        input_size: Model input size
        device: Device to run on
        flips: List of flip types ["horizontal", "vertical"]

    Returns:
        Averaged segmentation mask

    Example:
        >>> mask = run_inference_with_tta(model, "image.jpg")
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image}")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]

    # Generate depth once
    if depth_model is not None:
        depth = generate_depth(depth_model, image, device)
    else:
        depth = create_placeholder_depth(h, w)

    # Prepare base inputs
    img_tensor = preprocess_image(image, input_size, normalize=True)
    depth_tensor = preprocess_depth(depth, input_size)

    # Move to device
    img_tensor = img_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    predictions = []

    # Original
    model.eval()
    outputs = model.inference(img_tensor, depth_tensor)
    pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    while isinstance(pred, (list, tuple)) and len(pred) > 0:
        pred = pred[0]
    predictions.append(pred)

    # Horizontal flip
    if "horizontal" in flips:
        img_h = torch.flip(img_tensor, dims=[3])
        depth_h = torch.flip(depth_tensor, dims=[3])
        outputs = model.inference(img_h, depth_h)
        pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        while isinstance(pred, (list, tuple)) and len(pred) > 0:
            pred = pred[0]
        pred = torch.flip(pred, dims=[3])
        predictions.append(pred)

    # Vertical flip
    if "vertical" in flips:
        img_v = torch.flip(img_tensor, dims=[2])
        depth_v = torch.flip(depth_tensor, dims=[2])
        outputs = model.inference(img_v, depth_v)
        pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        while isinstance(pred, (list, tuple)) and len(pred) > 0:
            pred = pred[0]
        pred = torch.flip(pred, dims=[2])
        predictions.append(pred)

    # Average all predictions
    final_pred = torch.stack(predictions).mean(dim=0)

    # Postprocess
    result = postprocess_prediction(final_pred, (w, h))

    return result


def _process_image_batch(
    model: nn.Module,
    image_paths: list[Path | str],
    depth_model: nn.Module | None,
    input_size: int,
    device: torch.device | str,
) -> list[tuple[npt.NDArray[np.float32], tuple[int, int]]]:
    """
    Process a batch of images together (true batching).

    :param model: PDFNet model
    :param image_paths: List of image paths to process
    :param depth_model: Optional depth model
    :param input_size: Input size for model
    :param device: Device to use
    :returns: List of (mask, original_size) tuples
    """
    batch_images = []
    batch_depths = []
    original_sizes = []

    # Load and preprocess all images in batch
    for img_path in image_paths:
        # Load image
        if isinstance(img_path, (str, Path)):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise ValueError(f"Could not load image: {img_path}")
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            image = img_path

        h, w = image.shape[:2]
        original_sizes.append((w, h))

        # Generate depth
        if depth_model is not None:
            depth = generate_depth(depth_model, image, device)
        else:
            depth = create_placeholder_depth(h, w)

        # Preprocess
        img_tensor = preprocess_image(image, input_size, normalize=True)
        depth_tensor = preprocess_depth(depth, input_size)

        batch_images.append(img_tensor)
        batch_depths.append(depth_tensor)

    # Stack into batches
    image_batch = torch.cat(batch_images, dim=0).to(device)
    depth_batch = torch.cat(batch_depths, dim=0).to(device)

    # Run inference on entire batch
    model.eval()
    outputs = model.inference(image_batch, depth_batch)

    # Extract predictions
    if isinstance(outputs, (list, tuple)):
        pred = outputs[0]
    else:
        pred = outputs

    # Handle nested outputs
    while isinstance(pred, (list, tuple)) and len(pred) > 0:
        pred = pred[0]

    # Postprocess each result
    results = []
    for i, (w, h) in enumerate(original_sizes):
        result = postprocess_prediction(pred[i : i + 1], (w, h))
        results.append((result, (w, h)))

    return results


@torch.no_grad()
def run_batch_inference(
    model: nn.Module,
    images: Sequence[npt.NDArray[np.uint8] | Path | str],
    depth_model: nn.Module | None = None,
    input_size: int = 1024,
    device: torch.device | str = "cuda",
    batch_size: int = 4,
    show_progress: bool = True,
) -> list[npt.NDArray[np.float32]]:
    """
    Run inference on multiple images using true batching.

    :param model: Loaded PDFNet model
    :param images: List of images or paths
    :param depth_model: Optional depth estimation model
    :param input_size: Model input size
    :param device: Device to run inference on
    :param batch_size: Number of images to process in each batch (default: 4)
    :param show_progress: Whether to show progress bar
    :returns: List of segmentation masks

    Example:
        >>> model = load_pdfnet_model("model.pth")
        >>> masks = run_batch_inference(model, ["img1.jpg", "img2.jpg"], batch_size=4)
    """
    results: list[npt.NDArray[np.float32]] = []
    total_images = len(images)

    # Process in batches
    iterator = range(0, total_images, batch_size)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(
            iterator,
            desc="Processing batches",
            total=(total_images + batch_size - 1) // batch_size,
        )

    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, total_images)
        batch_paths = list(images[batch_start:batch_end])

        # Process this batch
        batch_results = _process_image_batch(
            model=model,
            image_paths=batch_paths,
            depth_model=depth_model,
            input_size=input_size,
            device=device,
        )

        # Extract just the masks
        for mask, _ in batch_results:
            results.append(mask)

        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def process_directory(
    model: nn.Module,
    input_dir: Path | str,
    output_dir: Path | str,
    depth_model: nn.Module | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    input_size: int = 1024,
    device: torch.device | str = "cuda",
    batch_size: int = 4,
    use_tta: bool = False,
) -> None:
    """
    Process all images in a directory using true batching.

    :param model: Loaded PDFNet model
    :param input_dir: Input directory path
    :param output_dir: Output directory path
    :param depth_model: Optional depth model
    :param extensions: Image file extensions to process
    :param input_size: Model input size
    :param device: Device to run on
    :param batch_size: Number of images to process in each batch (default: 4)
    :param use_tta: Use test-time augmentation

    Example:
        >>> model = load_pdfnet_model("model.pth")
        >>> process_directory(model, "input/", "output/", batch_size=4)
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
        logger.warning(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process")
    logger.info(f"Batch size: {batch_size}")

    if use_tta:
        # TTA doesn't support batching yet, process individually
        logger.info("Note: TTA mode processes images individually")
        from tqdm import tqdm

        for img_path in tqdm(image_files, desc="Processing with TTA"):
            mask = run_inference_with_tta(
                model=model,
                image=img_path,
                depth_model=depth_model,
                input_size=input_size,
                device=device,
            )
            output_file = output_path / f"{img_path.stem}.png"
            save_prediction(mask, output_file)
    else:
        # Use true batching
        from tqdm import tqdm

        total_batches = (len(image_files) + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(0, len(image_files), batch_size),
            desc="Processing batches",
            total=total_batches,
        ):
            batch_paths = image_files[batch_idx : batch_idx + batch_size]

            # Process batch
            batch_results = _process_image_batch(
                model=model,
                image_paths=batch_paths,
                depth_model=depth_model,
                input_size=input_size,
                device=device,
            )

            # Save results
            for img_path, (mask, _) in zip(batch_paths, batch_results):
                output_file = output_path / f"{img_path.stem}.png"
                save_prediction(mask, output_file)

            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(f"Results saved to {output_dir}")


# ==============================================================================
# CLASS-BASED API - Convenience wrapper around functional API
# ==============================================================================


class PDFNetInference:
    """
    Class-based inference engine for PDFNet.

    This is a convenience wrapper around the functional API above.
    For simple use cases, consider using the functions directly:
    - load_pdfnet_model()
    - run_inference()
    - process_directory()

    Example:
        >>> engine = PDFNetInference()
        >>> mask = engine.predict_single("image.jpg")
    """

    model: nn.Module
    device: torch.device
    config: PDFNetConfig
    moge_model: nn.Module | None
    input_size: int

    def __init__(
        self,
        config: PDFNetConfig | None = None,
        checkpoint_path: Path | str | None = None,
    ) -> None:
        """
        Initialize inference engine.

        Args:
            config: PDFNet configuration object (creates default if None)
            checkpoint_path: Override path to model checkpoint
        """
        self.config = config or PDFNetConfig()
        self.device = self._setup_device()
        self.input_size = self.config.data.input_size

        # Load models
        ckpt_path = checkpoint_path or self.config.inference.checkpoint_path
        self.model = load_pdfnet_model(
            checkpoint_path=ckpt_path,
            model_name=self.config.model.name,
            device=self.device,
            strict=False,
        )

        # Load depth model if configured
        if self.config.inference.use_moge:
            self.moge_model = load_moge_depth_model(
                checkpoint_path=self.config.inference.moge_model, device=self.device
            )
        else:
            self.moge_model = None

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device_str = self.config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_str)

    @torch.no_grad()
    def generate_depth(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Generate depth map for an image.

        Args:
            image: Input image as numpy array (H, W, 3) RGB

        Returns:
            Depth map as numpy array (H, W) with values [0, 255]
        """
        if self.moge_model is not None:
            return generate_depth(self.moge_model, image, self.device)
        else:
            h, w = image.shape[:2]
            return create_placeholder_depth(h, w)

    @torch.no_grad()
    def predict_single(
        self,
        image: npt.NDArray[np.uint8] | Path | str,
        depth: npt.NDArray[np.uint8] | None = None,
        return_depth: bool = False,
    ) -> (
        npt.NDArray[np.float32] | tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]
    ):
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
        # Load image if path provided (needed for depth generation)
        image_array = image
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Could not load image: {image}")
            image_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        mask = run_inference(
            model=self.model,
            image=image,
            depth=depth,
            depth_model=self.moge_model,
            input_size=self.input_size,
            device=self.device,
        )

        if return_depth:
            if depth is None:
                depth = self.generate_depth(image_array)
            return mask, depth

        return mask

    @torch.no_grad()
    def predict_with_tta(
        self,
        image: npt.NDArray[np.uint8] | Path | str,
        flips: list[str] = ["horizontal", "vertical"],
    ) -> npt.NDArray[np.float32]:
        """
        Predict segmentation with test-time augmentation.

        Args:
            image: Input image array or path
            flips: List of flip types to apply ["horizontal", "vertical"]

        Returns:
            Segmentation mask as float array [0, 1] (averaged over augmentations)

        Example:
            >>> engine = PDFNetInference()
            >>> mask = engine.predict_with_tta("image.jpg")
        """
        return run_inference_with_tta(
            model=self.model,
            image=image,
            depth_model=self.moge_model,
            input_size=self.input_size,
            device=self.device,
            flips=flips,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        images: list[npt.NDArray[np.uint8] | Path | str],
        batch_size: int = 4,
        use_tta: bool = False,
        progress: bool = True,
    ) -> list[npt.NDArray[np.float32]]:
        """
        Predict segmentation for multiple images using true batching.

        :param images: List of images or paths
        :param batch_size: Number of images to process in each batch (default: 4)
        :param use_tta: Use test-time augmentation (not supported in batch mode)
        :param progress: Show progress bar
        :returns: List of segmentation masks
        """
        if use_tta:
            warnings.warn("TTA not supported in batch mode, running without TTA")

        return run_batch_inference(
            model=self.model,
            images=images,
            depth_model=self.moge_model,
            input_size=self.input_size,
            device=self.device,
            batch_size=batch_size,
            show_progress=progress,
        )

    def predict_directory(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
        batch_size: int = 4,
        use_tta: bool = False,
        save_depth: bool = False,
    ) -> None:
        """
        Process all images in a directory using true batching.

        :param input_dir: Input directory path
        :param output_dir: Output directory path
        :param extensions: Image file extensions to process
        :param batch_size: Number of images to process in each batch (default: 4)
        :param use_tta: Use test-time augmentation
        :param save_depth: Save depth maps (not implemented in this version)
        """
        if save_depth:
            warnings.warn("save_depth not implemented in this version")

        process_directory(
            model=self.model,
            input_dir=input_dir,
            output_dir=output_dir,
            depth_model=self.moge_model,
            extensions=extensions,
            input_size=self.input_size,
            device=self.device,
            batch_size=batch_size,
            use_tta=use_tta,
        )


__all__ = [
    # Functional API - Preprocessing
    "preprocess_image",
    "preprocess_depth",
    "postprocess_prediction",
    # Functional API - Inference
    "run_inference",
    "run_inference_with_tta",
    "run_batch_inference",
    "process_directory",
    # Functional API - Utilities
    "generate_depth",
    "create_placeholder_depth",
    "save_prediction",
    # Class-based API
    "PDFNetInference",
]
