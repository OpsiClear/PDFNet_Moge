"""
Unified inference module for PDFNet.

This module consolidates all inference functionality including single image,
batch processing, and test-time augmentation.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple
import ttach as tta
from tqdm import tqdm
from PIL import Image

from .models.PDFNet import build_model
from .config import Config, load_config
from .data.transforms import Normalize, Resize, ToTensor


class PDFNetInference:
    """Unified inference class for PDFNet."""

    def __init__(self, config: Optional[Union[str, Config]] = None, checkpoint_path: Optional[str] = None):
        """
        Initialize inference engine.

        Args:
            config: Configuration object or path to config file
            checkpoint_path: Override checkpoint path from config
        """
        # Load configuration
        if isinstance(config, str):
            self.config = load_config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            self.config = load_config()

        # Setup device
        self.device = torch.device(
            self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        )

        # Load model
        self.model = self._load_model(checkpoint_path)

        # Setup transforms
        self.transform = self._setup_transforms()

        # Load depth estimator if needed
        self.moge_model = None
        if self.config.get('inference.use_moge', True):
            self._load_moge()

    def _load_model(self, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
        """Load PDFNet model with checkpoint."""
        # Build model
        from .args import get_args_parser
        import argparse
        parser = argparse.ArgumentParser('PDFNet', parents=[get_args_parser()])
        args = parser.parse_args(args=[])
        args.model = self.config.get('model.name', 'PDFNet_swinB')

        model, _ = build_model(args)

        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = self.config.get('inference.checkpoint_path', 'checkpoints/PDFNet_Best.pth')

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}")

        model = model.to(self.device).eval()
        return model

    def _load_moge(self):
        """Load MoGe depth estimation model."""
        try:
            from moge.model import MoGeModel
            moge_path = self.config.get('inference.moge_model',
                                         'checkpoints/moge/moge-2-vitl-normal/model.pt')
            if Path(moge_path).exists():
                print("Loading MoGe model for depth estimation...")
                self.moge_model = MoGeModel.from_pretrained(moge_path)
                self.moge_model = self.moge_model.to(self.device).eval()
            else:
                print(f"MoGe model not found at {moge_path}")
        except ImportError:
            print("MoGe not available, will use placeholder depth")

    def _setup_transforms(self):
        """Setup image transforms."""
        normalize = Normalize(
            mean=self.config.get('data.normalize_mean', [0.485, 0.456, 0.406]),
            std=self.config.get('data.normalize_std', [0.229, 0.224, 0.225])
        )
        return normalize

    def generate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map for image.

        Args:
            image: Input image as numpy array (H, W, 3)

        Returns:
            Depth map as numpy array (H, W)
        """
        if self.moge_model is not None:
            return self._generate_moge_depth(image)
        else:
            # Placeholder depth
            return np.full((image.shape[0], image.shape[1]), 128, dtype=np.uint8)

    def _generate_moge_depth(self, image: np.ndarray) -> np.ndarray:
        """Generate depth using MoGe model."""
        h, w = image.shape[:2]
        input_size = self.config.get('inference.moge_input_size', 518)

        # Prepare image
        img_resized = cv2.resize(image, (input_size, input_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.moge_model.infer(img_tensor)
            depth = output["depth"]

            # Handle different depth formats
            if depth.dim() == 4:
                depth = depth.squeeze(0).squeeze(0)
            elif depth.dim() == 3:
                depth = depth.squeeze(0)

            # Resize to original size
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            # Convert and normalize
            depth_np = depth.cpu().numpy()
            depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=1.0, neginf=0.0)

            # Normalize to 0-255
            depth_min, depth_max = depth_np.min(), depth_np.max()
            if depth_min == depth_max:
                depth_normalized = np.full_like(depth_np, 128.0)
            else:
                depth_normalized = (depth_np - depth_min) / (depth_max - depth_min) * 255.0

            return np.clip(depth_normalized, 0, 255).astype(np.uint8)

    def predict(
        self,
        image: Union[str, np.ndarray],
        depth: Optional[np.ndarray] = None,
        use_tta: bool = False
    ) -> np.ndarray:
        """
        Predict segmentation for a single image.

        Args:
            image: Image path or numpy array
            depth: Optional pre-computed depth map
            use_tta: Use test-time augmentation

        Returns:
            Segmentation mask as numpy array
        """
        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # Generate depth if not provided
        if depth is None:
            depth = self.generate_depth(image)

        # Prepare inputs
        input_size = self.config.get('data.input_size', 1024)
        img_resized = cv2.resize(image, (input_size, input_size))
        depth_resized = cv2.resize(depth, (input_size, input_size))

        # Convert to tensors
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255
        img_tensor = self.transform({'image': img_tensor})['image']
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0).unsqueeze(0).float() / 255
        depth_tensor = depth_tensor.to(self.device)

        # Run inference
        if use_tta:
            prediction = self._predict_with_tta(img_tensor, depth_tensor)
        else:
            prediction = self._predict_single(img_tensor, depth_tensor)

        # Resize to original size
        prediction = cv2.resize(prediction, (w, h))

        return prediction

    def _predict_single(self, image: torch.Tensor, depth: torch.Tensor) -> np.ndarray:
        """Single image prediction."""
        with torch.no_grad():
            outputs = self.model.inference(image, depth)

            # Extract prediction
            if isinstance(outputs, (list, tuple)):
                pred = outputs[0]
            else:
                pred = outputs

            while isinstance(pred, (list, tuple)) and len(pred) > 0:
                pred = pred[0]

            pred = pred.squeeze().cpu().numpy()

        return pred

    def _predict_with_tta(self, image: torch.Tensor, depth: torch.Tensor) -> np.ndarray:
        """Prediction with test-time augmentation."""
        tta_transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(
                scales=self.config.get('inference.tta_scales', [0.75, 1.0, 1.25]),
                interpolation='bilinear'
            ),
        ])

        predictions = []

        for transformer in tta_transforms:
            # Transform inputs
            aug_image = transformer.augment_image(image)
            aug_depth = transformer.augment_image(depth)

            # Get prediction
            with torch.no_grad():
                outputs = self.model.inference(aug_image, aug_depth)
                if isinstance(outputs, (list, tuple)):
                    pred = outputs[0]
                else:
                    pred = outputs

                while isinstance(pred, (list, tuple)) and len(pred) > 0:
                    pred = pred[0]

            # Reverse transform
            pred_deaug = transformer.deaugment_mask(pred)
            predictions.append(pred_deaug)

        # Average predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred.squeeze().cpu().numpy()

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 1,
        use_tta: bool = False,
        progress: bool = True
    ) -> List[np.ndarray]:
        """
        Predict segmentation for multiple images.

        Args:
            images: List of image paths or numpy arrays
            batch_size: Batch size for processing
            use_tta: Use test-time augmentation
            progress: Show progress bar

        Returns:
            List of segmentation masks
        """
        results = []
        iterator = tqdm(images, desc="Processing images") if progress else images

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = []

            for img in batch:
                pred = self.predict(img, use_tta=use_tta)
                batch_results.append(pred)

            results.extend(batch_results)

        return results

    def predict_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
        use_tta: bool = False,
        batch_size: int = 1
    ):
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            extensions: Image file extensions to process
            use_tta: Use test-time augmentation
            batch_size: Batch size for processing
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))

        print(f"Found {len(image_files)} images to process")

        # Process images
        for img_path in tqdm(image_files, desc="Processing"):
            # Predict
            pred = self.predict(str(img_path), use_tta=use_tta)

            # Save result
            output_file = output_path / f"{img_path.stem}.png"
            pred_img = (pred * 255).astype(np.uint8)
            cv2.imwrite(str(output_file), pred_img)

        print(f"Results saved to {output_dir}")


def run_inference(
    input_path: str,
    output_path: Optional[str] = None,
    config: Optional[str] = None,
    checkpoint: Optional[str] = None,
    use_tta: bool = False,
    batch_size: int = 1,
    device: str = 'cuda'
) -> Union[np.ndarray, List[np.ndarray], None]:
    """
    Convenience function for running inference.

    Args:
        input_path: Input image, directory, or list of images
        output_path: Output path for results
        config: Config file path
        checkpoint: Checkpoint path
        use_tta: Use test-time augmentation
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        Predictions or None if saved to disk
    """
    # Setup configuration
    cfg = load_config(config) if config else load_config()
    cfg.update({'device': device})
    if checkpoint:
        cfg.update({'inference': {'checkpoint_path': checkpoint}})

    # Create inference engine
    engine = PDFNetInference(cfg)

    input_p = Path(input_path)

    if input_p.is_file():
        # Single image
        pred = engine.predict(str(input_p), use_tta=use_tta)

        if output_path:
            pred_img = (pred * 255).astype(np.uint8)
            cv2.imwrite(output_path, pred_img)
            print(f"Result saved to {output_path}")

        return pred

    elif input_p.is_dir():
        # Directory of images
        if output_path is None:
            output_path = 'results'

        engine.predict_directory(
            str(input_p),
            output_path,
            use_tta=use_tta,
            batch_size=batch_size
        )
        return None

    else:
        raise ValueError(f"Invalid input path: {input_path}")