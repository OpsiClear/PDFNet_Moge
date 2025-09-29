#!/usr/bin/env python3
"""
PDFNet Demo Script

This script provides a simple demo interface for PDFNet inference.
It wraps the apply_pdfnet functionality for package-based usage.
"""

import argparse
import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from .models.PDFNet import build_model
from .args import get_args_parser
from .dataloaders.Mydataset import GOSNormalize


def run_inference(input_path, output_path, checkpoint_path=None, device='cuda', visualize=False):
    """
    Run PDFNet inference on a single image.

    Args:
        input_path: Path to input image
        output_path: Path to save output segmentation
        checkpoint_path: Path to model checkpoint
        device: Device to use ('cuda' or 'cpu')
        visualize: If True, show visualization of results

    Returns:
        Segmentation mask as numpy array
    """
    # Check if MoGe is available for depth estimation
    try:
        from moge.model import MoGeModel
        use_moge = True
    except ImportError:
        print("Warning: MoGe not available. Using placeholder depth map.")
        use_moge = False

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    parser = argparse.ArgumentParser('PDFNet Demo', parents=[get_args_parser()])
    args = parser.parse_args(args=[])
    model, model_name = build_model(args)

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
    else:
        # Try default checkpoint location
        default_checkpoint = 'checkpoints/PDFNet_Best.pth'
        if os.path.exists(default_checkpoint):
            print(f"Loading default checkpoint: {default_checkpoint}")
            model.load_state_dict(torch.load(default_checkpoint, map_location='cpu'), strict=False)
        else:
            print("Warning: No checkpoint loaded, using random weights")

    model = model.to(device).eval()

    # Load and prepare image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    print(f"Image size: {W}x{H}")

    # Generate or create depth map
    if use_moge:
        print("Generating depth map with MoGe...")
        moge_model = MoGeModel.from_pretrained('checkpoints/moge/moge-2-vitl-normal/model.pt')
        moge_model = moge_model.to(device).eval()

        # Process image with MoGe
        input_size = 518
        img_resized = cv2.resize(img, (input_size, input_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            moge_output = moge_model.infer(img_tensor)
            depth = moge_output["depth"]

            if depth.dim() == 4:
                depth = depth.squeeze(0).squeeze(0)
            elif depth.dim() == 3:
                depth = depth.squeeze(0)

            # Resize back to original size
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            depth_np = depth.cpu().numpy()
            depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=1.0, neginf=0.0)

            # Normalize to 0-255
            depth_min, depth_max = depth_np.min(), depth_np.max()
            if depth_min == depth_max:
                depth_normalized = np.full_like(depth_np, 128.0)
            else:
                depth_normalized = (depth_np - depth_min) / (depth_max - depth_min) * 255.0

            depth = np.clip(depth_normalized, 0, 255).astype(np.uint8)
    else:
        # Create placeholder depth map
        print("Creating placeholder depth map...")
        depth = np.full((H, W), 128, dtype=np.uint8)

    # Prepare inputs for model
    img_resized = cv2.resize(img, (1024, 1024))
    depth_resized = cv2.resize(depth, (1024, 1024))

    # Apply normalization
    transforms = GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255
    img_tensor = transforms(img_tensor).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0).unsqueeze(0).to(device) / 255

    # Run inference
    print("Running PDFNet inference...")
    with torch.no_grad():
        outputs = model.inference(img_tensor, depth_tensor)
        if isinstance(outputs, (list, tuple)):
            prediction = outputs[0]
        else:
            prediction = outputs

        # Extract the prediction
        while isinstance(prediction, (list, tuple)) and len(prediction) > 0:
            prediction = prediction[0]

        prediction = prediction.squeeze().cpu()

    # Resize to original size
    prediction_np = prediction.numpy()
    prediction_resized = cv2.resize(prediction_np, (W, H))

    # Convert to image and save
    prediction_img = (prediction_resized * 255).astype(np.uint8)
    cv2.imwrite(output_path, prediction_img)
    print(f"Result saved to: {output_path}")

    # Visualize if requested
    if visualize:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(depth, cmap='plasma')
            axes[1].set_title("Depth Map")
            axes[1].axis('off')

            axes[2].imshow(prediction_resized, cmap='gray')
            axes[2].set_title("PDFNet Segmentation")
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not available for visualization")

    return prediction_resized


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(description='PDFNet Demo - Run inference on single images')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output segmentation path')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Model checkpoint path (default: checkpoints/PDFNet_Best.pth)')
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Show visualization of results')

    args = parser.parse_args()

    try:
        run_inference(
            args.input,
            args.output,
            args.checkpoint,
            args.device,
            args.visualize
        )
        print("Demo completed successfully!")
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())