#!/usr/bin/env python3
"""
Apply PDFNet to your images for dichotomous segmentation.

Usage:
    python apply_pdfnet.py --input path/to/image.jpg --output result.png
    python apply_pdfnet.py --input_dir path/to/images/ --output_dir results/
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize

# Import PDFNet components
from src.pdfnet.models.PDFNet import build_model
from src.pdfnet.args import get_args_parser
from moge.model import MoGeModel


class GOSNormalize(object):
    """Normalization for PDFNet input."""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return normalize(image, self.mean, self.std)


def process_image_with_moge(image_np, model, device):
    """Process a single image with MoGe to get depth map."""
    height, width = image_np.shape[:2]
    
    # Prepare image for MoGe
    input_size = 518
    image_resized = cv2.resize(image_np, (input_size, input_size))
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        
        # Extract depth map
        if isinstance(output, dict):
            depth = output.get('depth', output.get('pred_depth', None))
        else:
            depth = output
        
        if depth is None:
            raise ValueError("Could not extract depth from MoGe output")
        
        # Resize depth back to original image size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0) if depth.dim() == 3 else depth,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Convert to numpy
        depth_np = depth.cpu().numpy()
        
        # Normalize depth to 0-255 range
        depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255.0
        
        return depth_normalized.astype(np.uint8)


def apply_pdfnet_single(image_path, output_path, pdfnet_model, moge_model, transforms, device):
    """Apply PDFNet to a single image."""
    print(f"Processing: {image_path}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return False
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    # Generate depth map with MoGe
    print("  Generating depth map...")
    depth = process_image_with_moge(img, moge_model, device)
    
    # Resize for PDFNet input (1024x1024)
    img_resized = cv2.resize(img, (1024, 1024))
    depth_resized = cv2.resize(depth, (1024, 1024))
    
    # Prepare tensors
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255
    img_tensor = transforms(img_tensor).unsqueeze(0)
    depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0).unsqueeze(0) / 255
    
    # Run PDFNet inference
    print("  Running PDFNet segmentation...")
    with torch.no_grad():
        segmentation_result = pdfnet_model.inference(
            img_tensor.to(device), 
            depth_tensor.to(device)
        )[0][0][0].cpu()
    
    # Resize result back to original size
    segmentation_map = cv2.resize(np.array(segmentation_result), (W, H))
    
    # Convert to binary mask (0-255)
    binary_mask = (segmentation_map * 255).astype(np.uint8)
    
    # Save result
    cv2.imwrite(str(output_path), binary_mask)
    print(f"  Saved result: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Apply PDFNet for dichotomous image segmentation')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--input_dir', type=str, help='Input directory with images')
    parser.add_argument('--output', type=str, help='Output image path')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/PDFNet_Best.pth',
                       help='Path to PDFNet checkpoint')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.input_dir:
        print("Error: Please specify either --input or --input_dir")
        return
    
    if args.input and not args.output:
        print("Error: Please specify --output when using --input")
        return
        
    if args.input_dir and not args.output_dir:
        print("Error: Please specify --output_dir when using --input_dir")
        return
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load PDFNet model
    print("Loading PDFNet model...")
    pdfnet_parser = argparse.ArgumentParser(parents=[get_args_parser()])
    pdfnet_args = pdfnet_parser.parse_args(args=[])
    pdfnet_model, _ = build_model(pdfnet_args)
    
    if Path(args.checkpoint).exists():
        pdfnet_model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint not found: {args.checkpoint}")
        print("Using randomly initialized weights (results will be poor)")
    
    pdfnet_model = pdfnet_model.to(device).eval()
    
    # Load MoGe model
    print("Loading MoGe model...")
    moge_model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')
    moge_model = moge_model.to(device).eval()
    
    # Setup transforms
    transforms = GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # Process single image
    if args.input:
        input_path = Path(args.input)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = apply_pdfnet_single(
            input_path, output_path, pdfnet_model, moge_model, transforms, device
        )
        
        if success and args.visualize:
            # Show results
            original = cv2.imread(str(input_path))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            result = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(original)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(result, cmap='gray')
            axes[1].set_title('PDFNet Segmentation')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    # Process directory
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to process")
        
        success_count = 0
        for image_file in image_files:
            output_file = output_dir / f"{image_file.stem}_segmented{image_file.suffix}"
            
            if apply_pdfnet_single(
                image_file, output_file, pdfnet_model, moge_model, transforms, device
            ):
                success_count += 1
        
        print(f"\nCompleted: {success_count}/{len(image_files)} images processed successfully")
    
    print("Done!")


if __name__ == '__main__':
    main()
