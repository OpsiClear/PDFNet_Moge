# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDFNet is a PyTorch implementation of a Patch-Depth Fusion Network for high-precision Dichotomous Image Segmentation (DIS). The model combines fine-grained patch strategy with depth integrity-prior from MoGe (Monocular Geometry) depth estimation to achieve state-of-the-art segmentation results.

## Common Development Commands

### Installation and Setup
```bash
# Using UV (recommended)
uv pip install -e .

# Download model weights
python download.py

# Traditional pip installation (requires manual MoGe setup)
pip install -r requirements.txt  # Note: This method requires manual MoGe configuration
```

### Training
```bash
# Train with default settings
pdfnet-train --data_path DATA/DIS-DATA --model PDFNet_swinB

# Or using the package directly
python -m pdfnet.train --data_path DATA/DIS-DATA --model PDFNet_swinB

# Training arguments are defined in src/pdfnet/args.py
```

### Testing and Metrics
```bash
# Run testing script
python -m pdfnet.metric_tools.Test

# Metrics calculation
python -m pdfnet.metric_tools.soc_metrics
```

### Inference
```bash
# Apply PDFNet to images
python apply_pdfnet.py --input path/to/image.jpg --output result.png
python apply_pdfnet.py --input_dir path/to/images/ --output_dir results/

# Interactive demo with Jupyter
# Use demo.ipynb for interactive testing with TTA support
```

## Architecture and Key Components

### Model Architecture
PDFNet combines three core components:
1. **Multi-modal Fusion**: Image and depth map inputs are fused for enhanced object perception
2. **Patch Strategy**: Fine-grained patch selection and enhancement for detail sensitivity
3. **Depth Refinement**: Shared encoder with depth refinement decoder for capturing subtle depth information

### Core File Structure
- `src/pdfnet/models/PDFNet.py`: Main model implementation with build_model() factory
- `src/pdfnet/models/swin_transformer.py`: Swin Transformer backbone implementation
- `src/pdfnet/train.py`: Training loop with mixed precision and EMA support
- `src/pdfnet/dataloaders/Mydataset.py`: Dataset loading and augmentation pipeline
- `apply_pdfnet.py`: Standalone inference script with MoGe depth generation
- `demo.ipynb`: Interactive demonstration with Test-Time Augmentation (TTA)

### Depth Processing Pipeline
The project uses MoGe (moge-2-vitl-normal variant) for depth estimation instead of DepthAnything V2. The depth processing workflow:
1. Input image → MoGe model → Pseudo-depth map
2. Depth maps are normalized to 0-255 range with NaN/Inf handling
3. Depth integrity-prior loss enhances uniformity in segmentation results

### Dataset Structure
Expected DIS-5K dataset structure:
```
PDFNet/
└── DATA/
    └── DIS-DATA/
        ├── DIS-TE1/
        ├── DIS-TE2/
        ├── DIS-TE3/
        ├── DIS-TE4/
        ├── DIS-TR/
        └── DIS-VD/
            ├── images/
            └── masks/
```

### Model Checkpoints
- Download Swin-B pretrained weights to `checkpoints/` directory
- Trained PDFNet models are saved to `runs/` directory during training
- Best model checkpoint: `checkpoints/PDFNet_Best.pth`

## Important Implementation Details

### Training Configuration
- Default batch size: 1 (due to high memory requirements)
- Input size: 1024×1024
- Optimizer: AdamW with learning rate scheduling
- Mixed precision training supported via torch.cuda.amp
- Gradient accumulation available via --update_freq parameter

### Memory Requirements
- Training: RTX 4090 or equivalent (24GB VRAM recommended)
- Inference at 1024×1024: ~4.9GB VRAM
- TTA inference requires additional memory for augmentation transforms

### Test-Time Augmentation (TTA)
The demo notebook includes TTA support with horizontal and vertical flips for improved accuracy at the cost of inference speed (typically 2-3x slower).

### Metric Evaluation
The project uses multiple metrics for evaluation:
- F1 score (F1torch.py)
- SOC metrics (soc_metrics.py)
- Additional metrics defined in metrics.py

## Development Notes

- The project uses UV for dependency management (see pyproject.toml)
- MoGe is installed as a git dependency from Microsoft's repository
- CUDA 12.8 PyTorch builds are configured in pyproject.toml
- The codebase supports both package-style imports (pdfnet.*) and direct script execution