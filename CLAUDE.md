# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDFNet is a PyTorch implementation of a Patch-Depth Fusion Network for high-precision Dichotomous Image Segmentation (DIS). The model combines fine-grained patch strategy with depth integrity-prior from MoGe (Monocular Geometry) depth estimation to achieve state-of-the-art segmentation results.

## Common Development Commands

### Installation and Setup
```bash
# Using UV (recommended - run from project root)
uv pip install -e .

# Download model weights
uv run download.py

# Or install directly from GitHub
pip install git+https://github.com/OpsiClear/PDFNet_Moge.git
```

### Training
```bash
# Training still uses legacy argparse structure (not the main CLI)
# Direct training command (NO longer use pdfnet-train, it doesn't exist):
uv run python -m pdfnet.train --data_path DATA/DIS-DATA --model PDFNet_swinB --epochs 100

# Or via the tyro CLI (which wraps the legacy trainer):
uv run pdfnet.py train --config-file config.yaml

# Key training arguments (see src/pdfnet/train.py for full list):
# --model: PDFNet_swinB, PDFNet_swinL, PDFNet_swinT
# --batch_size: Default 1 (high memory requirements)
# --epochs: Default 100
# --lr: Learning rate, default 1e-4
# --data_path: Path to DIS-DATA directory
# --input_size: Input image size, default 1024
```

### Benchmarking and Metrics
```bash
# Benchmark on DIS datasets using tyro CLI
uv run pdfnet.py benchmark --checkpoint checkpoints/PDFNet_Best.pth --data-path DATA/DIS-DATA

# Benchmark with TTA
uv run pdfnet.py benchmark --use-tta --datasets DIS-TE1 DIS-TE2

# Or directly via benchmark module
uv run python -m pdfnet.metric_tools.benchmark

# Evaluate predictions
uv run pdfnet.py evaluate --pred-dir results/ --gt-dir DATA/DIS-DATA
```

### Inference
```bash
# Run inference on single image (local development)
uv run pdfnet.py infer --input path/to/image.jpg --output result.png

# Run inference on directory
uv run pdfnet.py infer --input path/to/images/ --output results/

# With test-time augmentation for better accuracy
uv run pdfnet.py infer --input image.jpg --use-tta

# Visualize results interactively
uv run pdfnet.py infer --input image.jpg --visualize

# After pip installation, use:
pdfnet infer --input image.jpg --output result.png

# Interactive demo with Jupyter
jupyter notebook demo.ipynb
```

## Architecture and Key Components

### Model Architecture
PDFNet combines three core components:
1. **Multi-modal Fusion**: Image and depth map inputs are fused for enhanced object perception
2. **Patch Strategy**: Fine-grained patch selection and enhancement for detail sensitivity
3. **Depth Refinement**: Shared encoder with depth refinement decoder for capturing subtle depth information

### Core File Structure
- `pdfnet.py`: Local development CLI entry point (tyro-based)
- `src/pdfnet/__main__.py`: Package CLI entry point (makes `pdfnet` command work after pip install)
- `src/pdfnet/config.py`: Type-safe configuration using Python 3.12 dataclasses with tyro annotations
- `src/pdfnet/types.py`: Central type definitions (Tensor, ImageArray, PathLike, etc.)
- `src/pdfnet/models/PDFNet.py`: Main model with PDFNet_swinB/L/T, includes FSE (Fusion), CoA (Cross-attention), PDF_depth_decoder
- `src/pdfnet/models/swin_transformer.py`: Swin Transformer backbone (SwinB, SwinL, SwinT)
- `src/pdfnet/models/utils.py`: Model utilities (RMSNorm, SwiGLU, losses, upsampling)
- `src/pdfnet/train.py`: Legacy argparse-based training loop with mixed precision and EMA
- `src/pdfnet/inference.py`: Complete inference API (functional + class-based PDFNetInference)
- `src/pdfnet/model_loader.py`: Isolated model loading (load_pdfnet_model, load_moge_depth_model)
- `src/pdfnet/dataloaders/dis_dataset.py`: DIS dataset with build_dataset() factory
- `src/pdfnet/data/transforms.py`: Type-safe transforms (Resize, RandomCrop, ColorJitter, etc.)
- `src/pdfnet/core/losses.py`: Type-safe loss functions (StructureLoss, IOULoss, SSIMLoss, IntegrityPriorLoss)
- `src/pdfnet/metric_tools/benchmark.py`: Benchmark runner for DIS datasets
- `src/pdfnet/metric_tools/soc_metrics.py`: SOC metric computation
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

### Package Management and Dependencies
- **UV**: Used for fast, reliable dependency management (see pyproject.toml)
- **MoGe**: Installed as git dependency from Microsoft's repository (replaces DepthAnything V2)
- **PyTorch**: CUDA 12.8 builds configured via uv.sources for Linux/Windows
- **Execution modes**: Supports both `python -m pdfnet` and `uv run pdfnet.py`

### CLI Architecture (Tyro-based)
- **Local dev**: `pdfnet.py` (root) → Main CLI during development
- **Installed**: `src/pdfnet/__main__.py` → Entry point after `pip install`
- Both use tyro for type-safe subcommands (train, infer, benchmark, evaluate, download, config)
- Training uses legacy argparse internally but is wrapped by TrainCommand for consistency

### Configuration System
- Type-safe dataclasses with Python 3.12 type hints throughout
- Central config in `src/pdfnet/config.py` (PDFNetConfig with nested configs)
- YAML file support via config.save()/config.load()
- Tyro annotations (@tyro.conf.arg) provide CLI help text

### Code Organization Principles
- **Isolated inference**: `inference.py` + `model_loader.py` have no training dependencies
- **Type safety**: Central types in `src/pdfnet/types.py` (removed after refactor, now using native types)
- **Functional + OOP**: Inference provides both functional API (run_inference) and class API (PDFNetInference)
- **Loss functions**: Modular in `core/losses.py` (StructureLoss, IOULoss, SSIMLoss, IntegrityPriorLoss)

### Key Implementation Details
- **Batch inference**: True batching support (default: batch_size=4), processes multiple images in parallel on GPU
- **TTA**: Horizontal + vertical flips for improved accuracy (processes individually, no batching)
- **Depth processing**: MoGe generates pseudo-depth maps, normalized 0-255 with NaN/Inf handling
- **Model states**: Training uses model.forward() with losses, inference uses model.inference() (no gradients)