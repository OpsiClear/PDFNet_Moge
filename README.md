# PDFNet - Type-Safe Dichotomous Image Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2503.06100-red)](https://arxiv.org/abs/2503.06100)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Modern, type-safe implementation of **PDFNet** (Patch-Depth Fusion Network) for high-precision dichotomous image segmentation. This fork features a complete rewrite with Python 3.12 type hints, unified CLI using tyro, and streamlined codebase architecture.

> **Original Paper:** [Patch-Depth Fusion: Dichotomous Image Segmentation via Fine-Grained Patch Strategy and Depth Integrity-Prior](https://arxiv.org/abs/2503.06100)
> **Original Authors:** Xianjie Liu, Keren Fu, Qijun Zhao
> **Original Repository:** [Tennine2077/PDFNet](https://github.com/Tennine2077/PDFNet)

## Key Features

✅ **Type-Safe Implementation** - Full Python 3.12 type hints throughout
✅ **Unified CLI** - Single `pdfnet` command with subcommands (train, infer, test, evaluate)
✅ **Modern Architecture** - Clean, maintainable codebase with 40% less code
✅ **MoGe Integration** - Uses Microsoft's MoGe for superior depth estimation
✅ **Easy Installation** - Install directly from GitHub with uv or pip
✅ **TTA Support** - Test-time augmentation for improved accuracy

## Quick Start

### Installation

```bash
# Install with uv (recommended - faster)
uv add git+https://github.com/OpsiClear/PDFNet_Moge.git

# Or with pip
pip install git+https://github.com/OpsiClear/PDFNet_Moge.git

# Development installation
git clone https://github.com/OpsiClear/PDFNet_Moge.git
cd PDFNet_Moge
uv pip install -e .
```

### Download Model Weights

```bash
# Download PDFNet and Swin-B weights
pdfnet download --weights

# Show dataset download instructions
pdfnet download --dataset-info
```

### Basic Usage

```bash
# Run inference on a single image
pdfnet infer --input image.jpg --output result.png

# Run inference on a directory with TTA
pdfnet infer --input images/ --output results/ --use-tta

# Visualize results
pdfnet infer --input image.jpg --visualize

# Train a model
pdfnet train --config-file config.yaml

# Benchmark on DIS datasets
pdfnet benchmark --checkpoint checkpoints/PDFNet_Best.pth --data-path DATA/DIS-DATA

# Evaluate predictions
pdfnet evaluate --pred-dir results/ --gt-dir DATA/DIS-DATA
```

## CLI Commands

The unified `pdfnet` CLI provides all functionality:

| Command | Description |
|---------|-------------|
| `pdfnet train` | Train PDFNet models with custom configurations |
| `pdfnet infer` | Run inference on images (single/batch, with optional TTA) |
| `pdfnet benchmark` | Benchmark model on standard DIS datasets with metrics |
| `pdfnet evaluate` | Evaluate predictions against ground truth |
| `pdfnet config` | Configuration management (show, create, validate) |
| `pdfnet download` | Download model weights and get dataset info |

Get help for any command:
```bash
pdfnet --help
pdfnet infer --help
```

## Python API

Use PDFNet in your Python scripts:

```python
from pdfnet.inference import PDFNetInference
from pdfnet.config import PDFNetConfig

# Create configuration
config = PDFNetConfig()
config.inference.checkpoint_path = "checkpoints/PDFNet_Best.pth"
config.inference.use_tta = True

# Initialize inference engine
engine = PDFNetInference(config)

# Run inference
result = engine.predict("image.jpg")

# Process directory
engine.predict_directory("input_dir/", "output_dir/")
```

## Project Structure

```
PDFNet_Moge/
├── pdfnet.py              # CLI entry point (local dev)
├── src/pdfnet/
│   ├── __main__.py        # Package CLI entry point
│   ├── config.py          # Type-safe configuration
│   ├── inference.py       # Inference engine with TTA
│   ├── train.py           # Training loop
│   ├── models/
│   │   ├── PDFNet.py      # Model architecture
│   │   └── swin_transformer.py
│   ├── data/
│   │   └── transforms.py  # Type-safe data transforms
│   ├── core/
│   │   └── losses.py      # Type-safe loss functions
│   ├── dataloaders/
│   │   └── dis_dataset.py # DIS dataset loading
│   └── metric_tools/      # Evaluation utilities
├── demo.ipynb             # Interactive demo
└── CLAUDE.md              # Development guide
```

## Dataset Preparation

Download the [DIS-5K dataset](https://github.com/xuebinqin/DIS) and organize as:

```
PDFNet_Moge/
└── DATA/
    └── DIS-DATA/
        ├── DIS-TR/         # Training set
        ├── DIS-VD/         # Validation set
        ├── DIS-TE1/        # Test set 1
        ├── DIS-TE2/        # Test set 2
        ├── DIS-TE3/        # Test set 3
        └── DIS-TE4/        # Test set 4
            ├── images/
            └── masks/
```

## Model Weights

| Model | Download |
|-------|----------|
| PDFNet (DIS-5K) | [Google Drive](https://drive.google.com/drive/folders/1dqkFVR4TElSRFNHhu6er45OQkoHhJsZz) |
| Swin-B Backbone | [GitHub Release](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) |

Place weights in `checkpoints/` directory.

## Training

```bash
# Train with default configuration
pdfnet train

# Train with custom config
pdfnet train --config-file my_config.yaml

# Resume training
pdfnet train --resume checkpoints/last.pth
```

Training configuration can be managed via YAML files or the type-safe `PDFNetConfig` dataclass.

## Performance

PDFNet achieves state-of-the-art results on DIS-5K dataset:

- **Memory Efficient**: 1024×1024 inference uses ~4.9GB VRAM
- **Fast Training**: ~2 days on RTX 4090 for DIS-5K
- **Small Model**: <11% parameters of diffusion-based methods
- **High Accuracy**: Matches or exceeds diffusion methods

For detailed benchmarks, see the [original paper](https://arxiv.org/abs/2503.06100).

## What's New in This Fork

### Architecture Improvements
- ✅ **Type-Safe Codebase** - Python 3.12 type hints using `tyro` for CLI
- ✅ **Unified CLI** - Single entry point replacing 3 separate scripts
- ✅ **40% Code Reduction** - Removed duplicates and dead code
- ✅ **Clean Structure** - Organized modules (core/, data/, models/)
- ✅ **Better Imports** - Proper package structure for pip installation

### Removed Legacy Code
- ❌ Old argparse CLI (`pdfnet_cli.py`)
- ❌ Standalone inference script (`apply_pdfnet.py`)
- ❌ Redundant demo script (`demo.py`)
- ❌ Unused constants file
- ❌ Duplicate utility functions

### New Features
- 🎯 Type-safe configuration with dataclasses
- 🎯 Modular inference engine
- 🎯 Test-time augmentation support
- 🎯 Batch processing for directories
- 🎯 Pip installable from GitHub
- 🎯 `python -m pdfnet` execution support

## Configuration Management

```bash
# Show default configuration
pdfnet config --action show

# Create custom config file
pdfnet config --action create --output my_config.yaml

# Validate configuration
pdfnet config --action validate --config-file my_config.yaml
```

## Requirements

- Python 3.12+
- PyTorch 2.0+ with CUDA support (recommended)
- 8GB+ GPU VRAM for inference
- 24GB+ GPU VRAM for training (RTX 4090 or equivalent)

All dependencies are automatically installed via `uv add` or `pip install`.

## Interactive Demo

Try PDFNet with the Jupyter notebook:

```bash
jupyter notebook demo.ipynb
```

The demo includes:
- Single image inference
- Batch processing
- Test-time augmentation examples
- Visualization tools

## Citation

If you use PDFNet in your research, please cite the original paper:

```bibtex
@misc{liu2025patchdepthfusiondichotomousimage,
  title={Patch-Depth Fusion: Dichotomous Image Segmentation via Fine-Grained Patch Strategy and Depth Integrity-Prior},
  author={Xianjie Liu and Keren Fu and Qijun Zhao},
  year={2025},
  eprint={2503.06100},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.06100}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original PDFNet paper and implementation by Xianjie Liu, Keren Fu, and Qijun Zhao
- [MoGe](https://github.com/microsoft/MoGe) by Microsoft for depth estimation
- [DIS-5K](https://github.com/xuebinqin/DIS) dataset by Xuebin Qin et al.
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) by Microsoft

## Related Resources

- 📖 [Original Paper](https://arxiv.org/abs/2503.06100)
- 🤗 [Hugging Face Space](https://huggingface.co/spaces/Tennineee/PDFNet)
- 📚 [Awesome Dichotomous Image Segmentation](https://github.com/Tennine2077/Awesome-Dichotomous-Image-Segmentation)
- 🔧 [Development Guide](CLAUDE.md)
- 📦 [Installation Guide](INSTALL.md)

## Contributing

Contributions are welcome! This fork focuses on:
- Type safety and code quality
- CLI/API improvements
- Documentation
- Bug fixes

Please open an issue or pull request on GitHub.

## Support

- **Issues**: [GitHub Issues](https://github.com/OpsiClear/PDFNet_Moge/issues)
- **Original Repository**: [Tennine2077/PDFNet](https://github.com/Tennine2077/PDFNet)

---

**Note**: This is a modernized fork focusing on code quality and developer experience. For the original implementation, please visit [Tennine2077/PDFNet](https://github.com/Tennine2077/PDFNet).