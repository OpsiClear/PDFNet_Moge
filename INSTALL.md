# PDFNet Installation Guide

## Installation from GitHub

### Option 1: Install directly from GitHub (Recommended)

```bash
# Install the latest version from main branch
pip install git+https://github.com/OpsiClear/PDFNet_Moge.git

# Or install with uv (faster)
uv pip install git+https://github.com/OpsiClear/PDFNet_Moge.git
```

### Option 2: Install specific branch or commit

```bash
# Install from specific branch
pip install git+https://github.com/OpsiClear/PDFNet_Moge.git@branch-name

# Install from specific commit
pip install git+https://github.com/OpsiClear/PDFNet_Moge.git@commit-hash

# Install from specific tag/release
pip install git+https://github.com/OpsiClear/PDFNet_Moge.git@v0.1.0
```

### Option 3: Development Installation (Editable Mode)

```bash
# Clone the repository
git clone https://github.com/OpsiClear/PDFNet_Moge.git
cd PDFNet_Moge

# Install in editable mode
pip install -e .

# Or with uv
uv pip install -e .
```

## What Gets Installed

After installation, you'll have:

1. **`pdfnet` command** - CLI tool available system-wide
2. **`pdfnet` Python package** - Importable in your Python scripts
3. **All dependencies** - Including PyTorch, MoGe, tyro, etc.

## Usage After Installation

### Command-Line Interface

```bash
# Train a model
pdfnet train --config-file config.yaml

# Run inference
pdfnet infer --input image.jpg --output result.png

# Test on datasets
pdfnet test --checkpoint model.pth --data-path DATA/

# Evaluate predictions
pdfnet evaluate --pred-dir results/ --gt-dir DATA/

# Configuration management
pdfnet config --action show

# Download weights
pdfnet download --weights
```

### Python API

```python
from pdfnet.models.PDFNet import build_model
from pdfnet.inference import PDFNetInference
from pdfnet.config import PDFNetConfig

# Create configuration
config = PDFNetConfig()

# Initialize inference engine
engine = PDFNetInference(config)

# Run inference
result = engine.predict("image.jpg")
```

### Module Execution

You can also run as a Python module:

```bash
# Alternative to 'pdfnet' command
python -m pdfnet train --config-file config.yaml
python -m pdfnet infer --input image.jpg
```

## Requirements

- **Python**: 3.12 or higher
- **CUDA**: Optional but recommended for GPU acceleration
- **Platform**: Linux, Windows, macOS

## Dependencies

Core dependencies are automatically installed:
- PyTorch (with CUDA 12.8 support on Linux/Windows)
- torchvision
- MoGe (from Microsoft GitHub)
- tyro (for CLI)
- timm, einops, Pillow, opencv-python
- numpy, scipy, scikit-learn
- matplotlib, tqdm
- And more (see pyproject.toml)

## Verify Installation

```bash
# Check if pdfnet command is available
pdfnet --help

# Check version
python -c "import pdfnet; print('PDFNet installed successfully')"

# Run a quick test
pdfnet download --dataset-info
```

## Troubleshooting

### Command not found: pdfnet

If `pdfnet` command is not found after installation:

1. Make sure pip/uv install location is in your PATH
2. Try using `python -m pdfnet` instead
3. Reinstall with `pip install --force-reinstall git+https://github.com/OpsiClear/PDFNet_Moge.git`

### Import errors

If you get import errors:
```bash
# Reinstall with all dependencies
pip install --force-reinstall --no-cache-dir git+https://github.com/OpsiClear/PDFNet_Moge.git
```

### PyTorch CUDA version mismatch

The package installs PyTorch with CUDA 12.8 by default. If you need a different version:

```bash
# Install PyTorch separately first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install PDFNet
pip install --no-deps git+https://github.com/OpsiClear/PDFNet_Moge.git

# Install remaining dependencies
pip install tyro numpy opencv-python pillow tqdm matplotlib scikit-learn einops timm
pip install git+https://github.com/microsoft/MoGe.git
```

## Uninstallation

```bash
pip uninstall pdfnet
```

## Development Setup

For contributing or development:

```bash
# Clone and create virtual environment
git clone https://github.com/OpsiClear/PDFNet_Moge.git
cd PDFNet_Moge
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
uv pip install -e .

# Now you can edit code and changes take effect immediately
```