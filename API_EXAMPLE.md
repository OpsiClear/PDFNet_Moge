# PDFNet Python API Examples

Complete guide for using PDFNet as a Python library after installation.

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/OpsiClear/PDFNet_Moge.git

# Or with uv (faster)
uv pip install git+https://github.com/OpsiClear/PDFNet_Moge.git

# Development installation (editable)
git clone https://github.com/OpsiClear/PDFNet_Moge.git
cd PDFNet_Moge
uv pip install -e .
```

## Quick Start

### Simple Inference (Recommended)

```python
from pdfnet import PDFNetInference

# Initialize inference engine
# NOTE: Requires MoGe depth model by default
# Checkpoint must exist at: checkpoints/moge/moge-2-vitl-normal/model.pt
engine = PDFNetInference()

# Run inference on a single image
mask = engine.predict_single("input.jpg")

# Save result
import cv2
cv2.imwrite("output.png", (mask * 255).astype('uint8'))

# Or with visualization
import matplotlib.pyplot as plt
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()
```

### Inference Without MoGe (Faster, Lower Accuracy)

```python
from pdfnet import PDFNetInference
from pdfnet.config import PDFNetConfig

# Disable MoGe depth estimation
config = PDFNetConfig()
config.inference.use_moge = False

engine = PDFNetInference(config)
mask = engine.predict_single("input.jpg")
```

### Batch Processing

```python
from pdfnet import PDFNetInference

engine = PDFNetInference()

# Process multiple images (uses true batching on GPU)
masks = engine.predict_batch(
    ["image1.jpg", "image2.jpg", "image3.jpg"],
    batch_size=4  # Process 4 images at once
)

# Process entire directory
engine.predict_directory(
    input_dir="input_images/",
    output_dir="results/",
    batch_size=4
)
```

### Test-Time Augmentation (TTA)

```python
from pdfnet import PDFNetInference

engine = PDFNetInference()

# Higher accuracy with TTA (slower)
mask = engine.predict_with_tta("input.jpg")
```

---

## Functional API (Advanced)

For fine-grained control, use the functional API directly.

### 1. Model Loading

```python
from pdfnet import load_pdfnet_model, load_moge_depth_model, get_model_info

# Load PDFNet model
model = load_pdfnet_model(
    checkpoint_path="checkpoints/PDFNet_Best.pth",
    model_name="PDFNet_swinB",
    device="cuda",
    strict=False
)

# Load MoGe depth estimation model
# REQUIRED if using depth estimation (recommended for best accuracy)
# Will raise ImportError if MoGe not installed
# Will raise FileNotFoundError if checkpoint missing
try:
    depth_model = load_moge_depth_model(
        checkpoint_path="checkpoints/moge/moge-2-vitl-normal/model.pt",
        device="cuda"
    )
except (ImportError, FileNotFoundError) as e:
    print(f"MoGe not available: {e}")
    depth_model = None  # Proceed without depth estimation

# Get model information
info = get_model_info(model)
print(f"Model parameters: {info['total_params']:,}")
print(f"Trainable params: {info['trainable_params']:,}")
```

### 2. Single Image Inference

```python
from pdfnet import load_pdfnet_model, load_moge_depth_model, run_inference
import cv2

# Load models
model = load_pdfnet_model("checkpoints/PDFNet_Best.pth")
depth_model = load_moge_depth_model()  # Optional but recommended

# Run inference
mask = run_inference(
    model=model,
    image="input.jpg",  # Can be path or numpy array
    depth_model=depth_model,  # Optional
    input_size=1024,
    device="cuda"
)

# mask is a numpy array (H, W) with values [0, 1]
cv2.imwrite("output.png", (mask * 255).astype('uint8'))
```

### 3. Batch Inference (True Batching)

```python
from pdfnet import load_pdfnet_model, run_batch_inference

model = load_pdfnet_model("checkpoints/PDFNet_Best.pth")

# Process multiple images in parallel on GPU
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
masks = run_batch_inference(
    model=model,
    images=image_paths,
    batch_size=4,  # Process 4 images at once
    device="cuda",
    show_progress=True
)

# Save results
for i, mask in enumerate(masks):
    cv2.imwrite(f"output_{i}.png", (mask * 255).astype('uint8'))
```

### 4. Test-Time Augmentation

```python
from pdfnet import load_pdfnet_model, run_inference_with_tta

model = load_pdfnet_model("checkpoints/PDFNet_Best.pth")

# Run with horizontal and vertical flips for better accuracy
mask = run_inference_with_tta(
    model=model,
    image="input.jpg",
    flips=["horizontal", "vertical"],
    device="cuda"
)
```

### 5. Directory Processing

```python
from pdfnet import load_pdfnet_model, process_directory

model = load_pdfnet_model("checkpoints/PDFNet_Best.pth")

# Process all images in a directory
process_directory(
    model=model,
    input_dir="input_images/",
    output_dir="results/",
    extensions=(".jpg", ".jpeg", ".png"),
    batch_size=4,  # Batch processing for speed
    device="cuda",
    use_tta=False  # Set True for higher accuracy (slower)
)
```

### 6. Preprocessing & Postprocessing

```python
from pdfnet import preprocess_image, preprocess_depth, postprocess_prediction
import cv2
import torch

# Load and preprocess image
image = cv2.imread("input.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_tensor = preprocess_image(image_rgb, input_size=1024, normalize=True)
# Returns: torch.Tensor (1, 3, 1024, 1024)

# Preprocess depth map (if you have one)
depth = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE)
depth_tensor = preprocess_depth(depth, input_size=1024)
# Returns: torch.Tensor (1, 1, 1024, 1024)

# After model inference, postprocess prediction
with torch.no_grad():
    outputs = model.inference(img_tensor.cuda(), depth_tensor.cuda())
    prediction = outputs[0]  # Get first output

# Postprocess back to original size
mask = postprocess_prediction(prediction, original_size=(image.shape[1], image.shape[0]))
# Returns: numpy array (H, W) with values [0, 1]
```

### 7. Depth Map Generation

```python
from pdfnet import load_moge_depth_model, generate_depth, create_placeholder_depth
import cv2

# Load depth estimation model
depth_model = load_moge_depth_model()

# Generate depth map from image
image = cv2.imread("input.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

depth = generate_depth(
    depth_model=depth_model,
    image=image_rgb,
    device="cuda"
)
# Returns: numpy array (H, W) with values 0-255

# Save depth map
cv2.imwrite("depth.png", depth)

# Or create a placeholder depth (radial gradient)
placeholder = create_placeholder_depth(height=1024, width=1024)
```

### 8. Save Predictions

```python
from pdfnet import save_prediction
import numpy as np

# Save as grayscale (0-255)
mask = np.random.rand(1024, 1024)  # Example mask
save_prediction(mask, "output.png")

# Save as binary mask
save_prediction(mask, "binary_mask.png", as_binary=True, threshold=0.5)
```

---

## Class-Based API

The `PDFNetInference` class provides a convenient wrapper.

### Basic Usage

```python
from pdfnet import PDFNetInference
from pdfnet.config import PDFNetConfig

# Option 1: Use default configuration
engine = PDFNetInference()

# Option 2: Custom configuration
config = PDFNetConfig()
config.inference.checkpoint_path = "custom_checkpoint.pth"
config.inference.use_tta = True
config.inference.batch_size = 8
config.device = "cuda"

engine = PDFNetInference(config)
```

### Available Methods

```python
# Single image inference
mask = engine.predict_single("input.jpg")
mask, depth = engine.predict_single("input.jpg", return_depth=True)

# Test-time augmentation
mask = engine.predict_with_tta("input.jpg")

# Batch processing (true batching on GPU)
masks = engine.predict_batch(
    ["img1.jpg", "img2.jpg", "img3.jpg"],
    batch_size=4
)

# Directory processing
engine.predict_directory(
    input_dir="images/",
    output_dir="results/",
    batch_size=4,
    use_tta=False
)

# Generate depth map
depth = engine.generate_depth(image_array)
```

---

## Training API

### Train a Model

```python
from pdfnet.config import PDFNetConfig
from pdfnet.train import train_from_config

# Create configuration
config = PDFNetConfig()
config.model.name = "PDFNet_swinB"
config.training.epochs = 100
config.training.batch_size = 1
config.training.optimizer.lr = 1e-4
config.data.root_path = "DATA/DIS-DATA"
config.device = "cuda"

# Start training
train_from_config(
    config=config,
    resume="",  # Optional: path to checkpoint to resume
    finetune=""  # Optional: path to checkpoint to finetune
)
```

### Build Models and Datasets

```python
from pdfnet import build_model, build_dataset, DISDataset
from pdfnet.config import PDFNetConfig

# Create config-like object with required attributes
class Args:
    model = "PDFNet_swinB"
    device = "cuda"
    emb = 128
    drop_path = 0.1
    back_bone_channels_stage1 = 128
    back_bone_channels_stage2 = 256
    back_bone_channels_stage3 = 512
    back_bone_channels_stage4 = 1024
    DEBUG = False

args = Args()

# Build model
model, model_name = build_model(args)
print(f"Built model: {model_name}")

# Build dataset
dataset = build_dataset(is_train=True, args=args)
print(f"Dataset size: {len(dataset)}")
```

---

## Configuration Management

### Create and Load Configurations

```python
from pdfnet.config import PDFNetConfig
from pathlib import Path

# Create default configuration
config = PDFNetConfig()

# Modify configuration
config.model.name = "PDFNet_swinB"
config.training.epochs = 50
config.training.batch_size = 2
config.training.optimizer.lr = 5e-5
config.data.root_path = Path("DATA/DIS-DATA")
config.device = "cuda"

# Save to YAML
config.save("my_config.yaml")

# Load from YAML
loaded_config = PDFNetConfig.load("my_config.yaml")

# Access nested configs
print(f"Learning rate: {loaded_config.training.optimizer.lr}")
print(f"Model: {loaded_config.model.name}")
print(f"Epochs: {loaded_config.training.epochs}")
```

### Configuration Structure

```python
from pdfnet.config import PDFNetConfig

config = PDFNetConfig()

# Model configuration
config.model.name = "PDFNet_swinB"  # or PDFNet_swinL, PDFNet_swinT
config.model.input_size = 1024
config.model.drop_path = 0.1

# Training configuration
config.training.batch_size = 1
config.training.epochs = 100
config.training.num_workers = 8
config.training.seed = 0
config.training.eval_metric = "F1"  # or "MAE", "IoU"

# Optimizer
config.training.optimizer.lr = 1e-4
config.training.optimizer.weight_decay = 0.05
config.training.optimizer.type = "adamw"

# Scheduler
config.training.scheduler.type = "cosine"
config.training.scheduler.warmup_epochs = 5
config.training.scheduler.min_lr = 1e-5

# Data configuration
config.data.root_path = Path("DATA/DIS-DATA")
config.data.dataset = "DIS"
config.data.input_size = 1024

# Inference configuration
config.inference.checkpoint_path = Path("checkpoints/PDFNet_Best.pth")
config.inference.use_tta = False
config.inference.batch_size = 1
config.inference.use_moge = True

# Output configuration
config.output.save_dir = Path("runs")
config.output.checkpoint_dir = Path("checkpoints")
config.output.log_dir = Path("logs")
```

---

## Complete Examples

### Example 1: Simple Segmentation Script

```python
#!/usr/bin/env python3
"""Simple PDFNet segmentation script."""

import cv2
from pathlib import Path
from pdfnet import PDFNetInference

def main():
    # Initialize engine
    engine = PDFNetInference()

    # Process single image
    input_path = "input.jpg"
    output_path = "output.png"

    print(f"Processing {input_path}...")
    mask = engine.predict_single(input_path)

    # Save result
    cv2.imwrite(output_path, (mask * 255).astype('uint8'))
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
```

### Example 2: Batch Processing with Progress

```python
#!/usr/bin/env python3
"""Batch process images with PDFNet."""

import cv2
from pathlib import Path
from pdfnet import PDFNetInference

def main():
    # Setup
    input_dir = Path("input_images")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Find all images
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")

    # Initialize engine
    engine = PDFNetInference()

    # Process in batches
    masks = engine.predict_batch(
        [str(f) for f in image_files],
        batch_size=4,
        progress=True
    )

    # Save results
    for img_file, mask in zip(image_files, masks):
        output_path = output_dir / f"{img_file.stem}_mask.png"
        cv2.imwrite(str(output_path), (mask * 255).astype('uint8'))

    print(f"Saved {len(masks)} results to {output_dir}")

if __name__ == "__main__":
    main()
```

### Example 3: Custom Configuration

```python
#!/usr/bin/env python3
"""PDFNet inference with custom configuration."""

from pathlib import Path
from pdfnet import PDFNetInference
from pdfnet.config import PDFNetConfig

def main():
    # Create custom configuration
    config = PDFNetConfig()
    config.inference.checkpoint_path = Path("custom_model.pth")
    config.inference.use_tta = True  # Enable TTA
    config.inference.batch_size = 8
    config.device = "cuda"

    # Initialize with custom config
    engine = PDFNetInference(config)

    # Process directory
    engine.predict_directory(
        input_dir="images/",
        output_dir="results/",
        batch_size=8,
        use_tta=True
    )

    print("Processing complete!")

if __name__ == "__main__":
    main()
```

### Example 4: Advanced - Manual Pipeline

```python
#!/usr/bin/env python3
"""Advanced PDFNet usage with manual control."""

import cv2
import torch
import numpy as np
from pdfnet import (
    load_pdfnet_model,
    load_moge_depth_model,
    preprocess_image,
    preprocess_depth,
    postprocess_prediction,
    generate_depth
)

def main():
    # Load models
    print("Loading models...")
    model = load_pdfnet_model("checkpoints/PDFNet_Best.pth", device="cuda")
    depth_model = load_moge_depth_model(device="cuda")

    # Load image
    image = cv2.imread("input.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Generate depth
    print("Generating depth map...")
    depth = generate_depth(depth_model, image_rgb, device="cuda")

    # Preprocess
    img_tensor = preprocess_image(image_rgb, input_size=1024).cuda()
    depth_tensor = preprocess_depth(depth, input_size=1024).cuda()

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model.inference(img_tensor, depth_tensor)
        prediction = outputs[0]  # Get first output (main prediction)

    # Postprocess
    mask = postprocess_prediction(prediction, original_size=(w, h))

    # Save results
    cv2.imwrite("output_mask.png", (mask * 255).astype(np.uint8))
    cv2.imwrite("depth_map.png", depth)

    print("Done!")

if __name__ == "__main__":
    main()
```

### Example 5: Integration with Other Libraries

```python
#!/usr/bin/env python3
"""Integrate PDFNet with other vision tasks."""

import cv2
import numpy as np
from pdfnet import PDFNetInference

def apply_mask_to_image(image, mask, background=(0, 255, 0)):
    """Apply segmentation mask to image with colored background."""
    # Convert mask to 3-channel
    mask_3ch = np.stack([mask] * 3, axis=2)

    # Create background
    bg = np.full_like(image, background, dtype=np.uint8)

    # Blend
    result = (image * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
    return result

def main():
    # Initialize PDFNet
    engine = PDFNetInference()

    # Load image
    image = cv2.imread("input.jpg")

    # Get segmentation mask
    mask = engine.predict_single("input.jpg")

    # Apply mask
    result = apply_mask_to_image(image, mask)

    # Save
    cv2.imwrite("result.jpg", result)

    # Also save just the foreground
    foreground = (image * np.stack([mask] * 3, axis=2)).astype(np.uint8)
    cv2.imwrite("foreground.png", foreground)

if __name__ == "__main__":
    main()
```

---

## API Reference Summary

### Inference Functions
- `run_inference(model, image, depth_model, ...)` - Single image inference
- `run_batch_inference(model, images, batch_size, ...)` - Batch inference with true batching
- `run_inference_with_tta(model, image, ...)` - Inference with test-time augmentation
- `process_directory(model, input_dir, output_dir, ...)` - Process entire directory

### Preprocessing Functions
- `preprocess_image(image, input_size, normalize)` - Prepare image for model
- `preprocess_depth(depth, input_size)` - Prepare depth map for model
- `postprocess_prediction(prediction, original_size)` - Convert model output to mask

### Model Loading Functions
- `load_pdfnet_model(checkpoint_path, model_name, device, strict)` - Load PDFNet model
- `load_moge_depth_model(checkpoint_path, device)` - Load MoGe depth model
- `get_model_info(model)` - Get model statistics

### Utility Functions
- `generate_depth(depth_model, image, device)` - Generate depth map
- `create_placeholder_depth(height, width)` - Create simple depth map
- `save_prediction(prediction, output_path, as_binary, threshold)` - Save mask to file

### Class-Based API
- `PDFNetInference(config, checkpoint_path)` - Inference engine class
  - `.predict_single(image, depth, return_depth)` - Single image
  - `.predict_batch(images, batch_size, use_tta, progress)` - Batch processing
  - `.predict_with_tta(image, flips)` - With TTA
  - `.predict_directory(input_dir, output_dir, ...)` - Directory processing
  - `.generate_depth(image)` - Generate depth map

### Training Functions
- `train_from_config(config, resume, finetune)` - Train with type-safe config
- `build_model(args)` - Build PDFNet model
- `build_dataset(is_train, args)` - Build DIS dataset
- `DISDataset` - PyTorch Dataset class

### Configuration
- `PDFNetConfig()` - Main configuration dataclass
  - `.model` - Model architecture settings
  - `.training` - Training hyperparameters
  - `.data` - Dataset configuration
  - `.inference` - Inference settings
  - `.output` - Output directories
  - `.save(path)` - Save to YAML
  - `.load(path)` - Load from YAML

---

## Tips and Best Practices

1. **Use batch processing** for multiple images - it's much faster than processing one by one
2. **Enable TTA** for critical applications where accuracy matters more than speed
3. **Use MoGe depth estimation** (enabled by default) for best results
4. **Adjust batch_size** based on your GPU memory (4-8 is good for most GPUs)
5. **Use the class-based API** (`PDFNetInference`) for simplicity
6. **Use the functional API** when you need fine-grained control
7. **Save configurations** to YAML files for reproducibility

---

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended for fast inference)
- See `pyproject.toml` for full dependency list

## Installation Issues?

If you encounter issues, try:
```bash
# Force reinstall
pip install --force-reinstall git+https://github.com/OpsiClear/PDFNet_Moge.git

# Or install without dependencies first, then separately
pip install --no-deps git+https://github.com/OpsiClear/PDFNet_Moge.git
pip install torch torchvision opencv-python numpy ...
```

---

## More Information

- **CLI Guide**: See `CLAUDE.md` for command-line usage
- **Installation Guide**: See `INSTALL.md` for detailed installation instructions
- **Paper**: [arXiv:2503.06100](https://arxiv.org/abs/2503.06100)
- **GitHub**: [OpsiClear/PDFNet_Moge](https://github.com/OpsiClear/PDFNet_Moge)
