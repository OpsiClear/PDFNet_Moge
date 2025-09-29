#!/usr/bin/env python3
"""
Download models for PDFNet.

This script downloads the MoGe models and Swin Transformer weights required for PDFNet.
"""

import sys
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict

import tyro

try:
    from huggingface_hub import snapshot_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class MoGeModelConfig(TypedDict):
    """Configuration for a MoGe model to download."""

    name: str
    repo_id: str
    description: str
    version: str
    metric_scale: bool
    normal_map: bool
    params: str


class ModelConfig(TypedDict):
    """Configuration for other models to download."""

    name: str
    url: str
    filename: str
    description: str


@dataclass
class DownloadConfig:
    """Configuration for downloading models."""

    models_dir: str = "checkpoints"
    """Directory to store model checkpoints"""

    skip_existing: bool = True
    """Skip download if files already exist"""

    download_moge_models: bool = True
    """Download MoGe pretrained models"""

    download_swin_weights: bool = True
    """Download Swin Transformer weights"""

    download_pdfnet_checkpoint: bool = True
    """Download pre-trained PDFNet checkpoint"""

    moge_models: str = "recommended"
    """Which MoGe models to download: recommended, all, or comma-separated model names"""


def create_directory(directory: Path) -> None:
    """Create directory if it doesn't exist.

    :param directory: Path to the directory
    """
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {directory}")


def download_file(url: str, output_path: Path, description: str = "") -> None:
    """Download a file from URL.

    :param url: URL to download from
    :param output_path: Local path to save the file
    :param description: Description of what's being downloaded
    """
    try:
        print(f"Downloading {description or output_path.name}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded: {output_path.name}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)


def download_moge_model(
    model_config: MoGeModelConfig, models_dir: Path, skip_existing: bool = False
) -> None:
    """Download a MoGe model from Hugging Face.

    :param model_config: Configuration for the model to download
    :param models_dir: Directory to store models
    :param skip_existing: Skip download if model already exists
    """
    if not HF_AVAILABLE:
        print(
            f"Warning: Skipping {model_config['name']} (huggingface_hub not available)"
        )
        print("Install with: pip install huggingface_hub")
        return

    moge_models_dir = models_dir / "moge"
    moge_models_dir.mkdir(parents=True, exist_ok=True)
    model_path = moge_models_dir / model_config["name"]

    if skip_existing and model_path.exists() and any(model_path.iterdir()):
        print(f"Skipping {model_config['name']} (already exists)")
        return

    try:
        print(
            f"Downloading {model_config['name']} ({model_config['params']}) - {model_config['description']}"
        )
        snapshot_download(
            repo_id=model_config["repo_id"],
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )
        print(f"[+] Downloaded {model_config['name']}")
    except Exception as e:
        print(f"Error downloading {model_config['name']}: {e}")
        if model_path.exists():
            import shutil

            shutil.rmtree(model_path)


def get_moge_models() -> list[MoGeModelConfig]:
    """Get the list of available MoGe models.

    :return: List of MoGe model configurations
    """
    return [
        {
            "name": "moge-2-vitl-normal",
            "repo_id": "Ruicheng/moge-2-vitl-normal",
            "description": "MoGe-2 ViT-Large with metric scale and normal maps (RECOMMENDED for PDFNet)",
            "version": "v2",
            "metric_scale": True,
            "normal_map": True,
            "params": "331M",
        },
        {
            "name": "moge-2-vitb-normal",
            "repo_id": "Ruicheng/moge-2-vitb-normal",
            "description": "MoGe-2 ViT-Base with metric scale and normal maps",
            "version": "v2",
            "metric_scale": True,
            "normal_map": True,
            "params": "104M",
        },
        {
            "name": "moge-2-vits-normal",
            "repo_id": "Ruicheng/moge-2-vits-normal",
            "description": "MoGe-2 ViT-Small with metric scale and normal maps",
            "version": "v2",
            "metric_scale": True,
            "normal_map": True,
            "params": "35M",
        },
    ]


def get_other_models() -> list[ModelConfig]:
    """Get the list of other required models.

    :return: List of other model configurations
    """
    return [
        {
            "name": "swin_base_patch4_window12_384_22k",
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
            "filename": "swin_base_patch4_window12_384_22k.pth",
            "description": "Swin Transformer Base weights (required for PDFNet backbone)",
        }
    ]


def get_pdfnet_checkpoints() -> list[ModelConfig]:
    """Get the list of PDFNet pre-trained checkpoints.

    :return: List of PDFNet checkpoint configurations
    """
    return [
        {
            "name": "PDFNet_DIS5K",
            "url": "https://drive.google.com/uc?id=1dqkFVR4TElSRFNHhu6er45OQkoHhJsZz",
            "filename": "PDFNet_Best.pth",
            "description": "PDFNet trained on DIS-5K dataset (RECOMMENDED)",
        }
    ]


def download_from_google_drive(
    file_id: str, output_path: Path, description: str = ""
) -> bool:
    """Download a file from Google Drive.

    :param file_id: Google Drive file ID
    :param output_path: Local path to save the file
    :param description: Description of what's being downloaded
    :return: True if successful, False otherwise
    """
    try:
        import gdown

        print(f"Downloading {description or output_path.name} from Google Drive...")

        # Extract file ID from URL if needed
        if "drive.google.com" in file_id:
            # Extract ID from Google Drive URL
            if "/folders/" in file_id:
                # This is a folder, we need to handle it differently
                print(
                    "Warning: Google Drive folder detected. Please download manually."
                )
                print(f"URL: {file_id}")
                return False
            else:
                # Extract file ID from share URL
                import re

                match = re.search(r"/d/([a-zA-Z0-9-_]+)", file_id)
                if match:
                    file_id = match.group(1)

        # Download with gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)
        print(f"[+] Downloaded {output_path.name}")
        return True

    except ImportError:
        print("Warning: gdown not available. Install with: pip install gdown")
        print(
            "Please manually download from: https://drive.google.com/drive/folders/1dqkFVR4TElSRFNHhu6er45OQkoHhJsZz"
        )
        return False
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        print(
            "Please manually download from: https://drive.google.com/drive/folders/1dqkFVR4TElSRFNHhu6er45OQkoHhJsZz"
        )
        return False


def create_utility_dirs(project_root: Path) -> None:
    """Create utility directories for PDFNet.

    :param project_root: Path to the project root directory
    """
    print("\nSetting up utility directories...")

    # Create additional utility directories
    for dir_name in ["runs", "valid_sample"]:
        (project_root / dir_name).mkdir(exist_ok=True)

    print("[+] Created utility directories")


def main(config: DownloadConfig) -> None:
    """Main function to download and setup models.

    :param config: Download configuration
    """
    project_root = Path(__file__).parent
    models_dir = project_root / config.models_dir

    print("=" * 60)
    print("PDFNet Model Setup")
    print("=" * 60)
    print(f"Models directory: {models_dir}")
    print()

    # Create directories
    create_directory(models_dir)

    # Download MoGe models
    if config.download_moge_models:
        print("\n[*] Downloading MoGe models...")

        available_models = get_moge_models()

        # Select models based on config
        if config.moge_models == "recommended":
            # Only download the model used in PDFNet
            selected_models = [
                m for m in available_models if m["name"] == "moge-2-vitl-normal"
            ]
        elif config.moge_models == "all":
            selected_models = available_models
        else:
            # Allow specifying model names explicitly
            model_names = [name.strip() for name in config.moge_models.split(",")]
            selected_models = [m for m in available_models if m["name"] in model_names]

        if not selected_models:
            print(f"Warning: No models found for: {config.moge_models}")
        else:
            for model in selected_models:
                download_moge_model(model, models_dir, config.skip_existing)

    # Download Swin weights
    if config.download_swin_weights:
        print("\n[*] Downloading Swin Transformer weights...")

        other_models = get_other_models()
        for model in other_models:
            output_path = models_dir / model["filename"]

            if config.skip_existing and output_path.exists():
                print(f"Skipping {model['name']} (already exists)")
                continue

            download_file(model["url"], output_path, model["description"])
            print(f"[+] Downloaded {model['name']}")

    # Download PDFNet checkpoints
    if config.download_pdfnet_checkpoint:
        print("\n[*] Downloading PDFNet pre-trained checkpoint...")

        pdfnet_checkpoints = get_pdfnet_checkpoints()
        for checkpoint in pdfnet_checkpoints:
            output_path = models_dir / checkpoint["filename"]

            if config.skip_existing and output_path.exists():
                print(f"Skipping {checkpoint['name']} (already exists)")
                continue

            # The Google Drive link is a folder, so we need to provide manual instructions
            print(f"\n{checkpoint['description']}")
            print("=" * 60)
            print("MANUAL DOWNLOAD REQUIRED:")
            print(
                "1. Go to: https://drive.google.com/drive/folders/1dqkFVR4TElSRFNHhu6er45OQkoHhJsZz"
            )
            print("2. Download the PDFNet checkpoint file")
            print(f"3. Save it as: {output_path}")
            print("=" * 60)

            # Try automatic download if gdown is available
            success = download_from_google_drive(
                "1dqkFVR4TElSRFNHhu6er45OQkoHhJsZz",
                output_path,
                checkpoint["description"],
            )
            if not success:
                print(
                    f"Note: You'll need to manually download {checkpoint['filename']} for inference"
                )

    # Create utility directories
    create_utility_dirs(project_root)

    print("\n" + "=" * 60)
    print("[SUCCESS] Model setup complete!")
    print("=" * 60)
    print(f"Models: {models_dir}")

    if config.download_moge_models:
        print(
            "MoGe models: Use MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')"
        )

    # Check if PDFNet checkpoint exists
    pdfnet_checkpoint = models_dir / "PDFNet_Best.pth"
    if pdfnet_checkpoint.exists():
        print("PDFNet checkpoint: Ready for inference!")
        print("\nNext steps:")
        print("1. Try the demo: jupyter notebook demo.ipynb")
        print("2. Or use: python apply_pdfnet.py --input image.jpg --output result.png")
    else:
        print("PDFNet checkpoint: Manual download required (see instructions above)")
        print("\nNext steps:")
        print("1. Download PDFNet_Best.pth manually from Google Drive")
        print("2. Prepare your dataset in DATA/DIS-DATA/ structure")
        print("3. Run MoGe/Depth-prepare.ipynb to generate depth maps")
        print("4. Start training with: pdfnet-train --data_path DATA/DIS-DATA")
    print()


if __name__ == "__main__":
    tyro.cli(main)
