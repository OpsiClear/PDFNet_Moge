#!/usr/bin/env python3
"""
PDFNet CLI using tyro for type-safe command-line interface.

Usage:
    uv run pdfnet.py train --model.name PDFNet_swinB --training.epochs 100
    uv run pdfnet.py infer --input image.jpg --output result.png
    uv run pdfnet.py evaluate --pred-dir results/ --gt-dir DATA/
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

# Fix encoding issues on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import tyro
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pdfnet.config_types import PDFNetConfig


@dataclass
class TrainCommand:
    """Train a PDFNet model."""

    config: PDFNetConfig = field(default_factory=PDFNetConfig)
    """Training configuration."""

    config_file: Path | None = None
    """Load configuration from YAML file."""

    resume: Path | None = None
    """Resume training from checkpoint."""

    def run(self) -> None:
        """Execute training."""
        from src.pdfnet.train import train_main
        from src.pdfnet.args import get_args_parser
        import argparse

        # Load config from file if provided
        if self.config_file and self.config_file.exists():
            self.config = PDFNetConfig.load(self.config_file)

        # Create legacy args for compatibility
        parser = argparse.ArgumentParser(parents=[get_args_parser()])
        args = parser.parse_args(args=[])

        # Update args from typed config
        args.model = self.config.model.name
        args.batch_size = self.config.training.batch_size
        args.epochs = self.config.training.epochs
        args.lr = self.config.training.optimizer.lr
        args.weight_decay = self.config.training.optimizer.weight_decay
        args.data_path = str(self.config.data.root_path)
        args.input_size = self.config.data.input_size
        args.device = self.config.device
        args.num_workers = self.config.training.num_workers
        args.seed = self.config.training.seed
        args.eval_metric = self.config.training.eval_metric
        args.checkpoints_save_path = str(self.config.output.checkpoint_dir)
        args.output_dir = str(self.config.output.save_dir)
        args.DEBUG = self.config.debug

        if self.resume:
            args.resume = str(self.resume)

        # Print configuration
        print("Training Configuration:")
        print(f"  Model: {self.config.model.name}")
        print(f"  Epochs: {self.config.training.epochs}")
        print(f"  Batch Size: {self.config.training.batch_size}")
        print(f"  Learning Rate: {self.config.training.optimizer.lr}")
        print(f"  Dataset: {self.config.data.dataset} @ {self.config.data.root_path}")
        print(f"  Device: {self.config.device}")

        # Run training
        train_main(args)


@dataclass
class InferCommand:
    """Run inference on images."""

    input: Path
    """Input image or directory path."""

    output: Path | None = None
    """Output path for results (auto-generated if not specified)."""

    checkpoint: Path | None = Path("checkpoints/PDFNet_Best.pth")
    """Model checkpoint path."""

    batch_size: int = 1
    """Batch size for directory processing."""

    use_tta: bool = False
    """Enable test-time augmentation."""

    device: Literal["cuda", "cpu", "auto"] = "auto"
    """Device for inference."""

    use_moge: bool = True
    """Use MoGe for depth estimation."""

    visualize: bool = False
    """Visualize results (for single images)."""

    def run(self) -> None:
        """Execute inference."""
        from src.pdfnet.inference import PDFNetInference
        from src.pdfnet.config_types import PDFNetConfig
        import cv2
        import numpy as np

        # Setup device
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        # Create config
        config = PDFNetConfig()
        config.inference.checkpoint_path = self.checkpoint
        config.inference.use_tta = self.use_tta
        config.inference.use_moge = self.use_moge
        config.inference.batch_size = self.batch_size
        config.device = device

        # Initialize inference engine
        print(f"Loading model from: {self.checkpoint}")
        engine = PDFNetInference(config)

        # Determine output path
        if self.output is None:
            if self.input.is_file():
                self.output = Path(f"{self.input.stem}_result.png")
            else:
                self.output = Path("results")

        # Process based on input type
        if self.input.is_file():
            print(f"Processing image: {self.input}")
            result = engine.predict(str(self.input), use_tta=self.use_tta)

            # Save result
            result_img = (result * 255).astype(np.uint8)
            cv2.imwrite(str(self.output), result_img)
            print(f"Result saved to: {self.output}")

            # Visualize if requested
            if self.visualize:
                import matplotlib.pyplot as plt
                img = cv2.imread(str(self.input))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].imshow(img)
                axes[0].set_title("Original")
                axes[0].axis('off')
                axes[1].imshow(result, cmap='gray')
                axes[1].set_title("Segmentation")
                axes[1].axis('off')
                plt.tight_layout()
                plt.show()

        elif self.input.is_dir():
            print(f"Processing directory: {self.input}")
            engine.predict_directory(
                str(self.input),
                str(self.output),
                use_tta=self.use_tta,
                batch_size=self.batch_size
            )

        else:
            raise ValueError(f"Invalid input path: {self.input}")


@dataclass
class TestCommand:
    """Test model on evaluation datasets."""

    checkpoint: Path = Path("checkpoints/PDFNet_Best.pth")
    """Model checkpoint path."""

    data_path: Path = Path("DATA/DIS-DATA")
    """Dataset root path."""

    output_dir: Path = Path("results")
    """Output directory for results."""

    datasets: list[str] = field(default_factory=lambda: ["DIS-VD", "DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"])
    """Datasets to test on."""

    batch_size: int = 1
    """Batch size for testing."""

    use_tta: bool = False
    """Enable test-time augmentation."""

    compute_metrics: bool = True
    """Compute evaluation metrics."""

    device: Literal["cuda", "cpu", "auto"] = "auto"
    """Device for testing."""

    def run(self) -> None:
        """Execute testing."""
        from src.pdfnet.metric_tools.Test import test_pdfnet
        from src.pdfnet.args import get_args_parser
        import argparse

        # Create legacy args
        parser = argparse.ArgumentParser(parents=[get_args_parser()])
        args = parser.parse_args(args=[])

        # Update from typed config
        args.checkpoint_path = str(self.checkpoint)
        args.data_path = str(self.data_path)
        args.output_dir = str(self.output_dir)
        args.test_batch_size = self.batch_size
        args.use_tta = self.use_tta
        args.compute_metrics = self.compute_metrics
        args.input_size = 1024

        if self.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            args.device = self.device

        # Set test datasets
        args.datasets = self.datasets

        print("Testing Configuration:")
        print(f"  Checkpoint: {self.checkpoint}")
        print(f"  Datasets: {', '.join(self.datasets)}")
        print(f"  Output: {self.output_dir}")
        print(f"  TTA: {self.use_tta}")
        print(f"  Device: {args.device}")

        # Run testing
        test_pdfnet(args)


@dataclass
class EvaluateCommand:
    """Evaluate predictions against ground truth."""

    pred_dir: Path
    """Directory containing predictions."""

    gt_dir: Path | None = None
    """Directory containing ground truth."""

    output_dir: Path | None = None
    """Output directory for results."""

    datasets: list[str] | None = None
    """Dataset names to evaluate."""

    n_jobs: int = 12
    """Number of parallel jobs."""

    def run(self) -> None:
        """Execute evaluation."""
        from src.pdfnet.metric_tools.soc_metrics import compute_metrics

        print(f"Evaluating predictions in: {self.pred_dir}")

        # Build ground truth mapping
        gt_dir_dict = None
        if self.gt_dir:
            gt_dir_dict = {}
            if self.datasets:
                for dataset in self.datasets:
                    dataset_gt = self.gt_dir / dataset / "masks"
                    if dataset_gt.exists():
                        gt_dir_dict[dataset] = str(dataset_gt)
                        print(f"  Found GT for {dataset}: {dataset_gt}")
            else:
                # Auto-detect datasets
                for subdir in self.pred_dir.iterdir():
                    if subdir.is_dir():
                        dataset_name = subdir.name
                        dataset_gt = self.gt_dir / dataset_name / "masks"
                        if dataset_gt.exists():
                            gt_dir_dict[dataset_name] = str(dataset_gt)
                            print(f"  Found GT for {dataset_name}: {dataset_gt}")

        # Run evaluation
        output_dir = self.output_dir or self.pred_dir
        compute_metrics(
            pred_dir=str(self.pred_dir),
            gt_dir_dict=gt_dir_dict,
            output_dir=str(output_dir),
            n_jobs=self.n_jobs
        )


@dataclass
class DownloadCommand:
    """Download model weights and show dataset instructions."""

    weights: bool = True
    """Download model weights."""

    dataset_info: bool = False
    """Show dataset download instructions."""

    def run(self) -> None:
        """Execute download."""
        if self.weights:
            print("Downloading model weights...")
            import subprocess
            result = subprocess.run(["uv", "run", "download.py"], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Download completed successfully!")
                print(result.stdout)
            else:
                print("❌ Download failed!")
                print(result.stderr)

        if self.dataset_info:
            print("\n📦 Dataset Download Instructions:")
            print("=" * 50)
            print("\nDIS-5K Dataset:")
            print("  Website: https://github.com/xuebinqin/DIS")
            print("  Place in: DATA/DIS-DATA/")
            print("\nExpected structure:")
            print("  DATA/")
            print("  └── DIS-DATA/")
            print("      ├── DIS-TR/")
            print("      ├── DIS-VD/")
            print("      ├── DIS-TE1/")
            print("      ├── DIS-TE2/")
            print("      ├── DIS-TE3/")
            print("      └── DIS-TE4/")


@dataclass
class ConfigCommand:
    """Configuration management utilities."""

    action: Literal["show", "create", "validate"] = "show"
    """Action to perform."""

    config_file: Path | None = None
    """Configuration file path."""

    output: Path | None = None
    """Output path for created config."""

    def run(self) -> None:
        """Execute configuration command."""
        if self.action == "show":
            if self.config_file and self.config_file.exists():
                config = PDFNetConfig.load(self.config_file)
                print(f"Configuration from {self.config_file}:")
            else:
                config = PDFNetConfig()
                print("Default configuration:")

            import yaml
            from dataclasses import asdict
            print(yaml.dump(asdict(config), default_flow_style=False))

        elif self.action == "create":
            output_path = self.output or Path("config/custom.yaml")
            config = PDFNetConfig()
            config.save(output_path)
            print(f"✅ Configuration saved to: {output_path}")

        elif self.action == "validate":
            if not self.config_file or not self.config_file.exists():
                print("❌ Config file not found!")
                return

            try:
                config = PDFNetConfig.load(self.config_file)
                print(f"✅ Configuration is valid: {self.config_file}")

                # Check paths
                missing_paths = []
                if config.model.pretrained_swin and not config.model.pretrained_swin.exists():
                    missing_paths.append(f"  ⚠️  Pretrained weights: {config.model.pretrained_swin}")
                if not config.data.root_path.exists():
                    missing_paths.append(f"  ⚠️  Data root: {config.data.root_path}")

                if missing_paths:
                    print("\nMissing paths:")
                    for path in missing_paths:
                        print(path)

            except Exception as e:
                print(f"❌ Invalid configuration: {e}")


def main() -> None:
    """Main CLI entry point using tyro."""

    # Create CLI with subcommands
    cli = tyro.cli(
        tyro.extras.subcommand_cli_from_dict({
            "train": TrainCommand,
            "infer": InferCommand,
            "test": TestCommand,
            "evaluate": EvaluateCommand,
            "download": DownloadCommand,
            "config": ConfigCommand,
        }),
        description="PDFNet - Deep Learning for Dichotomous Image Segmentation",
        version="1.0.0",
    )

    # Execute the selected command
    cli.run()


if __name__ == "__main__":
    main()