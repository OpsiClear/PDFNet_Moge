#!/usr/bin/env python3
"""
PDFNet CLI using tyro for type-safe command-line interface.

Usage:
    uv run pdfnet.py train --model.name PDFNet_swinB --training.epochs 100
    uv run pdfnet.py infer --input image.jpg --output result.png
    uv run pdfnet.py evaluate --pred-dir results/ --gt-dir DATA/
"""


import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

# Fix encoding issues on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import tyro
import torch
import logging

from pdfnet.config import PDFNetConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainCommand:
    """Train a PDFNet model using type-safe configuration."""

    config_file: Path | None = None
    """Load configuration from YAML file (uses defaults if not specified)."""

    resume: Path | None = None
    """Resume training from checkpoint."""

    def run(self) -> None:
        """Execute training with type-safe configuration."""
        try:
            from pdfnet.train import train_from_config
        except ImportError:
            logger.error("Training module not available")
            return

        # Load configuration
        if self.config_file and self.config_file.exists():
            config = PDFNetConfig.load(self.config_file)
            logger.info("=" * 60)
            logger.info(f"Training with config from: {self.config_file}")
            logger.info("=" * 60)
        else:
            config = PDFNetConfig()
            logger.info("=" * 60)
            logger.info("Training with default configuration")
            logger.info("=" * 60)
            logger.info("Use --config-file to specify a custom configuration")
            logger.info("=" * 60)

        # Print config summary
        logger.info(f"Model:         {config.model.name}")
        logger.info(f"Dataset:       {config.data.root_path}")
        logger.info(f"Epochs:        {config.training.epochs}")
        logger.info(f"Batch size:    {config.training.batch_size}")
        logger.info(f"Learning rate: {config.training.optimizer.lr}")
        logger.info(f"Device:        {config.device}")
        logger.info(f"Checkpoints:   {config.output.checkpoint_dir}")
        if self.resume:
            logger.info(f"Resume from:   {self.resume}")
        logger.info("=" * 60)

        # Run training with type-safe config
        train_from_config(
            config=config,
            resume=str(self.resume) if self.resume else '',
            finetune=''
        )


@dataclass
class InferCommand:
    """Run inference on images or directories."""

    input: Path
    """Input image or directory path."""

    output: Path | None = None
    """Output path for results (auto-generated if not specified)."""

    checkpoint: Path | None = Path("checkpoints/PDFNet_Best.pth")
    """Model checkpoint path."""

    batch_size: int = 4
    """Number of images to process in each batch (1-32)."""

    use_tta: bool = False
    """Enable test-time augmentation for better accuracy (slower)."""

    device: Literal["cuda", "cpu", "auto"] = "auto"
    """Device for inference (auto selects CUDA if available)."""

    use_moge: bool = True
    """Use MoGe depth estimation model for better results."""

    visualize: bool = False
    """Display visualization for single image results."""

    def run(self) -> None:
        """Execute inference."""
        from pdfnet.inference import PDFNetInference
        from pdfnet.config import PDFNetConfig
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
        logger.info(f"Loading model from: {self.checkpoint}")
        engine = PDFNetInference(config)

        # Determine output path
        if self.output is None:
            if self.input.is_file():
                self.output = Path(f"{self.input.stem}_result.png")
            else:
                self.output = Path("results")

        # Process based on input type
        if self.input.is_file():
            logger.info(f"Processing image: {self.input}")
            result = engine.predict(str(self.input), use_tta=self.use_tta)

            # Save result
            result_img = (result * 255).astype(np.uint8)
            cv2.imwrite(str(self.output), result_img)
            logger.info(f"Result saved to: {self.output}")

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
            logger.info(f"Processing directory: {self.input}")
            engine.predict_directory(
                str(self.input),
                str(self.output),
                use_tta=self.use_tta,
                batch_size=self.batch_size
            )

        else:
            raise ValueError(f"Invalid input path: {self.input}")


@dataclass
class BenchmarkCommand:
    """Benchmark model performance on standard DIS datasets."""

    checkpoint: Path = Path("checkpoints/PDFNet_Best.pth")
    """Path to model checkpoint file."""

    data_path: Path = Path("DATA/DIS-DATA")
    """Path to DIS dataset root directory."""

    output_dir: Path = Path("results")
    """Output directory for benchmark results."""

    datasets: list[str] | None = None
    """Dataset names to benchmark (default: all DIS test sets: DIS-TE1,TE2,TE3,TE4)."""

    batch_size: int = 4
    """Number of images to process in each batch (1-32)."""

    use_tta: bool = False
    """Enable test-time augmentation (slower but more accurate)."""

    compute_metrics: bool = True
    """Compute evaluation metrics (MAE, F1, etc.) after inference."""

    device: Literal["cuda", "cpu", "auto"] = "auto"
    """Computation device (auto selects CUDA if available)."""

    debug: bool = False
    """Debug mode: process only 5 images per dataset for quick testing."""

    def run(self) -> None:
        """Execute benchmarking."""
        from pdfnet.metric_tools.benchmark import benchmark_pdfnet

        # Run benchmark
        benchmark_pdfnet(
            checkpoint_path=self.checkpoint,
            data_path=self.data_path,
            output_dir=self.output_dir,
            datasets=self.datasets,
            batch_size=self.batch_size,
            use_tta=self.use_tta,
            device=self.device,
            compute_metrics=self.compute_metrics,
            debug=self.debug
        )


@dataclass
class EvaluateCommand:
    """Evaluate prediction results against ground truth masks."""

    pred_dir: Path
    """Directory containing prediction masks (PNG/JPG files)."""

    gt_dir: Path | None = None
    """Ground truth directory (uses default DIS-5K paths if not specified)."""

    output_dir: Path | None = None
    """Output directory for metrics CSV and summary files."""

    datasets: list[str] | None = None
    """Dataset names to evaluate (e.g., DIS-TE1 DIS-TE2)."""

    n_jobs: int = 12
    """Number of parallel workers for metric computation (1-64)."""

    def run(self) -> None:
        """Execute evaluation."""
        from pdfnet.metric_tools.soc_metrics import compute_metrics

        logger.info(f"Evaluating predictions in: {self.pred_dir}")

        # Build ground truth mapping
        gt_dir_dict = None
        if self.gt_dir:
            gt_dir_dict = {}
            if self.datasets:
                for dataset in self.datasets:
                    dataset_gt = self.gt_dir / dataset / "masks"
                    if dataset_gt.exists():
                        gt_dir_dict[dataset] = str(dataset_gt)
                        logger.info(f"Found GT for {dataset}: {dataset_gt}")
            else:
                # Auto-detect datasets
                for subdir in self.pred_dir.iterdir():
                    if subdir.is_dir():
                        dataset_name = subdir.name
                        dataset_gt = self.gt_dir / dataset_name / "masks"
                        if dataset_gt.exists():
                            gt_dir_dict[dataset_name] = str(dataset_gt)
                            logger.info(f"Found GT for {dataset_name}: {dataset_gt}")

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
    """Download model weights (PDFNet and Swin Transformer)."""

    dataset_info: bool = False
    """Show instructions for downloading DIS-5K dataset."""

    def run(self) -> None:
        """Execute download."""
        if self.weights:
            logger.info("Downloading model weights...")
            import subprocess
            result = subprocess.run(["uv", "run", "download.py"], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Download completed successfully")
                logger.info(result.stdout)
            else:
                logger.error("Download failed")
                logger.error(result.stderr)

        if self.dataset_info:
            logger.info("\nDataset Download Instructions:")
            logger.info("=" * 50)
            logger.info("\nDIS-5K Dataset:")
            logger.info("  Website: https://github.com/xuebinqin/DIS")
            logger.info("  Place in: DATA/DIS-DATA/")
            logger.info("\nExpected structure:")
            logger.info("  DATA/")
            logger.info("  └── DIS-DATA/")
            logger.info("      ├── DIS-TR/")
            logger.info("      ├── DIS-VD/")
            logger.info("      ├── DIS-TE1/")
            logger.info("      ├── DIS-TE2/")
            logger.info("      ├── DIS-TE3/")
            logger.info("      └── DIS-TE4/")


@dataclass
class ConfigCommand:
    """Manage PDFNet configuration files."""

    action: Literal["show", "create", "validate"] = "show"
    """Action to perform: show, create, or validate."""

    config_file: Path | None = None
    """Configuration file path (YAML format)."""

    output: Path | None = None
    """Output path for newly created config file."""

    def run(self) -> None:
        """Execute configuration command."""
        if self.action == "show":
            if self.config_file and self.config_file.exists():
                config = PDFNetConfig.load(self.config_file)
                logger.info(f"Configuration from {self.config_file}:")
            else:
                config = PDFNetConfig()
                logger.info("Default configuration:")

            import yaml
            from dataclasses import asdict
            logger.info(yaml.dump(asdict(config), default_flow_style=False))

        elif self.action == "create":
            output_path = self.output or Path("config/custom.yaml")
            config = PDFNetConfig()
            config.save(output_path)
            logger.info(f"Configuration saved to: {output_path}")

        elif self.action == "validate":
            if not self.config_file or not self.config_file.exists():
                logger.error("Config file not found")
                return

            try:
                config = PDFNetConfig.load(self.config_file)
                logger.info(f"Configuration is valid: {self.config_file}")

                # Check paths
                missing_paths = []
                if config.model.pretrained_swin and not config.model.pretrained_swin.exists():
                    missing_paths.append(f"  Pretrained weights: {config.model.pretrained_swin}")
                if not config.data.root_path.exists():
                    missing_paths.append(f"  Data root: {config.data.root_path}")

                if missing_paths:
                    logger.warning("Missing paths:")
                    for path in missing_paths:
                        logger.warning(path)

            except Exception as e:
                logger.error(f"Invalid configuration: {e}")


def main() -> None:
    """Main CLI entry point using tyro."""

    # Create CLI with subcommands
    cli = tyro.cli(
        tyro.extras.subcommand_cli_from_dict({
            "train": TrainCommand,
            "infer": InferCommand,
            "benchmark": BenchmarkCommand,
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