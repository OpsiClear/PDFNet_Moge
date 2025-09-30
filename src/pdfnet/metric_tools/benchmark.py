"""
PDFNet Benchmark Script

Evaluates PDFNet models on standard DIS benchmark datasets using the
type-safe inference engine with optional TTA support.
"""

import torch
from pathlib import Path
from tqdm import tqdm
import datetime
from typing import Literal
import logging

from ..inference import PDFNetInference
from ..config import PDFNetConfig

logger = logging.getLogger(__name__)


def benchmark_pdfnet(
    checkpoint_path: str | Path,
    data_path: str | Path = "DATA/DIS-DATA",
    output_dir: str | Path = "results",
    datasets: list[str] | None = None,
    batch_size: int = 1,
    use_tta: bool = False,
    device: Literal["cuda", "cpu", "auto"] = "auto",
    compute_metrics: bool = True,
    debug: bool = False,
) -> dict[str, float]:
    """
    Benchmark PDFNet model on standard DIS datasets.

    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Root path to DIS dataset
        output_dir: Directory to save results
        datasets: List of dataset names to test (default: all DIS test sets)
        batch_size: Batch size for inference
        use_tta: Enable test-time augmentation
        device: Device for inference
        compute_metrics: Compute evaluation metrics after testing
        debug: Enable debug mode (process only 5 images per dataset)

    Returns:
        Dictionary of timing statistics per dataset
    """

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default datasets
    if datasets is None:
        datasets = ["DIS-VD", "DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"]

    # Create configuration
    config = PDFNetConfig()
    config.inference.checkpoint_path = Path(checkpoint_path)
    config.inference.batch_size = batch_size
    config.inference.use_tta = use_tta
    config.inference.device = device
    config.device = device

    # Initialize inference engine
    logger.info(f"Loading model from: {checkpoint_path}")
    logger.info(f"Device: {device}")
    logger.info(f"TTA: {'Enabled' if use_tta else 'Disabled'}")
    logger.info(f"Batch size: {batch_size}")

    engine = PDFNetInference(config)

    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(output_dir) / f"benchmark_{timestamp}"

    if not debug:
        save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {save_dir}")

    # Benchmark statistics
    stats = {}
    data_root = Path(data_path)

    # Process each dataset
    for dataset_name in datasets:
        dataset_path = data_root / dataset_name / "images"

        if not dataset_path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Benchmarking: {dataset_name}")
        logger.info(f"{'=' * 60}")

        # Create output directory for this dataset
        dataset_output = save_dir / dataset_name
        if not debug:
            dataset_output.mkdir(parents=True, exist_ok=True)

        # Get all images
        image_files = sorted(
            list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
        )

        if not image_files:
            logger.warning(f"No images found in {dataset_path}")
            continue

        if debug:
            image_files = image_files[:5]  # Limit to 5 images in debug mode
            logger.info(f"DEBUG MODE: Processing only {len(image_files)} images")

        # Timing setup
        timings = []
        starter, ender = None, None
        if torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

        # Process images
        with tqdm(total=len(image_files), desc=dataset_name) as pbar:
            for img_path in image_files:
                # Start timing
                if starter:
                    starter.record()

                # Run inference
                result = engine.predict(str(img_path), use_tta=use_tta)

                # End timing
                if ender:
                    ender.record()
                    torch.cuda.synchronize()
                    elapsed = starter.elapsed_time(ender)
                    timings.append(elapsed)

                # Save result
                if not debug:
                    output_path = dataset_output / f"{img_path.stem}.png"
                    engine.save_prediction(result, output_path)

                pbar.update(1)

        # Calculate statistics
        if timings:
            mean_time = sum(timings) / len(timings)
            fps = 1000.0 / mean_time if mean_time > 0 else 0
            stats[dataset_name] = {
                "mean_time_ms": mean_time,
                "fps": fps,
                "num_images": len(image_files),
            }

            logger.info(f"\n{dataset_name} Statistics:")
            logger.info(f"  Images processed: {len(image_files)}")
            logger.info(f"  Mean time: {mean_time:.2f} ms/image")
            logger.info(f"  FPS: {fps:.2f}")
        else:
            logger.warning(f"No timing data collected for {dataset_name}")

        torch.cuda.empty_cache()

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Benchmark Summary")
    logger.info(f"{'=' * 60}")

    if stats:
        total_images = sum(s["num_images"] for s in stats.values())
        avg_fps = sum(s["fps"] for s in stats.values()) / len(stats)

        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Average FPS across datasets: {avg_fps:.2f}")
        logger.info("Per-dataset breakdown:")
        for dataset, data in stats.items():
            logger.info(
                f"  {dataset:10s}: {data['fps']:.2f} FPS ({data['num_images']} images)"
            )
    else:
        logger.info("No statistics collected")

    if not debug:
        logger.info(f"Results saved to: {save_dir}")

    # Compute metrics if requested
    if compute_metrics and not debug:
        logger.info(f"\n{'=' * 60}")
        logger.info("Computing Evaluation Metrics")
        logger.info(f"{'=' * 60}")

        try:
            from .soc_metrics import compute_metrics

            # Build ground truth mapping
            gt_dir_dict = {}
            for dataset_name in datasets:
                gt_path = data_root / dataset_name / "masks"
                if gt_path.exists():
                    gt_dir_dict[dataset_name] = str(gt_path)

            if gt_dir_dict:
                compute_metrics(
                    pred_dir=str(save_dir),
                    gt_dir_dict=gt_dir_dict,
                    output_dir=str(save_dir),
                )
                logger.info(f"Metrics saved to: {save_dir}")
            else:
                logger.warning("No ground truth masks found for metric computation")

        except ImportError:
            logger.warning("Metrics computation not available (soc_metrics module not found)")
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")

    return stats


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("PDFNet Benchmark")
    logger.info("=" * 70)
    logger.info("This module should be called through the tyro CLI:")
    logger.info("  uv run pdfnet.py benchmark --checkpoint checkpoints/PDFNet_Best.pth")
    logger.info("  python -m pdfnet benchmark --checkpoint checkpoints/PDFNet_Best.pth")
    logger.info("=" * 70)
    import sys
    sys.exit(1)
