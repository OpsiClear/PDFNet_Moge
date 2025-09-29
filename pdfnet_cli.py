#!/usr/bin/env python3
"""
PDFNet Unified CLI Interface.

A single entry point for all PDFNet operations including training, evaluation,
and inference.

Usage:
    uv run pdfnet_cli.py train --config config/default.yaml
    uv run pdfnet_cli.py test --checkpoint checkpoints/model.pth
    uv run pdfnet_cli.py infer --input image.jpg --output result.png
    uv run pdfnet_cli.py evaluate --pred_dir results/ --gt_dir DATA/
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.pdfnet.config import load_config, Config
from src.pdfnet.inference import PDFNetInference, run_inference


def train_command(args):
    """Execute training."""
    from src.pdfnet.train import train_main
    from src.pdfnet.args import get_args_parser

    # Load configuration
    config = load_config(args.config)

    # Create training args from config
    train_parser = argparse.ArgumentParser('PDFNet Training', parents=[get_args_parser()])
    train_args = train_parser.parse_args(args=[])

    # Update args from config
    train_args.model = config.get('model.name', 'PDFNet_swinB')
    train_args.batch_size = config.get('training.batch_size', 1)
    train_args.epochs = config.get('training.epochs', 100)
    train_args.lr = config.get('training.lr', 1e-4)
    train_args.data_path = config.get('data.root_path', 'DATA/DIS-DATA')
    train_args.input_size = config.get('data.input_size', 1024)
    train_args.device = config.get('device', 'cuda')
    train_args.num_workers = config.get('training.num_workers', 8)

    # Override with CLI args if provided
    if args.epochs:
        train_args.epochs = args.epochs
    if args.batch_size:
        train_args.batch_size = args.batch_size
    if args.lr:
        train_args.lr = args.lr
    if args.data_path:
        train_args.data_path = args.data_path

    print(f"Starting training with config: {args.config}")
    print(f"Model: {train_args.model}")
    print(f"Epochs: {train_args.epochs}")
    print(f"Batch size: {train_args.batch_size}")
    print(f"Learning rate: {train_args.lr}")

    train_main(train_args)


def test_command(args):
    """Execute testing/evaluation on test sets."""
    from src.pdfnet.metric_tools.Test import test_pdfnet
    from src.pdfnet.args import get_args_parser

    # Load configuration
    config = load_config(args.config)

    # Create test args
    test_parser = argparse.ArgumentParser('PDFNet Testing', parents=[get_args_parser()])
    test_args = test_parser.parse_args(args=[])

    # Update from config and CLI
    test_args.checkpoint_path = args.checkpoint or config.get('inference.checkpoint_path')
    test_args.data_path = args.data_path or config.get('data.root_path')
    test_args.output_dir = args.output_dir or 'results'
    test_args.test_batch_size = args.batch_size or 1
    test_args.use_tta = args.tta
    test_args.device = config.get('device', 'cuda')
    test_args.input_size = config.get('data.input_size', 1024)
    test_args.compute_metrics = args.metrics

    # Set test datasets
    if args.datasets:
        test_args.test_dataset = args.datasets[0] if len(args.datasets) == 1 else None
        test_args.datasets = args.datasets
    else:
        test_args.test_dis_vd = True
        test_args.test_dis_te = True

    print(f"Testing with checkpoint: {test_args.checkpoint_path}")
    print(f"Output directory: {test_args.output_dir}")
    print(f"TTA enabled: {test_args.use_tta}")

    test_pdfnet(test_args)


def infer_command(args):
    """Execute inference on images."""
    print(f"Running inference on: {args.input}")

    # Determine output path
    output = args.output
    if output is None:
        input_p = Path(args.input)
        if input_p.is_file():
            output = f"{input_p.stem}_result.png"
        else:
            output = "results"

    # Run inference
    result = run_inference(
        input_path=args.input,
        output_path=output,
        config=args.config,
        checkpoint=args.checkpoint,
        use_tta=args.tta,
        batch_size=args.batch_size,
        device=args.device or 'cuda'
    )

    if output:
        print(f"Results saved to: {output}")


def evaluate_command(args):
    """Evaluate predictions against ground truth."""
    from src.pdfnet.metric_tools.soc_metrics import compute_metrics

    print(f"Evaluating predictions in: {args.pred_dir}")

    # Build ground truth mapping if provided
    gt_dir_dict = None
    if args.gt_dir:
        from pathlib import Path
        gt_path = Path(args.gt_dir)

        if args.datasets:
            gt_dir_dict = {}
            for dataset in args.datasets:
                dataset_gt = gt_path / dataset / 'masks'
                if dataset_gt.exists():
                    gt_dir_dict[dataset] = str(dataset_gt)
                    print(f"Found GT for {dataset}: {dataset_gt}")

    # Run evaluation
    compute_metrics(
        pred_dir=args.pred_dir,
        gt_dir_dict=gt_dir_dict,
        output_dir=args.output_dir or args.pred_dir,
        n_jobs=args.n_jobs
    )


def download_command(args):
    """Download model weights and datasets."""
    print("Downloading model weights...")

    # Run download script
    import subprocess
    result = subprocess.run(['uv', 'run', 'download.py'], capture_output=True, text=True)

    if result.returncode == 0:
        print("Download completed successfully!")
        print(result.stdout)
    else:
        print("Download failed!")
        print(result.stderr)
        return 1

    if args.dataset:
        print(f"\nTo download the DIS-5K dataset, please visit:")
        print("https://github.com/xuebinqin/DIS")
        print("\nPlace the dataset in DATA/DIS-DATA/ directory")


def config_command(args):
    """Show or create configuration."""
    if args.show:
        # Show current config
        config = load_config(args.config)
        print("Current configuration:")
        print("-" * 50)
        print(config)

    elif args.create:
        # Create new config file
        from src.pdfnet.config import get_default_config
        import yaml

        output_path = args.output or 'config/custom.yaml'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        config_dict = get_default_config()

        # Apply any overrides
        if args.set:
            for setting in args.set:
                key, value = setting.split('=')
                keys = key.split('.')
                d = config_dict
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
                # Try to parse value
                try:
                    d[keys[-1]] = eval(value)
                except:
                    d[keys[-1]] = value

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"Configuration saved to: {output_path}")


def main():
    """Main entry point for PDFNet CLI."""
    parser = argparse.ArgumentParser(
        description='PDFNet - Unified CLI for training, testing, and inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  uv run pdfnet_cli.py train --config config/default.yaml --epochs 50

  # Testing
  uv run pdfnet_cli.py test --checkpoint checkpoints/model.pth --tta

  # Inference
  uv run pdfnet_cli.py infer --input image.jpg --output result.png
  uv run pdfnet_cli.py infer --input images/ --output results/ --tta

  # Evaluation
  uv run pdfnet_cli.py evaluate --pred_dir results/ --gt_dir DATA/DIS-DATA

  # Configuration
  uv run pdfnet_cli.py config --show
  uv run pdfnet_cli.py config --create --set training.lr=0.0001
        """
    )

    # Global arguments
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    # Create subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train PDFNet model')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--data-path', type=str, help='Dataset path')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.set_defaults(func=train_command)

    # Testing command
    test_parser = subparsers.add_parser('test', help='Test model on datasets')
    test_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    test_parser.add_argument('--data-path', type=str, help='Dataset root path')
    test_parser.add_argument('--output-dir', type=str, help='Output directory')
    test_parser.add_argument('--datasets', nargs='+', help='Datasets to test on')
    test_parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    test_parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    test_parser.add_argument('--metrics', action='store_true', help='Compute metrics')
    test_parser.set_defaults(func=test_command)

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on images')
    infer_parser.add_argument('--input', '-i', type=str, required=True,
                              help='Input image or directory')
    infer_parser.add_argument('--output', '-o', type=str,
                              help='Output path (default: auto)')
    infer_parser.add_argument('--checkpoint', '-c', type=str,
                              help='Model checkpoint path')
    infer_parser.add_argument('--batch-size', type=int, default=1,
                              help='Batch size for directory processing')
    infer_parser.add_argument('--tta', action='store_true',
                              help='Use test-time augmentation')
    infer_parser.set_defaults(func=infer_command)

    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate predictions')
    eval_parser.add_argument('--pred-dir', type=str, required=True,
                             help='Directory with predictions')
    eval_parser.add_argument('--gt-dir', type=str,
                             help='Directory with ground truth')
    eval_parser.add_argument('--output-dir', type=str,
                             help='Output directory for results')
    eval_parser.add_argument('--datasets', nargs='+',
                             help='Dataset names to evaluate')
    eval_parser.add_argument('--n-jobs', type=int, default=12,
                             help='Number of parallel jobs')
    eval_parser.set_defaults(func=evaluate_command)

    # Download command
    download_parser = subparsers.add_parser('download', help='Download model weights')
    download_parser.add_argument('--dataset', action='store_true',
                                 help='Show dataset download instructions')
    download_parser.set_defaults(func=download_command)

    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configurations')
    config_parser.add_argument('--show', action='store_true',
                               help='Show current configuration')
    config_parser.add_argument('--create', action='store_true',
                               help='Create new configuration file')
    config_parser.add_argument('--output', type=str,
                               help='Output path for new config')
    config_parser.add_argument('--set', nargs='+',
                               help='Set config values (key=value)')
    config_parser.set_defaults(func=config_command)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command:
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())