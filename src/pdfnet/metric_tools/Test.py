"""
PDFNet Model Testing and Evaluation Script

This script evaluates PDFNet models on test datasets with optional TTA support.
"""

import torch
import torchvision.transforms as transforms
import sys
from PIL import Image
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import datetime
import ttach as tta
from pathlib import Path

from ..dataloaders.Mydataset import MyDataset, GOSNormalize
from ..models.PDFNet import build_model
from ..args import get_args_parser


def test_pdfnet(args):
    """Main testing function for PDFNet model evaluation."""

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build model
    model, model_name = build_model(args)

    # Load checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'), strict=False)
    else:
        print("Warning: No checkpoint path provided or file doesn't exist")

    model = model.to(device)
    model.eval()

    # Setup test directories from args
    test_dirs = {}
    if args.test_dataset:
        test_dirs[args.test_dataset] = args.test_data_path
    else:
        # Default test directories (can be configured via args)
        if args.test_dis_vd:
            test_dirs["DIS-VD"] = os.path.join(args.data_path, "DIS-VD/images")
        if args.test_dis_te:
            for i in range(1, 5):
                test_dirs[f"DIS-TE{i}"] = os.path.join(args.data_path, f"DIS-TE{i}/images")

    # Setup output directory
    test_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{model_name}_{test_time}"
    save_dir = args.output_dir if args.output_dir else f"results/{file_name}"

    if not args.DEBUG:
        os.makedirs(save_dir, exist_ok=True)

    to_pil = transforms.ToPILImage()

    # Setup TTA transforms if enabled
    tta_transforms = None
    if args.use_tta:
        tta_transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
        ])

    # Timing setup if CUDA available
    starter, ender = None, None
    if torch.cuda.is_available():
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    # Process each test dataset
    for dataset_name, dataset_path in test_dirs.items():
        if not os.path.exists(dataset_path):
            print(f"Warning: Test dataset path doesn't exist: {dataset_path}")
            continue

        print(f"\nTesting on {dataset_name}...")

        if not args.DEBUG:
            os.makedirs(f"{save_dir}/{dataset_name}", exist_ok=True)

        # Create dataset
        test_dataset = MyDataset(
            root=dataset_path,
            transform=[GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
            chached=False,
            size=[args.input_size, args.input_size],
            use_gt=False
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=True if args.num_workers > 0 else False
        )

        with torch.no_grad():
            iter_pbar = tqdm(total=len(test_loader), desc=dataset_name)
            all_times = []

            for i, data in enumerate(test_loader):
                image_name = data['image_name']
                image_size = data['image_size']
                inputs = data['image'].to(device)
                depth = data['depth'].to(device)

                if args.use_tta and tta_transforms:
                    # TTA inference
                    masks = []
                    for transformer in tta_transforms:
                        rgb_trans = transformer.augment_image(inputs)
                        depth_trans = transformer.augment_image(depth)

                        if starter:
                            starter.record()

                        pred_grad_sigmoid, pred_grad = model.inference(rgb_trans, depth_trans)

                        if ender:
                            ender.record()
                            torch.cuda.synchronize()
                            curr_time = starter.elapsed_time(ender)
                            all_times.append(curr_time)

                        deaug_mask = transformer.deaugment_mask(pred_grad)
                        masks.append(deaug_mask)

                    prediction = torch.mean(torch.stack(masks, dim=0), dim=0)
                else:
                    # Standard inference
                    if starter:
                        starter.record()

                    pred_grad_sigmoid, prediction = model.inference(inputs, depth)

                    if ender:
                        ender.record()
                        torch.cuda.synchronize()
                        curr_time = starter.elapsed_time(ender)
                        all_times.append(curr_time)

                prediction = prediction.sigmoid()

                # Save predictions
                for k in range(inputs.shape[0]):
                    save_name = Path(image_name[k]).stem + '.png'
                    w_, h_ = image_size

                    pred_pil = to_pil(prediction[k].squeeze(0).cpu())
                    pred_pil = pred_pil.resize((h_, w_), Image.BILINEAR)

                    if not args.DEBUG:
                        pred_pil.save(f"{save_dir}/{dataset_name}/{save_name}")

                    if args.DEBUG:
                        break

                iter_pbar.update()

                if args.DEBUG and i >= 5:  # Process only 5 batches in debug mode
                    break

            iter_pbar.close()
            torch.cuda.empty_cache()

            if all_times:
                mean_time = sum(all_times) / len(all_times)
                print(f"{dataset_name} - Inference time: {mean_time:.2f}ms/iter, FPS: {1000/mean_time:.2f}")

    print(f"\nTesting completed. Results saved to: {save_dir}")

    # Run metrics evaluation if requested
    if args.compute_metrics:
        print("\nComputing metrics...")
        from .soc_metrics import compute_metrics
        compute_metrics(save_dir, test_dirs)


def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser('PDFNet Testing', parents=[get_args_parser()])

    # Add test-specific arguments
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/PDFNet_Best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--test_data_path', type=str, default=None,
                        help='Path to test dataset')
    parser.add_argument('--test_dataset', type=str, default=None,
                        help='Name of test dataset')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--use_tta', action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute evaluation metrics after testing')
    parser.add_argument('--test_dis_vd', action='store_true',
                        help='Test on DIS-VD dataset')
    parser.add_argument('--test_dis_te', action='store_true',
                        help='Test on DIS-TE1-4 datasets')

    args = parser.parse_args()

    # Set defaults if not provided
    if not args.device:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_pdfnet(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())