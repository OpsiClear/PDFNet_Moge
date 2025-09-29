"""
SOC (Salient Object Characterization) metrics evaluation module.

This module provides functions to compute various metrics for evaluating
salient object detection models, including F-measure, MAE, S-measure, E-measure, etc.
"""

import os
import cv2
from tqdm import tqdm
from .metrics import Fmeasure, WeightedFmeasure, Smeasure, Emeasure, MAE
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

_EPS = 1e-16
_TYPE = np.float64


def get_image_files(path, exclude_ext='.pkl'):
    """
    Get all image files in a directory, excluding certain extensions.

    Args:
        path: Directory path to search
        exclude_ext: File extension to exclude (default: '.pkl')

    Returns:
        List of file paths
    """
    file_list = []
    for filepath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            full_path = os.path.join(filepath, filename)
            if exclude_ext not in full_path:
                file_list.append(full_path)
    return file_list


def once_compute(gt_root, gt_name, pred_root, FM, WFM, SM, EM, MAE):
    """Compute metrics for a single image pair."""
    gt_path = os.path.join(gt_root, gt_name)
    pred_path = os.path.join(pred_root, gt_name)

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or pred is None:
        print(f"Warning: Could not read images - GT: {gt_path}, Pred: {pred_path}")
        return None

    gtsize = gt.shape
    predsize = pred.shape

    if gtsize[0] == predsize[1] and gtsize[1] == predsize[0] and gtsize[0] != gtsize[1]:
        print(f"Warning: Transposed dimensions detected in {pred_path}")

    if predsize[0] != gtsize[0] or predsize[1] != gtsize[1]:
        pred = cv2.resize(pred, (gtsize[1], gtsize[0]))

    precisions, recalls = FM.step(pred=pred, gt=gt)
    wfm = WFM.step(pred=pred, gt=gt)
    mae = MAE.step(pred=pred, gt=gt)
    sm = SM.step(pred=pred, gt=gt)
    em = EM.step(pred=pred, gt=gt)

    return {
        'precisions': precisions,
        'recalls': recalls,
        'wfm': wfm,
        'mae': mae,
        'sm': sm,
        'em': em,
    }


def once_get(gt_root, pred_root, FM, WFM, SM, EM, MAE, testdir, i, n_jobs):
    """Process all images in a directory and compute aggregate metrics."""
    gt_name_list = get_image_files(pred_root)
    gt_name_list = sorted([x.split('/')[-1] for x in gt_name_list])

    results = Parallel(n_jobs=n_jobs)(
        delayed(once_compute)(gt_root, gt_name, pred_root, FM, WFM, SM, EM, MAE)
        for gt_name in tqdm(gt_name_list, total=len(gt_name_list), desc=f"Processing {testdir}")
    )

    # Filter out None results (failed computations)
    results = [r for r in results if r is not None]

    if not results:
        print(f"Warning: No valid results for {testdir}")
        return pd.DataFrame()

    precisions, recalls, wfm, sm, em, mae = [], [], [], [], [], []
    for result in results:
        precisions.append([result['precisions']])
        recalls.append([result['recalls']])
        wfm.append([result['wfm']])
        mae.append([result['mae']])
        sm.append([result['sm']])
        em.append([result['em']])

    precisions = np.array(precisions, dtype=_TYPE)
    recalls = np.array(recalls, dtype=_TYPE)
    precision = precisions.mean(axis=0)
    recall = recalls.mean(axis=0)
    fmeasure = 1.3 * precision * recall / (0.3 * precision + recall + _EPS)

    wfm = np.mean(np.array(wfm, dtype=_TYPE))
    mae = np.mean(np.array(mae, dtype=_TYPE))
    sm = np.mean(np.array(sm, dtype=_TYPE))
    em = np.mean(np.array(em, dtype=_TYPE), axis=0)

    results = {
        'maxFm': fmeasure.max(),
        'wFmeasure': wfm,
        'MAE': mae,
        'Smeasure': sm,
        'meanEm': em.mean(),
    }

    results_df = pd.DataFrame.from_dict([results]).T

    print(
        f"Results for {testdir}_{i}:",
        f"maxFm: {fmeasure.max():.3f},",
        f"wFmeasure: {wfm:.3f},",
        f"MAE: {mae:.3f},",
        f"Smeasure: {sm:.3f},",
        f"meanEm: {em.mean():.3f}"
    )

    return results_df


def compute_metrics(pred_dir, gt_dir_dict=None, output_dir=None, n_jobs=12):
    """
    Compute SOC metrics for predictions against ground truth.

    Args:
        pred_dir: Directory containing predictions (can have subdirectories for different datasets)
        gt_dir_dict: Dictionary mapping dataset names to ground truth directories
        output_dir: Directory to save results CSV (default: pred_dir)
        n_jobs: Number of parallel jobs for computation
    """
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    MAE_metric = MAE()

    if output_dir is None:
        output_dir = pred_dir

    # If no ground truth directories provided, use default structure
    if gt_dir_dict is None:
        gt_dir_dict = {}
        # Check for common dataset structures
        pred_path = Path(pred_dir)
        for subdir in pred_path.iterdir():
            if subdir.is_dir():
                dataset_name = subdir.name
                # Try to find corresponding ground truth
                potential_gt_paths = [
                    Path("DATA") / "DIS-DATA" / dataset_name / "masks",
                    Path("DATA") / dataset_name / "masks",
                    Path("data") / dataset_name / "masks",
                ]
                for gt_path in potential_gt_paths:
                    if gt_path.exists():
                        gt_dir_dict[dataset_name] = str(gt_path)
                        break

    if not gt_dir_dict:
        print("Warning: No ground truth directories found or specified")
        return

    allfile = pd.DataFrame()

    for i, (dataset_name, gt_root) in enumerate(gt_dir_dict.items()):
        pred_root = os.path.join(pred_dir, dataset_name)

        if not os.path.exists(pred_root):
            print(f"Warning: Prediction directory doesn't exist: {pred_root}")
            continue

        if not os.path.exists(gt_root):
            print(f"Warning: Ground truth directory doesn't exist: {gt_root}")
            continue

        print(f"\nEvaluating {dataset_name}...")
        onefile = once_get(gt_root, pred_root, FM, WFM, SM, EM, MAE_metric, dataset_name, i, n_jobs)

        if not onefile.empty:
            allfile = pd.concat([allfile.T, onefile.T]).T

    if not allfile.empty:
        output_path = os.path.join(output_dir, "metrics_results.csv")
        allfile.to_csv(output_path)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nNo results to save")


def main():
    """Main entry point for metrics evaluation."""
    parser = argparse.ArgumentParser(description='Compute SOC metrics for salient object detection')

    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing predictions')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Directory containing ground truth (or will auto-detect)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: pred_dir)')
    parser.add_argument('--n_jobs', type=int, default=12,
                        help='Number of parallel jobs')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='List of dataset names to evaluate')

    args = parser.parse_args()

    # Build ground truth directory dictionary
    gt_dir_dict = {}
    if args.gt_dir and args.datasets:
        for dataset in args.datasets:
            gt_path = os.path.join(args.gt_dir, dataset, 'masks')
            if os.path.exists(gt_path):
                gt_dir_dict[dataset] = gt_path

    compute_metrics(args.pred_dir, gt_dir_dict, args.output_dir, args.n_jobs)


if __name__ == '__main__':
    main()