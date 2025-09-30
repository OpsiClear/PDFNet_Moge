import os

from skimage import io
import torch
import numpy as np
import cv2
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)


def mae_torch(pred, gt):
    h, w = gt.shape[0:2]
    sumError = torch.sum(torch.absolute(torch.sub(pred.float(), gt.float())))
    maeError = torch.divide(sumError, float(h) * float(w) * 255.0 + 1e-4)

    return maeError


def maximal_f_measure_torch(pd, gt):
    gtNum = torch.sum((gt > 128).float() * 1)

    pp = pd[gt > 128]
    nn = pd[gt <= 128]

    pp_hist = torch.histc(pp, bins=255, min=0, max=255)
    nn_hist = torch.histc(nn, bins=255, min=0, max=255)

    pp_hist_flip = torch.flipud(pp_hist)
    nn_hist_flip = torch.flipud(nn_hist)

    pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
    nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

    precision = (pp_hist_flip_cum) / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
    recall = (pp_hist_flip_cum) / (gtNum + 1e-4)
    f_measure = (2 * precision * recall) / (precision + recall + 1e-4)

    max_f_measure, threshold = torch.max(f_measure, dim=0)

    return max_f_measure.item(), threshold.item()


def calculate_meam(image1, image2):
    image1_equalized = cv2.equalizeHist(image1)
    image2_equalized = cv2.equalizeHist(image2)

    correlation_coefficient, _ = pearsonr(
        image1_equalized.flatten(), image2_equalized.flatten()
    )

    meam_value = correlation_coefficient * np.mean(
        np.minimum(image1_equalized, image2_equalized)
    )

    return meam_value


def f1score_torch(pd, gt):
    # print(gt.shape)
    gtNum = torch.sum((gt > 128).float() * 1)  ## number of ground truth pixels

    pp = pd[gt > 128]
    nn = pd[gt <= 128]

    pp_hist = torch.histc(pp, bins=255, min=0, max=255)
    nn_hist = torch.histc(nn, bins=255, min=0, max=255)

    pp_hist_flip = torch.flipud(pp_hist)
    nn_hist_flip = torch.flipud(nn_hist)

    pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
    nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

    precision = (
        (pp_hist_flip_cum) / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
    )  # torch.divide(pp_hist_flip_cum,torch.sum(torch.sum(pp_hist_flip_cum, nn_hist_flip_cum), 1e-4))
    recall = (pp_hist_flip_cum) / (gtNum + 1e-4)
    f1 = (1 + 0.3) * precision * recall / (0.3 * precision + recall + 1e-4)

    return (
        torch.reshape(precision, (1, precision.shape[0])),
        torch.reshape(recall, (1, recall.shape[0])),
        torch.reshape(f1, (1, f1.shape[0])),
    )


def f1_mae_torch(pred, gt, valid_dataset, idx, mybins, hypar):
    import time

    tic = time.time()

    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    pre, rec, f1 = f1score_torch(pred, gt)
    mae = mae_torch(pred, gt)

    # hypar["valid_out_dir"] = hypar["valid_out_dir"]+"-eval" ###
    if hypar["valid_out_dir"] != "":
        if not os.path.exists(hypar["valid_out_dir"]):
            os.mkdir(hypar["valid_out_dir"])
        dataset_folder = os.path.join(
            hypar["valid_out_dir"], valid_dataset.dataset["data_name"][idx]
        )
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)
        io.imsave(
            os.path.join(
                dataset_folder, valid_dataset.dataset["im_name"][idx] + ".png"
            ),
            pred.cpu().data.numpy().astype(np.uint8),
        )
    logger.debug(f"Processed: {valid_dataset.dataset['im_name'][idx]}.png")
    logger.debug(f"Evaluation time: {time.time() - tic:.4f}s")

    return (
        pre.cpu().data.numpy(),
        rec.cpu().data.numpy(),
        f1.cpu().data.numpy(),
        mae.cpu().data.numpy(),
    )
