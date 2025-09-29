"""
Unified loss functions module for PDFNet.

This module consolidates all loss functions into a clean, organized structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class BaseLoss(nn.Module):
    """Base class for all loss functions."""

    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BCELoss(BaseLoss):
    """Binary Cross Entropy Loss with optional weighting."""

    def __init__(self, weight: float = 1.0, reduction: str = 'mean', pos_weight: Optional[float] = None):
        super().__init__(weight, reduction)
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pos_weight:
            pos_weight = torch.ones_like(target) * self.pos_weight
            loss = F.binary_cross_entropy_with_logits(
                pred, target, pos_weight=pos_weight, reduction=self.reduction
            )
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction=self.reduction)
        return loss * self.weight


class IoULoss(BaseLoss):
    """Intersection over Union Loss."""

    def __init__(self, weight: float = 1.0, smooth: float = 1e-6):
        super().__init__(weight)
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - iou.mean()
        return loss * self.weight


class DiceLoss(BaseLoss):
    """Dice Loss for segmentation."""

    def __init__(self, weight: float = 1.0, smooth: float = 1e-6):
        super().__init__(weight)
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )
        loss = 1 - dice.mean()
        return loss * self.weight


class SSIMLoss(BaseLoss):
    """Structural Similarity Index Loss."""

    def __init__(self, weight: float = 1.0, window_size: int = 11, sigma: float = 1.5):
        super().__init__(weight)
        self.window_size = window_size
        self.sigma = sigma
        self.window = self._create_window(window_size, sigma)

    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian window for SSIM."""
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / (2.0*sigma**2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        return gauss.unsqueeze(1).mm(gauss.unsqueeze(1).t()).unsqueeze(0).unsqueeze(0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)

        # Move window to same device as input
        window = self.window.to(pred.device)

        # Compute SSIM
        mu1 = F.conv2d(pred, window, padding=self.window_size//2)
        mu2 = F.conv2d(target, window, padding=self.window_size//2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred*pred, window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target*target, window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred*target, window, padding=self.window_size//2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        loss = 1 - ssim.mean()
        return loss * self.weight


class IntegrityPriorLoss(BaseLoss):
    """Integrity Prior Loss for depth consistency."""

    def __init__(self, weight: float = 0.3, epsilon: float = 1e-8):
        super().__init__(weight)
        self.epsilon = epsilon
        self.max_variance = 0.05
        self.max_grad = 0.05

        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32
        ))
        self.register_buffer('sobel_y', torch.tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32
        ))

    def forward(self, pred: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute integrity prior loss.

        Args:
            pred: Predicted segmentation
            depth: Depth map

        Returns:
            Loss value
        """
        pred_sig = torch.sigmoid(pred)

        # Compute gradients
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + self.epsilon)

        # Normalize gradients
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + self.epsilon)

        # Compute local variance in depth
        kernel_size = 3
        depth_patches = F.unfold(depth, kernel_size, padding=1)
        depth_mean = depth_patches.mean(dim=1, keepdim=True)
        depth_var = ((depth_patches - depth_mean) ** 2).mean(dim=1)
        depth_var = depth_var.view(depth.shape[0], 1, depth.shape[2], depth.shape[3])

        # Normalize variance
        depth_var = (depth_var - depth_var.min()) / (depth_var.max() - depth_var.min() + self.epsilon)

        # Integrity score
        integrity = 1 - torch.maximum(
            torch.minimum(grad_mag, torch.tensor(self.max_grad).to(grad_mag.device)),
            torch.minimum(depth_var, torch.tensor(self.max_variance).to(depth_var.device))
        )

        # Weighted loss
        loss = F.binary_cross_entropy(pred_sig, pred_sig.detach(), reduction='none')
        loss = (loss * integrity).mean()

        return loss * self.weight


class FocalLoss(BaseLoss):
    """Focal Loss for handling class imbalance."""

    def __init__(self, weight: float = 1.0, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__(weight)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            loss = focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum()
        else:
            loss = focal_loss

        return loss * self.weight


class CombinedLoss(nn.Module):
    """Combined loss function using multiple loss components."""

    def __init__(self, loss_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.losses = nn.ModuleDict()

        # Default configuration
        if loss_config is None:
            loss_config = {
                'bce': {'weight': 1.0},
                'iou': {'weight': 0.5},
                'ssim': {'weight': 0.5},
            }

        # Initialize losses
        self.loss_classes = {
            'bce': BCELoss,
            'iou': IoULoss,
            'dice': DiceLoss,
            'ssim': SSIMLoss,
            'focal': FocalLoss,
            'integrity': IntegrityPriorLoss,
        }

        for name, config in loss_config.items():
            if name in self.loss_classes:
                self.losses[name] = self.loss_classes[name](**config)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        depth: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Predictions
            target: Ground truth
            depth: Optional depth map for integrity loss

        Returns:
            Dictionary with individual losses and total loss
        """
        loss_dict = {}
        total_loss = 0

        for name, loss_fn in self.losses.items():
            if name == 'integrity' and depth is not None:
                loss_val = loss_fn(pred, depth)
            else:
                loss_val = loss_fn(pred, target)

            loss_dict[name] = loss_val
            total_loss += loss_val

        loss_dict['total'] = total_loss
        return loss_dict


class LossFactory:
    """Factory class for creating loss functions."""

    _losses = {
        'bce': BCELoss,
        'iou': IoULoss,
        'dice': DiceLoss,
        'ssim': SSIMLoss,
        'focal': FocalLoss,
        'integrity': IntegrityPriorLoss,
        'combined': CombinedLoss,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """
        Create a loss function by name.

        Args:
            name: Loss function name
            **kwargs: Loss function parameters

        Returns:
            Loss function instance
        """
        if name not in cls._losses:
            raise ValueError(f"Unknown loss: {name}. Available: {list(cls._losses.keys())}")

        return cls._losses[name](**kwargs)

    @classmethod
    def register(cls, name: str, loss_class: type):
        """Register a new loss function."""
        cls._losses[name] = loss_class

    @classmethod
    def available(cls) -> list:
        """Get list of available loss functions."""
        return list(cls._losses.keys())