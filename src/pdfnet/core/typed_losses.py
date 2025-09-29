"""
Type-safe loss functions using Python 3.12 type hints.

This module provides fully typed loss functions for training PDFNet models.
"""

from __future__ import annotations

from typing import Literal, TypeAlias, overload
from collections.abc import Mapping
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Type aliases
Tensor: TypeAlias = torch.Tensor
LossDict: TypeAlias = dict[str, Tensor]
ReductionType: TypeAlias = Literal["none", "mean", "sum"]


class BaseLoss(nn.Module, ABC):
    """Abstract base class for loss functions with type safety."""

    weight: float
    reduction: ReductionType

    def __init__(
        self,
        weight: float = 1.0,
        reduction: ReductionType = "mean"
    ) -> None:
        """
        Initialize base loss.

        Args:
            weight: Loss weight for multi-objective training
            reduction: Reduction method for batch losses
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    @abstractmethod
    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the actual loss value."""
        ...

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass with weight application.

        Args:
            pred: Predictions tensor
            target: Ground truth tensor

        Returns:
            Weighted loss value
        """
        loss = self.compute_loss(pred, target)
        return loss * self.weight


class BCEWithLogitsLoss(BaseLoss):
    """Binary Cross Entropy with Logits Loss."""

    pos_weight: float | None

    def __init__(
        self,
        weight: float = 1.0,
        reduction: ReductionType = "mean",
        pos_weight: float | None = None
    ) -> None:
        """
        Initialize BCE loss.

        Args:
            weight: Loss weight
            reduction: Reduction method
            pos_weight: Weight for positive class
        """
        super().__init__(weight, reduction)
        self.pos_weight = pos_weight

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute BCE with logits loss."""
        if self.pos_weight is not None:
            pos_weight_tensor = torch.ones_like(target) * self.pos_weight
            return F.binary_cross_entropy_with_logits(
                pred, target,
                pos_weight=pos_weight_tensor,
                reduction=self.reduction
            )
        return F.binary_cross_entropy_with_logits(
            pred, target, reduction=self.reduction
        )


class IoULoss(BaseLoss):
    """Intersection over Union Loss."""

    smooth: float

    def __init__(
        self,
        weight: float = 1.0,
        smooth: float = 1e-6,
        reduction: ReductionType = "mean"
    ) -> None:
        """
        Initialize IoU loss.

        Args:
            weight: Loss weight
            smooth: Smoothing factor for numerical stability
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.smooth = smooth

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute IoU loss."""
        pred_sigmoid = torch.sigmoid(pred)

        # Flatten spatial dimensions
        pred_flat = pred_sigmoid.view(pred_sigmoid.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Compute IoU
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        loss = 1 - iou

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(BaseLoss):
    """Dice Loss for segmentation."""

    smooth: float

    def __init__(
        self,
        weight: float = 1.0,
        smooth: float = 1e-6,
        reduction: ReductionType = "mean"
    ) -> None:
        """
        Initialize Dice loss.

        Args:
            weight: Loss weight
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.smooth = smooth

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Dice loss."""
        pred_sigmoid = torch.sigmoid(pred)

        # Flatten spatial dimensions
        pred_flat = pred_sigmoid.view(pred_sigmoid.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )

        loss = 1 - dice

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SSIMLoss(BaseLoss):
    """Structural Similarity Index Loss."""

    window_size: int
    sigma: float
    _window: Tensor | None

    def __init__(
        self,
        weight: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
        reduction: ReductionType = "mean"
    ) -> None:
        """
        Initialize SSIM loss.

        Args:
            weight: Loss weight
            window_size: Size of Gaussian window
            sigma: Standard deviation for Gaussian window
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.window_size = window_size
        self.sigma = sigma
        self._window = None

    def _get_gaussian_window(self, device: torch.device) -> Tensor:
        """Get or create Gaussian window."""
        if self._window is None or self._window.device != device:
            coords = torch.arange(self.window_size, device=device)
            coords = coords - self.window_size // 2

            gauss = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
            gauss = gauss / gauss.sum()

            window = gauss.unsqueeze(1).mm(gauss.unsqueeze(0))
            self._window = window.unsqueeze(0).unsqueeze(0)

        return self._window

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute SSIM loss."""
        pred_sigmoid = torch.sigmoid(pred)
        window = self._get_gaussian_window(pred.device)

        # Compute SSIM components
        mu1 = F.conv2d(pred_sigmoid, window, padding=self.window_size//2)
        mu2 = F.conv2d(target, window, padding=self.window_size//2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred_sigmoid*pred_sigmoid, window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target*target, window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred_sigmoid*target, window, padding=self.window_size//2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        loss = 1 - ssim_map

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(BaseLoss):
    """Focal Loss for handling class imbalance."""

    alpha: float
    gamma: float

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: ReductionType = "mean"
    ) -> None:
        """
        Initialize Focal loss.

        Args:
            weight: Loss weight
            alpha: Weighting factor for class balance
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Focal loss."""
        pred_sigmoid = torch.sigmoid(pred)

        # Compute focal terms
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Compute focal loss
        ce_loss = F.binary_cross_entropy(pred_sigmoid, target, reduction="none")
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class IntegrityPriorLoss(BaseLoss):
    """Integrity Prior Loss for depth consistency."""

    epsilon: float
    max_variance: float
    max_grad: float

    def __init__(
        self,
        weight: float = 0.3,
        epsilon: float = 1e-8,
        max_variance: float = 0.05,
        max_grad: float = 0.05,
        reduction: ReductionType = "mean"
    ) -> None:
        """
        Initialize Integrity Prior loss.

        Args:
            weight: Loss weight
            epsilon: Small constant for numerical stability
            max_variance: Maximum variance threshold
            max_grad: Maximum gradient threshold
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.epsilon = epsilon
        self.max_variance = max_variance
        self.max_grad = max_grad

        # Register Sobel kernels
        self.register_buffer("sobel_x", torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32
        ))
        self.register_buffer("sobel_y", torch.tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32
        ))

    def compute_loss(self, pred: Tensor, depth: Tensor) -> Tensor:
        """
        Compute integrity prior loss.

        Note: This expects depth as the second argument instead of target.
        """
        pred_sigmoid = torch.sigmoid(pred)

        # Compute depth gradients
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + self.epsilon)

        # Normalize gradients
        grad_min = grad_mag.view(grad_mag.size(0), -1).min(dim=1, keepdim=True)[0]
        grad_max = grad_mag.view(grad_mag.size(0), -1).max(dim=1, keepdim=True)[0]
        grad_mag = (grad_mag - grad_min.view(-1, 1, 1, 1)) / (
            grad_max.view(-1, 1, 1, 1) - grad_min.view(-1, 1, 1, 1) + self.epsilon
        )

        # Compute local variance
        kernel_size = 3
        depth_unfold = F.unfold(depth, kernel_size, padding=1)
        depth_mean = depth_unfold.mean(dim=1, keepdim=True)
        depth_var = ((depth_unfold - depth_mean) ** 2).mean(dim=1)
        depth_var = depth_var.view(depth.shape[0], 1, depth.shape[2], depth.shape[3])

        # Normalize variance
        var_min = depth_var.view(depth_var.size(0), -1).min(dim=1, keepdim=True)[0]
        var_max = depth_var.view(depth_var.size(0), -1).max(dim=1, keepdim=True)[0]
        depth_var = (depth_var - var_min.view(-1, 1, 1, 1)) / (
            var_max.view(-1, 1, 1, 1) - var_min.view(-1, 1, 1, 1) + self.epsilon
        )

        # Compute integrity score
        integrity = 1 - torch.maximum(
            torch.minimum(grad_mag, torch.tensor(self.max_grad, device=grad_mag.device)),
            torch.minimum(depth_var, torch.tensor(self.max_variance, device=depth_var.device))
        )

        # Weighted BCE loss
        loss = F.binary_cross_entropy(pred_sigmoid, pred_sigmoid.detach(), reduction="none")
        loss = loss * integrity

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@dataclass
class LossConfig:
    """Configuration for combined loss."""

    bce: dict[str, float] = field(default_factory=lambda: {"weight": 1.0})
    iou: dict[str, float] = field(default_factory=lambda: {"weight": 0.5})
    ssim: dict[str, float] = field(default_factory=lambda: {"weight": 0.5})
    dice: dict[str, float] | None = None
    focal: dict[str, float] | None = None
    integrity: dict[str, float] | None = None


class CombinedLoss(nn.Module):
    """Combined loss using multiple loss components."""

    losses: nn.ModuleDict
    loss_types: dict[str, type[BaseLoss]]

    def __init__(self, config: LossConfig | None = None) -> None:
        """
        Initialize combined loss.

        Args:
            config: Loss configuration
        """
        super().__init__()

        if config is None:
            config = LossConfig()

        self.losses = nn.ModuleDict()
        self.loss_types = {
            "bce": BCEWithLogitsLoss,
            "iou": IoULoss,
            "dice": DiceLoss,
            "ssim": SSIMLoss,
            "focal": FocalLoss,
            "integrity": IntegrityPriorLoss,
        }

        # Initialize configured losses
        for name, params in [
            ("bce", config.bce),
            ("iou", config.iou),
            ("ssim", config.ssim),
            ("dice", config.dice),
            ("focal", config.focal),
            ("integrity", config.integrity),
        ]:
            if params is not None:
                loss_class = self.loss_types[name]
                self.losses[name] = loss_class(**params)

    @overload
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        *,
        depth: None = None,
        return_dict: Literal[False] = False
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        *,
        depth: Tensor | None = None,
        return_dict: Literal[True]
    ) -> LossDict: ...

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        *,
        depth: Tensor | None = None,
        return_dict: bool = False
    ) -> Tensor | LossDict:
        """
        Compute combined loss.

        Args:
            pred: Predictions tensor
            target: Ground truth tensor
            depth: Optional depth map for integrity loss
            return_dict: Return individual losses as dict

        Returns:
            Total loss or dictionary of losses
        """
        loss_dict: LossDict = {}
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for name, loss_fn in self.losses.items():
            if name == "integrity" and depth is not None:
                # Integrity loss uses depth instead of target
                loss_val = loss_fn(pred, depth)
            elif name != "integrity":
                loss_val = loss_fn(pred, target)
            else:
                continue  # Skip integrity if no depth provided

            loss_dict[name] = loss_val
            total_loss = total_loss + loss_val

        if return_dict:
            loss_dict["total"] = total_loss
            return loss_dict

        return total_loss


class LossFactory:
    """Factory for creating loss functions."""

    _registry: dict[str, type[nn.Module]] = {
        "bce": BCEWithLogitsLoss,
        "iou": IoULoss,
        "dice": DiceLoss,
        "ssim": SSIMLoss,
        "focal": FocalLoss,
        "integrity": IntegrityPriorLoss,
        "combined": CombinedLoss,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> nn.Module:
        """
        Create a loss function by name.

        Args:
            name: Loss function name
            **kwargs: Loss function parameters

        Returns:
            Loss function instance

        Raises:
            ValueError: If loss name is not registered
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown loss: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )

        return cls._registry[name](**kwargs)

    @classmethod
    def register(cls, name: str, loss_class: type[nn.Module]) -> None:
        """Register a new loss function."""
        cls._registry[name] = loss_class

    @classmethod
    def available(cls) -> list[str]:
        """Get list of available loss functions."""
        return list(cls._registry.keys())