"""
Type-safe transforms module using Python 3.12 type hints.

This module provides fully typed data augmentation and transformation classes
for training and inference pipelines.
"""

from typing import TypedDict, Protocol
from abc import ABC, abstractmethod
from collections.abc import Sequence
import random

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class SampleDict(TypedDict, total=False):
    """Type definition for data samples."""

    image: torch.Tensor
    gt: torch.Tensor
    depth: torch.Tensor
    label: torch.Tensor
    image_name: str
    image_size: tuple[int, int]


class Transform(Protocol):
    """Protocol for transform callables."""

    def __call__(self, sample: SampleDict) -> SampleDict: ...


class BaseTransform(ABC):
    """Abstract base class for probabilistic transforms."""

    prob: float

    def __init__(self, prob: float = 1.0) -> None:
        """
        Initialize transform with probability.

        Args:
            prob: Probability of applying this transform [0, 1]
        """
        self.prob = prob

    def should_apply(self) -> bool:
        """Check if transform should be applied based on probability."""
        return random.random() < self.prob

    @abstractmethod
    def apply(self, sample: SampleDict) -> SampleDict:
        """Apply the transformation to the sample."""
        ...

    def __call__(self, sample: SampleDict) -> SampleDict:
        """
        Apply transform with probability.

        Args:
            sample: Input sample dictionary

        Returns:
            Transformed sample
        """
        if self.should_apply():
            return self.apply(sample)
        return sample


class ComposeTransforms:
    """Compose multiple transforms sequentially."""

    transforms: Sequence[Transform]

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """
        Initialize composition.

        Args:
            transforms: Sequence of transforms to apply
        """
        self.transforms = list(transforms)

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __len__(self) -> int:
        """Get number of transforms."""
        return len(self.transforms)

    def __getitem__(self, index: int) -> Transform:
        """Get transform at index."""
        return self.transforms[index]


class ToTensor:
    """Convert numpy arrays or PIL images to tensors."""

    def __call__(self, sample: SampleDict) -> SampleDict:
        """
        Convert sample components to tensors.

        Args:
            sample: Input sample with numpy/PIL components

        Returns:
            Sample with tensor components
        """
        to_tensor = T.ToTensor()

        if "image" in sample:
            sample["image"] = self._to_tensor(sample["image"], to_tensor)

        if "gt" in sample:
            sample["gt"] = self._to_tensor(sample["gt"], to_tensor)

        if "depth" in sample:
            sample["depth"] = self._to_tensor(sample["depth"], to_tensor)

        return sample

    def _to_tensor(
        self, data: npt.NDArray | torch.Tensor | Image.Image, converter: T.ToTensor
    ) -> torch.Tensor:
        """Convert various data types to tensor."""
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 3 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            return tensor
        elif isinstance(data, Image.Image):
            return converter(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")


class Normalize:
    """Normalize image tensors."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """
        Initialize normalization.

        Args:
            mean: Per-channel mean values
            std: Per-channel standard deviation values
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Apply normalization to image."""
        if "image" in sample:
            sample["image"] = TF.normalize(sample["image"], self.mean, self.std)
        return sample


class Resize:
    """Resize images and masks."""

    size: tuple[int, int]

    def __init__(self, size: int | tuple[int, int]) -> None:
        """
        Initialize resize transform.

        Args:
            size: Target size (height, width) or single value for square
        """
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Resize image and ground truth."""
        if "image" in sample:
            sample["image"] = F.interpolate(
                sample["image"].unsqueeze(0),
                size=self.size,
                mode="bilinear",
                align_corners=False,
            )[0]

        if "gt" in sample:
            sample["gt"] = F.interpolate(
                sample["gt"].unsqueeze(0), size=self.size, mode="nearest"
            )[0]

        if "depth" in sample:
            sample["depth"] = F.interpolate(
                sample["depth"].unsqueeze(0),
                size=self.size,
                mode="bilinear",
                align_corners=False,
            )[0]

        return sample


class RandomFlip(BaseTransform):
    """Random horizontal and vertical flips."""

    horizontal: bool
    vertical: bool

    def __init__(
        self, prob: float = 0.5, horizontal: bool = True, vertical: bool = False
    ) -> None:
        """
        Initialize flip transform.

        Args:
            prob: Probability of applying flip
            horizontal: Enable horizontal flip
            vertical: Enable vertical flip
        """
        super().__init__(prob)
        self.horizontal = horizontal
        self.vertical = vertical

    def apply(self, sample: SampleDict) -> SampleDict:
        """Apply random flips."""
        image = sample.get("image")
        gt = sample.get("gt")
        depth = sample.get("depth")

        if image is None:
            return sample

        # Horizontal flip
        if self.horizontal and random.random() > 0.5:
            image = TF.hflip(image)
            if gt is not None:
                gt = TF.hflip(gt)
            if depth is not None:
                depth = TF.hflip(depth)

        # Vertical flip
        if self.vertical and random.random() > 0.5:
            image = TF.vflip(image)
            if gt is not None:
                gt = TF.vflip(gt)
            if depth is not None:
                depth = TF.vflip(depth)

        sample["image"] = image
        if gt is not None:
            sample["gt"] = gt
        if depth is not None:
            sample["depth"] = depth

        return sample


class RandomRotation(BaseTransform):
    """Random rotation with center crop."""

    degrees: float

    def __init__(self, prob: float = 0.5, degrees: float = 30) -> None:
        """
        Initialize rotation transform.

        Args:
            prob: Probability of applying rotation
            degrees: Maximum rotation angle
        """
        super().__init__(prob)
        self.degrees = degrees

    def apply(self, sample: SampleDict) -> SampleDict:
        """Apply random rotation."""
        angle = random.uniform(-self.degrees, self.degrees)

        image = sample.get("image")
        gt = sample.get("gt")
        depth = sample.get("depth")

        if image is None:
            return sample

        # Apply rotation
        image = TF.rotate(image, angle)
        if gt is not None:
            gt = TF.rotate(gt, angle)
        if depth is not None:
            depth = TF.rotate(depth, angle)

        # Center crop to remove black borders
        if angle != 0:
            image, gt, depth = self._center_crop_after_rotate(image, gt, depth, angle)

        sample["image"] = image
        if gt is not None:
            sample["gt"] = gt
        if depth is not None:
            sample["depth"] = depth

        return sample

    def _center_crop_after_rotate(
        self,
        image: torch.Tensor,
        gt: torch.Tensor | None,
        depth: torch.Tensor | None,
        angle: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Center crop to remove black borders after rotation."""
        _, h, w = image.shape
        angle_rad = abs(np.radians(angle % 180))

        if angle_rad > np.pi / 2:
            angle_rad = np.pi - angle_rad

        new_size = int(min(h, w) / (np.cos(angle_rad) + np.sin(angle_rad)))

        # Center crop
        image = TF.center_crop(image, new_size)
        if gt is not None:
            gt = TF.center_crop(gt, new_size)
        if depth is not None:
            depth = TF.center_crop(depth, new_size)

        # Resize back
        image = F.interpolate(image.unsqueeze(0), size=(h, w), mode="bilinear")[0]

        if gt is not None:
            gt = F.interpolate(gt.unsqueeze(0), size=(h, w), mode="nearest")[0]

        if depth is not None:
            depth = F.interpolate(depth.unsqueeze(0), size=(h, w), mode="bilinear")[0]

        return image, gt, depth


class ColorJitter(BaseTransform):
    """Color jittering for images."""

    jitter: T.ColorJitter

    def __init__(
        self,
        prob: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ) -> None:
        """
        Initialize color jitter.

        Args:
            prob: Probability of applying jitter
            brightness: Brightness jitter strength
            contrast: Contrast jitter strength
            saturation: Saturation jitter strength
            hue: Hue jitter strength
        """
        super().__init__(prob)
        self.jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def apply(self, sample: SampleDict) -> SampleDict:
        """Apply color jittering."""
        if "image" in sample:
            sample["image"] = self.jitter(sample["image"])
        return sample


class GaussianNoise(BaseTransform):
    """Add Gaussian noise to images."""

    max_std: float

    def __init__(self, prob: float = 0.5, max_std: float = 0.1) -> None:
        """
        Initialize Gaussian noise.

        Args:
            prob: Probability of applying noise
            max_std: Maximum standard deviation for noise
        """
        super().__init__(prob)
        self.max_std = max_std

    def apply(self, sample: SampleDict) -> SampleDict:
        """Add Gaussian noise."""
        if "image" in sample:
            image = sample["image"]
            std = torch.rand(1).item() * self.max_std
            noise = torch.randn_like(image) * std
            sample["image"] = torch.clamp(image + noise, 0, 1)
        return sample


class RandomCrop(BaseTransform):
    """Random crop with resize."""

    crop_ratio: tuple[float, float]

    def __init__(
        self, prob: float = 0.5, crop_ratio: tuple[float, float] = (0.8, 1.0)
    ) -> None:
        """
        Initialize random crop.

        Args:
            prob: Probability of applying crop
            crop_ratio: Range of crop size ratios
        """
        super().__init__(prob)
        self.crop_ratio = crop_ratio

    def apply(self, sample: SampleDict) -> SampleDict:
        """Apply random crop."""
        image = sample.get("image")
        gt = sample.get("gt")
        depth = sample.get("depth")

        if image is None:
            return sample

        _, h, w = image.shape
        crop_factor = random.uniform(*self.crop_ratio)
        new_h = int(h * crop_factor)
        new_w = int(w * crop_factor)

        # Random crop position
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        # Crop
        image = TF.crop(image, top, left, new_h, new_w)
        if gt is not None:
            gt = TF.crop(gt, top, left, new_h, new_w)
        if depth is not None:
            depth = TF.crop(depth, top, left, new_h, new_w)

        # Resize back
        image = F.interpolate(image.unsqueeze(0), size=(h, w), mode="bilinear")[0]

        if gt is not None:
            gt = F.interpolate(gt.unsqueeze(0), size=(h, w), mode="nearest")[0]

        if depth is not None:
            depth = F.interpolate(depth.unsqueeze(0), size=(h, w), mode="bilinear")[0]

        sample["image"] = image
        if gt is not None:
            sample["gt"] = gt
        if depth is not None:
            sample["depth"] = depth

        return sample


def create_training_pipeline(
    input_size: int = 1024,
    augment: bool = True,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> ComposeTransforms:
    """
    Create standard training augmentation pipeline.

    Args:
        input_size: Target input size
        augment: Enable augmentation
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transform pipeline
    """
    transforms: list[Transform] = [ToTensor()]

    if augment:
        transforms.extend(
            [
                RandomFlip(prob=0.5, horizontal=True),
                RandomRotation(prob=0.3, degrees=30),
                RandomCrop(prob=0.3, crop_ratio=(0.8, 1.0)),
                ColorJitter(prob=0.5),
                GaussianNoise(prob=0.2, max_std=0.05),
            ]
        )

    transforms.extend(
        [
            Resize(input_size),
            Normalize(mean, std),
        ]
    )

    return ComposeTransforms(transforms)


def create_validation_pipeline(
    input_size: int = 1024,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> ComposeTransforms:
    """
    Create validation pipeline (no augmentation).

    Args:
        input_size: Target input size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transform pipeline
    """
    return ComposeTransforms(
        [
            ToTensor(),
            Resize(input_size),
            Normalize(mean, std),
        ]
    )
