"""
Unified transforms module for PDFNet.

This module consolidates all data augmentation and transformation classes.
"""

import torch
import numpy as np
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from PIL import Image, ImageEnhance


class ComposeTransforms:
    """Compose multiple transforms together."""

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class BaseTransform:
    """Base class for all transforms with probability control."""

    def __init__(self, prob=1.0):
        self.prob = prob

    def should_apply(self):
        return random.random() < self.prob

    def __call__(self, sample):
        if self.should_apply():
            return self.apply(sample)
        return sample

    def apply(self, sample):
        raise NotImplementedError


class RandomAffine(BaseTransform):
    """Random affine transformation."""

    def __init__(self, prob=0.5, degrees=30, translate=(0, 0.25),
                 scale=(0.8, 1.2), shear=15):
        super().__init__(prob)
        self.transform = transforms.RandomAffine(
            degrees=degrees, translate=translate,
            scale=scale, shear=shear, fill=0
        )

    def apply(self, sample):
        image, gt = sample['image'], sample['gt']
        # Apply to both image and ground truth
        combined = torch.cat([image, gt], dim=0)
        combined = self.transform(combined)
        sample['image'] = combined[:3, :, :]
        sample['gt'] = combined[3:, :, :]
        return sample


class RandomPerspective(BaseTransform):
    """Random perspective transformation."""

    def __init__(self, prob=0.5, distortion_scale=0.5):
        super().__init__(prob)
        self.transform = transforms.RandomPerspective(
            distortion_scale=distortion_scale, p=1.0
        )

    def apply(self, sample):
        image, gt = sample['image'], sample['gt']
        combined = torch.cat([image, gt], dim=0)
        combined = self.transform(combined)
        sample['image'] = combined[:3, :, :]
        sample['gt'] = combined[3:, :, :]
        return sample


class GaussianNoise(BaseTransform):
    """Add Gaussian noise to images."""

    def __init__(self, prob=0.5, max_std=0.2):
        super().__init__(prob)
        self.max_std = max_std

    def apply(self, sample):
        image = sample['image']
        noise = torch.randn(image.shape) * torch.rand(1) * self.max_std
        sample['image'] = image + noise
        return sample


class RandomRotation(BaseTransform):
    """Random rotation with center crop."""

    def __init__(self, prob=0.5, degrees=180):
        super().__init__(prob)
        self.degrees = degrees

    def apply(self, sample):
        angle = random.uniform(-self.degrees, self.degrees)
        image, gt = sample['image'], sample['gt']

        # Apply rotation
        image = transforms.functional.rotate(image, angle)
        gt = transforms.functional.rotate(gt, angle)

        # Calculate crop to remove black borders
        if angle != 0:
            image, gt = self._center_crop_after_rotate(image, gt, angle)

        sample['image'] = image
        sample['gt'] = gt
        return sample

    def _center_crop_after_rotate(self, image, gt, angle):
        """Center crop to remove black borders after rotation."""
        _, h, w = image.shape
        angle_rad = abs(np.radians(angle % 180))
        if angle_rad > np.pi/2:
            angle_rad = np.pi - angle_rad

        new_h = int(h / (np.cos(angle_rad) + np.sin(angle_rad)))
        new_w = new_h

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[:, top:top+new_h, left:left+new_w]
        gt = gt[:, top:top+new_h, left:left+new_w]

        # Resize back to original size
        image = F.interpolate(image.unsqueeze(0), size=(h, w), mode='bilinear')[0]
        gt = F.interpolate(gt.unsqueeze(0), size=(h, w), mode='bilinear')[0]

        return image, gt


class RandomFlip(BaseTransform):
    """Random horizontal and vertical flips."""

    def __init__(self, prob=0.5, h_flip=True, v_flip=False):
        super().__init__(prob)
        self.h_flip = h_flip
        self.v_flip = v_flip

    def apply(self, sample):
        image, gt = sample['image'], sample['gt']

        if self.h_flip and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            gt = transforms.functional.hflip(gt)

        if self.v_flip and random.random() > 0.5:
            image = transforms.functional.vflip(image)
            gt = transforms.functional.vflip(gt)

        sample['image'] = image
        sample['gt'] = gt
        return sample


class ColorJitter(BaseTransform):
    """Color jittering for images."""

    def __init__(self, prob=0.5, brightness=0.2, contrast=0.2,
                 saturation=0.2, hue=0.1):
        super().__init__(prob)
        self.transform = transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue
        )

    def apply(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample


class RandomCrop(BaseTransform):
    """Random crop with resize."""

    def __init__(self, prob=0.5, crop_ratio=(0.8, 1.0)):
        super().__init__(prob)
        self.crop_ratio = crop_ratio

    def apply(self, sample):
        image, gt = sample['image'], sample['gt']
        _, h, w = image.shape

        crop_factor = random.uniform(*self.crop_ratio)
        new_h = int(h * crop_factor)
        new_w = int(w * crop_factor)

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        image = image[:, top:top+new_h, left:left+new_w]
        gt = gt[:, top:top+new_h, left:left+new_w]

        # Resize back
        image = F.interpolate(image.unsqueeze(0), size=(h, w), mode='bilinear')[0]
        gt = F.interpolate(gt.unsqueeze(0), size=(h, w), mode='nearest')[0]

        sample['image'] = image
        sample['gt'] = gt
        return sample


class Normalize:
    """Normalize image tensors."""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        if 'image' in sample:
            sample['image'] = normalize(sample['image'], self.mean, self.std)
        return sample


class Resize:
    """Resize images and ground truth."""

    def __init__(self, size):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)

    def __call__(self, sample):
        image = sample['image']
        gt = sample['gt']

        # Resize
        image = F.interpolate(
            image.unsqueeze(0), size=self.size, mode='bilinear'
        )[0]
        gt = F.interpolate(
            gt.unsqueeze(0), size=self.size, mode='nearest'
        )[0]

        sample['image'] = image
        sample['gt'] = gt
        return sample


class ToTensor:
    """Convert numpy arrays or PIL images to tensors."""

    def __call__(self, sample):
        to_tensor = transforms.ToTensor()

        if 'image' in sample:
            if isinstance(sample['image'], np.ndarray):
                sample['image'] = torch.from_numpy(sample['image']).float()
                if sample['image'].dim() == 2:
                    sample['image'] = sample['image'].unsqueeze(0)
                elif sample['image'].dim() == 3 and sample['image'].shape[-1] == 3:
                    sample['image'] = sample['image'].permute(2, 0, 1)
            elif isinstance(sample['image'], Image.Image):
                sample['image'] = to_tensor(sample['image'])

        if 'gt' in sample:
            if isinstance(sample['gt'], np.ndarray):
                sample['gt'] = torch.from_numpy(sample['gt']).float()
                if sample['gt'].dim() == 2:
                    sample['gt'] = sample['gt'].unsqueeze(0)
            elif isinstance(sample['gt'], Image.Image):
                sample['gt'] = to_tensor(sample['gt'])

        return sample


class AugmentationPipeline:
    """
    Unified augmentation pipeline for training.

    Example:
        pipeline = AugmentationPipeline.create_training_pipeline()
        sample = pipeline(sample)
    """

    @staticmethod
    def create_training_pipeline(input_size=1024, augment=True):
        """Create standard training augmentation pipeline."""
        transforms_list = [ToTensor()]

        if augment:
            transforms_list.extend([
                RandomFlip(prob=0.5),
                RandomRotation(prob=0.3, degrees=30),
                RandomAffine(prob=0.3),
                RandomPerspective(prob=0.2),
                RandomCrop(prob=0.3),
                ColorJitter(prob=0.5),
                GaussianNoise(prob=0.2),
            ])

        transforms_list.extend([
            Resize(input_size),
            Normalize(),
        ])

        return ComposeTransforms(transforms_list)

    @staticmethod
    def create_validation_pipeline(input_size=1024):
        """Create standard validation pipeline (no augmentation)."""
        return ComposeTransforms([
            ToTensor(),
            Resize(input_size),
            Normalize(),
        ])

    @staticmethod
    def create_test_pipeline(input_size=1024):
        """Create test pipeline."""
        return ComposeTransforms([
            ToTensor(),
            Resize(input_size),
            Normalize(),
        ])


# Backward compatibility aliases
GOSNormalize = Normalize
GOSrandomAffine = RandomAffine
GOSrandomPerspective = RandomPerspective
GOSGaussianNoise = GaussianNoise
GOSrandomRotation = RandomRotation
GOSrandomFlip = RandomFlip
GOSrandomCrop = RandomCrop