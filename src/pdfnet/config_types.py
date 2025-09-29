"""
Type-safe configuration using Python 3.12 type hints and dataclasses.

This module defines all configuration types for PDFNet using modern Python
type hints and dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Annotated
from collections.abc import Sequence

import tyro


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: Literal["PDFNet_swinB", "PDFNet_swinL", "PDFNet_swinT"] = "PDFNet_swinB"
    """Model architecture to use."""

    input_size: Annotated[int, tyro.conf.arg(help="Input image size")] = 1024
    """Input image size (square)."""

    drop_path: Annotated[float, tyro.conf.arg(help="Drop path rate")] = 0.1
    """Drop path rate for regularization."""

    pretrained_swin: Path | None = Path("checkpoints/swin_base_patch4_window12_384_22k.pth")
    """Path to pretrained Swin Transformer weights."""

    num_classes: int = 1
    """Number of output classes (1 for binary segmentation)."""


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: Literal["sgd", "adam", "adamw"] = "adamw"
    """Optimizer type."""

    lr: Annotated[float, tyro.conf.arg(help="Learning rate")] = 1e-4
    """Initial learning rate."""

    weight_decay: float = 0.05
    """Weight decay for regularization."""

    momentum: float = 0.9
    """Momentum factor (for SGD)."""

    betas: tuple[float, float] = (0.9, 0.999)
    """Beta parameters for Adam optimizers."""

    eps: float = 1e-8
    """Epsilon for numerical stability."""


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    name: Literal["cosine", "step", "plateau", "linear"] = "cosine"
    """Scheduler type."""

    warmup_epochs: int = 5
    """Number of warmup epochs."""

    warmup_lr: float = 1e-6
    """Learning rate for warmup."""

    min_lr: float = 1e-5
    """Minimum learning rate."""

    decay_epochs: int = 30
    """Epochs between decay steps (for step scheduler)."""

    decay_rate: float = 0.1
    """Decay rate for step scheduler."""


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""

    enabled: bool = True
    """Enable data augmentation."""

    color_jitter: float = 0.4
    """Color jitter strength."""

    random_flip: bool = True
    """Enable random horizontal/vertical flips."""

    random_rotate: bool = True
    """Enable random rotation."""

    random_crop: bool = True
    """Enable random cropping."""

    mixup: float = 0.8
    """Mixup alpha parameter."""

    cutmix: float = 1.0
    """CutMix alpha parameter."""

    mixup_prob: float = 1.0
    """Probability of applying mixup/cutmix."""

    gaussian_noise: float = 0.2
    """Maximum standard deviation for Gaussian noise."""


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: Annotated[int, tyro.conf.arg(help="Training batch size")] = 1
    """Batch size for training."""

    epochs: Annotated[int, tyro.conf.arg(help="Number of training epochs")] = 100
    """Total number of training epochs."""

    num_workers: int = 8
    """Number of data loading workers."""

    seed: int = 0
    """Random seed for reproducibility."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer configuration."""

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Learning rate scheduler configuration."""

    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    """Data augmentation configuration."""

    clip_grad: float | None = None
    """Gradient clipping value (None to disable)."""

    update_freq: int = 1
    """Gradient accumulation steps."""

    use_amp: bool = True
    """Use automatic mixed precision training."""

    eval_freq: int = 1
    """Evaluate every N epochs."""

    eval_metric: Literal["F1", "MAE", "IoU"] = "F1"
    """Metric to use for model selection."""

    save_freq: int = 1
    """Save checkpoint every N epochs."""

    keep_checkpoints: int = 3
    """Number of best checkpoints to keep."""


@dataclass
class DataConfig:
    """Dataset configuration."""

    dataset: Literal["DIS", "HRSOD", "UHRSD"] = "DIS"
    """Dataset name."""

    root_path: Path = Path("DATA/DIS-DATA")
    """Root path to dataset."""

    train_split: str = "DIS-TR"
    """Training split name."""

    val_split: str = "DIS-VD"
    """Validation split name."""

    test_splits: list[str] = field(default_factory=lambda: ["DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"])
    """Test split names."""

    input_size: int = 1024
    """Input image size."""

    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    """Normalization mean values."""

    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    """Normalization standard deviation values."""

    cache_data: bool = False
    """Cache dataset in memory."""

    cache_dir: Path = Path("cache")
    """Directory for cached data."""


@dataclass
class InferenceConfig:
    """Inference configuration."""

    checkpoint_path: Path | None = Path("checkpoints/PDFNet_Best.pth")
    """Path to model checkpoint."""

    batch_size: int = 1
    """Batch size for inference."""

    use_tta: bool = False
    """Use test-time augmentation."""

    tta_scales: list[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    """Scales for test-time augmentation."""

    device: Literal["cuda", "cpu", "auto"] = "auto"
    """Device for inference (auto selects cuda if available)."""

    use_moge: bool = True
    """Use MoGe for depth estimation."""

    moge_model: Path | None = Path("checkpoints/moge/moge-2-vitl-normal/model.pt")
    """Path to MoGe model."""

    moge_input_size: int = 518
    """Input size for MoGe model."""


@dataclass
class OutputConfig:
    """Output and logging configuration."""

    save_dir: Path = Path("runs")
    """Directory for training outputs."""

    log_dir: Path = Path("logs")
    """Directory for logs."""

    checkpoint_dir: Path = Path("checkpoints")
    """Directory for checkpoints."""

    result_dir: Path = Path("results")
    """Directory for inference results."""

    use_tensorboard: bool = True
    """Enable TensorBoard logging."""

    log_freq: int = 10
    """Log frequency (iterations)."""

    verbose: bool = True
    """Enable verbose output."""


@dataclass
class LossConfig:
    """Loss function configuration."""

    bce_weight: float = 1.0
    """Weight for BCE loss."""

    iou_weight: float = 0.5
    """Weight for IoU loss."""

    ssim_weight: float = 0.5
    """Weight for SSIM loss."""

    integrity_weight: float = 0.3
    """Weight for integrity prior loss."""

    use_focal: bool = False
    """Use focal loss instead of BCE."""

    focal_alpha: float = 0.25
    """Alpha parameter for focal loss."""

    focal_gamma: float = 2.0
    """Gamma parameter for focal loss."""


@dataclass
class PDFNetConfig:
    """Complete PDFNet configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model architecture configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    """Dataset configuration."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    """Inference configuration."""

    output: OutputConfig = field(default_factory=OutputConfig)
    """Output and logging configuration."""

    loss: LossConfig = field(default_factory=LossConfig)
    """Loss function configuration."""

    device: Literal["cuda", "cpu", "auto"] = "auto"
    """Default device (auto selects cuda if available)."""

    distributed: bool = False
    """Enable distributed training."""

    world_size: int = 1
    """Number of distributed processes."""

    local_rank: int = -1
    """Local rank for distributed training."""

    debug: bool = False
    """Enable debug mode."""

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        import torch

        # Auto-select device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.inference.device == "auto":
            self.inference.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure paths exist
        for config_name in ["output", "data"]:
            config = getattr(self, config_name)
            for attr_name in dir(config):
                if not attr_name.startswith("_"):
                    attr_value = getattr(config, attr_name)
                    if isinstance(attr_value, Path) and attr_name.endswith("_dir"):
                        attr_value.mkdir(parents=True, exist_ok=True)

    def save(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path | str) -> "PDFNetConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Recursively create dataclass instances
        def create_dataclass(dc_type, data_dict):
            if not isinstance(data_dict, dict):
                return data_dict

            field_types = {f.name: f.type for f in dc_type.__dataclass_fields__.values()}
            kwargs = {}

            for key, value in data_dict.items():
                if key in field_types:
                    field_type = field_types[key]
                    if hasattr(field_type, "__dataclass_fields__"):
                        kwargs[key] = create_dataclass(field_type, value)
                    elif hasattr(field_type, "__origin__") and field_type.__origin__ is Path:
                        kwargs[key] = Path(value) if value is not None else None
                    else:
                        kwargs[key] = value

            return dc_type(**kwargs)

        return create_dataclass(cls, data)