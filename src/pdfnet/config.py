"""
Configuration management for PDFNet.

This module provides configuration loading and validation.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class Config:
    """Configuration class for PDFNet."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        self._config = config_dict
        self._process_config()

    def _process_config(self):
        """Process and set configuration attributes."""
        for key, value in self._config.items():
            if isinstance(value, dict):
                # Convert nested dicts to Config objects
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value

        deep_update(self._config, updates)
        self._process_config()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self._config

    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_file(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create config from argparse namespace."""
        # Load base config
        config_path = getattr(args, 'config', 'config/default.yaml')
        if os.path.exists(config_path):
            config = cls.from_file(config_path)
        else:
            config = cls({})

        # Override with command line arguments
        args_dict = vars(args)
        updates = {}

        # Map common arguments to config structure
        arg_mapping = {
            'batch_size': 'training.batch_size',
            'epochs': 'training.epochs',
            'lr': 'training.lr',
            'data_path': 'data.root_path',
            'input_size': 'data.input_size',
            'model': 'model.name',
            'checkpoint': 'inference.checkpoint_path',
            'device': 'device',
            'output_dir': 'output.save_dir',
        }

        for arg_key, config_key in arg_mapping.items():
            if arg_key in args_dict and args_dict[arg_key] is not None:
                keys = config_key.split('.')
                d = updates
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
                d[keys[-1]] = args_dict[arg_key]

        config.update(updates)
        return config

    def __repr__(self):
        return f"Config({self._config})"

    def __str__(self):
        return yaml.dump(self._config, default_flow_style=False)


def load_config(config_path: Optional[str] = None, **kwargs) -> Config:
    """
    Load configuration from file with optional overrides.

    Args:
        config_path: Path to config file (default: config/default.yaml)
        **kwargs: Override values

    Returns:
        Config object
    """
    if config_path is None:
        config_path = 'config/default.yaml'

    if os.path.exists(config_path):
        config = Config.from_file(config_path)
    else:
        # Use defaults if config file doesn't exist
        config = Config(get_default_config())

    # Apply overrides
    if kwargs:
        config.update(kwargs)

    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary."""
    return {
        'model': {
            'name': 'PDFNet_swinB',
            'input_size': 1024,
            'drop_path': 0.1,
        },
        'training': {
            'batch_size': 1,
            'epochs': 100,
            'lr': 1e-4,
            'optimizer': 'adamw',
            'weight_decay': 0.05,
        },
        'data': {
            'dataset': 'DIS',
            'root_path': 'DATA/DIS-DATA',
            'input_size': 1024,
        },
        'device': 'cuda',
    }


def merge_args_to_config(config: Config, args: argparse.Namespace) -> Config:
    """
    Merge command-line arguments into configuration.

    Args:
        config: Base configuration
        args: Command-line arguments

    Returns:
        Updated configuration
    """
    args_dict = vars(args)

    # Remove None values and special keys
    args_dict = {k: v for k, v in args_dict.items()
                 if v is not None and k not in ['config', 'func']}

    # Apply updates
    if args_dict:
        config.update(args_dict)

    return config