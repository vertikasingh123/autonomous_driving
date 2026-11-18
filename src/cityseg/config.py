"""Experiment configuration: defaults merged with an optional YAML file."""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "output_dir": "outputs/unet_r34",
    "data": {
        "root": "data/cityscapes",
        "crop_size": [512, 1024],
        "scale_range": [0.5, 2.0],
        "val_size": None,  # None = validate at the full 1024x2048 resolution
        "batch_size": 8,
        "val_batch_size": 4,
        "num_workers": 4,
    },
    "model": {
        "arch": "unet",
        "encoder": "resnet34",
        "encoder_weights": "imagenet",
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "dice_weight": 0.5,
        "warmup_steps": 500,
    },
    "trainer": {
        "max_epochs": 60,
        "precision": "16-mixed",
        "check_val_every_n_epoch": 2,
        "log_every_n_steps": 20,
    },
}


def _merge(base: dict[str, Any], update: dict[str, Any], path: str = "") -> dict[str, Any]:
    for key, value in update.items():
        if key not in base and path != "trainer.":
            # Typos in a config should fail loudly, not be silently ignored.
            # (The trainer section is free-form: it is passed to lightning.Trainer.)
            raise KeyError(f"Unknown config key: {path}{key}")
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value, f"{path}{key}.")
        else:
            base[key] = value
    return base


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            _merge(config, yaml.safe_load(handle) or {})
    return config
