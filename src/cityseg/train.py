"""Train a segmentation model on Cityscapes.

Example:
    python -m cityseg.train --config configs/unet_r34.yaml --data-root data/cityscapes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from .config import load_config
from .data import CityscapesDataModule
from .model import SegModel


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="YAML file overriding the default configuration")
    parser.add_argument("--data-root", help="Cityscapes root (contains leftImg8bit/ and gtFine/)")
    parser.add_argument("--output-dir", help="where checkpoints and logs are written")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--resume",
        help="checkpoint to resume from, or 'auto' to continue from "
        "<output-dir>/checkpoints/last.ckpt when it exists",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.epochs:
        config["trainer"]["max_epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if not torch.cuda.is_available():
        config["trainer"]["precision"] = "32-true"  # mixed precision needs a GPU

    L.seed_everything(config["seed"], workers=True)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    datamodule = CityscapesDataModule(**config["data"])
    model = SegModel(**config["model"])

    checkpoint_dir = output_dir / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            monitor="val/miou",
            mode="max",
            save_last=True,
            enable_version_counter=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    loggers = [
        CSVLogger(output_dir, name="csv"),
        TensorBoardLogger(output_dir, name="tensorboard"),
    ]
    trainer = L.Trainer(
        default_root_dir=output_dir, callbacks=callbacks, logger=loggers, **config["trainer"]
    )

    resume = args.resume
    if resume == "auto":
        last = checkpoint_dir / "last.ckpt"
        resume = str(last) if last.exists() else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume)

    best = trainer.checkpoint_callback.best_model_path
    if best:
        score = trainer.checkpoint_callback.best_model_score
        print(f"Best checkpoint: {best} (val mIoU {float(score):.4f})")


if __name__ == "__main__":
    main()
