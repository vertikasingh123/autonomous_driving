"""Cityscapes dataset, augmentation and data module."""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import lightning as L
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter

from .labels import IGNORE_INDEX, encode_labels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Transform = Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]]


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """RGB PIL image -> normalised float tensor of shape (3, H, W)."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD, dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(mask, dtype=np.int64))


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Invert :func:`image_to_tensor` normalisation; returns values in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


class TrainTransform:
    """Random scale + crop, horizontal flip and colour jitter.

    A window of ``crop_size / scale`` source pixels is cut out and resized to
    ``crop_size``, which is equivalent to rescaling the whole image and then
    cropping but far cheaper. Where the window extends beyond the image the
    mask is padded with ``IGNORE_INDEX`` so the padding never contributes to
    the loss.
    """

    def __init__(
        self,
        crop_size: Sequence[int] = (512, 1024),
        scale_range: Sequence[float] = (0.5, 2.0),
        flip_prob: float = 0.5,
        color_jitter: bool = True,
    ) -> None:
        self.crop_h, self.crop_w = crop_size
        self.scale_range = tuple(scale_range)
        self.flip_prob = flip_prob
        self.jitter = ColorJitter(0.3, 0.3, 0.3, 0.05) if color_jitter else None

    def __call__(self, image: Image.Image, mask: Image.Image):
        scale = random.uniform(*self.scale_range)
        win_w = max(1, round(self.crop_w / scale))
        win_h = max(1, round(self.crop_h / scale))

        pad_w = max(win_w - image.width, 0)
        pad_h = max(win_h - image.height, 0)
        if pad_w or pad_h:
            border = (0, 0, pad_w, pad_h)  # pad right and bottom
            image = ImageOps.expand(image, border, fill=0)
            mask = ImageOps.expand(mask, border, fill=IGNORE_INDEX)

        x = random.randint(0, image.width - win_w)
        y = random.randint(0, image.height - win_h)
        box = (x, y, x + win_w, y + win_h)
        size = (self.crop_w, self.crop_h)
        image = image.resize(size, Image.BILINEAR, box=box)
        mask = mask.resize(size, Image.NEAREST, box=box)

        if random.random() < self.flip_prob:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if self.jitter is not None:
            image = self.jitter(image)

        return image_to_tensor(image), mask_to_tensor(mask)


class EvalTransform:
    """Deterministic transform for validation and testing.

    The image can optionally be resized for speed, but the mask is always left
    at its original resolution: predictions are upsampled to it before scoring,
    which is how the Cityscapes benchmark is evaluated.
    """

    def __init__(self, size: Sequence[int] | None = None) -> None:
        self.size = tuple(size) if size else None

    def __call__(self, image: Image.Image, mask: Image.Image):
        if self.size is not None:
            height, width = self.size
            image = image.resize((width, height), Image.BILINEAR)
        return image_to_tensor(image), mask_to_tensor(mask)


class CityscapesSegmentation(Dataset):
    """Cityscapes fine annotations with labels already mapped to train ids.

    Expects the standard layout::

        root/leftImg8bit/{split}/{city}/*_leftImg8bit.png
        root/gtFine/{split}/{city}/*_gtFine_labelIds.png
    """

    def __init__(self, root: str | os.PathLike, split: str, transform: Transform) -> None:
        root = Path(root)
        image_dir = root / "leftImg8bit" / split
        self.images = sorted(image_dir.glob("*/*_leftImg8bit.png"))
        if not self.images:
            raise FileNotFoundError(f"No Cityscapes images found under {image_dir}")
        self.targets = [
            root
            / "gtFine"
            / split
            / path.parent.name
            / path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            for path in self.images
        ]
        missing = [t for t in self.targets if not t.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} label files missing, e.g. {missing[0]}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        label_ids = np.array(Image.open(self.targets[index]))
        mask = Image.fromarray(encode_labels(label_ids))
        return self.transform(image, mask)


class CityscapesDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = "data/cityscapes",
        crop_size: Sequence[int] = (512, 1024),
        scale_range: Sequence[float] = (0.5, 2.0),
        val_size: Sequence[int] | None = None,
        batch_size: int = 8,
        val_batch_size: int = 4,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.root = root
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.val_size = val_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_set = CityscapesSegmentation(
                self.root, "train", TrainTransform(self.crop_size, self.scale_range)
            )
        self.val_set = CityscapesSegmentation(self.root, "val", EvalTransform(self.val_size))

    def _loader(self, dataset: Dataset, batch_size: int, train: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, self.batch_size, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, self.val_batch_size, train=False)
