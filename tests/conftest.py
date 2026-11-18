"""Shared fixtures: a tiny synthetic dataset in the Cityscapes folder layout."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

# A mix of valid ids (road, sidewalk, person, car) and void ids (0, 1).
RAW_IDS = np.array([0, 1, 7, 8, 24, 26], dtype=np.uint8)
IMAGE_SIZE = (64, 128)  # (height, width)


def _write_sample(root, split: str, city: str, name: str, rng: np.random.Generator) -> None:
    height, width = IMAGE_SIZE
    image_dir = root / "leftImg8bit" / split / city
    label_dir = root / "gtFine" / split / city
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    # Blocky labels so that classes form regions rather than noise.
    blocks = rng.choice(RAW_IDS, size=(height // 16, width // 16))
    labels = np.kron(blocks, np.ones((16, 16), dtype=np.uint8))
    Image.fromarray(image).save(image_dir / f"{name}_leftImg8bit.png")
    Image.fromarray(labels).save(label_dir / f"{name}_gtFine_labelIds.png")


@pytest.fixture(scope="session")
def cityscapes_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("cityscapes")
    rng = np.random.default_rng(0)
    for index in range(4):
        _write_sample(root, "train", "aachen", f"aachen_{index:06d}_000019", rng)
    for index in range(2):
        _write_sample(root, "val", "frankfurt", f"frankfurt_{index:06d}_000019", rng)
    return root
