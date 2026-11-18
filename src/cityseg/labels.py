"""Cityscapes label definitions.

Cityscapes annotates 34 raw label ids, of which 19 are used for training and
evaluation. Every other id is "void": it is mapped to ``IGNORE_INDEX`` and
excluded from both the loss and the metrics, as in the official benchmark.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

IGNORE_INDEX = 255


class CityscapesClass(NamedTuple):
    name: str
    id: int  # raw id stored in *_gtFine_labelIds.png
    color: tuple[int, int, int]


# Order defines the train id (0..18).
CLASSES: tuple[CityscapesClass, ...] = (
    CityscapesClass("road", 7, (128, 64, 128)),
    CityscapesClass("sidewalk", 8, (244, 35, 232)),
    CityscapesClass("building", 11, (70, 70, 70)),
    CityscapesClass("wall", 12, (102, 102, 156)),
    CityscapesClass("fence", 13, (190, 153, 153)),
    CityscapesClass("pole", 17, (153, 153, 153)),
    CityscapesClass("traffic light", 19, (250, 170, 30)),
    CityscapesClass("traffic sign", 20, (220, 220, 0)),
    CityscapesClass("vegetation", 21, (107, 142, 35)),
    CityscapesClass("terrain", 22, (152, 251, 152)),
    CityscapesClass("sky", 23, (70, 130, 180)),
    CityscapesClass("person", 24, (220, 20, 60)),
    CityscapesClass("rider", 25, (255, 0, 0)),
    CityscapesClass("car", 26, (0, 0, 142)),
    CityscapesClass("truck", 27, (0, 0, 70)),
    CityscapesClass("bus", 28, (0, 60, 100)),
    CityscapesClass("train", 31, (0, 80, 100)),
    CityscapesClass("motorcycle", 32, (0, 0, 230)),
    CityscapesClass("bicycle", 33, (119, 11, 32)),
)

NUM_CLASSES = len(CLASSES)
CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in CLASSES)

# Raw label id -> train id. Anything not listed above becomes IGNORE_INDEX.
ID_TO_TRAIN_ID = np.full(256, IGNORE_INDEX, dtype=np.uint8)
# Train id -> raw label id (the format the Cityscapes test server expects).
TRAIN_ID_TO_ID = np.zeros(256, dtype=np.uint8)
# Train id -> RGB colour. Ignored pixels are drawn black.
PALETTE = np.zeros((256, 3), dtype=np.uint8)

for _train_id, _cls in enumerate(CLASSES):
    ID_TO_TRAIN_ID[_cls.id] = _train_id
    TRAIN_ID_TO_ID[_train_id] = _cls.id
    PALETTE[_train_id] = _cls.color


def encode_labels(label_ids: np.ndarray) -> np.ndarray:
    """Map a raw Cityscapes ``labelIds`` array to train ids (void -> 255)."""
    return ID_TO_TRAIN_ID[label_ids]


def colorize(train_ids: np.ndarray) -> np.ndarray:
    """Map an (H, W) array of train ids to an (H, W, 3) uint8 RGB image."""
    return PALETTE[train_ids]
