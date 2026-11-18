"""Dataset-level segmentation metrics from an accumulated confusion matrix."""

from __future__ import annotations

import torch
from torchmetrics import Metric

from .labels import IGNORE_INDEX, NUM_CLASSES


class SegmentationMetrics(Metric):
    """Accumulates a confusion matrix over the whole dataset.

    IoU is computed once from the summed matrix (the Cityscapes protocol)
    rather than averaged per image or per batch, which is biased and breaks
    whenever a class is absent from an image.
    """

    full_state_update = False

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state(
            "confmat",
            default=torch.zeros(num_classes, num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """``preds`` and ``target`` are integer class maps of identical shape."""
        keep = target != self.ignore_index
        index = target[keep].long() * self.num_classes + preds[keep].long()
        counts = torch.bincount(index, minlength=self.num_classes**2)
        self.confmat += counts.view(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, torch.Tensor]:
        confmat = self.confmat.double()  # rows: ground truth, columns: prediction
        true_positive = confmat.diag()
        union = confmat.sum(0) + confmat.sum(1) - true_positive
        # Classes that never occur in either prediction or ground truth are
        # NaN and left out of the mean.
        iou = torch.where(union > 0, true_positive / union, torch.nan)
        return {
            "iou": iou,
            "miou": iou.nanmean(),
            "pixel_acc": true_positive.sum() / confmat.sum().clamp(min=1),
        }
