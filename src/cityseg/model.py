"""Lightning module: segmentation network, loss, metrics and optimisation."""

from __future__ import annotations

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn

from .labels import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from .metrics import SegmentationMetrics


class SegModel(L.LightningModule):
    """Encoder-decoder segmentation model trained with cross-entropy + Dice.

    ``arch`` and ``encoder`` are passed to ``segmentation_models_pytorch``, so
    swapping U-Net/ResNet-34 for e.g. DeepLabV3+/ResNet-50 or a SegFormer is a
    config change only.
    """

    def __init__(
        self,
        arch: str = "unet",
        encoder: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        dice_weight: float = 0.5,
        warmup_steps: int = 500,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = smp.create_model(
            arch, encoder_name=encoder, encoder_weights=encoder_weights, classes=num_classes
        )
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)
        self.val_metrics = SegmentationMetrics(num_classes, IGNORE_INDEX)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.cross_entropy(logits, target)
        if self.hparams.dice_weight > 0:
            loss = loss + self.hparams.dice_weight * self.dice(logits, target)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, target = batch
        loss = self.loss(self(images), target)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        images, target = batch
        logits = self(images)
        if logits.shape[-2:] != target.shape[-2:]:
            # Score against the full-resolution labels.
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        self.log("val/loss", self.loss(logits, target), batch_size=images.shape[0])
        self.val_metrics.update(logits.argmax(1), target)

    def on_validation_epoch_end(self) -> None:
        results = self.val_metrics.compute()
        self.val_metrics.reset()
        self.log("val/miou", results["miou"].float(), prog_bar=True)
        self.log("val/pixel_acc", results["pixel_acc"].float())
        for name, iou in zip(CLASS_NAMES, results["iou"], strict=True):
            if not torch.isnan(iou):
                self.log(f"val_iou/{name}", iou.float())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup_steps = min(self.hparams.warmup_steps, total_steps // 10)

        def lr_factor(step: int) -> float:
            # Linear warm-up followed by polynomial ("poly") decay.
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(1.0 - progress, 0.0) ** 0.9

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
