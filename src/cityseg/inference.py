"""Inference helpers shared by evaluation, prediction, benchmarking and the demo."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import image_to_tensor
from .model import SegModel

STRIDE = 32  # encoder output stride; inputs are padded to a multiple of this


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint: str | os.PathLike, device: torch.device | None = None) -> SegModel:
    """Load a trained model in eval mode (without re-downloading ImageNet weights)."""
    device = device or default_device()
    model = SegModel.load_from_checkpoint(checkpoint, map_location=device, encoder_weights=None)
    return model.to(device).eval()


def _forward_padded(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    height, width = images.shape[-2:]
    pad_h = (-height) % STRIDE
    pad_w = (-width) % STRIDE
    if pad_h or pad_w:
        images = F.pad(images, (0, pad_w, 0, pad_h), mode="replicate")
    return model(images)[..., :height, :width]


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    flip: bool = False,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Class logits for a batch of normalised images.

    Args:
        flip: also predict the horizontally mirrored input and average
            (test-time augmentation).
        output_size: ``(height, width)`` to upsample the logits to. Upsampling
            happens on the logits, before the argmax, so class boundaries stay
            smooth instead of blocky.
    """
    device_type = images.device.type
    with torch.autocast(device_type, enabled=device_type == "cuda"):
        logits = _forward_padded(model, images)
        if flip:
            flipped = _forward_padded(model, images.flip(-1)).flip(-1)
            logits = (logits + flipped) / 2
    logits = logits.float()
    if output_size is not None and tuple(logits.shape[-2:]) != tuple(output_size):
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
    return logits


@torch.no_grad()
def segment_image(
    model: torch.nn.Module,
    image: Image.Image,
    flip: bool = False,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Segment one PIL image; returns an (H, W) uint8 array of train ids.

    ``size`` optionally sets the ``(height, width)`` the network runs at; the
    result is always returned at the original image resolution.
    """
    image = image.convert("RGB")
    original_size = (image.height, image.width)
    if size is not None:
        image = image.resize((size[1], size[0]), Image.BILINEAR)
    device = next(model.parameters()).device
    batch = image_to_tensor(image).unsqueeze(0).to(device)
    logits = predict_logits(model, batch, flip=flip, output_size=original_size)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def overlay(image: Image.Image, color_mask: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Blend a colour mask over the input image."""
    return Image.blend(image.convert("RGB"), Image.fromarray(color_mask), alpha)
