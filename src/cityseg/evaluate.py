"""Evaluate a checkpoint on the Cityscapes validation set.

Reports mean IoU, pixel accuracy and per-class IoU, and writes them to
``metrics.json`` and ``results.md`` next to the checkpoint (or ``--output-dir``).

Example:
    python -m cityseg.evaluate --checkpoint outputs/unet_r34/checkpoints/best.ckpt \
        --data-root data/cityscapes --flip
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CityscapesSegmentation, EvalTransform
from .inference import default_device, load_model, predict_logits
from .labels import CLASS_NAMES
from .metrics import SegmentationMetrics


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    flip: bool = False,
) -> dict:
    metrics = SegmentationMetrics().to(device)
    for images, target in loader:
        images, target = images.to(device), target.to(device)
        logits = predict_logits(model, images, flip=flip, output_size=target.shape[-2:])
        metrics.update(logits.argmax(1), target)
    results = metrics.compute()
    return {
        "miou": results["miou"].item(),
        "pixel_acc": results["pixel_acc"].item(),
        "iou": dict(zip(CLASS_NAMES, results["iou"].tolist(), strict=True)),
    }


def to_markdown(results: dict) -> str:
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| **mIoU** | **{100 * results['miou']:.2f}** |",
        f"| Pixel accuracy | {100 * results['pixel_acc']:.2f} |",
        "",
        "| Class | IoU (%) |",
        "|---|---|",
    ]
    for name, iou in results["iou"].items():
        value = "n/a" if iou is None or math.isnan(iou) else f"{100 * iou:.2f}"
        lines.append(f"| {name} | {value} |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/cityscapes")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="run the network at this resolution (default: full resolution); "
        "scoring is always against the full-resolution labels",
    )
    parser.add_argument("--flip", action="store_true", help="horizontal-flip test-time augmentation")  # noqa: E501
    parser.add_argument("--output-dir", help="default: the checkpoint's directory")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = default_device()
    model = load_model(args.checkpoint, device)
    dataset = CityscapesSegmentation(args.data_root, args.split, EvalTransform(args.size))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    results = evaluate(model, loader, device, flip=args.flip)
    results["settings"] = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "images": len(dataset),
        "size": args.size,
        "flip": args.flip,
    }

    markdown = to_markdown(results)
    print(markdown)
    output_dir = Path(args.output_dir or Path(args.checkpoint).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "results.md").write_text(markdown, encoding="utf-8")
    print(f"Saved metrics.json and results.md to {output_dir}")
    return results


if __name__ == "__main__":
    main()
