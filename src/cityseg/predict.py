"""Segment an image or a folder of images with a trained checkpoint.

Outputs keep the input's sub-folder structure and file names, and are always
written at the original image resolution:

    <output>/color/     colour-coded masks (Cityscapes palette)
    <output>/overlay/   masks blended over the input image
    <output>/labelids/  raw Cityscapes label ids (benchmark submission format)

Example:
    python -m cityseg.predict --checkpoint best.ckpt --input test/Images \
        --output results --save color overlay
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .inference import default_device, load_model, overlay, segment_image
from .labels import TRAIN_ID_TO_ID, colorize

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="image file or folder, searched recursively")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument(
        "--save",
        nargs="+",
        default=["color"],
        choices=["color", "overlay", "labelids"],
        help="which outputs to write (default: color)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="run the network at this resolution (default: the image's own)",
    )
    parser.add_argument("--flip", action="store_true", help="horizontal-flip test-time augmentation")  # noqa: E501
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    source = Path(args.input)
    images = find_images(source)
    if not images:
        raise FileNotFoundError(f"No images found in {source}")
    root = source.parent if source.is_file() else source
    output = Path(args.output)
    model = load_model(args.checkpoint, default_device())

    for index, path in enumerate(images, start=1):
        image = Image.open(path).convert("RGB")
        train_ids = segment_image(model, image, flip=args.flip, size=args.size)
        relative = path.relative_to(root).with_suffix(".png")
        results = {}
        if "color" in args.save or "overlay" in args.save:
            color = colorize(train_ids)
            if "color" in args.save:
                results["color"] = Image.fromarray(color)
            if "overlay" in args.save:
                results["overlay"] = overlay(image, color)
        if "labelids" in args.save:
            results["labelids"] = Image.fromarray(TRAIN_ID_TO_ID[train_ids])
        for kind, result in results.items():
            target = output / kind / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            result.save(target)
        print(f"[{index}/{len(images)}] {path.name}")

    print(f"Wrote {len(images)} prediction(s) to {output}")
    return len(images)


if __name__ == "__main__":
    main()
