"""Save a qualitative comparison grid: input | ground truth | prediction.

Example:
    python -m cityseg.visualize --checkpoint best.ckpt --data-root data/cityscapes \
        --num 4 --output assets/qualitative.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CityscapesSegmentation, EvalTransform
from .inference import default_device, load_model, segment_image
from .labels import colorize, encode_labels


def make_grid(rows: list[list[Image.Image]], tile_width: int = 512, gap: int = 4) -> Image.Image:
    tiles = [
        [tile.resize((tile_width, round(tile.height * tile_width / tile.width))) for tile in row]
        for row in rows
    ]
    columns = len(tiles[0])
    row_heights = [max(tile.height for tile in row) for row in tiles]
    width = columns * tile_width + (columns - 1) * gap
    height = sum(row_heights) + (len(tiles) - 1) * gap
    grid = Image.new("RGB", (width, height), "white")
    y = 0
    for row, row_height in zip(tiles, row_heights, strict=True):
        for column, tile in enumerate(row):
            grid.paste(tile, (column * (tile_width + gap), y))
        y += row_height + gap
    return grid


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/cityscapes")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num", type=int, default=4, help="number of images (rows)")
    parser.add_argument("--output", default="assets/qualitative.png")
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    model = load_model(args.checkpoint, default_device())
    dataset = CityscapesSegmentation(args.data_root, args.split, EvalTransform())
    # Evenly spaced samples, so the grid is reproducible and not cherry-picked.
    indices = np.linspace(0, len(dataset) - 1, min(args.num, len(dataset))).round().astype(int)

    rows = []
    for index in indices:
        image = Image.open(dataset.images[index]).convert("RGB")
        truth = encode_labels(np.array(Image.open(dataset.targets[index])))
        prediction = segment_image(model, image)
        rows.append(
            [image, Image.fromarray(colorize(truth)), Image.fromarray(colorize(prediction))]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_grid(rows).save(output)
    print(f"Saved {len(rows)}-row grid (input | ground truth | prediction) to {output}")
    return output


if __name__ == "__main__":
    main()
