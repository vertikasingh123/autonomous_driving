"""Measure model size and inference speed.

Example:
    python -m cityseg.benchmark --checkpoint best.ckpt --size 1024 2048
    python -m cityseg.benchmark --arch unet --encoder resnet34   # untrained, speed only
"""

from __future__ import annotations

import argparse
import time

import torch

from .inference import default_device, load_model
from .model import SegModel


@torch.no_grad()
def benchmark(
    model: torch.nn.Module,
    size: tuple[int, int],
    device: torch.device,
    runs: int = 50,
    warmup: int = 10,
) -> dict:
    model = model.to(device).eval()
    images = torch.randn(1, 3, *size, device=device)
    use_amp = device.type == "cuda"

    def step() -> None:
        with torch.autocast(device.type, enabled=use_amp):
            model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        step()
    start = time.perf_counter()
    for _ in range(runs):
        step()
    seconds = (time.perf_counter() - start) / runs

    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "resolution": f"{size[0]}x{size[1]}",
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "latency_ms": 1000 * seconds,
        "fps": 1 / seconds,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", help="trained checkpoint (otherwise --arch/--encoder)")
    parser.add_argument("--arch", default="unet")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--size", type=int, nargs=2, default=[1024, 2048], metavar=("H", "W"))
    parser.add_argument("--runs", type=int, default=50)
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = default_device()
    if args.checkpoint:
        model = load_model(args.checkpoint, device)
    else:
        model = SegModel(arch=args.arch, encoder=args.encoder, encoder_weights=None)
    result = benchmark(model, tuple(args.size), device, runs=args.runs)
    print(
        f"{result['device']} @ {result['resolution']}: "
        f"{result['params_m']:.1f} M params, "
        f"{result['latency_ms']:.1f} ms/image, {result['fps']:.1f} FPS"
    )
    return result


if __name__ == "__main__":
    main()
