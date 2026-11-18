"""End-to-end smoke test: train -> evaluate -> predict on synthetic data (CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from cityseg import benchmark, evaluate, predict, train, visualize
from cityseg.config import load_config
from cityseg.inference import predict_logits
from cityseg.labels import CLASSES, NUM_CLASSES
from cityseg.model import SegModel


@pytest.fixture(scope="module")
def trained(cityscapes_root, tmp_path_factory):
    work = tmp_path_factory.mktemp("run")
    config = {
        "output_dir": str(work / "out"),
        "data": {
            "root": str(cityscapes_root),
            "crop_size": [64, 128],
            "batch_size": 2,
            "val_batch_size": 1,
            "num_workers": 0,
        },
        "model": {"encoder_weights": None, "warmup_steps": 1},
        "trainer": {
            "max_epochs": 1,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 1,
            "accelerator": "cpu",
            "enable_progress_bar": False,
        },
    }
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    train.main(["--config", str(config_path)])
    return work / "out"


def test_training_writes_checkpoints(trained):
    assert (trained / "checkpoints" / "best.ckpt").exists()
    assert (trained / "checkpoints" / "last.ckpt").exists()
    assert (trained / "config.yaml").exists()


def test_evaluate_reports_metrics(trained, cityscapes_root, tmp_path):
    results = evaluate.main(
        [
            "--checkpoint", str(trained / "checkpoints" / "best.ckpt"),
            "--data-root", str(cityscapes_root),
            "--batch-size", "1",
            "--num-workers", "0",
            "--flip",
            "--output-dir", str(tmp_path),
        ]
    )  # fmt: skip
    assert 0.0 <= results["miou"] <= 1.0
    assert 0.0 <= results["pixel_acc"] <= 1.0
    assert len(results["iou"]) == NUM_CLASSES
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["settings"]["images"] == 2
    assert "mIoU" in (tmp_path / "results.md").read_text(encoding="utf-8")


def test_predict_keeps_size_and_structure(trained, tmp_path):
    # Deliberately not a multiple of 32, and nested in a sub-folder.
    source = tmp_path / "images"
    (source / "munster").mkdir(parents=True)
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 256, (70, 100, 3), dtype=np.uint8)).save(
        source / "munster" / "frame.jpg"
    )
    output = tmp_path / "results"
    count = predict.main(
        [
            "--checkpoint", str(trained / "checkpoints" / "best.ckpt"),
            "--input", str(source),
            "--output", str(output),
            "--save", "color", "overlay", "labelids",
        ]
    )  # fmt: skip
    assert count == 1
    for kind in ("color", "overlay", "labelids"):
        result = Image.open(output / kind / "munster" / "frame.png")
        assert result.size == (100, 70)
    label_ids = np.array(Image.open(output / "labelids" / "munster" / "frame.png"))
    assert set(np.unique(label_ids).tolist()) <= {c.id for c in CLASSES}


def test_visualize_and_benchmark(trained, cityscapes_root, tmp_path):
    checkpoint = str(trained / "checkpoints" / "best.ckpt")
    output = visualize.main(
        [
            "--checkpoint", checkpoint,
            "--data-root", str(cityscapes_root),
            "--num", "2",
            "--output", str(tmp_path / "grid.png"),
        ]
    )  # fmt: skip
    grid = Image.open(output)
    assert grid.width == 3 * 512 + 2 * 4  # three columns

    result = benchmark.main(["--checkpoint", checkpoint, "--size", "64", "128", "--runs", "2"])
    assert result["fps"] > 0 and result["params_m"] > 1


def test_flip_tta_is_consistent_for_mirrored_input():
    torch.manual_seed(0)
    model = SegModel(encoder_weights=None).eval()
    images = torch.randn(1, 3, 64, 96)
    logits = predict_logits(model, images, flip=True)
    mirrored = predict_logits(model, images.flip(-1), flip=True).flip(-1)
    assert logits.shape == (1, NUM_CLASSES, 64, 96)
    assert torch.allclose(logits, mirrored, atol=1e-4)


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  learning_rate: 0.1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(path)
