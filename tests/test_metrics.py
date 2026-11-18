import pytest
import torch

from cityseg.metrics import SegmentationMetrics


def test_iou_matches_hand_computation():
    metrics = SegmentationMetrics(num_classes=3, ignore_index=255)
    target = torch.tensor([[0, 0, 1, 1, 255]])
    preds = torch.tensor([[0, 1, 1, 1, 2]])
    metrics.update(preds, target)
    results = metrics.compute()

    # class 0: 1 hit, union 2 -> 0.5; class 1: 2 hits, union 3 -> 2/3.
    assert results["iou"][0].item() == pytest.approx(0.5)
    assert results["iou"][1].item() == pytest.approx(2 / 3)
    # class 2 only appears under an ignored pixel, so it is undefined ...
    assert torch.isnan(results["iou"][2])
    # ... and is excluded from the mean instead of poisoning it.
    assert results["miou"].item() == pytest.approx((0.5 + 2 / 3) / 2)
    assert results["pixel_acc"].item() == pytest.approx(3 / 4)


def test_accumulates_over_batches_not_averaged_per_batch():
    target_a, preds_a = torch.tensor([[0, 0, 0, 0]]), torch.tensor([[0, 0, 0, 0]])
    target_b, preds_b = torch.tensor([[0, 1]]), torch.tensor([[1, 1]])

    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(preds_a, target_a)
    metrics.update(preds_b, target_b)
    iou = metrics.compute()["iou"]

    # Dataset-level: class 0 = 4 / 5, class 1 = 1 / 2.
    assert iou[0].item() == pytest.approx(4 / 5)
    assert iou[1].item() == pytest.approx(1 / 2)


def test_perfect_prediction_and_reset():
    metrics = SegmentationMetrics(num_classes=4)
    target = torch.randint(0, 4, (2, 8, 8))
    metrics.update(target.clone(), target)
    assert metrics.compute()["miou"].item() == pytest.approx(1.0)
    metrics.reset()
    assert metrics.confmat.sum().item() == 0
