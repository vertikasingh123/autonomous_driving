import pytest
import torch
from PIL import Image

from cityseg.data import (
    CityscapesSegmentation,
    EvalTransform,
    TrainTransform,
    denormalize,
    image_to_tensor,
)
from cityseg.labels import IGNORE_INDEX, NUM_CLASSES


def test_dataset_returns_train_ids(cityscapes_root):
    dataset = CityscapesSegmentation(cityscapes_root, "val", EvalTransform())
    assert len(dataset) == 2
    image, mask = dataset[0]
    assert image.shape == (3, 64, 128) and image.dtype == torch.float32
    assert mask.shape == (64, 128) and mask.dtype == torch.int64
    values = set(mask.unique().tolist())
    assert values <= set(range(NUM_CLASSES)) | {IGNORE_INDEX}
    assert IGNORE_INDEX in values  # the fixture contains void pixels


def test_missing_split_raises(cityscapes_root):
    with pytest.raises(FileNotFoundError):
        CityscapesSegmentation(cityscapes_root, "test", EvalTransform())


def test_eval_transform_is_deterministic(cityscapes_root):
    dataset = CityscapesSegmentation(cityscapes_root, "val", EvalTransform())
    first, second = dataset[1], dataset[1]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_eval_transform_resizes_image_but_not_mask(cityscapes_root):
    dataset = CityscapesSegmentation(cityscapes_root, "val", EvalTransform((32, 64)))
    image, mask = dataset[0]
    assert image.shape == (3, 32, 64)
    assert mask.shape == (64, 128)


def test_train_transform_output_size(cityscapes_root):
    transform = TrainTransform(crop_size=(32, 96), scale_range=(0.5, 2.0))
    dataset = CityscapesSegmentation(cityscapes_root, "train", transform)
    for index in range(len(dataset)):
        image, mask = dataset[index]
        assert image.shape == (3, 32, 96)
        assert mask.shape == (32, 96)


def test_train_transform_pads_mask_with_ignore_index():
    image = Image.new("RGB", (20, 10), (255, 255, 255))
    mask = Image.new("L", (20, 10), 3)
    # scale 0.5 on a 40x40 crop needs an 80x80 window: far larger than the image.
    transform = TrainTransform(
        crop_size=(40, 40), scale_range=(0.5, 0.5), flip_prob=0.0, color_jitter=False
    )
    _, out = transform(image, mask)
    assert set(out.unique().tolist()) == {3, IGNORE_INDEX}
    assert (out[:5, :10] == 3).all()  # original content stays top-left
    assert (out[-10:, :] == IGNORE_INDEX).all()  # padding is ignored


def test_denormalize_inverts_normalisation():
    image = Image.new("RGB", (4, 4), (10, 128, 250))
    restored = denormalize(image_to_tensor(image))
    expected = torch.tensor([10, 128, 250]) / 255.0
    assert torch.allclose(restored[:, 0, 0], expected, atol=1e-6)
