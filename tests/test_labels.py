import numpy as np

from cityseg.labels import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    TRAIN_ID_TO_ID,
    colorize,
    encode_labels,
)


def test_nineteen_training_classes():
    assert NUM_CLASSES == 19
    assert CLASS_NAMES[0] == "road"
    assert CLASS_NAMES[-1] == "bicycle"


def test_void_ids_are_ignored_not_a_class():
    raw = np.array([[0, 1, 6, 9, 29, 30, 255]], dtype=np.uint8)
    assert (encode_labels(raw) == IGNORE_INDEX).all()


def test_valid_ids_map_to_train_ids_and_back():
    raw = np.array([[7, 8, 26, 33]], dtype=np.uint8)
    encoded = encode_labels(raw)
    assert encoded.tolist() == [[0, 1, 13, 18]]
    assert TRAIN_ID_TO_ID[encoded].tolist() == raw.tolist()


def test_colorize_uses_cityscapes_palette():
    color = colorize(np.array([[0, 10, IGNORE_INDEX]], dtype=np.uint8))
    assert color.shape == (1, 3, 3)
    assert color[0, 0].tolist() == [128, 64, 128]  # road
    assert color[0, 1].tolist() == [70, 130, 180]  # sky
    assert color[0, 2].tolist() == [0, 0, 0]  # ignored
