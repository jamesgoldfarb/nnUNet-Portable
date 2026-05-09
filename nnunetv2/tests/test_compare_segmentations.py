import numpy as np
import pytest

from nnunetv2.deployment.compare_segmentations import compare_arrays


def test_compare_arrays_reports_disagreement_and_dice():
    seg_a = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    seg_b = np.array([[0, 1], [2, 2]], dtype=np.uint8)

    report = compare_arrays(seg_a, seg_b)

    assert report["shape"] == [2, 2]
    assert report["labels_a"] == [0, 1, 2]
    assert report["labels_b"] == [0, 1, 2]
    assert report["voxel_disagreement_percent"] == 25.0
    assert report["dice_per_label"]["0"] == 1.0
    assert report["dice_per_label"]["1"] == pytest.approx(2 / 3)
    assert report["dice_per_label"]["2"] == pytest.approx(2 / 3)
    assert report["foreground_dice"] == 1.0


def test_compare_arrays_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_arrays(np.zeros((2, 2)), np.zeros((2, 3)))
