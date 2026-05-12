from pathlib import Path

import pytest

from nnunetv2.deployment.onnx_common import (
    checkpoint_path,
    fold_arg,
    fold_dir_name,
    normalize_fold,
    validate_configuration,
)


def test_normalize_fold_accepts_all_and_numeric_folds():
    assert normalize_fold("all") == "all"
    assert normalize_fold("fold_all") == "all"
    assert normalize_fold("0") == 0
    assert normalize_fold("fold_0") == 0
    assert normalize_fold("1") == 1
    assert normalize_fold(1) == 1

    with pytest.raises(RuntimeError, match="Fold must be"):
        normalize_fold("fold_nope")


def test_checkpoint_path_uses_nnunet_fold_directory_names():
    model_dir = Path("/tmp/model")

    assert checkpoint_path(model_dir, "all", "checkpoint_final.pth") == (
        model_dir / "fold_all" / "checkpoint_final.pth"
    )
    assert checkpoint_path(model_dir, "0", "checkpoint_final.pth") == (
        model_dir / "fold_0" / "checkpoint_final.pth"
    )
    assert checkpoint_path(model_dir, "fold_1", "checkpoint_final.pth") == (
        model_dir / "fold_1" / "checkpoint_final.pth"
    )
    assert fold_dir_name("1") == "fold_1"
    assert fold_arg("fold_1") == "1"
    assert fold_arg("fold_all") == "all"


def test_validate_configuration_accepts_supported_3d_configs():
    validate_configuration("3d_fullres")
    validate_configuration("3d_lowres")

    with pytest.raises(RuntimeError, match="Unsupported configuration"):
        validate_configuration("2d")
