import pytest

from nnunetv2.deployment.compare_backends import _compare_outputs, _segmentation_files


def test_segmentation_files_indexes_by_case_id(tmp_path):
    (tmp_path / "case_001.nii.gz").write_text("x")
    (tmp_path / "case_002.nii.gz").write_text("x")
    (tmp_path / "plans.json").write_text("{}")

    files = _segmentation_files(tmp_path, ".nii.gz")

    assert sorted(files) == ["case_001", "case_002"]


def test_compare_outputs_rejects_case_mismatch(tmp_path):
    pytorch_dir = tmp_path / "pytorch"
    onnx_dir = tmp_path / "onnxruntime"
    pytorch_dir.mkdir()
    onnx_dir.mkdir()
    (pytorch_dir / "case_a.nii.gz").write_text("x")
    (onnx_dir / "case_b.nii.gz").write_text("x")

    with pytest.raises(RuntimeError, match="Case matching failed"):
        _compare_outputs(pytorch_dir, onnx_dir, ".nii.gz")
