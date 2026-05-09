import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_onnx_console_entrypoints_are_registered():
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        scripts = tomllib.load(f)["project"]["scripts"]

    assert scripts["nnUNetv2_export_onnx"] == "nnunetv2.deployment.export_onnx:main"
    assert scripts["nnUNetv2_export_onnx_bundle"] == "nnunetv2.deployment.export_onnx_bundle:main"
    assert scripts["nnUNetv2_validate_onnx_random_patch"] == "nnunetv2.deployment.validate_onnx_random_patch:main"
    assert scripts["nnUNetv2_validate_onnx_preprocessed_patch"] == (
        "nnunetv2.deployment.validate_onnx_preprocessed_patch:main"
    )
    assert scripts["nnUNetv2_compare_segmentations"] == "nnunetv2.deployment.compare_segmentations:main"
    assert scripts["nnUNetv2_compare_backends"] == "nnunetv2.deployment.compare_backends:main"


def test_onnx_entrypoint_modules_show_help():
    modules = [
        "nnunetv2.deployment.export_onnx",
        "nnunetv2.deployment.export_onnx_bundle",
        "nnunetv2.deployment.validate_onnx_random_patch",
        "nnunetv2.deployment.validate_onnx_preprocessed_patch",
        "nnunetv2.deployment.compare_segmentations",
        "nnunetv2.deployment.compare_backends",
    ]
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
