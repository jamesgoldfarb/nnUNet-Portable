import json

from nnunetv2.deployment.export_onnx_bundle import create_bundle


def test_create_bundle_copies_expected_files_and_writes_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    output_bundle = tmp_path / "bundle"

    (model_dir / "plans.json").write_text(
        json.dumps({"configurations": {"3d_fullres": {"patch_size": [16, 32, 48]}}}),
        encoding="utf-8",
    )
    (model_dir / "dataset.json").write_text(json.dumps({"file_ending": ".nii.gz"}), encoding="utf-8")
    (onnx_dir / "model.onnx").write_bytes(b"onnx")
    (onnx_dir / "model_export.json").write_text(
        json.dumps(
            {
                "patch_size": [16, 32, 48],
                "num_input_channels": 2,
                "input_name": "input",
                "output_name": "logits",
            }
        ),
        encoding="utf-8",
    )

    create_bundle(model_dir, onnx_dir / "model.onnx", output_bundle, "checkpoint_final.pth", "3d_fullres", "all")

    assert sorted(path.name for path in output_bundle.iterdir()) == [
        "README.md",
        "dataset.json",
        "inference_config.json",
        "model.onnx",
        "model_export.json",
        "plans.json",
    ]
    config = json.loads((output_bundle / "inference_config.json").read_text(encoding="utf-8"))
    assert config["configuration"] == "3d_fullres"
    assert config["fold"] == "all"
    assert config["checkpoint"] == "checkpoint_final.pth"
    assert config["patch_size"] == [16, 32, 48]
    assert config["num_input_channels"] == 2
    assert config["input_tensor_name"] == "input"
    assert config["output_tensor_name"] == "logits"
    assert config["default_provider"] == "CPUExecutionProvider"
    assert "nnunetv2" in config["required_python_packages"]
