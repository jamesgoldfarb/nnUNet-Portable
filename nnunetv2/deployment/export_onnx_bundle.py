import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROVIDER = "CPUExecutionProvider"
INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a minimal portable nnU-Net ONNX inference bundle."
    )
    parser.add_argument("--model_dir", required=True, help="Trained nnUNetv2 model directory with plans.json and dataset.json.")
    parser.add_argument("--onnx_model", required=True, help="Already-exported ONNX model containing the network forward pass.")
    parser.add_argument("--output_bundle", required=True, help="Output bundle directory.")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth", help="Checkpoint filename. Default: checkpoint_final.pth")
    parser.add_argument("--configuration", default="3d_fullres", help="nnU-Net configuration. Default: 3d_fullres")
    parser.add_argument("--fold", default="all", help="Fold represented by the ONNX export. Default: all")
    return parser.parse_args()


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _configuration_from_plans(plans: dict[str, Any], configuration: str) -> dict[str, Any] | None:
    configurations = plans.get("configurations", {})
    if configuration not in configurations:
        return None
    config = dict(configurations[configuration])
    parent = config.get("inherits_from")
    if parent is not None and parent in configurations:
        inherited = dict(configurations[parent])
        inherited.update(config)
        config = inherited
    return config


def _bundle_metadata(
    plans: dict[str, Any],
    model_export: dict[str, Any] | None,
    configuration: str,
    fold: str,
    checkpoint: str,
) -> dict[str, Any]:
    plans_configuration = _configuration_from_plans(plans, configuration)
    patch_size = None
    num_input_channels = None
    input_name = INPUT_NAME
    output_name = OUTPUT_NAME

    if model_export is not None:
        patch_size = model_export.get("patch_size")
        num_input_channels = model_export.get("num_input_channels")
        input_name = model_export.get("input_name", input_name)
        output_name = model_export.get("output_name", output_name)

    if patch_size is None and plans_configuration is not None:
        patch_size = plans_configuration.get("patch_size")

    return {
        "configuration": configuration,
        "fold": fold,
        "checkpoint": checkpoint,
        "onnx_model_filename": "model.onnx",
        "model_export_json_filename": "model_export.json" if model_export is not None else None,
        "patch_size": patch_size,
        "num_input_channels": num_input_channels,
        "input_tensor_name": input_name,
        "output_tensor_name": output_name,
        "default_provider": DEFAULT_PROVIDER,
        "required_python_packages": [
            "nnunetv2",
            "onnxruntime",
            "numpy",
            "scipy",
            "SimpleITK",
            "nibabel",
        ],
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _readme_text(config: dict[str, Any]) -> str:
    return f"""# nnU-Net ONNX Inference Bundle

This bundle contains a fixed-shape ONNX export of the nnU-Net network forward pass.

## Contents

- `model.onnx`: ONNX model for patch-level raw logits.
- `model_export.json`: optional metadata from ONNX export, if available.
- `plans.json`: nnU-Net plans copied from the trained model folder.
- `dataset.json`: nnU-Net dataset metadata copied from the trained model folder.
- `inference_config.json`: bundle metadata and default runtime settings.

## Requirements

This bundle still requires an installed `nnunetv2` environment. The ONNX model contains only the network forward pass.
nnU-Net still performs image loading, preprocessing, sliding-window inference support, Gaussian blending, resampling,
postprocessing, and final segmentation export.

Install runtime packages compatible with your model, including `nnunetv2`, `onnxruntime`, `numpy`, and the image I/O
packages used by the dataset such as `SimpleITK` or `nibabel`.

## Example

```bash
nnUNetv2_predict \\
  -i imagesTs \\
  -o output_onnx \\
  -d DatasetXXX \\
  -c {config["configuration"]} \\
  -f {config["fold"]} \\
  --backend onnxruntime \\
  --onnx_model /path/to/this/bundle/model.onnx \\
  --ort_provider {config["default_provider"]}
```

The ONNX model is expected to use input tensor `{config["input_tensor_name"]}` and output tensor
`{config["output_tensor_name"]}`.
"""


def create_bundle(model_dir: Path, onnx_model: Path, output_bundle: Path, checkpoint: str, configuration: str, fold: str):
    plans_path = model_dir / "plans.json"
    dataset_path = model_dir / "dataset.json"
    model_export_path = onnx_model.parent / "model_export.json"

    if not model_dir.is_dir():
        raise RuntimeError(f"model_dir does not exist: {model_dir}")
    if not onnx_model.is_file():
        raise RuntimeError(f"ONNX model does not exist: {onnx_model}")
    if not plans_path.is_file():
        raise RuntimeError(f"plans.json not found in model_dir: {plans_path}")
    if not dataset_path.is_file():
        raise RuntimeError(f"dataset.json not found in model_dir: {dataset_path}")

    output_bundle.mkdir(parents=True, exist_ok=True)

    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    model_export = _load_json_if_present(model_export_path)
    inference_config = _bundle_metadata(plans, model_export, configuration, fold, checkpoint)

    shutil.copy2(onnx_model, output_bundle / "model.onnx")
    if model_export_path.is_file():
        shutil.copy2(model_export_path, output_bundle / "model_export.json")
    shutil.copy2(plans_path, output_bundle / "plans.json")
    shutil.copy2(dataset_path, output_bundle / "dataset.json")
    (output_bundle / "inference_config.json").write_text(
        json.dumps(inference_config, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_bundle / "README.md").write_text(_readme_text(inference_config), encoding="utf-8")
    return inference_config


def main() -> int:
    args = _parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    onnx_model = Path(args.onnx_model).expanduser().resolve()
    output_bundle = Path(args.output_bundle).expanduser().resolve()

    try:
        create_bundle(model_dir, onnx_model, output_bundle, args.checkpoint, args.configuration, args.fold)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Created ONNX inference bundle: {output_bundle}")
    print("Bundle contents:")
    for path in sorted(output_bundle.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
