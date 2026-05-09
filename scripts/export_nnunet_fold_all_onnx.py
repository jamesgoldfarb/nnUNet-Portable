#!/usr/bin/env python3
import argparse
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


INPUT_NAME = "input"
OUTPUT_NAME = "logits"
SUPPORTED_CONFIGURATION = "3d_fullres"


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unavailable"


def _versions(torch_module: Any = None, onnx_module: Any = None) -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": getattr(torch_module, "__version__", _package_version("torch")),
        "nnunetv2": _package_version("nnunetv2"),
        "onnx": getattr(onnx_module, "__version__", _package_version("onnx")),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fold_all/checkpoint_final.pth nnUNetv2 3d_fullres network forward pass to fixed-shape ONNX."
    )
    parser.add_argument("--model_dir", required=True, help="Trained nnUNetv2 model directory containing fold_all.")
    parser.add_argument("--output_onnx", required=True, help="Output ONNX file path.")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth", help="Checkpoint filename. Default: checkpoint_final.pth")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version. Default: 17")
    parser.add_argument("--device", default="cpu", help="Torch device for export. Default: cpu")
    return parser.parse_args()


def _validate_device(device_arg: str, torch_module: Any) -> Any:
    device = torch_module.device(device_arg)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device, but torch.cuda.is_available() is False.")
    return device


def _write_export_metadata(
    output_onnx: Path,
    model_dir: Path,
    checkpoint: str,
    patch_size: tuple[int, ...],
    num_input_channels: int,
    opset: int,
    torch_module: Any,
    onnx_module: Any,
) -> None:
    metadata = {
        "model_dir": str(model_dir),
        "checkpoint": checkpoint,
        "fold": "all",
        "configuration": SUPPORTED_CONFIGURATION,
        "patch_size": list(patch_size),
        "num_input_channels": num_input_channels,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "opset": opset,
        "python_version": platform.python_version(),
        "torch_version": torch_module.__version__,
        "nnunetv2_version": _package_version("nnunetv2"),
        "onnx_version": getattr(onnx_module, "__version__", _package_version("onnx")),
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_onnx.parent / "model_export.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote export metadata: {metadata_path}")


def main() -> int:
    args = _parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_onnx = Path(args.output_onnx).expanduser().resolve()
    checkpoint_path = model_dir / "fold_all" / args.checkpoint

    if not checkpoint_path.is_file():
        print(f"Error: required checkpoint not found: {checkpoint_path}", file=sys.stderr)
        print(f"Expected fold_all/{args.checkpoint} for this exporter scope.", file=sys.stderr)
        return 2

    torch = None
    onnx = None
    try:
        import torch
        import onnx
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    except Exception as exc:
        print("Error: failed to import required export dependencies.", file=sys.stderr)
        print("Install nnUNetv2, torch, onnx, and nnU-Net runtime dependencies before exporting.", file=sys.stderr)
        print(f"Full exception: {exc}", file=sys.stderr)
        return 2

    try:
        device = _validate_device(args.device, torch)

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=("all",), checkpoint_name=args.checkpoint)

        expected_configuration = predictor.plans_manager.get_configuration(SUPPORTED_CONFIGURATION).configuration
        if predictor.configuration_manager.configuration != expected_configuration:
            raise RuntimeError(f"Only {SUPPORTED_CONFIGURATION} is supported by this exporter.")

        if predictor.configuration_manager.previous_stage_name is not None:
            raise RuntimeError("Cascaded configurations are not supported by this exporter.")

        patch_size = tuple(int(i) for i in predictor.configuration_manager.patch_size)
        if len(patch_size) != 3:
            raise RuntimeError(f"Only 3D patch sizes are supported. Got patch_size={patch_size}")

        num_input_channels = determine_num_input_channels(
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        input_shape = (1, int(num_input_channels), *patch_size)

        network = predictor.network.to(device)
        network.eval()
        dummy_input = torch.zeros(input_shape, dtype=torch.float32, device=device)

        with torch.no_grad():
            torch_output = network(dummy_input)
        print(f"PyTorch output shape: {tuple(torch_output.shape)}")

        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        print(f"Exporting ONNX: {output_onnx}")

        with torch.no_grad():
            torch.onnx.export(
                network,
                dummy_input,
                str(output_onnx),
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                opset_version=args.opset,
                external_data=True,
            )

        print("Running ONNX checker")
        onnx.checker.check_model(str(output_onnx))
        _write_export_metadata(
            output_onnx,
            model_dir,
            args.checkpoint,
            patch_size,
            int(num_input_channels),
            args.opset,
            torch,
            onnx,
        )
        print("ONNX export complete")
        return 0

    except Exception as exc:
        versions = _versions(locals().get("torch"), locals().get("onnx"))
        network_class = type(getattr(locals().get("predictor", None), "network", None)).__name__
        print("ONNX export failed.", file=sys.stderr)
        print(f"Network class: {network_class}", file=sys.stderr)
        print(f"Input shape: {locals().get('input_shape', 'unavailable')}", file=sys.stderr)
        print(f"Opset: {args.opset}", file=sys.stderr)
        print(f"Package versions: {versions}", file=sys.stderr)
        print(f"Full exception: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
