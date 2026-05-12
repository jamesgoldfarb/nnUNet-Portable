import argparse
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict

from nnunetv2.deployment.onnx_common import (
    SUPPORTED_CONFIGURATION,
    SUPPORTED_CONFIGURATIONS,
    checkpoint_path,
    fold_arg,
    load_predictor_for_export,
)


INPUT_NAME = "input"
OUTPUT_NAME = "logits"


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
        description="Export an nnUNetv2 3D network forward pass to fixed-shape ONNX."
    )
    parser.add_argument("--model_dir", required=True, help="Trained nnUNetv2 model directory containing fold folders.")
    parser.add_argument("--output_onnx", required=True, help="Output ONNX file path.")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth", help="Checkpoint filename. Default: checkpoint_final.pth")
    parser.add_argument(
        "--configuration",
        default=SUPPORTED_CONFIGURATION,
        choices=SUPPORTED_CONFIGURATIONS,
        help="Expected nnU-Net configuration. Default: 3d_fullres",
    )
    parser.add_argument("--fold", default="all", help="Fold to export: all, 0, 1, etc. Default: all")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version. Default: 17")
    parser.add_argument("--device", default="cpu", help="Torch device for export. Default: cpu")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export network weights and dummy input as float16. Requires --device cuda. Default: fp32",
    )
    return parser.parse_args()


def _validate_device(device_arg: str, torch_module: Any) -> Any:
    device = torch_module.device(device_arg)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device, but torch.cuda.is_available() is False.")
    return device


def _validate_precision(fp16: bool, device: Any) -> None:
    if fp16 and device.type != "cuda":
        raise RuntimeError(
            "--fp16 export requires --device cuda. PyTorch CPU float16 convolution export is not supported reliably."
        )


def _write_export_metadata(
    output_onnx: Path,
    model_dir: Path,
    checkpoint: str,
    fold: str,
    configuration: str,
    patch_size: tuple[int, ...],
    num_input_channels: int,
    opset: int,
    precision: str,
    input_dtype: str,
    output_dtype: str,
    torch_module: Any,
    onnx_module: Any,
) -> None:
    metadata = {
        "model_dir": str(model_dir),
        "checkpoint": checkpoint,
        "fold": fold,
        "configuration": configuration,
        "patch_size": list(patch_size),
        "patch_size_order": "nnU-Net network tensor spatial order, matching input tensor shape [N, C, *patch_size]",
        "input_shape": [1, num_input_channels, *patch_size],
        "num_input_channels": num_input_channels,
        "precision": precision,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
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


def _get_onnx_input_shape(onnx_module: Any, output_onnx: Path, input_name: str) -> list[int] | None:
    model = onnx_module.load(str(output_onnx), load_external_data=False)
    for graph_input in model.graph.input:
        if graph_input.name != input_name:
            continue
        dims = []
        for dim in graph_input.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            else:
                return None
        return dims
    raise RuntimeError(f"ONNX graph input '{input_name}' was not found in {output_onnx}")


def main() -> int:
    args = _parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_onnx = Path(args.output_onnx).expanduser().resolve()
    try:
        resolved_checkpoint_path = checkpoint_path(model_dir, args.fold, args.checkpoint)
        normalized_fold_arg = fold_arg(args.fold)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not resolved_checkpoint_path.is_file():
        print(f"Error: required checkpoint not found: {resolved_checkpoint_path}", file=sys.stderr)
        print(f"Expected {resolved_checkpoint_path.parent.name}/{args.checkpoint}.", file=sys.stderr)
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
        _validate_precision(args.fp16, device)
        export_dtype = torch.float16 if args.fp16 else torch.float32
        precision = "fp16" if args.fp16 else "fp32"
        predictor, patch_size, num_input_channels = load_predictor_for_export(
            model_dir,
            args.checkpoint,
            torch,
            nnUNetPredictor,
            determine_num_input_channels,
            configuration=args.configuration,
            fold=args.fold,
        )
        input_shape = (1, num_input_channels, *patch_size)
        print(f"nnU-Net configuration used for export: {args.configuration}")
        print(f"nnU-Net fold used for export: {normalized_fold_arg}")
        print(f"nnU-Net patch size from plans.json used for export: {patch_size}")
        print(f"ONNX input shape used for export: {input_shape}")
        print(f"ONNX export precision: {precision}")

        network = predictor.network.to(device)
        if args.fp16:
            network = network.half()
        network.eval()
        dummy_input = torch.zeros(input_shape, dtype=export_dtype, device=device)

        with torch.no_grad():
            torch_output = network(dummy_input)
        print(f"PyTorch output shape: {tuple(torch_output.shape)}")
        print(f"PyTorch output dtype: {torch_output.dtype}")

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
        exported_input_shape = _get_onnx_input_shape(onnx, output_onnx, INPUT_NAME)
        if exported_input_shape is not None and exported_input_shape != list(input_shape):
            raise RuntimeError(
                f"Exported ONNX input shape {exported_input_shape} does not match dummy input shape {list(input_shape)}. "
                "Re-export with the same model_dir/configuration/fold used by nnUNetv2_predict."
            )
        _write_export_metadata(
            output_onnx,
            model_dir,
            args.checkpoint,
            normalized_fold_arg,
            args.configuration,
            patch_size,
            num_input_channels,
            args.opset,
            precision,
            str(dummy_input.dtype).replace("torch.", ""),
            str(torch_output.dtype).replace("torch.", ""),
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
