import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from nnunetv2.deployment.compare_segmentations import _load_segmentation, compare_arrays
from nnunetv2.deployment.onnx_common import (
    SUPPORTED_CONFIGURATIONS,
    checkpoint_path,
    normalize_fold,
    validate_configuration,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyTorch and ONNX Runtime nnU-Net predictions, then compare final segmentations."
    )
    parser.add_argument("--input", required=True, help="Input image folder in nnU-Net prediction format.")
    parser.add_argument("--output", required=True, help="Output folder. Subfolders pytorch and onnxruntime are created.")
    parser.add_argument("--model_dir", required=True, help="Trained nnUNetv2 model directory containing fold folders.")
    parser.add_argument(
        "--configuration",
        default="3d_fullres",
        choices=SUPPORTED_CONFIGURATIONS,
        help="Expected nnU-Net configuration. Default: 3d_fullres",
    )
    parser.add_argument("--fold", default="all", help="Fold to run. Default: all")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth", help="Checkpoint filename. Default: checkpoint_final.pth")
    parser.add_argument("--onnx_model", required=True, help="Fixed-shape ONNX model for ONNX Runtime backend.")
    parser.add_argument("--ort_provider", default="CPUExecutionProvider", help="ONNX Runtime provider. Default: CPUExecutionProvider")
    return parser.parse_args()


def _make_predictor(torch_module: Any, predictor_cls: Any, device: Any):
    return predictor_cls(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )


def _run_prediction(
    predictor_cls: Any,
    torch_module: Any,
    model_dir: Path,
    fold,
    checkpoint: str,
    input_dir: Path,
    output_dir: Path,
    device: Any,
    inference_backend: Any = None,
) -> tuple[Any, float]:
    predictor = _make_predictor(torch_module, predictor_cls, device)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(fold,), checkpoint_name=checkpoint)
    if inference_backend is not None:
        predictor.inference_backend = inference_backend

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    predictor.predict_from_files(str(input_dir), str(output_dir), save_probabilities=False, overwrite=True)
    runtime = time.perf_counter() - start
    return predictor, runtime


def _segmentation_files(output_dir: Path, file_ending: str) -> dict[str, Path]:
    files = {}
    for path in output_dir.iterdir():
        if path.is_file() and path.name.endswith(file_ending):
            case_id = path.name[:-len(file_ending)]
            files[case_id] = path
    return files


def _compare_outputs(pytorch_dir: Path, onnx_dir: Path, file_ending: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pytorch_files = _segmentation_files(pytorch_dir, file_ending)
    onnx_files = _segmentation_files(onnx_dir, file_ending)
    if not pytorch_files:
        raise RuntimeError(f"No PyTorch output segmentations found in {pytorch_dir}")
    if not onnx_files:
        raise RuntimeError(f"No ONNX Runtime output segmentations found in {onnx_dir}")

    pytorch_cases = set(pytorch_files)
    onnx_cases = set(onnx_files)
    if pytorch_cases != onnx_cases:
        raise RuntimeError(
            f"Case matching failed. Only in PyTorch: {sorted(pytorch_cases - onnx_cases)}. "
            f"Only in ONNX Runtime: {sorted(onnx_cases - pytorch_cases)}."
        )

    cases = []
    total_voxels = 0
    total_disagree = 0
    dice_by_label: dict[str, list[float]] = {}
    for case_id in sorted(pytorch_cases):
        pytorch_seg = _load_segmentation(pytorch_files[case_id])
        onnx_seg = _load_segmentation(onnx_files[case_id])
        comparison = compare_arrays(pytorch_seg, onnx_seg)
        voxel_count = int(pytorch_seg.size)
        total_voxels += voxel_count
        total_disagree += int(np.sum(pytorch_seg != onnx_seg))
        for label, dice in comparison["dice_per_label"].items():
            dice_by_label.setdefault(label, []).append(dice)
        comparison["case_id"] = case_id
        comparison["pytorch_segmentation"] = str(pytorch_files[case_id])
        comparison["onnxruntime_segmentation"] = str(onnx_files[case_id])
        cases.append(comparison)

    aggregate = {
        "voxel_disagreement_percent": float(total_disagree / total_voxels * 100.0) if total_voxels else 0.0,
        "dice_per_label": {label: float(np.mean(values)) for label, values in sorted(dice_by_label.items())},
    }
    foreground_values = [case["foreground_dice"] for case in cases if case["foreground_dice"] is not None]
    aggregate["foreground_dice"] = float(np.mean(foreground_values)) if foreground_values else None
    return cases, aggregate


def _print_summary(report: dict[str, Any]) -> None:
    print("Backend comparison")
    print(f"PyTorch runtime: {report['runtime_seconds']['pytorch']:.3f} s")
    print(f"ONNX Runtime runtime: {report['runtime_seconds']['onnxruntime']:.3f} s")
    print(f"Voxel disagreement: {report['aggregate']['voxel_disagreement_percent']:.6f}%")
    print("Dice per label:")
    for label, dice in report["aggregate"]["dice_per_label"].items():
        print(f"  label {label}: {dice:.8f}")
    if report["aggregate"]["foreground_dice"] is not None:
        print(f"Foreground Dice: {report['aggregate']['foreground_dice']:.8f}")
    print(f"PyTorch outputs: {report['output_dirs']['pytorch']}")
    print(f"ONNX Runtime outputs: {report['output_dirs']['onnxruntime']}")
    print(f"Comparison report: {report['comparison_report_json']}")


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    onnx_model = Path(args.onnx_model).expanduser().resolve()
    pytorch_dir = output_dir / "pytorch"
    onnx_dir = output_dir / "onnxruntime"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"Error: input folder does not exist: {input_dir}", file=sys.stderr)
        return 2
    try:
        resolved_checkpoint_path = checkpoint_path(model_dir, args.fold, args.checkpoint)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if not resolved_checkpoint_path.is_file():
        print(f"Error: checkpoint not found: {resolved_checkpoint_path}", file=sys.stderr)
        return 2
    if not onnx_model.is_file():
        print(f"Error: ONNX model does not exist: {onnx_model}", file=sys.stderr)
        return 2

    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.torch_inference_backend import OnnxRuntimeInferenceBackend
    except Exception as exc:
        print("Error: failed to import required backend comparison dependencies.", file=sys.stderr)
        print(f"Full exception: {exc}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold = normalize_fold(args.fold)
    try:
        validate_configuration(args.configuration)
        pytorch_predictor, pytorch_runtime = _run_prediction(
            nnUNetPredictor, torch, model_dir, fold, args.checkpoint, input_dir, pytorch_dir, device
        )
        expected_configuration = pytorch_predictor.plans_manager.get_configuration(args.configuration).configuration
        if pytorch_predictor.configuration_manager.configuration != expected_configuration:
            raise RuntimeError(f"Loaded model is not configuration {args.configuration}.")

        onnx_backend = OnnxRuntimeInferenceBackend(str(onnx_model), provider=args.ort_provider)
        onnx_predictor, onnx_runtime = _run_prediction(
            nnUNetPredictor,
            torch,
            model_dir,
            fold,
            args.checkpoint,
            input_dir,
            onnx_dir,
            device,
            inference_backend=onnx_backend,
        )
        if onnx_predictor.dataset_json["file_ending"] != pytorch_predictor.dataset_json["file_ending"]:
            raise RuntimeError("PyTorch and ONNX Runtime predictors disagree on output file ending.")

        cases, aggregate = _compare_outputs(pytorch_dir, onnx_dir, pytorch_predictor.dataset_json["file_ending"])
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    report_path = output_dir / "comparison_report.json"
    report = {
        "input": str(input_dir),
        "model_dir": str(model_dir),
        "configuration": args.configuration,
        "fold": args.fold,
        "checkpoint": args.checkpoint,
        "onnx_model": str(onnx_model),
        "ort_provider": args.ort_provider,
        "device": str(device),
        "runtime_seconds": {
            "pytorch": pytorch_runtime,
            "onnxruntime": onnx_runtime,
        },
        "output_dirs": {
            "pytorch": str(pytorch_dir),
            "onnxruntime": str(onnx_dir),
        },
        "aggregate": aggregate,
        "cases": cases,
        "comparison_report_json": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
