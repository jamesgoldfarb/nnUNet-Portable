#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark existing nnU-Net inference speed knobs without changing the model or exporting to ONNX."
    )
    parser.add_argument("--input", required=True, help="Input image folder in nnU-Net prediction format.")
    parser.add_argument("--output", required=True, help="Output folder for benchmark segmentations and reports.")
    parser.add_argument("--model_dir", required=True, help="Trained nnUNetv2 model directory containing fold folders.")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth", help="Checkpoint filename. Default: checkpoint_final.pth")
    parser.add_argument("--fold", default="all", help="Fold to benchmark. Default: all")
    parser.add_argument("--configuration", default="3d_fullres", help="Expected nnU-Net configuration. Default: 3d_fullres")
    return parser.parse_args()


def _normalize_fold(fold: str):
    return fold if fold == "all" else int(fold)


def _case_ids(input_folder: Path, file_ending: str, create_lists_from_splitted_dataset_folder_fn: Any) -> list[str]:
    input_lists = create_lists_from_splitted_dataset_folder_fn(str(input_folder), file_ending)
    return [Path(i[0]).name[:-(len(file_ending) + 5)] for i in input_lists]


def _load_segmentation(path: Path, reader_writer: Any) -> np.ndarray:
    seg, _ = reader_writer.read_seg(str(path))
    return np.asarray(seg).squeeze()


def _dice_per_label(reference: np.ndarray, candidate: np.ndarray, labels: list[int]) -> dict[str, float]:
    dice = {}
    for label in labels:
        ref_mask = reference == label
        cand_mask = candidate == label
        denom = int(ref_mask.sum() + cand_mask.sum())
        if denom == 0:
            dice[str(label)] = 1.0
        else:
            dice[str(label)] = float(2 * np.logical_and(ref_mask, cand_mask).sum() / denom)
    return dice


def _compare_outputs(
    reference_dir: Path,
    candidate_dir: Path,
    case_ids: list[str],
    file_ending: str,
    reader_writer: Any,
    labels: list[int],
) -> dict[str, Any]:
    total_voxels = 0
    total_disagree = 0
    dice_accumulator = {str(label): [] for label in labels}
    cases = []

    for case_id in case_ids:
        ref = _load_segmentation(reference_dir / f"{case_id}{file_ending}", reader_writer)
        cand = _load_segmentation(candidate_dir / f"{case_id}{file_ending}", reader_writer)
        if ref.shape != cand.shape:
            raise RuntimeError(f"Shape mismatch for case {case_id}: reference {ref.shape}, candidate {cand.shape}")

        disagreement = ref != cand
        voxel_count = int(ref.size)
        disagree_count = int(disagreement.sum())
        case_dice = _dice_per_label(ref, cand, labels)
        for label, value in case_dice.items():
            dice_accumulator[label].append(value)

        total_voxels += voxel_count
        total_disagree += disagree_count
        cases.append(
            {
                "case_id": case_id,
                "voxel_disagreement": float(disagree_count / voxel_count) if voxel_count else 0.0,
                "dice_per_label": case_dice,
            }
        )

    return {
        "voxel_disagreement": float(total_disagree / total_voxels) if total_voxels else 0.0,
        "dice_per_label": {
            label: float(np.mean(values)) if values else None for label, values in dice_accumulator.items()
        },
        "cases": cases,
    }


def _write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    json_path = output_dir / "benchmark_report.json"
    csv_path = output_dir / "benchmark_report.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    all_labels = sorted(
        {
            label
            for result in report["results"]
            for label in (result.get("comparison") or {}).get("dice_per_label", {}).keys()
        },
        key=lambda value: int(value),
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "status",
            "runtime_seconds",
            "use_mirroring",
            "tile_step_size",
            "perform_everything_on_device",
            "voxel_disagreement",
            *[f"dice_label_{label}" for label in all_labels],
            "output_dir",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in report["results"]:
            comparison = result.get("comparison") or {}
            dice = comparison.get("dice_per_label", {})
            row = {
                "name": result["name"],
                "status": result["status"],
                "runtime_seconds": result.get("runtime_seconds"),
                "use_mirroring": result["settings"].get("use_mirroring"),
                "tile_step_size": result["settings"].get("tile_step_size"),
                "perform_everything_on_device": result["settings"].get("perform_everything_on_device"),
                "voxel_disagreement": comparison.get("voxel_disagreement"),
                "output_dir": result.get("output_dir"),
                "error": result.get("error"),
            }
            for label in all_labels:
                row[f"dice_label_{label}"] = dice.get(label)
            writer.writerow(row)

    print(f"Wrote benchmark report: {json_path}")
    print(f"Wrote benchmark CSV: {csv_path}")


def _make_predictor(
    predictor_cls: Any,
    predictor_params: Any,
    torch_module: Any,
    device: Any,
    tile_step_size: float,
    use_mirroring: bool,
    perform_everything_on_device: bool,
):
    kwargs = {
        "use_gaussian": True,
        "device": device,
        "verbose": False,
        "verbose_preprocessing": False,
        "allow_tqdm": True,
    }
    if "tile_step_size" in predictor_params:
        kwargs["tile_step_size"] = tile_step_size
    if "use_mirroring" in predictor_params:
        kwargs["use_mirroring"] = use_mirroring
    if "perform_everything_on_device" in predictor_params:
        kwargs["perform_everything_on_device"] = perform_everything_on_device
    return predictor_cls(**kwargs)


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
    except Exception as exc:
        print("Error: failed to import required nnU-Net benchmark dependencies.", file=sys.stderr)
        print(f"Full exception: {exc}", file=sys.stderr)
        return 2

    checkpoint_path = model_dir / f"fold_{args.fold}" / args.checkpoint
    if not checkpoint_path.is_file():
        print(f"Error: required checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    predictor_params = inspect.signature(nnUNetPredictor).parameters
    knob_accessibility = {
        "use_mirroring": "use_mirroring" in predictor_params,
        "tile_step_size": "tile_step_size" in predictor_params,
        "perform_everything_on_device": "perform_everything_on_device" in predictor_params,
    }
    unavailable = {
        knob: "not exposed by nnUNetPredictor.__init__"
        for knob, accessible in knob_accessibility.items()
        if not accessible
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_perform_on_device = bool(device.type == "cuda")
    if not knob_accessibility["perform_everything_on_device"]:
        baseline_perform_on_device = False

    initial_predictor = _make_predictor(
        nnUNetPredictor,
        predictor_params,
        torch,
        device,
        tile_step_size=0.5,
        use_mirroring=True,
        perform_everything_on_device=baseline_perform_on_device,
    )
    initial_predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=(_normalize_fold(args.fold),),
        checkpoint_name=args.checkpoint,
    )
    expected_configuration = initial_predictor.plans_manager.get_configuration(args.configuration).configuration
    if initial_predictor.configuration_manager.configuration != expected_configuration:
        print(f"Error: loaded model is not configuration {args.configuration}.", file=sys.stderr)
        return 2

    file_ending = initial_predictor.dataset_json["file_ending"]
    cases = _case_ids(input_dir, file_ending, create_lists_from_splitted_dataset_folder)
    labels = [int(i) for i in initial_predictor.label_manager.all_labels]
    reader_writer = initial_predictor.plans_manager.image_reader_writer_class()

    settings = [
        {
            "name": "default",
            "use_mirroring": True,
            "tile_step_size": 0.5,
            "perform_everything_on_device": baseline_perform_on_device,
        }
    ]
    if knob_accessibility["use_mirroring"]:
        settings.append(
            {
                "name": "mirroring_disabled",
                "use_mirroring": False,
                "tile_step_size": 0.5,
                "perform_everything_on_device": baseline_perform_on_device,
            }
        )
    if knob_accessibility["tile_step_size"]:
        for step in (0.75, 1.0):
            settings.append(
                {
                    "name": f"tile_step_size_{str(step).replace('.', '_')}",
                    "use_mirroring": True,
                    "tile_step_size": step,
                    "perform_everything_on_device": baseline_perform_on_device,
                }
            )
    if knob_accessibility["perform_everything_on_device"] and device.type == "cuda":
        settings.append(
            {
                "name": "perform_everything_on_device_false",
                "use_mirroring": True,
                "tile_step_size": 0.5,
                "perform_everything_on_device": False,
            }
        )
    elif knob_accessibility["perform_everything_on_device"]:
        unavailable["perform_everything_on_device_true_vs_false"] = "not applicable because CUDA is unavailable"

    report = {
        "input": str(input_dir),
        "output": str(output_dir),
        "model_dir": str(model_dir),
        "checkpoint": args.checkpoint,
        "fold": args.fold,
        "configuration": args.configuration,
        "device": str(device),
        "cases": cases,
        "labels": labels,
        "knob_accessibility": knob_accessibility,
        "unavailable_options": unavailable,
        "results": [],
    }

    baseline_dir = None
    for setting in settings:
        setting_output_dir = output_dir / setting["name"]
        setting_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nRunning setting: {setting['name']}")
        print(f"Output: {setting_output_dir}")

        try:
            predictor = _make_predictor(
                nnUNetPredictor,
                predictor_params,
                torch,
                device,
                tile_step_size=setting["tile_step_size"],
                use_mirroring=setting["use_mirroring"],
                perform_everything_on_device=setting["perform_everything_on_device"],
            )
            predictor.initialize_from_trained_model_folder(
                str(model_dir),
                use_folds=(_normalize_fold(args.fold),),
                checkpoint_name=args.checkpoint,
            )
            start = time.perf_counter()
            predictor.predict_from_files(str(input_dir), str(setting_output_dir), save_probabilities=False, overwrite=True)
            runtime = time.perf_counter() - start

            comparison = None
            if setting["name"] == "default":
                baseline_dir = setting_output_dir
                comparison = {
                    "voxel_disagreement": 0.0,
                    "dice_per_label": {str(label): 1.0 for label in labels},
                    "cases": [],
                }
            else:
                comparison = _compare_outputs(
                    baseline_dir,
                    setting_output_dir,
                    cases,
                    file_ending,
                    reader_writer,
                    labels,
                )

            report["results"].append(
                {
                    "name": setting["name"],
                    "status": "ok",
                    "settings": setting,
                    "runtime_seconds": runtime,
                    "output_dir": str(setting_output_dir),
                    "comparison": comparison,
                }
            )
            _write_reports(output_dir, report)
        except Exception as exc:
            report["results"].append(
                {
                    "name": setting["name"],
                    "status": "failed",
                    "settings": setting,
                    "output_dir": str(setting_output_dir),
                    "error": str(exc),
                }
            )
            _write_reports(output_dir, report)
            print(f"Setting failed: {setting['name']}: {exc}", file=sys.stderr)

    return 0 if all(result["status"] == "ok" for result in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
