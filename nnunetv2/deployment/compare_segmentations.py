import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two final segmentation files, for example PyTorch nnU-Net vs ONNX Runtime output."
    )
    parser.add_argument("--seg_a", required=True, help="First segmentation file.")
    parser.add_argument("--seg_b", required=True, help="Second segmentation file.")
    parser.add_argument("--output_json", required=False, default=None, help="Optional path for comparison_report.json.")
    return parser.parse_args()


def _load_segmentation(path: Path) -> np.ndarray:
    from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_file_ending

    try:
        reader_writer_cls = determine_reader_writer_from_file_ending(
            "".join(path.suffixes) if path.name.endswith(".nii.gz") else path.suffix,
            str(path),
            allow_nonmatching_filename=True,
            verbose=False,
        )
        segmentation, _ = reader_writer_cls().read_seg(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not read segmentation file {path}: {exc}") from exc

    return np.asarray(segmentation).squeeze()


def _dice_for_label(seg_a: np.ndarray, seg_b: np.ndarray, label: int) -> float:
    a_mask = seg_a == label
    b_mask = seg_b == label
    denominator = int(a_mask.sum() + b_mask.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(a_mask, b_mask).sum() / denominator)


def compare_arrays(seg_a: np.ndarray, seg_b: np.ndarray) -> dict[str, Any]:
    if seg_a.shape != seg_b.shape:
        raise ValueError(f"Shape mismatch: seg_a {seg_a.shape}, seg_b {seg_b.shape}")

    labels_a = sorted(int(i) for i in np.unique(seg_a))
    labels_b = sorted(int(i) for i in np.unique(seg_b))
    labels_union = sorted(set(labels_a) | set(labels_b))
    foreground_labels = [label for label in labels_union if label != 0]
    voxel_disagreement = float(np.mean(seg_a != seg_b) * 100.0)
    dice_per_label = {str(label): _dice_for_label(seg_a, seg_b, label) for label in labels_union}
    foreground_dice = None
    if foreground_labels:
        foreground_dice = _dice_for_label(seg_a != 0, seg_b != 0, True)

    return {
        "shape": list(seg_a.shape),
        "labels_a": labels_a,
        "labels_b": labels_b,
        "voxel_disagreement_percent": voxel_disagreement,
        "dice_per_label": dice_per_label,
        "foreground_dice": foreground_dice,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print("Segmentation comparison")
    print(f"Shape: {tuple(report['shape'])}")
    print(f"Labels in seg_a: {report['labels_a']}")
    print(f"Labels in seg_b: {report['labels_b']}")
    print(f"Voxel disagreement: {report['voxel_disagreement_percent']:.6f}%")
    print("Dice per label:")
    for label, dice in report["dice_per_label"].items():
        print(f"  label {label}: {dice:.8f}")
    if report["foreground_dice"] is not None:
        print(f"Foreground Dice: {report['foreground_dice']:.8f}")
    else:
        print("Foreground Dice: not applicable")


def main() -> int:
    args = _parse_args()
    seg_a_path = Path(args.seg_a).expanduser().resolve()
    seg_b_path = Path(args.seg_b).expanduser().resolve()

    if not seg_a_path.is_file():
        print(f"Error: seg_a does not exist: {seg_a_path}", file=sys.stderr)
        return 2
    if not seg_b_path.is_file():
        print(f"Error: seg_b does not exist: {seg_b_path}", file=sys.stderr)
        return 2

    try:
        seg_a = _load_segmentation(seg_a_path)
        seg_b = _load_segmentation(seg_b_path)
        report = compare_arrays(seg_a, seg_b)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    report["seg_a"] = str(seg_a_path)
    report["seg_b"] = str(seg_b_path)
    _print_summary(report)

    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote comparison report: {output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
