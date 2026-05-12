import argparse
import sys
from pathlib import Path

from nnunetv2.deployment.onnx_common import (
    SUPPORTED_CONFIGURATION,
    SUPPORTED_CONFIGURATIONS,
    checkpoint_path,
    load_predictor_for_export,
    report_comparison,
    validate_provider,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nnUNetv2 PyTorch logits against ONNX Runtime logits on a preprocessed patch."
    )
    parser.add_argument("--model_dir", required=True, help="Trained nnUNetv2 model directory containing fold folders.")
    parser.add_argument("--onnx_model", required=True, help="Fixed-shape ONNX model exported from the same nnUNet model.")
    parser.add_argument(
        "--preprocessed_patch",
        required=True,
        help="Path to a preprocessed .npy patch or .npz file with a 'data' array.",
    )
    parser.add_argument("--checkpoint", default="checkpoint_final.pth", help="Checkpoint filename. Default: checkpoint_final.pth")
    parser.add_argument(
        "--configuration",
        default=SUPPORTED_CONFIGURATION,
        choices=SUPPORTED_CONFIGURATIONS,
        help="Expected nnU-Net configuration. Default: 3d_fullres",
    )
    parser.add_argument("--fold", default="all", help="Fold to validate: all, 0, 1, etc. Default: all")
    parser.add_argument("--provider", default="CPUExecutionProvider", help="ONNX Runtime provider. Default: CPUExecutionProvider")
    return parser.parse_args()


def _load_preprocessed_patch(path: Path):
    import numpy as np

    if path.suffix == ".npy":
        patch = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as loaded:
            if "data" not in loaded:
                raise RuntimeError(f"Expected key 'data' in npz file: {path}")
            patch = loaded["data"]
    else:
        raise RuntimeError(f"Expected .npy or .npz preprocessed patch. Got: {path}")

    patch = np.asarray(patch, dtype=np.float32)
    if patch.ndim == 4:
        patch = patch[None]
    if patch.ndim != 5:
        raise RuntimeError(f"Expected patch shape [C, D, H, W] or [1, C, D, H, W]. Got: {patch.shape}")
    if patch.shape[0] != 1:
        raise RuntimeError(f"Expected batch size 1. Got: {patch.shape}")
    return patch


def main() -> int:
    args = _parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    onnx_model = Path(args.onnx_model).expanduser().resolve()
    preprocessed_patch = Path(args.preprocessed_patch).expanduser().resolve()
    try:
        resolved_checkpoint_path = checkpoint_path(model_dir, args.fold, args.checkpoint)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not resolved_checkpoint_path.is_file():
        print(f"Error: required checkpoint not found: {resolved_checkpoint_path}", file=sys.stderr)
        print(f"Expected {resolved_checkpoint_path.parent.name}/{args.checkpoint}.", file=sys.stderr)
        return 2
    if not onnx_model.is_file():
        print(f"Error: ONNX model not found: {onnx_model}", file=sys.stderr)
        return 2
    if not preprocessed_patch.is_file():
        print(f"Error: preprocessed patch not found: {preprocessed_patch}", file=sys.stderr)
        return 2

    try:
        import onnxruntime as ort
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    except Exception as exc:
        print("Error: failed to import required validation dependencies.", file=sys.stderr)
        print("Install nnUNetv2, torch, onnxruntime, and nnU-Net runtime dependencies before validating.", file=sys.stderr)
        print(f"Full exception: {exc}", file=sys.stderr)
        return 2

    try:
        available_providers = validate_provider(ort, args.provider)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        predictor, patch_size, num_input_channels = load_predictor_for_export(
            model_dir,
            args.checkpoint,
            torch,
            nnUNetPredictor,
            determine_num_input_channels,
            configuration=args.configuration,
            fold=args.fold,
        )
        input_array = _load_preprocessed_patch(preprocessed_patch)
        expected_shape = (1, num_input_channels, *patch_size)
        if tuple(input_array.shape) != expected_shape:
            raise RuntimeError(f"Expected preprocessed patch shape {expected_shape}. Got: {tuple(input_array.shape)}")

        network = predictor.network.to(torch.device("cpu"))
        network.eval()
        with torch.no_grad():
            torch_output = network(torch.from_numpy(input_array)).detach().cpu().numpy()

        session = ort.InferenceSession(str(onnx_model), providers=[args.provider])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        onnx_output = session.run([output_name], {input_name: input_array})[0]

        return report_comparison(input_array.shape, torch_output, onnx_output, args.provider, available_providers)

    except Exception as exc:
        print("Validation failed.", file=sys.stderr)
        print(f"Full exception: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
