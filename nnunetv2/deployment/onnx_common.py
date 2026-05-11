from pathlib import Path
from typing import Any


SUPPORTED_CONFIGURATION = "3d_fullres"


def validate_provider(ort_module: Any, provider: str) -> list[str]:
    available_providers = ort_module.get_available_providers()
    print(f"Available ONNX Runtime providers: {available_providers}")
    if provider not in available_providers:
        raise RuntimeError(
            f"Requested ONNX Runtime provider is unavailable: {provider}. "
            f"Available providers: {available_providers}"
        )
    return available_providers


def load_fold_all_predictor(
    model_dir: Path,
    checkpoint: str,
    torch_module: Any,
    predictor_cls: Any,
    determine_num_input_channels_fn: Any,
    configuration: str = SUPPORTED_CONFIGURATION,
):
    predictor = predictor_cls(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch_module.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=("all",), checkpoint_name=checkpoint)

    expected_configuration = predictor.plans_manager.get_configuration(configuration).configuration
    if predictor.configuration_manager.configuration != expected_configuration:
        raise RuntimeError(f"Loaded model does not match requested configuration {configuration}.")
    if predictor.configuration_manager.previous_stage_name is not None:
        raise RuntimeError("Cascaded configurations are not supported.")

    patch_size = tuple(int(i) for i in predictor.configuration_manager.patch_size)
    if len(patch_size) != 3:
        raise RuntimeError(f"Only 3D patch sizes are supported. Got patch_size={patch_size}")

    num_input_channels = int(
        determine_num_input_channels_fn(
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
    )
    return predictor, patch_size, num_input_channels


def has_nan_or_inf(array: Any) -> bool:
    import numpy as np

    return bool(np.isnan(array).any() or np.isinf(array).any())


def report_comparison(input_shape, torch_output, onnx_output, provider: str, available_providers: list[str]) -> int:
    import numpy as np

    torch_has_nan_inf = has_nan_or_inf(torch_output)
    onnx_has_nan_inf = has_nan_or_inf(onnx_output)

    print(f"Input shape: {tuple(input_shape)}")
    print(f"PyTorch output shape: {tuple(torch_output.shape)}")
    print(f"ONNX output shape: {tuple(onnx_output.shape)}")
    print(f"Selected ONNX Runtime provider: {provider}")
    print(f"Available ONNX Runtime providers: {available_providers}")
    print(f"PyTorch output has NaN/Inf: {torch_has_nan_inf}")
    print(f"ONNX output has NaN/Inf: {onnx_has_nan_inf}")

    if tuple(torch_output.shape) != tuple(onnx_output.shape):
        print("Error: PyTorch and ONNX output shapes differ.")
        return 1
    if torch_has_nan_inf:
        print("Error: PyTorch output contains NaN or Inf.")
        return 1
    if onnx_has_nan_inf:
        print("Error: ONNX output contains NaN or Inf.")
        return 1

    abs_diff = np.abs(torch_output - onnx_output)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    relative_mean_diff = float(mean_abs_diff / max(float(np.abs(torch_output).mean()), np.finfo(np.float32).eps))
    torch_argmax = np.argmax(torch_output, axis=1)
    onnx_argmax = np.argmax(onnx_output, axis=1)
    argmax_disagreement = float(np.mean(torch_argmax != onnx_argmax) * 100.0)

    print(f"Max absolute difference: {max_abs_diff:.8g}")
    print(f"Mean absolute difference: {mean_abs_diff:.8g}")
    print(f"Relative mean difference: {relative_mean_diff:.8g}")
    print(f"Argmax disagreement percentage: {argmax_disagreement:.6f}%")
    return 0
