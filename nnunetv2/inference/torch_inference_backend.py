from pathlib import Path

import numpy as np
import torch


class TorchInferenceBackend:
    def __call__(self, network, x):
        return network(x)


class OnnxRuntimeInferenceBackend:
    def __init__(
            self,
            onnx_model_path: str,
            provider: str = "CPUExecutionProvider",
            input_name: str = "input",
            output_name: str = "logits",
    ):
        self.onnx_model_path = Path(onnx_model_path)
        if not self.onnx_model_path.is_file():
            raise FileNotFoundError(f"ONNX model path does not exist: {self.onnx_model_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required to use OnnxRuntimeInferenceBackend.") from exc

        self.available_providers = ort.get_available_providers()
        if provider not in self.available_providers:
            raise RuntimeError(
                f"ONNX Runtime provider '{provider}' is not available. "
                f"Available providers: {self.available_providers}"
            )

        self.provider = provider
        self.input_name = input_name
        self.output_name = output_name
        self.session = ort.InferenceSession(str(self.onnx_model_path), providers=[self.provider])

    def __call__(self, network, x: torch.Tensor) -> torch.Tensor:
        input_array = x.detach().cpu().numpy().astype(np.float32, copy=False)
        output_array = self.session.run([self.output_name], {self.input_name: input_array})[0]

        expected_prefix_and_spatial_shape = (x.shape[0], *x.shape[2:])
        actual_prefix_and_spatial_shape = (output_array.shape[0], *output_array.shape[2:])
        if actual_prefix_and_spatial_shape != expected_prefix_and_spatial_shape:
            raise RuntimeError(
                f"ONNX output shape {tuple(output_array.shape)} does not match expected "
                f"batch/spatial shape {expected_prefix_and_spatial_shape} for input shape {tuple(x.shape)}."
            )

        return torch.as_tensor(output_array, device=x.device, dtype=x.dtype)
