from pathlib import Path
from numbers import Integral

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
        self.expected_input_shape = self._get_expected_input_shape()

    def _get_expected_input_shape(self):
        for session_input in self.session.get_inputs():
            if session_input.name == self.input_name:
                shape = []
                for dim in session_input.shape:
                    shape.append(int(dim) if isinstance(dim, Integral) else None)
                return tuple(shape)
        available_inputs = [session_input.name for session_input in self.session.get_inputs()]
        raise RuntimeError(f"ONNX input '{self.input_name}' not found. Available inputs: {available_inputs}")

    def __call__(self, network, x: torch.Tensor) -> torch.Tensor:
        if self.expected_input_shape is not None and all(dim is not None for dim in self.expected_input_shape):
            expected_shape = tuple(self.expected_input_shape)
            if tuple(x.shape) != expected_shape:
                raise RuntimeError(
                    f"ONNX model expects input shape {expected_shape} but nnU-Net provided patch shape "
                    f"{tuple(x.shape)}. Re-export the ONNX model with the exact patch shape used by nnUNetv2_predict."
                )

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
