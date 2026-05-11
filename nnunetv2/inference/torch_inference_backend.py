from pathlib import Path
from numbers import Integral
from typing import Sequence

import numpy as np
import torch


ORT_INPUT_DTYPE_BY_TYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
}

ORT_TORCH_DTYPE_BY_TYPE = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
}


class TorchInferenceBackend:
    def __call__(self, network, x):
        return network(x)


class OnnxRuntimeInferenceBackend:
    def __init__(
            self,
            onnx_model_path: str,
            provider: str | Sequence[str] = "CPUExecutionProvider",
            input_name: str = "input",
            output_name: str = "logits",
            intra_op_num_threads: int = 8,
            inter_op_num_threads: int = 1,
    ):
        self.onnx_model_path = Path(onnx_model_path)
        if not self.onnx_model_path.is_file():
            raise FileNotFoundError(f"ONNX model path does not exist: {self.onnx_model_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required to use OnnxRuntimeInferenceBackend.") from exc

        self.available_providers = ort.get_available_providers()
        self.providers = self._normalize_providers(provider)
        unavailable_providers = [p for p in self.providers if p not in self.available_providers]
        if unavailable_providers:
            raise RuntimeError(
                f"ONNX Runtime provider(s) {unavailable_providers} are not available. "
                f"Available providers: {self.available_providers}"
            )

        self.provider = self.providers[0]
        self.input_name = input_name
        self.output_name = output_name
        self.session_options = self._create_session_options(ort, intra_op_num_threads, inter_op_num_threads)
        self.session = ort.InferenceSession(
            str(self.onnx_model_path),
            sess_options=self.session_options,
            providers=self.providers,
        )
        self.expected_input_shape = self._get_expected_input_shape()
        self.expected_input_dtype = self._get_expected_input_dtype()
        self.expected_input_torch_dtype = self._get_expected_input_torch_dtype()
        self.expected_output_shape = self._get_expected_output_shape()
        self.expected_output_dtype = self._get_expected_output_dtype()
        self.expected_output_torch_dtype = self._get_expected_output_torch_dtype()

    @staticmethod
    def _normalize_providers(provider: str | Sequence[str]) -> list[str]:
        if isinstance(provider, str):
            providers = [p.strip() for p in provider.split(",") if p.strip()]
        else:
            providers = list(provider)
        if not providers:
            raise RuntimeError("At least one ONNX Runtime provider must be specified.")
        return providers

    @staticmethod
    def _create_session_options(ort, intra_op_num_threads: int, inter_op_num_threads: int):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = intra_op_num_threads
        session_options.inter_op_num_threads = inter_op_num_threads
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return session_options

    def validate_expected_input_shape(self, expected_shape):
        if self.expected_input_shape is None or any(dim is None for dim in self.expected_input_shape):
            return
        if tuple(self.expected_input_shape) != tuple(expected_shape):
            raise RuntimeError(
                f"ONNX model expects input shape {tuple(self.expected_input_shape)} but this nnU-Net predictor "
                f"uses patch input shape {tuple(expected_shape)}. Make sure nnUNetv2_predict uses the same "
                "trainer/plans/configuration/fold as nnUNetv2_export_onnx. For this model you likely need to pass "
                "the matching -tr argument, then re-export if needed."
            )

    def _get_expected_input_shape(self):
        for session_input in self.session.get_inputs():
            if session_input.name == self.input_name:
                shape = []
                for dim in session_input.shape:
                    shape.append(int(dim) if isinstance(dim, Integral) else None)
                return tuple(shape)
        available_inputs = [session_input.name for session_input in self.session.get_inputs()]
        raise RuntimeError(f"ONNX input '{self.input_name}' not found. Available inputs: {available_inputs}")

    def _get_expected_input_dtype(self):
        return self._get_numpy_dtype(self.session.get_inputs(), self.input_name, "input")

    def _get_expected_input_torch_dtype(self):
        return self._get_torch_dtype(self.session.get_inputs(), self.input_name, "input")

    def _get_expected_output_shape(self):
        for session_output in self.session.get_outputs():
            if session_output.name == self.output_name:
                shape = []
                for dim in session_output.shape:
                    shape.append(int(dim) if isinstance(dim, Integral) else None)
                return tuple(shape)
        available_outputs = [session_output.name for session_output in self.session.get_outputs()]
        raise RuntimeError(f"ONNX output '{self.output_name}' not found. Available outputs: {available_outputs}")

    def _get_expected_output_dtype(self):
        return self._get_numpy_dtype(self.session.get_outputs(), self.output_name, "output")

    def _get_expected_output_torch_dtype(self):
        return self._get_torch_dtype(self.session.get_outputs(), self.output_name, "output")

    @staticmethod
    def _get_numpy_dtype(session_values, name, value_kind):
        for session_value in session_values:
            if session_value.name == name:
                value_type = getattr(session_value, "type", None)
                return ORT_INPUT_DTYPE_BY_TYPE.get(value_type, np.float32)
        available_values = [session_value.name for session_value in session_values]
        raise RuntimeError(f"ONNX {value_kind} '{name}' not found. Available {value_kind}s: {available_values}")

    @staticmethod
    def _get_torch_dtype(session_values, name, value_kind):
        for session_value in session_values:
            if session_value.name == name:
                value_type = getattr(session_value, "type", None)
                return ORT_TORCH_DTYPE_BY_TYPE.get(value_type, torch.float32)
        available_values = [session_value.name for session_value in session_values]
        raise RuntimeError(f"ONNX {value_kind} '{name}' not found. Available {value_kind}s: {available_values}")

    def __call__(self, network, x: torch.Tensor) -> torch.Tensor:
        if self.expected_input_shape is not None and all(dim is not None for dim in self.expected_input_shape):
            expected_shape = tuple(self.expected_input_shape)
            if tuple(x.shape) != expected_shape:
                raise RuntimeError(
                    f"ONNX model expects input shape {expected_shape} but nnU-Net provided patch shape "
                    f"{tuple(x.shape)}. Re-export the ONNX model with the exact patch shape used by nnUNetv2_predict."
                )

        if self._use_cuda_iobinding(x):
            return self._call_cuda_iobinding(x)

        input_array = x.detach().cpu().numpy().astype(self.expected_input_dtype, copy=False)
        output_array = self.session.run([self.output_name], {self.input_name: input_array})[0]

        self._validate_output_shape(tuple(output_array.shape), x)
        return torch.as_tensor(output_array, device=x.device, dtype=x.dtype)

    def _use_cuda_iobinding(self, x: torch.Tensor) -> bool:
        return self.provider == "CUDAExecutionProvider" and x.is_cuda

    def _call_cuda_iobinding(self, x: torch.Tensor) -> torch.Tensor:
        if self.expected_output_shape is None or any(dim is None for dim in self.expected_output_shape):
            raise RuntimeError(
                "CUDA ONNX Runtime I/O binding requires a fixed ONNX output shape. "
                "Re-export the model without dynamic axes."
            )

        output_shape = tuple(self.expected_output_shape)
        self._validate_output_shape(output_shape, x)

        input_tensor = x.detach()
        if input_tensor.dtype != self.expected_input_torch_dtype:
            input_tensor = input_tensor.to(dtype=self.expected_input_torch_dtype)
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        output_tensor = torch.empty(output_shape, dtype=self.expected_output_torch_dtype, device=x.device)
        device_id = x.device.index
        if device_id is None:
            device_id = torch.cuda.current_device()

        io_binding = self.session.io_binding()
        io_binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=device_id,
            element_type=self.expected_input_dtype,
            shape=tuple(input_tensor.shape),
            buffer_ptr=input_tensor.data_ptr(),
        )
        io_binding.bind_output(
            name=self.output_name,
            device_type="cuda",
            device_id=device_id,
            element_type=self.expected_output_dtype,
            shape=output_shape,
            buffer_ptr=output_tensor.data_ptr(),
        )
        self.session.run_with_iobinding(io_binding)

        if output_tensor.dtype != x.dtype:
            return output_tensor.to(dtype=x.dtype)
        return output_tensor

    @staticmethod
    def _validate_output_shape(output_shape, x: torch.Tensor) -> None:
        expected_prefix_and_spatial_shape = (x.shape[0], *x.shape[2:])
        actual_prefix_and_spatial_shape = (output_shape[0], *output_shape[2:])
        if actual_prefix_and_spatial_shape != expected_prefix_and_spatial_shape:
            raise RuntimeError(
                f"ONNX output shape {tuple(output_shape)} does not match expected "
                f"batch/spatial shape {expected_prefix_and_spatial_shape} for input shape {tuple(x.shape)}."
            )
