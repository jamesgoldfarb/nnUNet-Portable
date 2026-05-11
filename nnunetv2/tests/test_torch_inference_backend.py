import sys
import types

import numpy as np
import torch

from nnunetv2.inference.torch_inference_backend import OnnxRuntimeInferenceBackend, TorchInferenceBackend


class _FakeSessionInput:
    def __init__(self, name, shape, input_type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = input_type


class _FakeSessionOptions:
    def __init__(self):
        self.graph_optimization_level = None
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.execution_mode = None


def _fake_ort(session_cls, available_providers=None):
    return types.SimpleNamespace(
        get_available_providers=lambda: available_providers or ["CPUExecutionProvider"],
        InferenceSession=session_cls,
        SessionOptions=_FakeSessionOptions,
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL="ORT_ENABLE_ALL"),
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="ORT_SEQUENTIAL"),
    )


class _ToyNetwork(torch.nn.Module):
    def forward(self, x):
        return x * 2 + 1


def test_torch_inference_backend_matches_direct_network_call():
    network = _ToyNetwork()
    x = torch.randn((1, 2, 4, 5, 6), dtype=torch.float32)

    direct = network(x)
    wrapped = TorchInferenceBackend()(network, x)

    assert type(wrapped) is type(direct)
    assert wrapped.dtype == direct.dtype
    assert wrapped.shape == direct.shape
    assert wrapped.device == direct.device
    torch.testing.assert_close(wrapped, direct)


def test_onnx_runtime_inference_backend_returns_tensor_like_prediction(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    class _FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            self.sess_options = sess_options
            self.providers = providers

        def run(self, output_names, inputs):
            assert output_names == ["logits"]
            return [inputs["input"] + 1]

        def get_inputs(self):
            return [_FakeSessionInput("input", [1, 2, 4, 5, 6])]

        def get_outputs(self):
            return [_FakeSessionInput("logits", [1, 2, 4, 5, 6])]

    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(_FakeSession))

    x = torch.randn((1, 2, 4, 5, 6), dtype=torch.float32)
    backend = OnnxRuntimeInferenceBackend(str(model_path))
    wrapped = backend(None, x)

    assert backend.available_providers == ["CPUExecutionProvider"]
    assert backend.providers == ["CPUExecutionProvider"]
    assert backend.session.sess_options.graph_optimization_level == "ORT_ENABLE_ALL"
    assert backend.session.sess_options.intra_op_num_threads == 8
    assert backend.session.sess_options.inter_op_num_threads == 1
    assert backend.session.sess_options.execution_mode == "ORT_SEQUENTIAL"
    assert wrapped.dtype == x.dtype
    assert wrapped.shape == x.shape
    assert wrapped.device == x.device
    torch.testing.assert_close(wrapped, x + 1)


def test_onnx_runtime_inference_backend_casts_input_for_fp16_model(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    class _FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            self.sess_options = sess_options
            self.providers = providers

        def run(self, output_names, inputs):
            assert inputs["input"].dtype == np.float16
            return [inputs["input"] + np.float16(1)]

        def get_inputs(self):
            return [_FakeSessionInput("input", [1, 2, 4, 5, 6], "tensor(float16)")]

        def get_outputs(self):
            return [_FakeSessionInput("logits", [1, 2, 4, 5, 6], "tensor(float16)")]

    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(_FakeSession))

    x = torch.randn((1, 2, 4, 5, 6), dtype=torch.float32)
    backend = OnnxRuntimeInferenceBackend(str(model_path))
    wrapped = backend(None, x)

    assert backend.expected_input_dtype == np.float16
    assert wrapped.dtype == x.dtype
    assert wrapped.shape == x.shape
    assert wrapped.device == x.device


def test_onnx_runtime_inference_backend_rejects_input_shape_mismatch(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    class _FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            self.sess_options = sess_options
            self.providers = providers

        def run(self, output_names, inputs):
            raise AssertionError("Shape mismatch should fail before session.run")

        def get_inputs(self):
            return [_FakeSessionInput("input", [1, 2, 5, 4, 6])]

        def get_outputs(self):
            return [_FakeSessionInput("logits", [1, 2, 5, 4, 6])]

    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(_FakeSession))

    x = torch.randn((1, 2, 4, 5, 6), dtype=torch.float32)
    backend = OnnxRuntimeInferenceBackend(str(model_path))

    try:
        backend(None, x)
    except RuntimeError as exc:
        assert "expects input shape" in str(exc)
        assert "nnU-Net provided patch shape" in str(exc)
        assert "Re-export" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch to raise RuntimeError")


def test_onnx_runtime_inference_backend_reports_unavailable_provider(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(lambda *args, **kwargs: None))

    try:
        OnnxRuntimeInferenceBackend(str(model_path), provider="CUDAExecutionProvider")
    except RuntimeError as exc:
        assert "CUDAExecutionProvider" in str(exc)
        assert "CPUExecutionProvider" in str(exc)
    else:
        raise AssertionError("Expected unavailable provider to raise RuntimeError")


def test_onnx_runtime_inference_backend_accepts_provider_chain(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    class _FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            self.sess_options = sess_options
            self.providers = providers

        def get_inputs(self):
            return [_FakeSessionInput("input", [1, 2, 4, 5, 6])]

        def get_outputs(self):
            return [_FakeSessionInput("logits", [1, 2, 4, 5, 6])]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        _fake_ort(_FakeSession, ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
    )

    backend = OnnxRuntimeInferenceBackend(str(model_path), provider="CoreMLExecutionProvider,CPUExecutionProvider")

    assert backend.providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert backend.provider == "CoreMLExecutionProvider"
    assert backend.session.providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
