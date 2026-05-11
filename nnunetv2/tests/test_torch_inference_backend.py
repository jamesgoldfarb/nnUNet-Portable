import sys
import types

import torch

from nnunetv2.inference.torch_inference_backend import OnnxRuntimeInferenceBackend, TorchInferenceBackend


class _FakeSessionInput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


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
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def run(self, output_names, inputs):
            assert output_names == ["logits"]
            return [inputs["input"] + 1]

        def get_inputs(self):
            return [_FakeSessionInput("input", [1, 2, 4, 5, 6])]

    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=_FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    x = torch.randn((1, 2, 4, 5, 6), dtype=torch.float32)
    backend = OnnxRuntimeInferenceBackend(str(model_path))
    wrapped = backend(None, x)

    assert backend.available_providers == ["CPUExecutionProvider"]
    assert wrapped.dtype == x.dtype
    assert wrapped.shape == x.shape
    assert wrapped.device == x.device
    torch.testing.assert_close(wrapped, x + 1)


def test_onnx_runtime_inference_backend_rejects_input_shape_mismatch(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    class _FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def run(self, output_names, inputs):
            raise AssertionError("Shape mismatch should fail before session.run")

        def get_inputs(self):
            return [_FakeSessionInput("input", [1, 2, 5, 4, 6])]

    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=_FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

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
    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    try:
        OnnxRuntimeInferenceBackend(str(model_path), provider="CUDAExecutionProvider")
    except RuntimeError as exc:
        assert "CUDAExecutionProvider" in str(exc)
        assert "CPUExecutionProvider" in str(exc)
    else:
        raise AssertionError("Expected unavailable provider to raise RuntimeError")
