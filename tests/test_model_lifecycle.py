"""Tests for idle unload, CUDA OOM fallback, and CPU-correct model selection."""

from typing import Any

from fastapi.testclient import TestClient
import pytest

from whisper_server import server


class FakeModel:
    """Stand-in for WhisperModel that records how it was built."""

    def __init__(self, name: str, device: str, compute_type: str) -> None:
        self.name = name
        self.device = device
        self.compute_type = compute_type


def _cuda_oom() -> RuntimeError:
    return RuntimeError("CUDA failed with error out of memory")


def _builder(recorder: list[tuple[str, str, str]], fail_on_cuda: bool = False):
    def build(name: str, device: str, compute_type: str) -> FakeModel:
        recorder.append((name, device, compute_type))
        if fail_on_cuda and device != "cpu":
            raise _cuda_oom()
        return FakeModel(name, device, compute_type)

    return build


def test_cuda_oom_is_recognised():
    assert server._is_cuda_oom(_cuda_oom())
    assert server._is_cuda_oom(RuntimeError("cudaMalloc failed: out of memory"))
    assert server._is_cuda_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert not server._is_cuda_oom(RuntimeError("Invalid model name"))
    assert not server._is_cuda_oom(RuntimeError("CUDA driver version is insufficient"))


def test_cpu_compute_type_downgrades_gpu_only_type(monkeypatch):
    import ctranslate2

    monkeypatch.setattr(server.settings, "cpu_fallback_compute_type", "int8_float16")
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda device: {"float32", "int8", "int8_float32"}
    )

    assert server._cpu_compute_type() == "int8"


def test_cpu_compute_type_keeps_supported_type(monkeypatch):
    import ctranslate2

    monkeypatch.setattr(server.settings, "cpu_fallback_compute_type", "int8")
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda device: {"float32", "int8", "int8_float32"}
    )

    assert server._cpu_compute_type() == "int8"


def test_cpu_fallback_target_uses_cpu_sized_model(monkeypatch):
    monkeypatch.setattr(server.settings, "whisper_model", "large-v3-turbo")
    monkeypatch.setattr(server.settings, "cpu_fallback_model", "small")
    monkeypatch.setattr(server, "_cpu_compute_type", lambda: "int8")

    assert server._cpu_fallback_target() == ("small", "int8")


def test_empty_cpu_fallback_model_keeps_configured_model(monkeypatch):
    monkeypatch.setattr(server.settings, "whisper_model", "large-v3-turbo")
    monkeypatch.setattr(server.settings, "cpu_fallback_model", "")
    monkeypatch.setattr(server, "_cpu_compute_type", lambda: "int8")

    assert server._cpu_fallback_target() == ("large-v3-turbo", "int8")


def test_load_falls_back_to_cpu_model_on_cuda_oom(monkeypatch):
    builds: list[tuple[str, str, str]] = []
    monkeypatch.setattr(server.settings, "device", "cuda")
    monkeypatch.setattr(server.settings, "whisper_model", "large-v3-turbo")
    monkeypatch.setattr(server.settings, "compute_type", "int8_float16")
    monkeypatch.setattr(server.settings, "cpu_fallback_model", "small")
    monkeypatch.setattr(server.settings, "cuda_oom_fallback_cpu", True)
    monkeypatch.setattr(server, "_cpu_compute_type", lambda: "int8")
    monkeypatch.setattr(server, "_build_model", _builder(builds, fail_on_cuda=True))

    instance = server._load_model_sync()

    assert builds == [
        ("large-v3-turbo", "cuda", "int8_float16"),
        ("small", "cpu", "int8"),
    ]
    assert (instance.device, instance.name, instance.compute_type) == ("cpu", "small", "int8")
    assert server.model_cpu_fallback is True


def test_load_does_not_fall_back_when_disabled(monkeypatch):
    monkeypatch.setattr(server.settings, "device", "cuda")
    monkeypatch.setattr(server.settings, "cuda_oom_fallback_cpu", False)
    monkeypatch.setattr(server, "_build_model", _builder([], fail_on_cuda=True))

    with pytest.raises(RuntimeError):
        server._load_model_sync()
    assert server.model is None


def test_load_does_not_fall_back_on_unrelated_error(monkeypatch):
    def build(name: str, device: str, compute_type: str) -> FakeModel:
        raise RuntimeError("Invalid model size 'nonsense'")

    monkeypatch.setattr(server.settings, "device", "cuda")
    monkeypatch.setattr(server, "_build_model", build)

    with pytest.raises(RuntimeError):
        server._load_model_sync()


def test_cpu_device_sanitises_gpu_compute_type(monkeypatch):
    builds: list[tuple[str, str, str]] = []
    monkeypatch.setattr(server.settings, "device", "cpu")
    monkeypatch.setattr(server.settings, "whisper_model", "large-v3-turbo")
    monkeypatch.setattr(server.settings, "compute_type", "int8_float16")
    monkeypatch.setattr(server, "_cpu_compute_type", lambda: "int8")
    monkeypatch.setattr(server, "_build_model", _builder(builds))

    server._load_model_sync()

    # An explicit DEVICE=cpu keeps the configured model, but not a GPU-only type.
    assert builds == [("large-v3-turbo", "cpu", "int8")]
    assert server.model_cpu_fallback is False


def test_auto_device_without_gpu_loads_on_cpu(monkeypatch):
    builds: list[tuple[str, str, str]] = []
    monkeypatch.setattr(server.settings, "device", "auto")
    monkeypatch.setattr(server, "_cuda_device_count", lambda: 0)
    monkeypatch.setattr(server, "_cpu_compute_type", lambda: "int8")
    monkeypatch.setattr(server, "_build_model", _builder(builds))

    server._load_model_sync()

    assert builds[0][1] == "cpu"


def test_runtime_oom_retries_on_cpu(monkeypatch):
    builds: list[tuple[str, str, str]] = []
    monkeypatch.setattr(server.settings, "device", "cuda")
    monkeypatch.setattr(server.settings, "cpu_fallback_model", "small")
    monkeypatch.setattr(server, "_cpu_compute_type", lambda: "int8")
    monkeypatch.setattr(server, "_build_model", _builder(builds))

    attempts: list[str] = []

    def run(instance: FakeModel) -> str:
        attempts.append(instance.device)
        if instance.device != "cpu":
            raise _cuda_oom()
        return "done"

    assert server._run_with_cuda_fallback(run, what="test") == "done"
    assert attempts == ["cuda", "cpu"]
    assert builds == [
        (server.settings.whisper_model, "cuda", server.settings.compute_type),
        ("small", "cpu", "int8"),
    ]
    assert server.model_in_use == 0


def test_runtime_error_other_than_oom_propagates(monkeypatch):
    monkeypatch.setattr(server.settings, "device", "cuda")
    monkeypatch.setattr(server, "_build_model", _builder([]))

    def run(instance: FakeModel) -> str:
        raise ValueError("bad audio")

    with pytest.raises(ValueError):
        server._run_with_cuda_fallback(run, what="test")
    assert server.model_in_use == 0


def test_unload_is_skipped_while_model_is_in_use(monkeypatch):
    monkeypatch.setattr(server, "_build_model", _builder([]))
    server._load_model_sync()

    with server._model_in_use_guard():
        assert server._unload_model_sync() is False
        assert server.model is not None

    assert server._unload_model_sync() is True
    assert server.model is None


def test_health_does_not_wake_a_sleeping_model(monkeypatch):
    monkeypatch.setattr(server.settings, "health_wake_model", False)

    def fail_probe() -> dict[str, Any]:
        raise AssertionError("probe must not run while the model sleeps")

    monkeypatch.setattr(server, "_run_probe", fail_probe)

    with TestClient(server.app) as client:
        server.model = None
        payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["model_state"] == "sleeping"
    assert "probe_text" not in payload


def test_health_wake_argument_forces_the_probe(monkeypatch):
    calls: list[int] = []

    def probe() -> dict[str, Any]:
        calls.append(1)
        return {"probe_duration": 0.1, "probe_processing_seconds": 0.1, "probe_text": ""}

    monkeypatch.setattr(server, "_run_probe", probe)

    with TestClient(server.app) as client:
        server.model = None
        payload = client.get("/health", params={"wake": "true"}).json()

    assert calls == [1]
    assert payload["status"] == "ok"


def test_health_probe_does_not_postpone_the_unload(monkeypatch):
    monkeypatch.setattr(server, "_build_model", _builder([]))
    monkeypatch.setattr(
        server,
        "_run_probe",
        lambda: {"probe_duration": 0.1, "probe_processing_seconds": 0.1, "probe_text": ""},
    )

    with TestClient(server.app) as client:
        server._load_model_sync()
        server.model_last_used = 1.0
        client.get("/health")

    assert server.model_last_used == 1.0
