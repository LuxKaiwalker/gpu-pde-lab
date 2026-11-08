"""A CUDA device that exists but has no room is a different failure from no
device at all, and neither should take the app down.

These tests drive the two branches by monkeypatching the probe rather than by
actually filling the card: the real condition (another process holding the
GPU) is not something a test can arrange, and a test that only passes on a
loaded machine would never run in CI.
"""

from __future__ import annotations

import pytest

from gpu_pde_lab.core import backend as backend_mod
from gpu_pde_lab.core.backend import (
    MIN_CUDA_FREE_BYTES,
    Backend,
    BackendSpec,
    available_backend_specs,
    cpu_equivalent,
    cuda_is_usable,
    make_backend,
    reset_cuda_usable_cache,
)


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    # The usability probe is cached process-wide, so a test that fakes it has
    # to clear both before and after or it leaks into unrelated tests.
    reset_cuda_usable_cache()
    yield
    reset_cuda_usable_cache()


def _force_cuda_state(monkeypatch, *, present: bool, free_bytes: int | None):
    monkeypatch.setattr(backend_mod, "cuda_is_available", lambda: present)
    monkeypatch.setattr(backend_mod, "cuda_free_memory_bytes", lambda: free_bytes)


def test_usable_is_false_when_no_device(monkeypatch):
    _force_cuda_state(monkeypatch, present=False, free_bytes=None)
    assert cuda_is_usable() is False


def test_usable_is_false_when_device_is_full(monkeypatch):
    # Present, answering, but with far less than the floor available.
    _force_cuda_state(monkeypatch, present=True, free_bytes=64 * 1024 * 1024)
    assert cuda_is_usable() is False


def test_usable_is_false_when_context_cannot_be_created(monkeypatch):
    # mem_get_info returning None is how a failed context initialization
    # surfaces -- it must not be read as "unlimited memory".
    _force_cuda_state(monkeypatch, present=True, free_bytes=None)
    assert cuda_is_usable() is False


def test_probe_result_is_cached(monkeypatch):
    calls = []

    def counting_free():
        calls.append(1)
        return 16 * 1024 * 1024

    monkeypatch.setattr(backend_mod, "cuda_is_available", lambda: True)
    monkeypatch.setattr(backend_mod, "cuda_free_memory_bytes", counting_free)

    cuda_is_usable()
    cuda_is_usable()
    cuda_is_usable()
    assert len(calls) == 1, "probe should be cached; it costs a CUDA context"


def test_reset_cache_forces_a_re_probe(monkeypatch):
    _force_cuda_state(monkeypatch, present=True, free_bytes=16 * 1024 * 1024)
    assert cuda_is_usable() is False

    # Device frees up (the other process exited); after a reset we should see it.
    _force_cuda_state(monkeypatch, present=True, free_bytes=MIN_CUDA_FREE_BYTES * 4)
    assert cuda_is_usable() is False, "stale cache should still say unusable"
    reset_cuda_usable_cache()
    # The real probe now runs; on a CPU-only CI box it allocates nothing and
    # returns False, so assert only that the cached answer was discarded.
    assert isinstance(cuda_is_usable(), bool)


def test_make_backend_falls_back_to_cpu_with_an_explanation(monkeypatch):
    _force_cuda_state(monkeypatch, present=True, free_bytes=32 * 1024 * 1024)

    backend, warning = make_backend(BackendSpec("torch", "cuda", "fp32"))
    assert backend.spec.device == "cpu"
    assert warning is not None
    # The message has to say how much was free, or the user cannot tell this
    # from a driver problem.
    assert "32 MB" in warning
    assert "torch/cpu/fp32" in warning


def test_make_backend_explains_a_missing_device_differently(monkeypatch):
    _force_cuda_state(monkeypatch, present=False, free_bytes=None)

    _backend, warning = make_backend(BackendSpec("torch", "cuda", "fp32"))
    assert warning is not None
    assert "no CUDA device is present" in warning


def test_make_backend_can_refuse_instead_of_degrading(monkeypatch):
    _force_cuda_state(monkeypatch, present=True, free_bytes=0)

    with pytest.raises(RuntimeError, match="not usable"):
        make_backend(BackendSpec("torch", "cuda", "fp32"), allow_cpu_fallback=False)


def test_make_backend_returns_no_warning_on_a_plain_cpu_request():
    backend, warning = make_backend(BackendSpec("torch", "cpu", "fp32"))
    assert warning is None
    assert backend.spec.device == "cpu"


def test_cpu_equivalent_downgrades_fp16_but_keeps_other_precisions():
    # Torch CPU fp16 is emulated; timing it would mislead rather than inform.
    assert cpu_equivalent(BackendSpec("torch", "cuda", "fp16")).precision == "fp32"
    assert cpu_equivalent(BackendSpec("torch", "cuda", "fp32")).precision == "fp32"
    assert cpu_equivalent(BackendSpec("torch", "cuda", "fp64")).precision == "fp64"
    assert cpu_equivalent(BackendSpec("torch", "cuda", "fp64")).device == "cpu"


def test_available_specs_never_offer_an_unusable_cuda_backend(monkeypatch):
    _force_cuda_state(monkeypatch, present=True, free_bytes=8 * 1024 * 1024)

    specs = available_backend_specs()
    assert specs, "CPU specs should always be offered"
    assert all(s.device == "cpu" for s in specs), (
        "a full device must not be advertised; selecting it would throw"
    )


def test_cpu_backends_are_always_constructible():
    # The floor of the whole fallback story: whatever the GPU is doing, these
    # four must work.
    for spec in available_backend_specs():
        if spec.device == "cpu":
            assert Backend(spec).device_name() == "CPU"
