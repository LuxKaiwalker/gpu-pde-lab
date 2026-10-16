"""Guards the exact kind of gap this audit found: a PDE solver existing but
no experiment ever being registered/discoverable/constructible for it."""

from gpu_pde_lab.app import build_benchmark_factories, build_live_factories
from gpu_pde_lab.experiments.base import BenchmarkExperiment, LiveExperiment


def test_every_live_experiment_is_constructible():
    for name, factory in build_live_factories().items():
        exp = factory()
        assert isinstance(exp, LiveExperiment), f"{name} is not a LiveExperiment"


def test_every_benchmark_experiment_is_constructible():
    for name, factory in build_benchmark_factories().items():
        exp = factory()
        assert isinstance(exp, BenchmarkExperiment), f"{name} is not a BenchmarkExperiment"


def test_3d_heat_equation_is_registered_and_discoverable():
    live = build_live_factories()
    assert "3D Heat Equation" in live


def test_3d_heat_equation_reports_correct_dimensionality():
    from gpu_pde_lab.experiments.heat_3d_experiment import Heat3DExperiment

    exp = Heat3DExperiment()
    exp.backend_name = "CPU"
    exp.reset()

    assert exp.is_3d() is True
    volume = exp.volume_numpy()
    assert volume.ndim == 3
    assert volume.shape == (exp.n, exp.n, exp.n)
    assert exp.volume_grid() is not None


def test_only_3d_heat_equation_reports_is_3d_true():
    for name, factory in build_live_factories().items():
        exp = factory()
        if name == "3D Heat Equation":
            assert exp.is_3d() is True, name
        else:
            assert exp.is_3d() is False, f"{name} unexpectedly reports is_3d()=True"
