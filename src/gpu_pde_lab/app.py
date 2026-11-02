"""Entry point: builds the QApplication, registers all experiments, shows the
main window."""

from __future__ import annotations

import fcntl
import os
import sys
import tempfile

from PySide6.QtWidgets import QApplication

from .experiments.convergence_experiment import ConvergenceExperiment
from .experiments.cpu_gpu_experiment import CpuGpuExperiment
from .experiments.heat_3d_experiment import Heat3DExperiment
from .experiments.heat_experiment import HeatExperiment
from .experiments.kernel_comparison_experiment import KernelComparisonExperiment
from .experiments.memory_experiment import MemoryExperiment
from .experiments.poisson_experiment import PoissonExperiment
from .experiments.precision_experiment import PrecisionExperiment
from .experiments.stencil_experiment import StencilExperiment
from .experiments.transfer_experiment import TransferExperiment
from .experiments.wave_experiment import WaveExperiment
from .ui.main_window import MainWindow

try:
    from .experiments.fluid_experiment import FluidExperiment
    _HAS_FLUID = True
except ImportError:
    _HAS_FLUID = False


def build_live_factories() -> dict:
    factories = {
        "Heat Equation (+ Stability)": HeatExperiment,
        "3D Heat Equation": Heat3DExperiment,
        "Wave Equation": WaveExperiment,
        "Poisson Equation (Iterative Solvers)": PoissonExperiment,
        "Stencil Visualization": StencilExperiment,
    }
    if _HAS_FLUID:
        factories["2D Fluid Simulation"] = FluidExperiment
    return factories


def build_benchmark_factories() -> dict:
    return {
        "CPU vs GPU Benchmark": CpuGpuExperiment,
        "Host <-> Device Transfer": TransferExperiment,
        "GPU Memory": MemoryExperiment,
        "Kernel / Vectorization Comparison": KernelComparisonExperiment,
        "Resolution / Convergence Study": ConvergenceExperiment,
        "Precision Comparison": PrecisionExperiment,
    }


_lock_file = None  # must stay open for the process lifetime -- closing it drops the lock


def _acquire_single_instance_lock() -> bool:
    """Exclusive, non-blocking lock so a second launch can't stack another
    full torch/CUDA/PySide6 process (each holds several GB of grid buffers
    and a CUDA context) on top of one that's still running.

    Using flock() rather than a PID file means there's never a stale lock to
    clean up by hand: the kernel releases it the instant the process exits
    for any reason, including SIGKILL from the OOM killer.
    """
    global _lock_file
    lock_dir = os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir())
    _lock_file = open(os.path.join(lock_dir, "gpu_pde_lab.lock"), "w")
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _lock_file.close()
        _lock_file = None
        return False
    return True


def main() -> int:
    if not _acquire_single_instance_lock():
        print(
            "gpu_pde_lab is already running in another process -- refusing to "
            "start a second copy. Close the existing window first.",
            file=sys.stderr,
        )
        return 1

    app = QApplication(sys.argv)
    window = MainWindow(build_live_factories(), build_benchmark_factories())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
