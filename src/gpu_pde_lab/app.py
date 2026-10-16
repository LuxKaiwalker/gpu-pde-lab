"""Entry point: builds the QApplication, registers all experiments, shows the
main window."""

from __future__ import annotations

import sys

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


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow(build_live_factories(), build_benchmark_factories())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
