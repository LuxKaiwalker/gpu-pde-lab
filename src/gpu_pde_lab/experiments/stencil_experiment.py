"""Stencil visualization: click a cell to see exactly which neighbors feed
its Laplacian, and how that maps onto "one GPU thread per cell"."""

from __future__ import annotations

import numpy as np

from ..core.backend import Backend, BackendSpec
from ..core.boundary import BoundaryKind
from ..core.grid import Grid2D
from ..core.stencil import stencil_at
from ..initial_conditions import gaussian
from ..ui.controls import ControlPanel
from .base import LiveExperiment

DIAGRAM = r"""
```
            u[i, j-1]
                 |
                 |
 u[i-1, j] -- u[i, j] -- u[i+1, j]
                 |
                 |
            u[i, j+1]
```

```
GPU mapping (conceptual):

   thread (i, j)
        |
        v
   read u[i-1,j], u[i+1,j], u[i,j-1], u[i,j+1], u[i,j]   (5 memory loads)
        |
        v
   compute (sum - 4*center) / h^2                         (a few FLOPs)
        |
        v
   write u_new[i, j]                                      (1 memory store)
```
"""

EXPLANATION_HEADER = r"""
### What does one grid cell compute?

$$ \nabla^2 u_{i,j} \approx \frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2} $$

Click any cell in the field on the left. The stats panel will show that
cell's four neighbor values, its own value, and the resulting Laplacian --
exactly the data one GPU thread would read and the single value it would
write.

### Why can it run in parallel?

Look at the diagram: computing $(i,j)$ needs only $(i,j)$ and its four
direct neighbors from the *previous* timestep. It does **not** need the
*new* value of any other cell. That absence of inter-cell dependency within
one stencil pass is the entire reason this class of PDE solver maps onto a
GPU so well -- there is no serialization required between cells.

### What does the GPU actually do?

A CUDA kernel launch assigns a thread (or, in practice, a small vectorized
batch of elements per thread) to each output cell. All threads execute the
same instructions ("read 5 values, do a few FLOPs, write 1 value") on
different data -- the textbook definition of SIMT (single instruction,
multiple threads) execution. Because the arithmetic intensity per cell is so
low (5 loads + 1 store vs. ~6 FLOPs), the kernel's speed is set by memory
bandwidth, not compute throughput.
"""


class StencilExperiment(LiveExperiment):
    title = "Stencil Visualization"

    def __init__(self):
        self.n = 24
        self.boundary = BoundaryKind.NEUMANN
        self.selected_cell = (self.n // 2, self.n // 2)
        self._build()

    def _build(self) -> None:
        self.backend = Backend(BackendSpec("numpy", "cpu", "fp64"))
        self.grid = Grid2D(nx=self.n, ny=self.n)
        X, Y = np.meshgrid(np.linspace(0, 1, self.n), np.linspace(0, 1, self.n), indexing="ij")
        self.u = self.backend.array(gaussian(X, Y, amplitude=1.0, sigma=0.25))

    def reset(self) -> None:
        self._build()

    def step(self, dt: float) -> None:
        pass  # static field -- this experiment is about inspection, not evolution

    def field_numpy(self):
        return self.backend.to_numpy(self.u)

    def supports_click_injection(self) -> bool:
        return True

    def on_click(self, x_frac: float, y_frac: float) -> None:
        i = int(round(x_frac * (self.n - 1)))
        j = int(round(y_frac * (self.n - 1)))
        i = max(0, min(self.n - 1, i))
        j = max(0, min(self.n - 1, j))
        self.selected_cell = (i, j)

    def stats(self) -> dict:
        i, j = self.selected_cell
        info = stencil_at(self.field_numpy(), i, j, self.grid.dx, self.grid.dy, self.boundary)
        return {
            "Selected cell (i,j)": f"({i}, {j})",
            "u[i,j] (center)": f"{info['center']:.4f}",
            "u[i-1,j] (north)": f"{info['north']:.4f}",
            "u[i+1,j] (south)": f"{info['south']:.4f}",
            "u[i,j-1] (west)": f"{info['west']:.4f}",
            "u[i,j+1] (east)": f"{info['east']:.4f}",
            "Laplacian at cell": f"{info['laplacian']:.4f}",
            "Grid": f"{self.n} x {self.n}",
        }

    def explanation_markdown(self) -> str:
        return EXPLANATION_HEADER + "\n" + DIAGRAM

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_grid(text):
            self.n = int(text)
            on_change()

        panel.add_combo("Grid resolution", ["12", "16", "24", "32", "48"], current=str(self.n),
                         on_change=set_grid)
        return panel
