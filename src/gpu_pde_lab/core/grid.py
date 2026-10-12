"""Regular 2D grid geometry.

A :class:`Grid2D` only describes geometry (extent, resolution, spacing) --
it holds no field data. Field values live in :class:`~gpu_pde_lab.core.field.Field`,
which is bound to both a grid and a backend.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grid2D:
    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError("Grid must have at least 3 points per axis for a 5-point stencil")

    @property
    def dx(self) -> float:
        return self.lx / (self.nx - 1)

    @property
    def dy(self) -> float:
        return self.ly / (self.ny - 1)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def num_cells(self) -> int:
        return self.nx * self.ny

    def resized(self, nx: int, ny: int) -> "Grid2D":
        return Grid2D(nx=nx, ny=ny, lx=self.lx, ly=self.ly)

    def x_coords(self, backend):
        return backend.linspace(0.0, self.lx, self.nx)

    def y_coords(self, backend):
        return backend.linspace(0.0, self.ly, self.ny)

    def meshgrid(self, backend):
        return backend.meshgrid(self.x_coords(backend), self.y_coords(backend))
