"""Regular 2D/3D grid geometry.

A :class:`Grid2D`/:class:`Grid3D` only describes geometry (extent,
resolution, spacing) -- it holds no field data. Field values live in
:class:`~gpu_pde_lab.core.field.Field`, which is bound to both a grid and a
backend.
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


@dataclass(frozen=True)
class Grid3D:
    """The 3D analog of :class:`Grid2D`, for the 3D Heat Equation experiment.

    Deliberately minimal (no shared base class with ``Grid2D``, no
    backend-bound coordinate/meshgrid helpers): the only 3D PDE in this
    codebase (``pdes/heat_3d.py``) builds its own initial-condition
    coordinate arrays with plain NumPy, exactly like every 2D PDE already
    does in its own ``reset()`` -- so those helpers would be unused
    generalization, not a real need.
    """

    nx: int
    ny: int
    nz: int
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3 or self.nz < 3:
            raise ValueError("Grid must have at least 3 points per axis for a 7-point stencil")

    @property
    def dx(self) -> float:
        return self.lx / (self.nx - 1)

    @property
    def dy(self) -> float:
        return self.ly / (self.ny - 1)

    @property
    def dz(self) -> float:
        return self.lz / (self.nz - 1)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def num_cells(self) -> int:
        return self.nx * self.ny * self.nz
