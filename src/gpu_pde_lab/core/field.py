"""A field: a named array of values living on a grid, bound to a backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend import Backend
from .grid import Grid2D


@dataclass
class Field:
    """A 2D scalar field over a grid, stored as whatever array type the
    backend uses (``numpy.ndarray`` or ``torch.Tensor``)."""

    name: str
    grid: Grid2D
    backend: Backend
    data: Any

    @classmethod
    def zeros(cls, name: str, grid: Grid2D, backend: Backend) -> "Field":
        return cls(name=name, grid=grid, backend=backend, data=backend.zeros(grid.shape))

    @classmethod
    def from_numpy(cls, name: str, grid: Grid2D, backend: Backend, array) -> "Field":
        return cls(name=name, grid=grid, backend=backend, data=backend.array(array))

    def numpy(self):
        """Host-side float64 copy, suitable for plotting or saving."""
        return self.backend.to_numpy(self.data)

    def clone(self) -> "Field":
        return Field(self.name, self.grid, self.backend, self.backend.clone(self.data))

    def nbytes(self) -> int:
        return self.grid.num_cells * self.backend.bytes_per_element()

    def to_backend(self, other_backend: Backend) -> "Field":
        return Field(self.name, self.grid, other_backend, self.backend.to_device(self.data, other_backend))
