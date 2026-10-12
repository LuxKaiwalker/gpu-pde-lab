"""Shared interfaces for PDE models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Any

from ..core.backend import Backend
from ..core.boundary import BoundaryKind
from ..core.grid import Grid2D


@dataclass
class SimulationState:
    """Everything that evolves during a run: time, step count, and field data."""

    time: float = 0.0
    step: int = 0
    fields: dict[str, Any] = dc_field(default_factory=dict)

    def reset(self) -> None:
        self.time = 0.0
        self.step = 0


class PDEModel(ABC):
    """Base class for a PDE + discretization + explicit/iterative update rule.

    Subclasses own their own state (e.g. `u` for heat, `u`/`u_prev` for wave)
    and implement `step()` to advance it by one timestep using the grid,
    backend, and boundary condition provided at construction.
    """

    name: str = "pde"

    def __init__(self, grid: Grid2D, backend: Backend, boundary: BoundaryKind):
        self.grid = grid
        self.backend = backend
        self.boundary = boundary
        self.state = SimulationState()

    @abstractmethod
    def reset(self, **initial_condition_kwargs) -> None:
        """(Re)initialize field state and zero time/step counters."""

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance the simulation by one timestep of size dt."""

    @abstractmethod
    def primary_field_numpy(self):
        """Return the main visualized field as a host-side float64 NumPy array."""

    def dt_max(self) -> float | None:
        """Theoretical stability bound for the explicit scheme, if one exists."""
        return None
