"""Boundary condition application for the 5-point stencil grids.

Boundary conditions are applied by writing into the ghost/edge rows and
columns of a field array *after* an interior update, which is the simplest
correct way to combine them with an explicit stencil update:

1. Compute the Laplacian / update only on interior points.
2. Overwrite the boundary points according to the chosen BC.

Periodic boundaries are instead handled at the *stencil* level (wrap-around
neighbor indexing), since a periodic domain has no true "edge" -- see
:mod:`gpu_pde_lab.core.stencil`.
"""

from __future__ import annotations

from enum import Enum


class BoundaryKind(str, Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"


def apply_boundary(u, kind: BoundaryKind, dirichlet_value: float = 0.0):
    """Apply a boundary condition to the edges of field ``u`` in place (where possible).

    Dirichlet: fixed value at the boundary (e.g. walls held at a fixed temperature).
    Neumann:   zero normal derivative, i.e. edge = first interior neighbor
               (insulated / reflecting boundary -- no flux crosses it).
    Periodic:  not handled here; the stencil wraps indices directly instead,
               so this is a no-op for periodic domains.
    """
    if kind == BoundaryKind.PERIODIC:
        return u

    if kind == BoundaryKind.DIRICHLET:
        u[0, :] = dirichlet_value
        u[-1, :] = dirichlet_value
        u[:, 0] = dirichlet_value
        u[:, -1] = dirichlet_value
        return u

    if kind == BoundaryKind.NEUMANN:
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        return u

    raise ValueError(f"Unknown boundary kind: {kind}")
