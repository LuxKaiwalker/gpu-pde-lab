"""The multi-pass baseline: exactly what the eager solver does.

This is deliberately a copy of :meth:`gpu_pde_lab.pdes.heat.HeatEquation.step`
rather than a call into it. The comparison needs a pure
``(u, params) -> u_new`` function with no object state, so it can be captured
into a CUDA graph or handed to ``torch.compile``; and keeping the reference
implementation textually beside the optimized ones is what makes the diff
legible.

The expressions below mirror :mod:`gpu_pde_lab.core.stencil` *operation for
operation*, down to dividing by ``dx*dx`` rather than multiplying by a
precomputed reciprocal and to the order of the additions. Floating-point
addition is not associative and ``x / h`` is not ``x * (1/h)``, so any
tidying-up here would cost bit-exactness -- and bit-exactness is what lets
``tests/test_kernels.py`` assert this really is the same computation rather
than merely a close one.
"""

from __future__ import annotations

from ..core.boundary import BoundaryKind


def heat_step_baseline(u, dt: float, alpha: float, dx: float, dy: float,
                       boundary: BoundaryKind, dirichlet_value: float = 0.0):
    """One explicit FTCS heat step, returning a new array.

    Reproduces the eager path's memory behaviour on purpose: a full-sized
    Laplacian temporary, then a full-sized update, then boundary writes.
    """
    import torch

    if boundary == BoundaryKind.PERIODIC:
        up = torch.roll(u, -1, 0)
        down = torch.roll(u, 1, 0)
        left = torch.roll(u, 1, 1)
        right = torch.roll(u, -1, 1)
        lap = (up - 2 * u + down) / (dx * dx) + (left - 2 * u + right) / (dy * dy)
        return u + dt * alpha * lap

    lap = torch.zeros_like(u)
    center = u[1:-1, 1:-1]
    north = u[:-2, 1:-1]
    south = u[2:, 1:-1]
    west = u[1:-1, :-2]
    east = u[1:-1, 2:]
    lap[1:-1, 1:-1] = (north + south - 2 * center) / (dx * dx) + (east + west - 2 * center) / (dy * dy)
    out = u + dt * alpha * lap

    # Same sequential edge writes as core.boundary.apply_boundary, including
    # the corner values its ordering implies: `out[:, 0] = out[:, 1]` reads a
    # column that the preceding row writes have already touched, so each
    # corner ends up holding its diagonal interior neighbour.
    if boundary == BoundaryKind.DIRICHLET:
        out[0, :] = dirichlet_value
        out[-1, :] = dirichlet_value
        out[:, 0] = dirichlet_value
        out[:, -1] = dirichlet_value
    else:  # NEUMANN
        out[0, :] = out[1, :]
        out[-1, :] = out[-2, :]
        out[:, 0] = out[:, 1]
        out[:, -1] = out[:, -2]
    return out
