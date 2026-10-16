"""The 5-point Laplacian stencil.

    nabla^2 u[i,j] ~= (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / h^2

This single function is the computational core shared by the heat equation,
wave equation, and Jacobi/Gauss-Seidel/CG Poisson solvers. Every output cell
(i,j) depends only on its four direct neighbors -- there is no dependency
between non-adjacent cells within one stencil application. That locality is
exactly what makes the operation embarrassingly parallel: a GPU can assign
one thread (or vectorized lane) per grid cell and every thread reads its own
small neighborhood independently, with no inter-thread communication needed
to compute a single stencil pass.

On both NumPy and Torch, we implement the stencil using *shifted array
slices* rather than a Python double loop. This is what "vectorization" means
in practice: instead of looping over (i, j) and computing one scalar at a
time (which is how a single CPU core would naively do it, and how a single
GPU thread conceptually does it too), the whole grid is shifted and added in
one bulk operation. On a GPU, PyTorch dispatches this as one CUDA kernel
launch that runs thousands of threads concurrently, each thread computing
one (or a few) output cells -- that kernel launch is the "parallel computation"
step in the PDE -> discretization -> algorithm -> GPU pipeline.
"""

from __future__ import annotations

from .boundary import BoundaryKind


def laplacian(u, dx: float, dy: float, boundary: BoundaryKind):
    """Compute the 5-point Laplacian of ``u`` with the given boundary handling.

    Returns an array the same shape as ``u``. For Dirichlet/Neumann boundaries
    the edge values of the result are not meaningful (the caller is expected
    to only use/update interior points and then reapply the boundary
    condition separately -- see :func:`gpu_pde_lab.core.boundary.apply_boundary`).
    For periodic boundaries the Laplacian is valid everywhere, including the
    edges, because neighbor indices wrap around.
    """
    if boundary == BoundaryKind.PERIODIC:
        up = roll(u, -1, axis=0)
        down = roll(u, 1, axis=0)
        left = roll(u, 1, axis=1)
        right = roll(u, -1, axis=1)
        lap = (up - 2 * u + down) / (dx * dx) + (left - 2 * u + right) / (dy * dy)
        return lap

    # Non-periodic: compute only on the interior via slicing; pad the result
    # with zeros on the boundary ring since those values get overwritten by
    # apply_boundary() immediately afterwards anyway. `_zeros_like(u)` (a
    # pure allocate-and-fill) is used here rather than `u * 0.0`, which would
    # read the entire array just to discard it -- doubling the bandwidth of
    # what should be a cheap zero-fill, in the hottest path in the codebase
    # (every heat/wave/Poisson step funnels through this function).
    lap = _zeros_like(u)
    center = u[1:-1, 1:-1]
    north = u[:-2, 1:-1]
    south = u[2:, 1:-1]
    west = u[1:-1, :-2]
    east = u[1:-1, 2:]
    lap[1:-1, 1:-1] = (north + south - 2 * center) / (dx * dx) + (east + west - 2 * center) / (dy * dy)
    return lap


def laplacian_3d(u, dx: float, dy: float, dz: float, boundary: BoundaryKind):
    """3D analog of :func:`laplacian`: the 7-point stencil (center + 6 face
    neighbors) instead of the 5-point one (center + 4 edge neighbors).

    Same shifted-slice (not Python loop) approach, same boundary convention:
    non-periodic boundaries compute only the interior and leave the
    1-cell-thick face ring at zero (the caller re-applies the boundary
    condition via :func:`gpu_pde_lab.core.boundary.apply_boundary_3d`
    immediately afterward); periodic boundaries wrap on all 3 axes.
    """
    if boundary == BoundaryKind.PERIODIC:
        up = roll(u, -1, axis=0)
        down = roll(u, 1, axis=0)
        west = roll(u, 1, axis=1)
        east = roll(u, -1, axis=1)
        back = roll(u, 1, axis=2)
        front = roll(u, -1, axis=2)
        lap = ((up - 2 * u + down) / (dx * dx) + (west - 2 * u + east) / (dy * dy)
               + (back - 2 * u + front) / (dz * dz))
        return lap

    lap = _zeros_like(u)
    center = u[1:-1, 1:-1, 1:-1]
    north = u[:-2, 1:-1, 1:-1]
    south = u[2:, 1:-1, 1:-1]
    west = u[1:-1, :-2, 1:-1]
    east = u[1:-1, 2:, 1:-1]
    down = u[1:-1, 1:-1, :-2]
    up = u[1:-1, 1:-1, 2:]
    lap[1:-1, 1:-1, 1:-1] = (
        (north + south - 2 * center) / (dx * dx)
        + (east + west - 2 * center) / (dy * dy)
        + (up + down - 2 * center) / (dz * dz)
    )
    return lap


def _zeros_like(u):
    """Backend-agnostic zeros-with-same-shape/dtype/device, without reading
    `u`'s contents -- unlike `u * 0.0`, which touches the full array."""
    mod = type(u).__module__
    if mod.startswith("torch"):
        import torch

        return torch.zeros_like(u)
    import numpy as np

    return np.zeros_like(u)


def roll(u, shift: int, axis: int):
    """Backend-agnostic roll (wrap-around shift), used for periodic stencils."""
    # Works for both NumPy arrays (np.roll) and torch tensors (torch.roll);
    # both libraries expose a `.roll`-shaped free function with the same
    # (array, shift, axis) signature family, so we dispatch on module identity.
    mod = type(u).__module__
    if mod.startswith("torch"):
        import torch

        return torch.roll(u, shifts=shift, dims=axis)
    import numpy as np

    return np.roll(u, shift=shift, axis=axis)


def stencil_at(u, i: int, j: int, dx: float, dy: float, boundary: BoundaryKind) -> dict:
    """Introspect a single stencil application for the stencil-visualization experiment.

    Returns the neighbor values and the resulting Laplacian at one cell,
    without computing the whole-grid result -- this is the "zoom in on one
    GPU thread" view.
    """
    nx, ny = u.shape
    if boundary == BoundaryKind.PERIODIC:
        im1, ip1 = (i - 1) % nx, (i + 1) % nx
        jm1, jp1 = (j - 1) % ny, (j + 1) % ny
    else:
        im1, ip1 = max(i - 1, 0), min(i + 1, nx - 1)
        jm1, jp1 = max(j - 1, 0), min(j + 1, ny - 1)

    center = float(u[i, j])
    north = float(u[im1, j])
    south = float(u[ip1, j])
    west = float(u[i, jm1])
    east = float(u[i, jp1])
    lap = (north + south - 2 * center) / (dx * dx) + (east + west - 2 * center) / (dy * dy)
    return {
        "center": center,
        "north": north,
        "south": south,
        "west": west,
        "east": east,
        "laplacian": lap,
        "indices": {
            "center": (i, j),
            "north": (im1, j),
            "south": (ip1, j),
            "west": (i, jm1),
            "east": (i, jp1),
        },
    }
