"""Backend-agnostic bilinear sampling, used for semi-Lagrangian advection.

Given a field and floating-point *grid-index-space* coordinates (gx, gy),
returns the bilinearly-interpolated value at each coordinate. Coordinates
are clamped to the valid index range (clamp-to-edge), which acts as an
implicit free-slip wall for anything advected out of the domain.

Implemented with plain fancy indexing + arithmetic so the exact same code
path runs on NumPy arrays and Torch tensors (CPU or CUDA) -- no separate
torch.nn.functional.grid_sample special case, keeping the algorithm
transparent rather than delegating to a library primitive.
"""

from __future__ import annotations


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def bilinear_sample(field, gx, gy, backend):
    """Sample `field` (shape (nx, ny)) at coordinates (gx, gy) (same shape,
    float grid-index units). Returns an array of the same shape as gx/gy."""
    nx, ny = field.shape
    is_torch = _is_torch(field)

    # Clamp to [0, n-1] (inclusive) then floor; clamping x0/y0 to [0, n-2]
    # (rather than subtracting a small epsilon from the upper bound before
    # flooring) avoids float32 rounding silently reabsorbing the epsilon
    # near large index values and producing an out-of-bounds x0+1.
    if is_torch:
        import torch
        gx = torch.clamp(gx, 0.0, float(nx - 1))
        gy = torch.clamp(gy, 0.0, float(ny - 1))
        x0 = torch.clamp(torch.floor(gx).long(), 0, nx - 2)
        y0 = torch.clamp(torch.floor(gy).long(), 0, ny - 2)
    else:
        import numpy as np
        gx = np.clip(gx, 0.0, float(nx - 1))
        gy = np.clip(gy, 0.0, float(ny - 1))
        x0 = np.clip(np.floor(gx).astype(np.int64), 0, nx - 2)
        y0 = np.clip(np.floor(gy).astype(np.int64), 0, ny - 2)

    x1 = x0 + 1
    y1 = y0 + 1
    fx = gx - x0
    fy = gy - y0

    f00 = field[x0, y0]
    f10 = field[x1, y0]
    f01 = field[x0, y1]
    f11 = field[x1, y1]

    top = f00 * (1 - fx) + f10 * fx
    bottom = f01 * (1 - fx) + f11 * fx
    return top * (1 - fy) + bottom * fy
