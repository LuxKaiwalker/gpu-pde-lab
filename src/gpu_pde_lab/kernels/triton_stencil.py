"""A single fused Triton kernel for one explicit heat step.

Why this exists
---------------
The eager implementation expresses the stencil as shifted-slice arithmetic,
which is readable and portable and costs roughly four array-sized trips to
DRAM per step: read ``u`` to build the Laplacian, write the Laplacian
temporary, read it back for the update, write the new ``u`` -- plus a separate
launch for the boundary ring. A 5-point stencil is memory-bound, so that
factor is the whole performance story.

The lower bound for this computation is one read and one write. This kernel
hits it: each program loads its tile of ``u`` plus the one-cell halo it needs,
does the arithmetic in registers, and writes its tile of ``u_new``. Nothing
round-trips through memory in between, and the boundary condition is folded
into the same pass rather than costing a second launch.

Boundary handling without a second kernel
-----------------------------------------
Folding the boundary in is the one genuinely fiddly part, because a boundary
cell's value depends on an interior cell that a *different* program block
computes -- and there is no cross-block ordering inside a single launch.

The trick is to recompute rather than to read. For Neumann, every cell
evaluates the stencil at its own index *clamped into the interior*:

    ci = clamp(i, 1, nx-2),  cj = clamp(j, 1, ny-2)

An interior cell clamps to itself, so it is unaffected. An edge cell clamps to
its inward neighbour and so computes that neighbour's updated value, which is
exactly what ``apply_boundary``'s ``u[0, :] = u[1, :]`` produces. A corner
clamps on both axes and lands on its diagonal interior neighbour -- which is
also what the eager version's four sequential edge writes leave behind (see
``core/boundary.py``). The redundant work is O(n) cells out of O(n^2).

Dirichlet is simpler: the interior computes the stencil, the boundary ring
stores the fixed value.

Periodic needs no special case at all, which is why it is the cheapest of the
three: every index wraps, so there is no ring to treat differently.

Scope
-----
2D heat only, contiguous row-major, fp32/fp64. This is a measurement
instrument, not a replacement for :mod:`gpu_pde_lab.pdes` -- the eager solvers
stay the reference implementation because they run on every backend and are
what the correctness suite is written against. See ``doc.md`` for where this
sits in the project.
"""

from __future__ import annotations

from ..core.boundary import BoundaryKind

try:  # pragma: no cover - import guard, exercised by availability tests
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = exc


# Boundary kinds as integers, because a Triton kernel's compile-time
# specialization parameters have to be plain ints, not Python enums.
_BC_DIRICHLET = 0
_BC_NEUMANN = 1
_BC_PERIODIC = 2

_BC_CODES = {
    BoundaryKind.DIRICHLET: _BC_DIRICHLET,
    BoundaryKind.NEUMANN: _BC_NEUMANN,
    BoundaryKind.PERIODIC: _BC_PERIODIC,
}


def triton_is_available() -> bool:
    """True if Triton imported and a CUDA device is usable.

    Triton ships with torch but only compiles for an actual device, so both
    halves have to hold.
    """
    if triton is None:
        return False
    from ..core.backend import cuda_is_usable

    return cuda_is_usable()


def triton_unavailable_reason() -> str | None:
    """Why :func:`triton_is_available` is False, for reporting rather than guessing."""
    if triton is None:
        return f"Triton could not be imported: {_TRITON_IMPORT_ERROR}"
    from ..core.backend import cuda_free_memory_bytes, cuda_is_available, cuda_is_usable

    if cuda_is_usable():
        return None
    if not cuda_is_available():
        return "no CUDA device is present"
    free = cuda_free_memory_bytes()
    if free is None:
        return "the CUDA context could not be created (device likely full)"
    return f"only {free / 1024 ** 2:.0f} MB of device memory is free"


if triton is not None:  # pragma: no branch

    @triton.jit
    def _heat_step_kernel(
        u_ptr,
        out_ptr,
        coeff_ptr,
        nx,
        ny,
        row_stride,
        BC: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One fused FTCS heat step over a BLOCK_M x BLOCK_N tile.

        ``BC`` is ``tl.constexpr`` so each boundary condition compiles to its
        own specialized kernel with the branches resolved away -- a periodic
        run carries no Dirichlet comparison instructions and vice versa.

        The coefficients arrive through ``coeff_ptr`` rather than as scalar
        kernel arguments, which is not an arbitrary choice: Triton binds a
        Python ``float`` argument as **fp32**, so passing them directly
        silently truncates them and an fp64 field would be stepped with
        single-precision coefficients. Reading them from a tensor of the
        field's own dtype keeps the whole update in the requested precision.
        Cost is one 4-element load per program, amortized over the tile.
        """
        dx2 = tl.load(coeff_ptr + 0)
        dy2 = tl.load(coeff_ptr + 1)
        dt_alpha = tl.load(coeff_ptr + 2)
        dirichlet_value = tl.load(coeff_ptr + 3)

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        i = offs_m[:, None]
        j = offs_n[None, :]
        in_grid = (i < nx) & (j < ny)

        if BC == 2:  # periodic: every index wraps, no ring to special-case
            ci = i
            cj = j
            im = (ci - 1 + nx) % nx
            ip = (ci + 1) % nx
            jm = (cj - 1 + ny) % ny
            jp = (cj + 1) % ny
        else:
            # Clamp into the interior. Interior cells are unaffected; edge and
            # corner cells evaluate the stencil at the inward neighbour whose
            # value the boundary condition says they take.
            ci = tl.maximum(tl.minimum(i, nx - 2), 1)
            cj = tl.maximum(tl.minimum(j, ny - 2), 1)
            im = ci - 1
            ip = ci + 1
            jm = cj - 1
            jp = cj + 1

        # One read of u per cell (plus the halo overlap between tiles, which
        # the L2 cache absorbs). Everything below stays in registers.
        center = tl.load(u_ptr + ci * row_stride + cj, mask=in_grid, other=0.0)
        north = tl.load(u_ptr + im * row_stride + cj, mask=in_grid, other=0.0)
        south = tl.load(u_ptr + ip * row_stride + cj, mask=in_grid, other=0.0)
        west = tl.load(u_ptr + ci * row_stride + jm, mask=in_grid, other=0.0)
        east = tl.load(u_ptr + ci * row_stride + jp, mask=in_grid, other=0.0)

        lap = (north + south - 2 * center) / dx2 + (east + west - 2 * center) / dy2
        value = center + dt_alpha * lap

        if BC == 0:  # dirichlet: overwrite the ring with the fixed value
            on_edge = (i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1)
            value = tl.where(on_edge, dirichlet_value, value)

        tl.store(out_ptr + i * row_stride + j, value, mask=in_grid)


def make_triton_heat_step(boundary: BoundaryKind, block_m: int = 32, block_n: int = 64):
    """Return a ``(u, dt, alpha, dx, dy) -> u_new`` closure backed by the fused kernel.

    ``block_m``/``block_n`` default to a 32x64 tile: 2048 cells, which keeps
    enough rows in flight for the halo reads to hit in cache while staying
    inside the register budget at fp64. They are exposed because the best tile
    is card-specific, and the roofline harness sweeps them.
    """
    if triton is None:
        raise RuntimeError(f"Triton is not available: {triton_unavailable_reason()}")

    bc = _BC_CODES[boundary]

    # Coefficient tensors are cached by (dtype, values). Rebuilding one per
    # call would put a host-to-device copy inside every timed step -- a
    # self-inflicted version of exactly what the project's transfer
    # experiment exists to warn about. In steady state (fixed parameters,
    # many steps) this is built once.
    coeff_cache: dict[tuple, object] = {}

    def step(u, dt: float, alpha: float, dx: float, dy: float, dirichlet_value: float = 0.0):
        import torch

        if u.ndim != 2:
            raise ValueError(f"expected a 2D field, got shape {tuple(u.shape)}")
        if not u.is_cuda:
            raise ValueError("the Triton implementation requires a CUDA tensor")
        # The kernel indexes as `i * row_stride + j`, which is only the right
        # address arithmetic for a row-major contiguous tensor. Normalizing
        # beats silently reading the wrong cells.
        u = u.contiguous()

        key = (u.dtype, dx * dx, dy * dy, dt * alpha, dirichlet_value)
        coeffs = coeff_cache.get(key)
        if coeffs is None:
            coeffs = torch.tensor(
                [dx * dx, dy * dy, dt * alpha, dirichlet_value],
                dtype=u.dtype,
                device=u.device,
            )
            coeff_cache[key] = coeffs

        nx, ny = u.shape
        out = torch.empty_like(u)
        grid = (triton.cdiv(nx, block_m), triton.cdiv(ny, block_n))

        _heat_step_kernel[grid](
            u,
            out,
            coeffs,
            nx,
            ny,
            u.stride(0),
            BC=bc,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
        return out

    step.__doc__ = f"Fused Triton heat step, {boundary.value} boundary, {block_m}x{block_n} tile."
    return step
