"""Backend-independent 2D field editing primitives: brush painting, geometric
stamps, and a bounded undo/redo buffer.

Everything here operates on plain host ``numpy.ndarray`` instances in
continuous grid-index space (the same (row, col) float-index convention used
by :func:`gpu_pde_lab.core.interp.bilinear_sample`) -- there is no Qt and no
``Backend`` import. The interactive editor widget (``ui/field_editor_widget.py``)
is the only caller that knows about mouse events; it hands this module plain
arrays and float coordinates and gets plain arrays back. This split is what
makes the editing logic testable without a display and reusable for any PDE's
editable field (an initial condition, a source term, a velocity component).

All ops are bounding-box limited: a brush/stamp only ever touches the small
region it can possibly affect, not the whole grid, so painting stays cheap on
large grids even though it mutates plain NumPy arrays in place.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np

BlendMode = Literal["add", "erase", "set"]
Falloff = Literal["constant", "gaussian"]


def _blend(dst: np.ndarray, weight: np.ndarray, value: float, mode: BlendMode) -> None:
    if mode == "add":
        dst += value * weight
    elif mode == "erase":
        dst -= value * weight
    elif mode == "set":
        dst[:] = value * weight + dst * (1.0 - weight)
    else:
        raise ValueError(f"Unknown blend mode: {mode!r}")


def _clip_bbox(lo: float, hi: float, n: int) -> tuple[int, int]:
    i0 = max(int(np.floor(lo)), 0)
    i1 = min(int(np.ceil(hi)) + 1, n)
    return i0, i1


def paint_brush(field: np.ndarray, cx: float, cy: float, radius: float, strength: float,
                 mode: BlendMode = "add", falloff: Falloff = "gaussian") -> np.ndarray:
    """Paint a circular brush centered at continuous grid-index (cx, cy).

    ``falloff="gaussian"`` tapers smoothly to zero at ``radius`` (sigma =
    radius/2); ``falloff="constant"`` is a hard disc. Mutates and returns
    ``field`` in place.
    """
    nx, ny = field.shape
    r = max(float(radius), 1e-6)
    i0, i1 = _clip_bbox(cx - r, cx + r, nx)
    j0, j1 = _clip_bbox(cy - r, cy + r, ny)
    if i0 >= i1 or j0 >= j1:
        return field

    ii = np.arange(i0, i1, dtype=np.float64)[:, None]
    jj = np.arange(j0, j1, dtype=np.float64)[None, :]
    dist = np.sqrt((ii - cx) ** 2 + (jj - cy) ** 2)

    if falloff == "constant":
        weight = (dist <= r).astype(field.dtype)
    elif falloff == "gaussian":
        sigma = r / 2.0
        weight = np.exp(-(dist ** 2) / (2 * sigma * sigma))
        weight[dist > r] = 0.0
        weight = weight.astype(field.dtype)
    else:
        raise ValueError(f"Unknown falloff: {falloff!r}")

    _blend(field[i0:i1, j0:j1], weight, strength, mode)
    return field


def stamp_circle(field: np.ndarray, cx: float, cy: float, radius: float, value: float,
                  mode: BlendMode = "set") -> np.ndarray:
    """Stamp a hard-edged filled disc. Mutates and returns ``field`` in place."""
    nx, ny = field.shape
    r = max(float(radius), 1e-6)
    i0, i1 = _clip_bbox(cx - r, cx + r, nx)
    j0, j1 = _clip_bbox(cy - r, cy + r, ny)
    if i0 >= i1 or j0 >= j1:
        return field

    ii = np.arange(i0, i1, dtype=np.float64)[:, None]
    jj = np.arange(j0, j1, dtype=np.float64)[None, :]
    dist = np.sqrt((ii - cx) ** 2 + (jj - cy) ** 2)
    weight = (dist <= r).astype(field.dtype)
    _blend(field[i0:i1, j0:j1], weight, value, mode)
    return field


def stamp_rect(field: np.ndarray, x0: float, y0: float, x1: float, y1: float, value: float,
               mode: BlendMode = "set") -> np.ndarray:
    """Stamp an axis-aligned filled rectangle (corners in either order).
    Mutates and returns ``field`` in place."""
    nx, ny = field.shape
    xlo, xhi = (x0, x1) if x0 <= x1 else (x1, x0)
    ylo, yhi = (y0, y1) if y0 <= y1 else (y1, y0)
    i0, i1 = _clip_bbox(xlo, xhi, nx)
    j0, j1 = _clip_bbox(ylo, yhi, ny)
    if i0 >= i1 or j0 >= j1:
        return field

    weight = np.ones((i1 - i0, j1 - j0), dtype=field.dtype)
    _blend(field[i0:i1, j0:j1], weight, value, mode)
    return field


def stamp_line(field: np.ndarray, x0: float, y0: float, x1: float, y1: float, thickness: float,
               value: float, mode: BlendMode = "set") -> np.ndarray:
    """Stamp a filled segment from (x0, y0) to (x1, y1) with the given
    thickness (full width, in grid-index units). Mutates and returns
    ``field`` in place."""
    nx, ny = field.shape
    half = max(float(thickness) / 2.0, 0.5)
    i0, i1 = _clip_bbox(min(x0, x1) - half, max(x0, x1) + half, nx)
    j0, j1 = _clip_bbox(min(y0, y1) - half, max(y0, y1) + half, ny)
    if i0 >= i1 or j0 >= j1:
        return field

    ii = np.arange(i0, i1, dtype=np.float64)[:, None]
    jj = np.arange(j0, j1, dtype=np.float64)[None, :]
    dx, dy = x1 - x0, y1 - y0
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-12:
        dist = np.sqrt((ii - x0) ** 2 + (jj - y0) ** 2)
    else:
        t = ((ii - x0) * dx + (jj - y0) * dy) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        px = x0 + t * dx
        py = y0 + t * dy
        dist = np.sqrt((ii - px) ** 2 + (jj - py) ** 2)
    weight = (dist <= half).astype(field.dtype)
    _blend(field[i0:i1, j0:j1], weight, value, mode)
    return field


class EditableField:
    """One named, host-side editable buffer with bounded undo/redo.

    Deliberately keeps *snapshots* (not a general command stack): the task is
    hand-drawing a field for a few dozen cells at a time, not scripting
    arbitrary edits, so a bounded stack of full-array copies is both simpler
    and exactly what was asked for. ``undo_depth`` bounds memory use
    regardless of how many edits are made in a session.
    """

    def __init__(self, name: str, shape: tuple[int, int], dtype=np.float32,
                 initial: np.ndarray | None = None, undo_depth: int = 20):
        self.name = name
        self.dtype = dtype
        self.data: np.ndarray = (
            np.array(initial, dtype=dtype, copy=True) if initial is not None
            else np.zeros(shape, dtype=dtype)
        )
        self._undo: deque[np.ndarray] = deque(maxlen=undo_depth)
        self._redo: deque[np.ndarray] = deque(maxlen=undo_depth)

    def snapshot_for_undo(self) -> None:
        """Call once per committed edit *gesture* (not per mouse-move event)
        before mutating ``self.data``."""
        self._undo.append(self.data.copy())
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.data.copy())
        self.data = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.data.copy())
        self.data = self._redo.pop()
        return True

    def can_undo(self) -> bool:
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def clear(self, value: float = 0.0) -> None:
        self.snapshot_for_undo()
        self.data[:] = value

    def set_data(self, array: np.ndarray) -> None:
        """Replace the buffer wholesale (e.g. applying a preset), recording
        one undo step."""
        self.snapshot_for_undo()
        self.data = np.array(array, dtype=self.dtype, copy=True)

    def numpy(self) -> np.ndarray:
        """An independent copy -- never a view of the live buffer, matching
        the rest of the codebase's `Field.numpy()`/`Backend.to_numpy()`
        convention that snapshots for display/IO never alias live state."""
        return self.data.copy()
