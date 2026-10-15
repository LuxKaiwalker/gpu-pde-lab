"""Save/load for field-editor configurations: edited fields plus the PDE
parameters needed to reconstruct the experiment.

Single-file `.npz` format via plain ``numpy.savez``/``numpy.load`` with
``allow_pickle=False`` (the default) -- no ``pickle``, no ``eval``, nothing
that can execute arbitrary code on load. One array per editable field slot
(host, float32) plus one ``__meta__`` array: a length-1 byte buffer holding a
UTF-8 JSON blob with the PDE name, grid geometry, boundary kind, and
PDE-specific scalar parameters.

Every field read from a loaded file is validated before being handed back:
file size, declared vs. actual array shape, grid-dimension sanity bounds, and
finiteness. Any failure raises :class:`ConfigValidationError` rather than
partially applying a corrupt/malicious file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .boundary import BoundaryKind

MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_GRID_DIM = 4096
MIN_GRID_DIM = 3
META_KEY = "__meta__"
_REQUIRED_META_KEYS = {"pde_name", "nx", "ny", "lx", "ly", "boundary", "params", "field_keys"}


class ConfigValidationError(ValueError):
    """Raised for any malformed, oversized, or inconsistent config file."""


@dataclass
class ExperimentConfig:
    pde_name: str
    nx: int
    ny: int
    lx: float
    ly: float
    boundary: str
    params: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, np.ndarray] = field(default_factory=dict)


def save_config(path: str | Path, config: ExperimentConfig) -> None:
    meta = {
        "pde_name": config.pde_name,
        "nx": config.nx,
        "ny": config.ny,
        "lx": config.lx,
        "ly": config.ly,
        "boundary": config.boundary,
        "params": config.params,
        "field_keys": list(config.fields.keys()),
    }
    meta_bytes = json.dumps(meta).encode("utf-8")

    arrays: dict[str, np.ndarray] = {
        key: np.asarray(arr, dtype=np.float32) for key, arr in config.fields.items()
    }
    arrays[META_KEY] = np.frombuffer(meta_bytes, dtype=np.uint8)
    np.savez(path, **arrays)


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigValidationError(f"No such config file: {path}")

    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ConfigValidationError(f"Config file too large ({size} bytes > {MAX_FILE_BYTES})")

    try:
        npz = np.load(p, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - any load failure is a validation failure here
        raise ConfigValidationError(f"Could not read {path!r} as a config file: {exc}") from exc

    try:
        if META_KEY not in npz:
            raise ConfigValidationError("Config file is missing its metadata block")

        try:
            meta = json.loads(npz[META_KEY].tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigValidationError(f"Corrupt metadata block: {exc}") from exc

        if not isinstance(meta, dict) or not _REQUIRED_META_KEYS.issubset(meta):
            missing = _REQUIRED_META_KEYS - (set(meta) if isinstance(meta, dict) else set())
            raise ConfigValidationError(f"Metadata missing required keys: {missing}")

        nx, ny = meta["nx"], meta["ny"]
        if not (isinstance(nx, int) and isinstance(ny, int)):
            raise ConfigValidationError("nx/ny must be integers")
        if not (MIN_GRID_DIM <= nx <= MAX_GRID_DIM and MIN_GRID_DIM <= ny <= MAX_GRID_DIM):
            raise ConfigValidationError(f"Grid dimensions out of range: {nx} x {ny}")

        try:
            BoundaryKind(meta["boundary"])
        except ValueError as exc:
            raise ConfigValidationError(f"Unknown boundary kind: {meta['boundary']!r}") from exc

        if not isinstance(meta.get("field_keys"), list):
            raise ConfigValidationError("field_keys must be a list")

        fields: dict[str, np.ndarray] = {}
        for key in meta["field_keys"]:
            if key not in npz:
                raise ConfigValidationError(f"Missing declared field array: {key!r}")
            arr = npz[key]
            if arr.shape != (nx, ny):
                raise ConfigValidationError(
                    f"Field {key!r} shape {arr.shape} does not match declared grid ({nx}, {ny})"
                )
            if not np.all(np.isfinite(arr)):
                raise ConfigValidationError(f"Field {key!r} contains non-finite values")
            fields[key] = arr.astype(np.float64)

        return ExperimentConfig(
            pde_name=str(meta["pde_name"]),
            nx=nx, ny=ny,
            lx=float(meta["lx"]), ly=float(meta["ly"]),
            boundary=str(meta["boundary"]),
            params=dict(meta["params"]) if isinstance(meta["params"], dict) else {},
            fields=fields,
        )
    finally:
        npz.close()
