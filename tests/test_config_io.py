import json

import numpy as np
import pytest

from gpu_pde_lab.core.config_io import (
    ConfigValidationError,
    ExperimentConfig,
    load_config,
    save_config,
)


def _simple_config(nx=8, ny=8):
    return ExperimentConfig(
        pde_name="heat",
        nx=nx, ny=ny, lx=1.0, ly=1.0,
        boundary="neumann",
        params={"alpha": 0.1},
        fields={"u0": np.random.default_rng(0).standard_normal((nx, ny))},
    )


def test_save_load_round_trip_preserves_fields_and_params(tmp_path):
    cfg = _simple_config()
    path = tmp_path / "test.npz"
    save_config(path, cfg)

    loaded = load_config(path)
    assert loaded.pde_name == "heat"
    assert loaded.nx == 8 and loaded.ny == 8
    assert loaded.boundary == "neumann"
    assert loaded.params == {"alpha": 0.1}
    assert np.allclose(loaded.fields["u0"], cfg.fields["u0"], atol=1e-5)  # float32 round trip


def test_load_rejects_oversized_file(tmp_path, monkeypatch):
    cfg = _simple_config()
    path = tmp_path / "test.npz"
    save_config(path, cfg)
    monkeypatch.setattr("gpu_pde_lab.core.config_io.MAX_FILE_BYTES", 1)
    with pytest.raises(ConfigValidationError):
        load_config(path)


def test_load_rejects_shape_mismatch(tmp_path):
    cfg = _simple_config(nx=8, ny=8)
    path = tmp_path / "test.npz"
    save_config(path, cfg)

    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta = json.loads(arrays["__meta__"].tobytes().decode("utf-8"))
    meta["nx"] = 99
    meta["ny"] = 99
    arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)

    with pytest.raises(ConfigValidationError):
        load_config(path)


def test_load_rejects_non_finite_values(tmp_path):
    cfg = _simple_config(nx=4, ny=4)
    cfg.fields["u0"][0, 0] = np.inf
    path = tmp_path / "test.npz"
    save_config(path, cfg)
    with pytest.raises(ConfigValidationError):
        load_config(path)


def test_load_rejects_unknown_boundary(tmp_path):
    cfg = _simple_config()
    cfg.boundary = "not_a_real_boundary"
    path = tmp_path / "test.npz"
    save_config(path, cfg)
    with pytest.raises(ConfigValidationError):
        load_config(path)


def test_load_rejects_missing_metadata(tmp_path):
    path = tmp_path / "test.npz"
    np.savez(path, u0=np.zeros((4, 4), dtype=np.float32))  # no __meta__
    with pytest.raises(ConfigValidationError):
        load_config(path)


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(ConfigValidationError):
        load_config(tmp_path / "does_not_exist.npz")
