"""Tests for the trajectory-evaluation module (CI-safe, no data download)."""
from __future__ import annotations

import numpy as np

from monocular_vo.eval import ate_rmse, rpe_translation, scale_error, umeyama


def test_umeyama_recovers_known_transform() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    angle = np.deg2rad(30.0)
    R_true = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t_true = np.array([1.0, -2.0, 0.5])
    scale_true = 2.5
    dst = (scale_true * (R_true @ src.T)).T + t_true

    result = umeyama(src, dst, with_scale=True)
    assert np.allclose(result.R, R_true, atol=1e-8)
    assert np.allclose(result.t, t_true, atol=1e-8)
    assert np.isclose(result.scale, scale_true, atol=1e-8)


def test_ate_zero_for_identical_trajectories() -> None:
    gt = np.linspace([0, 0, 0], [10, 1, 0], 100)
    assert ate_rmse(gt.copy(), gt) < 1e-9


def test_ate_invariant_to_scale_when_alignment_with_scale() -> None:
    gt = np.linspace([0, 0, 0], [10, 0, 0], 100)
    est = gt * 1.5  # uniformly scaled
    # With scale alignment, ATE should be ~0
    assert ate_rmse(est, gt, with_scale=True) < 1e-9
    # Without scale alignment, ATE should pick up the scale error
    assert ate_rmse(est, gt, with_scale=False) > 0.5


def test_rpe_zero_for_identical_step_trajectories() -> None:
    rng = np.random.default_rng(2)
    gt = np.cumsum(rng.normal(scale=0.1, size=(50, 3)), axis=0)
    assert rpe_translation(gt.copy(), gt, delta=1) < 1e-9


def test_scale_error_sign_and_magnitude() -> None:
    gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    est_longer = gt * 1.1
    est_shorter = gt * 0.9
    assert np.isclose(scale_error(est_longer, gt), 10.0)
    assert np.isclose(scale_error(est_shorter, gt), -10.0)
