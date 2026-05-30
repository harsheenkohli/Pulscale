"""Evaluation metrics for prediction intervals.

The two core metrics for conformal evaluation are coverage and width.
We also expose per-subject decomposition and a calibration-curve helper for
the reliability diagram.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Empirical marginal coverage = fraction of points where lower <= y <= upper.

    Returns shape (n_targets,) for multi-target predictions, or scalar for 1-D.
    """
    in_band = (y_true >= lower) & (y_true <= upper)
    return in_band.mean(axis=0)


def mean_width(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Mean band width = mean(upper - lower)."""
    return (upper - lower).mean(axis=0)


def median_width(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Median band width — robust to outliers."""
    return np.median(upper - lower, axis=0)


def coverage_gap(empirical: np.ndarray, nominal: float) -> np.ndarray:
    """|empirical - nominal|. Single number summarizing miscalibration."""
    return np.abs(empirical - nominal)


def per_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    subjects: np.ndarray,
    target_names: list[str],
) -> pd.DataFrame:
    """Compute coverage, width, MAE per subject. Long-format output."""
    rows = []
    for sid in np.unique(subjects):
        mask = subjects == sid
        yt = y_true[mask]
        yp = y_pred[mask]
        lo = lower[mask]
        up = upper[mask]
        cov = coverage(yt, lo, up)
        w = mean_width(lo, up)
        mae = np.abs(yt - yp).mean(axis=0)
        for i, name in enumerate(target_names):
            rows.append({
                "subject": sid,
                "target": name,
                "coverage": float(np.atleast_1d(cov)[i]),
                "mean_width": float(np.atleast_1d(w)[i]),
                "mae": float(np.atleast_1d(mae)[i]),
                "n_test": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def calibration_curve_data(
    y_true_calib: np.ndarray,
    y_pred_calib: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    alpha_grid: np.ndarray | None = None,
    target_names: list[str] | None = None,
) -> pd.DataFrame:
    """For each alpha, compute split-conformal coverage on the test set.

    Returns long-format DataFrame: alpha, target, nominal_coverage, empirical_coverage, mean_width.
    Use with seaborn lineplot for the reliability diagram.
    """
    from conformal_recovery.conformal.methods import SplitConformal

    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 0.5, 10)
    n_targets = (
        y_true_calib.shape[1] if y_true_calib.ndim > 1 else 1
    )
    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]

    rows = []
    for alpha in alpha_grid:
        sc = SplitConformal(alpha=alpha)
        sc.calibrate(y_true_calib, y_pred_calib)
        lo, up = sc.intervals(y_pred_test)
        emp_cov = coverage(y_true_test, lo, up)
        widths = mean_width(lo, up)
        for i, name in enumerate(target_names):
            rows.append({
                "alpha": float(alpha),
                "target": name,
                "nominal_coverage": float(1 - alpha),
                "empirical_coverage": float(np.atleast_1d(emp_cov)[i]),
                "mean_width": float(np.atleast_1d(widths)[i]),
            })
    return pd.DataFrame(rows)
