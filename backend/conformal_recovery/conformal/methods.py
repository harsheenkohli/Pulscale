"""Uncertainty quantification methods.

Each class follows the same interface:

    method = MethodClass(alpha=0.1)
    method.calibrate(y_calib, y_pred_calib)
    lower, upper = method.intervals(y_pred_test)

This lets us compare them apples-to-apples on the same trained base model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats as stats


# -----------------------------------------------------------------------------
# Split Conformal Prediction (our primary contribution)
# -----------------------------------------------------------------------------


@dataclass
class SplitConformal:
    """Split conformal prediction (Angelopoulos & Bates 2021, §1).

    Computes a constant-width band that guarantees marginal coverage 1-alpha
    under exchangeability, no distributional assumptions.

    Procedure:
      1. Train base model on "proper training" set.
      2. On held-out calibration set, compute absolute residuals
         r_i = |y_i - y_pred_i|.
      3. Take the (1-alpha) quantile of {r_i} (with finite-sample (n+1)/n
         correction): q_hat = ceil((n+1)(1-alpha)) / n -th quantile of r_i.
      4. Test interval at point ŷ is [ŷ - q_hat, ŷ + q_hat].
    """

    alpha: float = 0.1

    def __post_init__(self):
        self.q_hat: np.ndarray | None = None

    def calibrate(self, y_calib: np.ndarray, y_pred_calib: np.ndarray) -> "SplitConformal":
        """Compute band half-widths from calibration residuals.

        Args:
            y_calib: ground-truth values, shape (n,) or (n, n_targets).
            y_pred_calib: model predictions on the same points.

        Returns:
            self, with `q_hat` populated.
        """
        residuals = np.abs(y_calib - y_pred_calib)
        n = len(residuals)
        # Finite-sample-corrected quantile level.
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = np.quantile(residuals, q_level, axis=0)
        return self

    def intervals(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_hat is None:
            raise RuntimeError("Must call .calibrate() before .intervals()")
        return y_pred - self.q_hat, y_pred + self.q_hat


# -----------------------------------------------------------------------------
# Empirical-quantile baseline (asymmetric band, no formal guarantee)
# -----------------------------------------------------------------------------


@dataclass
class EmpiricalQuantile:
    """Asymmetric band using α/2 and 1-α/2 percentiles of *signed* residuals.

    Common informal practice: same calibration data, but asymmetric. Lacks the
    formal coverage guarantee of split conformal but often works similarly well
    in practice. Here as a baseline to show conformal's value.
    """

    alpha: float = 0.1

    def __post_init__(self):
        self.q_lo: np.ndarray | None = None
        self.q_hi: np.ndarray | None = None

    def calibrate(self, y_calib: np.ndarray, y_pred_calib: np.ndarray) -> "EmpiricalQuantile":
        residuals = y_calib - y_pred_calib
        self.q_lo = np.quantile(residuals, self.alpha / 2, axis=0)
        self.q_hi = np.quantile(residuals, 1 - self.alpha / 2, axis=0)
        return self

    def intervals(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_lo is None or self.q_hi is None:
            raise RuntimeError("Must call .calibrate() before .intervals()")
        return y_pred + self.q_lo, y_pred + self.q_hi


# -----------------------------------------------------------------------------
# Gaussian ±zσ baseline (assumes Gaussian errors)
# -----------------------------------------------------------------------------


@dataclass
class GaussianCI:
    """Band of ±z·σ where σ = std of calibration residuals.

    Assumes residuals are zero-mean Gaussian. When the assumption fails
    (heavy-tailed errors, biased model), coverage is wrong. Here as a baseline
    to show what naive parametric UQ produces.
    """

    alpha: float = 0.1

    def __post_init__(self):
        self.z: float = float(stats.norm.ppf(1 - self.alpha / 2))
        self.sigma: np.ndarray | None = None

    def calibrate(self, y_calib: np.ndarray, y_pred_calib: np.ndarray) -> "GaussianCI":
        self.sigma = (y_calib - y_pred_calib).std(axis=0, ddof=1)
        return self

    def intervals(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.sigma is None:
            raise RuntimeError("Must call .calibrate() before .intervals()")
        half = self.z * self.sigma
        return y_pred - half, y_pred + half


# -----------------------------------------------------------------------------
# Conformalized Quantile Regression (CQR)
# -----------------------------------------------------------------------------


@dataclass
class CQR:
    """Conformalized Quantile Regression (Romano, Patterson, Candès 2019).

    Combines quantile regression (predicts conditional quantiles directly) with
    conformal calibration (gives formal coverage guarantee). Produces *adaptive*
    band widths — narrower when the model is confident, wider when not — unlike
    SplitConformal's constant-width bands.

    Procedure:
      1. Train base model with quantile loss to predict q_low (e.g. alpha/2)
         and q_high (1 - alpha/2) directly.
      2. On calibration set, compute conformity score per point:
            s_i = max(q_low_pred(x_i) - y_i,  y_i - q_high_pred(x_i))
         Negative score = both quantiles bracket truth; positive = at least one
         quantile is too tight on its side.
      3. Take eta = (1-alpha)-quantile of {s_i} (with finite-sample correction).
      4. Final interval at test point: [q_low_pred - eta, q_high_pred + eta].

    The eta correction *grows or shrinks both ends symmetrically* relative to
    the base quantile predictions, preserving their input-dependent shape.
    """

    alpha: float = 0.1

    def __post_init__(self):
        self.eta: np.ndarray | None = None

    def calibrate(
        self,
        y_calib: np.ndarray,
        q_low_calib: np.ndarray,
        q_high_calib: np.ndarray,
    ) -> "CQR":
        """Compute the conformity-score correction eta.

        Args:
            y_calib: ground truth, (n,) or (n, n_targets).
            q_low_calib: model's predicted low quantile on calibration.
            q_high_calib: model's predicted high quantile on calibration.

        Returns:
            self, with `eta` populated. Shape matches per-target.
        """
        scores = np.maximum(q_low_calib - y_calib, y_calib - q_high_calib)
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.eta = np.quantile(scores, q_level, axis=0)
        return self

    def intervals(
        self,
        q_low_pred: np.ndarray,
        q_high_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.eta is None:
            raise RuntimeError("Must call .calibrate() before .intervals()")
        return q_low_pred - self.eta, q_high_pred + self.eta


# -----------------------------------------------------------------------------
# Adaptive Conformal Inference (Xu & Xie 2021)
# -----------------------------------------------------------------------------


@dataclass
class AdaptiveConformal:
    """Adaptive Conformal Inference (Xu & Xie 2021).

    Maintains a time-varying alpha that adjusts based on recent miscoverage
    events. Designed for non-exchangeable / drifting distributions, which
    physiological time-series often violate.

    Update rule (after observing y_t):
        alpha_{t+1} = alpha_t + gamma * (target_alpha - 1{y_t in interval_t})

    The interval at step t is the SplitConformal interval computed with the
    current alpha_t but using the *running* calibration residuals.
    """

    alpha: float = 0.1            # target miscoverage
    gamma: float = 0.005          # learning rate (Xu & Xie 2021 use 0.005)
    init_residuals: np.ndarray | None = None   # initial calibration residuals (1D or 2D)

    def __post_init__(self):
        self.alpha_t = float(self.alpha)
        self._residuals: list[float] | None = None
        self._target_dim: int = 0
        if self.init_residuals is not None:
            arr = np.atleast_2d(self.init_residuals)
            self._residuals = list(arr.ravel())
            self._target_dim = arr.shape[1] if arr.ndim > 1 else 1

    def calibrate(self, y_calib: np.ndarray, y_pred_calib: np.ndarray) -> "AdaptiveConformal":
        """Initialize the residual buffer from calibration set."""
        self._residuals = np.abs(y_calib - y_pred_calib)
        self._target_dim = self._residuals.shape[1] if self._residuals.ndim > 1 else 1
        return self

    def step(self, y_pred_t: np.ndarray, y_true_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the interval at time t, then update alpha based on coverage.

        Returns (lower, upper) for time t. After this call, alpha_{t+1} is updated.
        """
        if self._residuals is None:
            raise RuntimeError("Call .calibrate() before .step()")
        residuals = np.asarray(self._residuals)

        # Compute current interval using alpha_t-quantile of residuals
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha_t)) / n
        q_level = float(min(max(q_level, 0.0), 1.0))
        q_hat = np.quantile(residuals, q_level, axis=0)
        lower, upper = y_pred_t - q_hat, y_pred_t + q_hat

        # Check coverage and update alpha
        in_band = ((y_true_t >= lower) & (y_true_t <= upper)).all()
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - (0.0 if in_band else 1.0))
        # Clamp to [0, 1]
        self.alpha_t = float(min(max(self.alpha_t, 1e-3), 1 - 1e-3))

        # Append today's residual into running buffer
        new_residual = np.abs(y_true_t - y_pred_t)
        self._residuals = np.concatenate([residuals, new_residual.reshape(1, -1) if new_residual.ndim > 0 else [[new_residual]]], axis=0)

        return lower, upper


# -----------------------------------------------------------------------------
# Helper: train base model + apply UQ method on a fold
# -----------------------------------------------------------------------------


def split_train_calibration(n: int, calib_frac: float = 0.25, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Random split of n indices into proper-training and calibration sets."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_calib = max(1, int(round(n * calib_frac)))
    calib_idx = idx[:n_calib]
    train_idx = idx[n_calib:]
    return train_idx, calib_idx
