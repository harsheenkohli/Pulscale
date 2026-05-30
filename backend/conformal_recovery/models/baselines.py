"""Simple baselines.

These are the no-ML reference points the TCN must beat. They share the same
(X, y) interface as the TCN trainer so we can plug them into the same eval
harness.

Each baseline class implements:
    .fit(X_train, y_train)              -- mostly stateless, but keep the API
    .predict(X_test)                    -- (n, n_targets) array
    .predict_intervals(X_test, alpha)   -- optional, point ± half-band
"""

from __future__ import annotations

import numpy as np


class NaivePersistence:
    """y_pred[t] = X[t, -1, :n_targets].

    Predicts that tomorrow's deviation equals the last observed deviation in
    the input window.  Assumes the first n_targets columns of X are the same
    as the targets, in the same order — caller responsibility.
    """

    def __init__(self, n_targets: int):
        self.n_targets = n_targets

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaivePersistence":
        return self  # stateless

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, -1, : self.n_targets]


class TrailingMean:
    """y_pred[t] = mean of last `window` days of the target columns in X.

    Uses the input window's most recent `window` days. With window=7 on a
    7-step input, that's just the mean of the whole window.
    """

    def __init__(self, n_targets: int, window: int = 7):
        self.n_targets = n_targets
        self.window = window

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrailingMean":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        sub = X[:, -self.window :, : self.n_targets]
        return sub.mean(axis=1)


class GlobalMean:
    """Predicts the training-set mean for every example (regression-to-mean).

    A degenerate baseline. Useful as a "is the model learning anything at all"
    sanity check — any model that beats this is doing something.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GlobalMean":
        self.mean_ = y.mean(axis=0, keepdims=True)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Must call .fit() before .predict()")
        return np.repeat(self.mean_, len(X), axis=0)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-target MAE. Returns shape (n_targets,)."""
    return np.abs(y_true - y_pred).mean(axis=0)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-target RMSE. Returns shape (n_targets,)."""
    return np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
