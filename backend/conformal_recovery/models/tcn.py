"""Temporal Convolutional Network (TCN) forecaster.

Small TCN appropriate for a 7-day input window. Two dilated causal conv
blocks with dilations [1, 2] give a receptive field of 7 days — exactly
matching the input size. Stacking more blocks or using larger dilations
would be wasteful and prone to overfitting on our ~926-window dataset.

Reference: Bai, Kolter, Koltun (2018), "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling", arXiv:1803.01271.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """Causal 1D convolution: output at time t depends only on inputs at <= t.

    Implemented by padding only on the left and trimming the right side after
    the convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Trim right padding to enforce causality.
        return out[..., : -self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """Two causal conv layers + dropout + ReLU + residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu(self.conv1(x)))
        out = self.dropout2(self.relu(self.conv2(out)))
        return self.relu(out + self.residual(x))


class TCN(nn.Module):
    """Stack of TCN blocks + linear head producing point predictions."""

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        hidden: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_ch = n_features
        for d in dilations:
            layers.append(TCNBlock(in_ch, hidden, kernel_size, d, dropout))
            in_ch = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch, seq_len, features) -> conv expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        h = self.tcn(x)
        last = h[:, :, -1]  # take output at the final timestep
        return self.head(last)


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.1   # of train -> internal validation split
    early_stop_patience: int = 10
    device: str = "cpu"     # set to "mps" or "cuda" if available
    seed: int = 42


def _make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    val_frac: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xtr = torch.from_numpy(X[tr_idx]).float()
    ytr = torch.from_numpy(y[tr_idx]).float()
    Xva = torch.from_numpy(X[val_idx]).float()
    yva = torch.from_numpy(y[val_idx]).float()

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader


def fit(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig | None = None,
    verbose: bool = False,
) -> dict:
    """Train `model` on (X, y) with internal val split and early stopping.

    Returns a dict of training history keys: train_loss, val_loss (per epoch),
    plus best_epoch and best_val_loss.
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    tr_loader, va_loader = _make_loaders(X, y, cfg.batch_size, cfg.val_frac, cfg.seed)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_state, best_val, patience_left = None, float("inf"), cfg.early_stop_patience

    for epoch in range(cfg.epochs):
        model.train()
        tr_losses = []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_losses.append(loss_fn(model(xb), yb).item())

        tl, vl = float(np.mean(tr_losses)), float(np.mean(va_losses))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if verbose:
            print(f"  epoch {epoch:3d}  train={tl:.4f}  val={vl:.4f}")

        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stop_patience
            history["best_epoch"] = epoch
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    return history


def predict(model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    """Run inference on X. Returns (n, n_targets) numpy array."""
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature z-score using training-set statistics only (no leakage).

    Returns (X_train_std, X_test_std, mean, std).
    """
    # Compute mean/std over (samples, time) per feature
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)  # avoid division by zero on constant features
    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std
    return Xtr, Xte, mean, std


# -----------------------------------------------------------------------------
# Quantile-regression variant (for CQR — Conformalized Quantile Regression)
# -----------------------------------------------------------------------------


class TCNQuantile(nn.Module):
    """TCN that outputs `n_targets * 2` values: low and high quantile estimates
    per target. Same backbone as TCN, just a wider output head."""

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        hidden: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_targets = n_targets
        layers = []
        in_ch = n_features
        for d in dilations:
            layers.append(TCNBlock(in_ch, hidden, kernel_size, d, dropout))
            in_ch = hidden
        self.tcn = nn.Sequential(*layers)
        # Output 2 quantiles per target: [low_t0, low_t1, ..., high_t0, high_t1, ...]
        self.head = nn.Linear(hidden, n_targets * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.tcn(x)
        last = h[:, :, -1]
        return self.head(last)  # shape (batch, n_targets * 2)


def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball loss (a.k.a. quantile loss) for the tau-quantile.

    rho_tau(u) = u * (tau - 1{u < 0})

    Asymmetric: penalizes under-prediction more heavily for tau > 0.5,
    over-prediction more heavily for tau < 0.5.
    """
    diff = y_true - y_pred
    return torch.maximum(tau * diff, (tau - 1) * diff).mean()


def quantile_loss_dual(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    tau_low: float = 0.05,
    tau_high: float = 0.95,
) -> torch.Tensor:
    """Combined pinball loss for two quantiles simultaneously.

    Args:
        y_pred: (batch, n_targets * 2) — first half is low quantile, second is high.
        y_true: (batch, n_targets) — actual target values.
    """
    n_targets = y_true.shape[1]
    pred_low = y_pred[:, :n_targets]
    pred_high = y_pred[:, n_targets:]
    return pinball_loss(pred_low, y_true, tau_low) + pinball_loss(pred_high, y_true, tau_high)


def fit_quantile(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    tau_low: float = 0.05,
    tau_high: float = 0.95,
    config: TrainConfig | None = None,
    verbose: bool = False,
) -> dict:
    """Train a quantile-regression model with combined pinball loss.

    Same training loop as `fit()`, but uses the dual quantile loss instead
    of MSE. Validation loss is also pinball.
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    tr_loader, va_loader = _make_loaders(X, y, cfg.batch_size, cfg.val_frac, cfg.seed)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": []}
    best_state, best_val, patience_left = None, float("inf"), cfg.early_stop_patience

    for epoch in range(cfg.epochs):
        model.train()
        tr_losses = []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = quantile_loss_dual(pred, yb, tau_low, tau_high)
            loss.backward()
            optim.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_losses.append(quantile_loss_dual(model(xb), yb, tau_low, tau_high).item())

        tl, vl = float(np.mean(tr_losses)), float(np.mean(va_losses))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if verbose:
            print(f"  epoch {epoch:3d}  train={tl:.4f}  val={vl:.4f}")

        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stop_patience
            history["best_epoch"] = epoch
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    return history


def predict_quantile(
    model: nn.Module,
    X: np.ndarray,
    n_targets: int,
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a quantile model. Returns (low_quantile, high_quantile).
    Each has shape (n, n_targets)."""
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            out.append(model(xb).cpu().numpy())
    arr = np.concatenate(out, axis=0)
    return arr[:, :n_targets], arr[:, n_targets:]
