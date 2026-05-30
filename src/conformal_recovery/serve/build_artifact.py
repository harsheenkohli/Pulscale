"""Build the production model artifact for the deployed API.

Unlike LOSO evaluation (which trains a fresh model per held-out subject),
deployment serves brand-new users — so we train one final 3-TCN ensemble on
*all* of the cleaned PMData cohort (12 subjects after dropping p04), with a
held-out calibration set for split conformal.

The artifact written to disk contains everything the API needs to:
  1. Standardize a new user's input window the same way the model was trained
  2. Run inference through all 3 ensemble members and average
  3. Add back the user's baseline (passed in as part of the request)
  4. Apply split conformal bands using the calibration residuals
  5. Optionally show one of a few sample PMData subjects

Run:  python -m conformal_recovery.serve.build_artifact
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conformal_recovery.data.features import make_windows, prepare_features
from conformal_recovery.data.loaders import load_all_subjects
from conformal_recovery.data.splits import filter_subjects
from conformal_recovery.models.tcn import TCN, TrainConfig, fit, predict, standardize
from conformal_recovery.conformal.methods import SplitConformal, split_train_calibration


# Same locked spec as in the paper
WINDOW = 14
HIDDEN = 32
DILATIONS = (1, 2, 4)
DROPOUT = 0.3
N_MODELS = 3
ENSEMBLE_SEEDS = (42, 142, 242)
ALPHA = 0.1
EPOCHS = 60
CALIB_FRAC = 0.25
DROP_SUBJECTS = ["p04"]

FEATURE_COLS = [
    "rhr_dev", "sleep_efficiency_dev",
    "rhr_ma3", "rhr_ma7",
    "sleep_minutes_dev",
    "steps_dev", "calories_dev",
    "strain_proxy", "strain_minutes",
    "days_since_workout",
    "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
]
TARGET_COLS = ["rhr_dev", "sleep_efficiency_dev"]


def build_and_save(pmdata_root: Path, out_path: Path, sample_subject_ids: list[str] | None = None):
    """Train the production ensemble, calibrate, and save the artifact."""
    print(f"Loading PMData from {pmdata_root}...")
    raw = load_all_subjects(pmdata_root)
    feats = prepare_features(raw)
    print(f"  Loaded {raw['subject'].nunique()} subjects, {len(raw)} subject-days")

    print("Building windows...")
    X, y, meta = make_windows(feats, FEATURE_COLS, TARGET_COLS, window=WINDOW)
    meta_clean = filter_subjects(meta, drop=DROP_SUBJECTS)
    keep = meta_clean.index.values
    X, y, meta_clean = X[keep], y[keep], meta_clean.reset_index(drop=True)
    print(f"  Cohort: {meta_clean['subject'].nunique()} subjects, {len(meta_clean)} windows")

    # 75/25 train/calibration split
    sub_tr, sub_calib = split_train_calibration(len(X), calib_frac=CALIB_FRAC, seed=42)
    Xtr, ytr = X[sub_tr], y[sub_tr]
    Xcalib, ycalib = X[sub_calib], y[sub_calib]
    print(f"  Train={len(Xtr)}, Calibration={len(Xcalib)}")

    # Standardization (fit on train)
    Xtr_s, Xcalib_s, x_mean, x_std = standardize(Xtr, Xcalib)

    # Residual prediction targets
    last_day_tr = Xtr[:, -1, :2]
    last_day_calib = Xcalib[:, -1, :2]
    ytr_resid = ytr - last_day_tr
    y_mean = ytr_resid.mean(axis=0, keepdims=True)
    y_std = ytr_resid.std(axis=0, keepdims=True) + 1e-8
    ytr_resid_s = (ytr_resid - y_mean) / y_std

    # Train the ensemble
    state_dicts = []
    calib_preds = []
    for k, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\nTraining ensemble member {k+1}/{N_MODELS} (seed={seed})...")
        torch.manual_seed(seed)
        model = TCN(
            n_features=X.shape[2],
            n_targets=2,
            hidden=HIDDEN,
            dilations=DILATIONS,
            dropout=DROPOUT,
        )
        cfg = TrainConfig(
            epochs=EPOCHS,
            batch_size=32,
            learning_rate=1e-3,
            device="cpu",  # production = CPU; deploy environments rarely have GPU
            seed=seed,
        )
        history = fit(model, Xtr_s, ytr_resid_s, cfg, verbose=False)
        print(f"  best epoch {history.get('best_epoch')}, val loss {history['best_val_loss']:.4f}")

        pc = predict(model, Xcalib_s, device="cpu") * y_std + y_mean + last_day_calib
        calib_preds.append(pc)
        state_dicts.append({k: v.cpu() for k, v in model.state_dict().items()})

    pred_calib = np.mean(calib_preds, axis=0)

    # Calibrate split conformal
    print("\nCalibrating split conformal...")
    sc = SplitConformal(alpha=ALPHA)
    sc.calibrate(ycalib, pred_calib)
    print(f"  q_hat = {sc.q_hat}")

    # Sample subjects: write last-N days of each chosen subject for the demo
    if sample_subject_ids is None:
        sample_subject_ids = ["p01", "p06", "p15", "p16"]  # cleanest subjects
    samples = {}
    for sid in sample_subject_ids:
        if sid in DROP_SUBJECTS:
            continue
        sub = feats[feats["subject"] == sid].sort_values("date").tail(60).reset_index(drop=True)
        if len(sub) < WINDOW + 1:
            continue
        samples[sid] = {
            "dates": [str(d.date()) for d in sub["date"]],
            "rhr": sub["rhr"].tolist(),
            "rhr_baseline": sub["rhr_baseline"].tolist(),
            "sleep_efficiency": sub["sleep_efficiency"].tolist(),
            "sleep_efficiency_baseline": sub["sleep_efficiency_baseline"].tolist(),
            "steps": sub["steps"].tolist(),
            "strain_proxy": sub["strain_proxy"].tolist(),
            # Pre-built last 14-day window of features (so the API can do
            # one-shot inference without re-running the feature pipeline)
            "last_window_features": sub.iloc[-WINDOW:][FEATURE_COLS].astype(float).values.tolist(),
        }
        print(f"  Bundled sample: {sid} with last {len(sub)} days")

    # Save the artifact
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": "0.1.0",
        "config": {
            "window": WINDOW,
            "hidden": HIDDEN,
            "dilations": list(DILATIONS),
            "dropout": DROPOUT,
            "n_models": N_MODELS,
            "alpha": ALPHA,
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
        },
        "state_dicts": state_dicts,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean_residual": y_mean.astype(np.float32),
        "y_std_residual": y_std.astype(np.float32),
        "split_conformal_q_hat": sc.q_hat.astype(np.float32),
        "samples": samples,
    }
    torch.save(artifact, out_path)
    print(f"\nArtifact saved to {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")
    return artifact


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]
    pmdata_root = ROOT / "data" / "pmdata"
    out_path = ROOT / "backend" / "model_artifact.pt"
    build_and_save(pmdata_root, out_path)
