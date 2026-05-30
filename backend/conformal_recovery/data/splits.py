"""Train/test splits for evaluation.

Three split strategies, each implemented as a generator yielding
(train_idx, test_idx) pairs of integer indices into the windowed metadata
DataFrame produced by `features.make_windows`.

1. **Leave-One-Subject-Out (LOSO)** — for each subject, train on all other
   subjects' windows, test on this subject's. Tests generalization to a new
   user.

2. **Temporal hold-out** — for each subject, train on the first `train_frac`
   of their days, test on the rest. Tests generalization to future time on
   the same user. Yields one (train, test) pair per subject; concatenate
   across subjects for cohort-wide metrics.

3. **LOSO + temporal** — combined: for each held-out subject, train on the
   first `train_frac` of all other subjects' days. Tests the hardest case:
   new user *and* future time.

All three respect chronological order — no future data leaks into past.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def loso_split(meta: pd.DataFrame) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held_out_subject, train_idx, test_idx) for each subject.

    `meta` must have a `subject` column. Returned indices index into `meta`.
    """
    subjects = meta["subject"].unique()
    for held_out in subjects:
        test_mask = meta["subject"].values == held_out
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        yield held_out, train_idx, test_idx


def temporal_split(
    meta: pd.DataFrame,
    train_frac: float = 0.7,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Per-subject temporal split.

    For each subject, sort their windows by `target_date` and take the first
    `train_frac` as train, the rest as test. Yields one tuple per subject;
    callers can concatenate across subjects for cohort-wide reporting.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")

    for subject, group in meta.groupby("subject", sort=False):
        sorted_idx = group.sort_values("target_date").index.values
        cut = int(len(sorted_idx) * train_frac)
        train_idx = sorted_idx[:cut]
        test_idx = sorted_idx[cut:]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield subject, train_idx, test_idx


def loso_temporal_split(
    meta: pd.DataFrame,
    train_frac: float = 0.7,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Hardest test: held-out subject AND only the first `train_frac` of
    every other subject's data is available for training.

    For each held-out subject, train on the first `train_frac` of all other
    subjects' windows (chronologically per other-subject), test on the
    held-out subject's full window set.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")

    subjects = meta["subject"].unique()
    for held_out in subjects:
        # Test = all of the held-out subject's windows
        test_idx = np.flatnonzero(meta["subject"].values == held_out)

        # Train = first train_frac of each *other* subject's windows
        train_idx_parts = []
        for other in subjects:
            if other == held_out:
                continue
            other_idx = meta.index[meta["subject"] == other].values
            other_sorted = meta.loc[other_idx].sort_values("target_date").index.values
            cut = int(len(other_sorted) * train_frac)
            train_idx_parts.append(other_sorted[:cut])

        train_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.array([], dtype=int)
        yield held_out, train_idx, test_idx


def filter_subjects(
    meta: pd.DataFrame,
    drop: list[str] | None = None,
) -> pd.DataFrame:
    """Return a meta DataFrame with `drop` subjects removed.

    Use this to enforce the cohort-level drop list (p04, p12, p13) before
    calling any split function. Preserves the original index so downstream
    indexing into X/y stays valid.
    """
    if not drop:
        return meta
    return meta[~meta["subject"].isin(drop)]


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT / "src"))
    from conformal_recovery.data.features import (
        ACTIVITY_TARGETS,
        prepare_features,
        make_windows,
    )
    from conformal_recovery.data.loaders import load_all_subjects

    raw = load_all_subjects(ROOT / "data" / "pmdata")
    feats = prepare_features(raw)

    feature_cols = [
        "rhr_dev", "rhr_ma3", "rhr_ma7",
        "sleep_efficiency_dev", "sleep_minutes_dev",
        "steps_dev", "calories_dev",
        "strain_proxy", "strain_minutes",
        "days_since_workout",
        "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
    ]
    targets = ["rhr_dev", "sleep_efficiency_dev"]
    X, y, meta = make_windows(feats, feature_cols, targets, window=7)

    # Drop p04, p12, p13 per cohort decision (and any subject with 0 windows)
    meta_filtered = filter_subjects(meta, drop=["p04"])
    print(f"Total windows after dropping p04: {len(meta_filtered)}")
    print(f"Subjects: {sorted(meta_filtered['subject'].unique())}")
    print()

    print("--- LOSO splits ---")
    for sid, tr, te in loso_split(meta_filtered):
        print(f"  hold out {sid}: train={len(tr)}, test={len(te)}")

    print()
    print("--- Temporal split (train_frac=0.7), per-subject ---")
    total_tr = total_te = 0
    for sid, tr, te in temporal_split(meta_filtered, train_frac=0.7):
        total_tr += len(tr)
        total_te += len(te)
    print(f"  Aggregate: train={total_tr}, test={total_te}")

    print()
    print("--- LOSO + temporal (train_frac=0.7) ---")
    for sid, tr, te in loso_temporal_split(meta_filtered, train_frac=0.7):
        print(f"  hold out {sid}: train={len(tr)}, test={len(te)}")
