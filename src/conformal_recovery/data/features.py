"""Feature engineering for the recovery and activity forecasters.

This module turns the daily DataFrame produced by `loaders.load_subject` into
model-ready features. The pipeline:

    daily DataFrame
      → personal baselines (rolling 30-day means per subject)
      → baseline-centered features (deviation from each subject's normal)
      → strain features (workout impulse, calorie-based proxy, optional Banister TRIMP)
      → calendar features (day-of-week)
      → trailing means (3- and 7-day)
      → short-gap imputation (forward-fill up to 2 days)
      → 7-day windowed (X, y) arrays for the model

The recovery model uses the *full* feature set (RHR + sleep features).
The activity model uses a phone-only subset (no HR/sleep features) to support
users without a wrist-worn device.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ROLLING_BASELINE_DAYS = 30
ROLLING_BASELINE_MIN_PERIODS = 14  # need at least 2 weeks of data for a baseline
INPUT_WINDOW_DAYS = 7
SHORT_GAP_FILL_LIMIT = 2  # forward-fill missing days up to this many in a row

# Recovery-model target columns (Watch-class data required)
RECOVERY_TARGETS = ["rhr", "sleep_efficiency"]

# Activity-model targets (phone-only data sufficient)
ACTIVITY_TARGETS = ["steps"]

# Which columns get baseline-centered. Anything that's bounded and has a
# meaningful "personal normal" (resting HR, sleep efficiency, daily steps).
COLUMNS_TO_PERSONALIZE = [
    "rhr",
    "sleep_efficiency",
    "sleep_minutes",
    "steps",
    "calories",
    "very_active_min",
    "moderately_active_min",
    "lightly_active_min",
]


# -----------------------------------------------------------------------------
# Per-subject helpers (operate on a single subject's chronological DataFrame)
# -----------------------------------------------------------------------------


def _add_personal_baselines_one_subject(
    df: pd.DataFrame,
    columns: Iterable[str],
    window: int = ROLLING_BASELINE_DAYS,
    min_periods: int = ROLLING_BASELINE_MIN_PERIODS,
) -> pd.DataFrame:
    """Add `<col>_baseline` and `<col>_dev` columns for each input column.

    `_baseline`: rolling mean over the past `window` days (right-aligned, so the
    baseline at day t excludes day t to avoid leakage).
    `_dev`: raw value minus baseline = "how far is today from your personal normal".

    Operates on one subject's chronologically-ordered DataFrame.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    for col in columns:
        if col not in df.columns:
            continue
        # `closed="left"` excludes the current day from its own baseline -- this
        # prevents trivial leakage where today's RHR is part of "your normal".
        baseline = (
            df[col]
            .rolling(window=window, min_periods=min_periods, closed="left")
            .mean()
        )
        df[f"{col}_baseline"] = baseline
        df[f"{col}_dev"] = df[col] - baseline
    return df


def _add_calendar_features_one_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week one-hots and a weekend indicator."""
    df = df.copy()
    dow = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
    for i, name in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]):
        df[f"dow_{name}"] = (dow == i).astype(int)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def _add_trailing_means_one_subject(
    df: pd.DataFrame,
    columns: Iterable[str],
    windows: Iterable[int] = (3, 7),
) -> pd.DataFrame:
    """Add `<col>_ma<window>` columns -- trailing means over each window length."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_ma{w}"] = (
                df[col].rolling(window=w, min_periods=max(1, w // 2), closed="left").mean()
            )
    return df


def _add_long_context_emas_one_subject(
    df: pd.DataFrame,
    columns: Iterable[str],
    alphas: Iterable[float] = (0.05, 0.15),
) -> pd.DataFrame:
    """Add exponentially-weighted moving averages over long horizons.

    EMAs let the model see beyond the 7-day input window. alpha=0.05 is roughly
    a 30-day half-life; alpha=0.15 is roughly a 14-day half-life.

    The EMA is shifted by one day so day-t's feature does not include day-t's
    own value (no leakage).
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    for col in columns:
        if col not in df.columns:
            continue
        for alpha in alphas:
            half_life_days = int(-1 / np.log(1 - alpha))
            df[f"{col}_ema{half_life_days}d"] = (
                df[col].ewm(alpha=alpha, adjust=False).mean().shift(1)
            )
    return df


def _add_strain_proxy_one_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Calorie-based strain proxy: `(active_calories) × (workout_minutes)`.

    Phone-friendly substitute for Banister TRIMP (which needs continuous HR).
    Active calories ≈ total calories minus the predictable basal portion. Since
    PMData only gives total calories, we use a coarse proxy: the very-active and
    moderately-active minutes weighted, scaled by total calories.
    """
    df = df.copy()
    intense_minutes = df["very_active_min"].fillna(0) + 0.5 * df["moderately_active_min"].fillna(0)
    workout_minutes = df["workout_minutes"].fillna(0)
    # Pick whichever is larger -- workout log if present, else the intensity-minutes proxy.
    df["strain_minutes"] = np.maximum(workout_minutes, intense_minutes)
    df["strain_proxy"] = df["strain_minutes"] * df["calories"].fillna(0) / 1000.0
    return df


def _add_time_since_workout_one_subject(df: pd.DataFrame, cap_days: int = 14) -> pd.DataFrame:
    """Days since the last logged workout. Capped to avoid huge values for inactive subjects."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    has_workout = (df["n_workouts"].fillna(0) > 0).values
    days_since = np.full(len(df), cap_days, dtype=float)
    counter = cap_days
    for i in range(len(df)):
        if has_workout[i]:
            counter = 0
        days_since[i] = counter
        counter = min(counter + 1, cap_days)
    df["days_since_workout"] = days_since
    return df


def _impute_short_gaps_one_subject(
    df: pd.DataFrame,
    columns: Iterable[str],
    limit: int = SHORT_GAP_FILL_LIMIT,
) -> pd.DataFrame:
    """Forward-fill missing values up to `limit` consecutive days. Longer gaps stay NaN."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].ffill(limit=limit)
    return df


# -----------------------------------------------------------------------------
# Cohort-level pipeline (operates on a multi-subject DataFrame)
# -----------------------------------------------------------------------------


def prepare_features(
    df: pd.DataFrame,
    personalize_columns: Iterable[str] = tuple(COLUMNS_TO_PERSONALIZE),
    trailing_mean_columns: Iterable[str] = ("rhr", "sleep_efficiency", "steps", "strain_proxy"),
    impute_columns: Iterable[str] = ("rhr", "sleep_efficiency", "steps", "calories"),
    long_context_ema_columns: Iterable[str] = ("rhr_dev", "sleep_efficiency_dev"),
) -> pd.DataFrame:
    """Apply the full feature pipeline. Operates per subject (groupby), then concatenates.

    Args:
        df: long-format DataFrame across one or more subjects, must have a `subject` column.
        personalize_columns: columns to baseline-center.
        trailing_mean_columns: columns for which to add 3- and 7-day trailing means.
        impute_columns: columns whose short missing gaps to forward-fill.
        long_context_ema_columns: columns for which to add 14-day and 30-day-horizon
            EMAs. These features let the model see beyond the 7-day input window.

    Returns:
        DataFrame with all engineered features, sorted by (subject, date).
    """
    if "subject" not in df.columns:
        raise ValueError("Input must contain a 'subject' column.")

    out_frames = []
    for subject, sub in df.groupby("subject", sort=False):
        sub = sub.sort_values("date").reset_index(drop=True)
        # 1. Strain proxy (uses raw values; do this before imputation so we don't strain-up gaps)
        sub = _add_strain_proxy_one_subject(sub)
        # 2. Impute short gaps in the columns we care about
        sub = _impute_short_gaps_one_subject(sub, impute_columns)
        # 3. Personal baselines (after imputation so a 1-day gap doesn't break the rolling window)
        sub = _add_personal_baselines_one_subject(sub, personalize_columns)
        # 4. Trailing means
        sub = _add_trailing_means_one_subject(sub, trailing_mean_columns)
        # 5. Long-context EMAs over deviation features (beyond the 7-day window)
        sub = _add_long_context_emas_one_subject(sub, long_context_ema_columns)
        # 6. Calendar features
        sub = _add_calendar_features_one_subject(sub)
        # 7. Time since last workout
        sub = _add_time_since_workout_one_subject(sub)
        out_frames.append(sub)

    return pd.concat(out_frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Windowing for sequence models (TCN/LSTM)
# -----------------------------------------------------------------------------


def make_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    window: int = INPUT_WINDOW_DAYS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build sliding-window (X, y) arrays from a per-subject feature DataFrame.

    For each valid window of `window` consecutive days where day `t` has all
    target values present and days `[t-window, t-1]` have all feature values
    present, emit:
        X[i] = features over days [t-window, t-1], shape (window, n_features)
        y[i] = targets at day t, shape (n_targets,)

    Also returns a metadata DataFrame with subject + target_date for each row,
    needed for LOSO/temporal split bookkeeping.
    """
    if "subject" not in df.columns:
        raise ValueError("Input must contain a 'subject' column.")

    Xs, ys, metas = [], [], []
    for subject, sub in df.groupby("subject", sort=False):
        sub = sub.sort_values("date").reset_index(drop=True)
        # Cast to float so np.isnan works (pandas may produce object dtype if
        # any source column is nullable-int or otherwise non-numeric).
        feats = sub[feature_columns].astype(float).values
        tgts = sub[target_columns].astype(float).values
        dates = sub["date"].values

        for t in range(window, len(sub)):
            X_win = feats[t - window : t]
            y_win = tgts[t]
            # pd.isna handles every dtype safely; np.isnan does not.
            if pd.isna(X_win).any() or pd.isna(y_win).any():
                continue
            Xs.append(X_win)
            ys.append(y_win)
            metas.append((subject, dates[t]))

    if not Xs:
        return (
            np.empty((0, window, len(feature_columns))),
            np.empty((0, len(target_columns))),
            pd.DataFrame(columns=["subject", "target_date"]),
        )

    X = np.stack(Xs)
    y = np.stack(ys)
    meta = pd.DataFrame(metas, columns=["subject", "target_date"])
    return X, y, meta


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    from pathlib import Path
    import sys

    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT / "src"))
    from conformal_recovery.data.loaders import load_all_subjects

    print("Loading all PMData subjects...")
    raw = load_all_subjects(ROOT / "data" / "pmdata")
    print(f"Raw: {len(raw)} subject-days across {raw['subject'].nunique()} subjects")

    print("\nApplying feature pipeline...")
    feats = prepare_features(raw)
    print(f"After feature pipeline: {len(feats)} rows, {feats.shape[1]} columns")

    # Pick feature columns that should exist after the pipeline
    rhr_features = [
        "rhr_dev",
        "rhr_ma3",
        "rhr_ma7",
        "sleep_efficiency_dev",
        "sleep_minutes_dev",
        "steps_dev",
        "calories_dev",
        "strain_proxy",
        "strain_minutes",
        "days_since_workout",
        "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
    ]
    targets = ["rhr_dev", "sleep_efficiency_dev"]

    print("\nWindowing features into 7-day input → next-day target...")
    X, y, meta = make_windows(feats, rhr_features, targets, window=7)
    print(f"  X shape: {X.shape}    (n_windows, 7 days, n_features)")
    print(f"  y shape: {y.shape}    (n_windows, n_targets)")
    print(f"  meta:    {len(meta)} rows -- subjects: {meta['subject'].nunique()}")
    print(f"  Per-subject window count:")
    print(meta.groupby("subject").size().to_string())
