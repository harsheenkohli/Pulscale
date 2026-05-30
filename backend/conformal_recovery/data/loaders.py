"""PMData loaders.

Each PMData subject's `fitbit/` folder holds JSON files for daily and
intraday physiological signals. This module loads them into a single
daily-frequency DataFrame keyed by date.

Daily output schema (per subject):
    date                pandas.Timestamp (no time component)
    rhr                 daily resting heart rate (bpm), with implausible
                        readings (< 35 or > 95) converted to NaN — these
                        are sensor errors that Fitbit encodes as 0
    sleep_minutes       total sleep time in minutes
    sleep_efficiency    (minutes asleep / time in bed) * 100
    steps               total daily steps
    distance_km         total daily distance in kilometers (raw values
                        are in 10⁻⁵ km units; we divide by 100,000)
    calories            total daily calories burned
    very_active_min     minutes in "very active" zone
    moderately_active_min
    lightly_active_min
    sedentary_min
    n_workouts          number of logged workout sessions
    workout_minutes     total minutes across workouts
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# Physiologically plausible RHR range. Values outside this are sensor errors
# (Fitbit encodes "no data" as 0, which would otherwise break personalization
# baselines). 35 bpm covers elite endurance athletes; 95 bpm covers tachycardic
# but still-ambulatory adults. PMData subjects fall well inside these.
RHR_MIN_BPM = 35
RHR_MAX_BPM = 95


def _read_json(path: Path) -> list:
    """Load a Fitbit JSON file. Returns [] if missing or empty."""
    if not path.exists():
        return []
    with path.open("r") as f:
        return json.load(f)


def _daily_rhr(fitbit_dir: Path) -> pd.DataFrame:
    """Load daily resting heart rate."""
    records = _read_json(fitbit_dir / "resting_heart_rate.json")
    rows = []
    for r in records:
        # PMData format: {'dateTime': '2019-11-01 00:00:00',
        #                 'value': {'date': '11/01/19', 'value': 53.74, 'error': ...}}
        date = pd.Timestamp(r["dateTime"]).normalize()
        rhr = r["value"]["value"]
        rows.append((date, rhr))
    return pd.DataFrame(rows, columns=["date", "rhr"])


def _daily_sleep(fitbit_dir: Path) -> pd.DataFrame:
    """Load sleep records and aggregate to one row per `dateOfSleep`."""
    records = _read_json(fitbit_dir / "sleep.json")
    rows = []
    for r in records:
        # Fitbit can log naps; we keep only the main sleep.
        if not r.get("mainSleep", False):
            continue
        date = pd.Timestamp(r["dateOfSleep"]).normalize()
        rows.append(
            {
                "date": date,
                "sleep_minutes": r["minutesAsleep"],
                "sleep_efficiency": r["efficiency"],
                "minutes_awake": r["minutesAwake"],
                "time_in_bed": r["timeInBed"],
            }
        )
    return pd.DataFrame(rows)


def _aggregate_minute_series(
    fitbit_dir: Path, filename: str, value_key: str = "value"
) -> pd.DataFrame:
    """Aggregate a minute-level series (steps, calories, distance) to daily totals."""
    records = _read_json(fitbit_dir / filename)
    if not records:
        return pd.DataFrame(columns=["date", "total"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["dateTime"]).dt.normalize()
    df["value"] = pd.to_numeric(df[value_key], errors="coerce")
    daily = df.groupby("date")["value"].sum().reset_index()
    daily.columns = ["date", "total"]
    return daily


def _daily_active_minutes(fitbit_dir: Path) -> pd.DataFrame:
    """Load each activity-minutes JSON and merge into one DataFrame."""
    levels = {
        "very_active_min": "very_active_minutes.json",
        "moderately_active_min": "moderately_active_minutes.json",
        "lightly_active_min": "lightly_active_minutes.json",
        "sedentary_min": "sedentary_minutes.json",
    }
    out = None
    for col, fname in levels.items():
        records = _read_json(fitbit_dir / fname)
        if not records:
            continue
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["dateTime"]).dt.normalize()
        df[col] = pd.to_numeric(df["value"], errors="coerce")
        daily = df.groupby("date")[col].sum().reset_index()
        out = daily if out is None else out.merge(daily, on="date", how="outer")
    return out if out is not None else pd.DataFrame(columns=["date"])


def _daily_workouts(fitbit_dir: Path) -> pd.DataFrame:
    """Load workout sessions and aggregate to count + total minutes per day."""
    records = _read_json(fitbit_dir / "exercise.json")
    if not records:
        return pd.DataFrame(columns=["date", "n_workouts", "workout_minutes"])
    rows = []
    for r in records:
        # Some records may have different shapes; defensively parse.
        start = r.get("startTime") or r.get("dateTime")
        duration_ms = r.get("duration", 0)
        if start is None:
            continue
        date = pd.Timestamp(start).normalize()
        rows.append((date, duration_ms / 60000.0))  # ms -> minutes
    df = pd.DataFrame(rows, columns=["date", "workout_minutes"])
    daily = (
        df.groupby("date")
        .agg(n_workouts=("workout_minutes", "size"), workout_minutes=("workout_minutes", "sum"))
        .reset_index()
    )
    return daily


def load_subject(subject_dir: Path | str) -> pd.DataFrame:
    """Load all daily-aggregated features for one PMData subject.

    Args:
        subject_dir: Path to e.g. data/pmdata/p01

    Returns:
        DataFrame with one row per day, sorted by date. Days with no
        data in any source are dropped.
    """
    subject_dir = Path(subject_dir)
    fitbit_dir = subject_dir / "fitbit"
    if not fitbit_dir.exists():
        raise FileNotFoundError(f"No fitbit/ folder in {subject_dir}")

    rhr = _daily_rhr(fitbit_dir)
    sleep = _daily_sleep(fitbit_dir)

    steps = _aggregate_minute_series(fitbit_dir, "steps.json").rename(
        columns={"total": "steps"}
    )
    distance = _aggregate_minute_series(fitbit_dir, "distance.json").rename(
        columns={"total": "distance_km"}
    )
    # Fitbit distance.json values are in 10^-5 km (effectively cm); convert to km.
    if not distance.empty:
        distance["distance_km"] = distance["distance_km"] / 100000.0
    calories = _aggregate_minute_series(fitbit_dir, "calories.json").rename(
        columns={"total": "calories"}
    )

    active = _daily_active_minutes(fitbit_dir)
    workouts = _daily_workouts(fitbit_dir)

    out = rhr
    for df in (sleep, steps, distance, calories, active, workouts):
        if df.empty:
            continue
        out = out.merge(df, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)
    out["subject"] = subject_dir.name

    # Convert physiologically-implausible RHR readings (Fitbit's 0-encoded
    # sensor errors) to NaN BEFORE any downstream baseline computation.
    out["rhr"] = out["rhr"].where(out["rhr"].between(RHR_MIN_BPM, RHR_MAX_BPM))

    # Some source JSONs (notably exercise.json) can produce duplicate rows on
    # days with multiple records at the same timestamp. Collapse to one row
    # per date, keeping the first (sleep / RHR are already daily-aggregated).
    out = out.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    return out


def load_all_subjects(pmdata_root: Path | str) -> pd.DataFrame:
    """Load every subject's daily DataFrame; concatenate."""
    pmdata_root = Path(pmdata_root)
    subject_dirs = sorted(d for d in pmdata_root.iterdir() if d.is_dir() and d.name.startswith("p"))
    frames = [load_subject(d) for d in subject_dirs]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    # Smoke test: load one subject and print summary.
    import sys

    root = Path(__file__).resolve().parents[3]
    p01 = root / "data" / "pmdata" / "p01"
    df = load_subject(p01)
    print(f"Loaded {len(df)} days for {p01.name}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print()
    print("Columns + non-null counts:")
    print(df.count())
    print()
    print("First 3 rows:")
    print(df.head(3).to_string())
