"""FastAPI backend for the Conformal Strain–Recovery Forecaster demo.

Endpoints:
  GET  /health                      -> liveness check
  GET  /sample-subjects             -> list of available demo subjects with metadata
  GET  /sample-subject/{sid}        -> full 60-day history for plotting
  POST /predict                     -> recovery forecast with conformal bands (sample subject)
  POST /recommend-workout           -> highest slider value that doesn't trigger warning
  POST /predict-from-stats          -> recovery forecast from manually-entered baseline + yesterday
  POST /upload-apple-health         -> recovery forecast from an Apple Health export.xml

Workflow:
  1. App startup loads `model_artifact.pt` once into memory
  2. Each /predict request swaps in the requested sample subject's last 14-day
     window, applies the optional workout-strain override (slider), runs the
     ensemble + conformal layer, returns the calibrated band

The ACI variant is not exposed via this API — ACI is a sequential method that
requires observed outcomes to update state. The demo uses Split Conformal,
which is well-defined for one-shot what-if queries. The paper reports both.
"""

from __future__ import annotations

import csv as csv_module
import logging
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from conformal_recovery.models.tcn import TCN

logger = logging.getLogger("uvicorn")

# -----------------------------------------------------------------------------
# Artifact loading (once at startup)
# -----------------------------------------------------------------------------

ARTIFACT_PATH = Path(__file__).parent / "model_artifact.pt"

class _AppState:
    artifact: dict | None = None
    models: list[TCN] = []
    feature_cols: list[str] = []
    target_cols: list[str] = []
    samples: dict = {}

state = _AppState()


def load_artifact():
    """Load the model artifact and instantiate the ensemble."""
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {ARTIFACT_PATH}. "
            "Run `python -m conformal_recovery.serve.build_artifact` first."
        )
    artifact = torch.load(ARTIFACT_PATH, map_location="cpu", weights_only=False)
    cfg = artifact["config"]

    models = []
    n_features = len(cfg["feature_cols"])
    for sd in artifact["state_dicts"]:
        m = TCN(
            n_features=n_features,
            n_targets=len(cfg["target_cols"]),
            hidden=cfg["hidden"],
            dilations=tuple(cfg["dilations"]),
            dropout=cfg["dropout"],
        )
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    state.artifact = artifact
    state.models = models
    state.feature_cols = cfg["feature_cols"]
    state.target_cols = cfg["target_cols"]
    state.samples = artifact["samples"]
    logger.info(f"Loaded artifact: {len(models)} ensemble members, "
                f"{len(state.samples)} sample subjects, "
                f"q_hat={artifact['split_conformal_q_hat']}")


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title="Conformal Recovery Forecaster", version="0.1.0")

# Allow the Vercel frontend to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten before final deploy
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    load_artifact()


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    n_models: int
    samples: list[str]


class SubjectMeta(BaseModel):
    subject_id: str
    n_days: int
    rhr_baseline: float
    sleep_baseline: float


class SubjectHistory(BaseModel):
    subject_id: str
    dates: list[str]
    rhr: list[float | None]
    rhr_baseline: list[float | None]
    sleep_efficiency: list[float | None]
    sleep_efficiency_baseline: list[float | None]
    steps: list[float | None]
    strain_proxy: list[float | None]


class PredictRequest(BaseModel):
    subject_id: str
    # Slider value 1-10. Replaces the strain feature on the LAST day of the
    # input window so the model "sees" tomorrow's planned strain.
    planned_strain_slider: float = Field(default=5.0, ge=1.0, le=10.0)


class PredictResponse(BaseModel):
    subject_id: str
    target_date: str                      # the date we predict for
    rhr: dict                              # {point, lower, upper, baseline}
    sleep_efficiency: dict
    warning_level: Literal["green", "yellow", "red"]
    warning_message: str


class RecommendRequest(BaseModel):
    subject_id: str


class RecommendResponse(BaseModel):
    subject_id: str
    recommended_max_slider: float
    sweep: list[dict]                      # all 10 (slider, warning_level) pairs


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------


def _standardize(window_features: np.ndarray) -> np.ndarray:
    """Apply the artifact's standardization (no leakage; uses train stats)."""
    return (window_features - state.artifact["x_mean"]) / state.artifact["x_std"]


def _ensemble_predict_residual(window_features_std: np.ndarray) -> np.ndarray:
    """Run the ensemble. Returns mean over models in standardized residual space."""
    x = torch.from_numpy(window_features_std).float().unsqueeze(0)  # (1, T, F)
    preds = []
    with torch.no_grad():
        for m in state.models:
            preds.append(m(x).numpy()[0])
    return np.mean(preds, axis=0)


def _slider_to_strain(slider: float, sample_strain_history: list[float]) -> float:
    """Map a 1-10 slider to a TRIMP-equivalent strain value, calibrated to user.

    slider=1 -> rest day (very low strain); slider=10 -> max workout (90th
    percentile of subject's history). Linear in between.
    """
    arr = np.asarray([s for s in sample_strain_history if s is not None and s > 0])
    if len(arr) == 0:
        max_strain = 100.0  # fallback if subject never logs strain
    else:
        max_strain = float(np.percentile(arr, 90))
    return (slider - 1) / 9.0 * max_strain


def _compute_warning(
    rhr_baseline: float, rhr_pred: float, rhr_lower: float, rhr_upper: float,
    sleep_baseline: float, sleep_pred: float, sleep_lower: float, sleep_upper: float,
) -> tuple[str, str]:
    """Two-tier overtraining warning logic from the locked spec."""
    rhr_red = rhr_lower > rhr_baseline + 5
    sleep_red = sleep_upper < sleep_baseline - 5
    rhr_yellow = rhr_pred > rhr_baseline + 3
    sleep_yellow = sleep_pred < sleep_baseline - 3

    if rhr_red or sleep_red:
        return "red", "Strong signs you'll be under-recovered tomorrow. Consider rest or light activity."
    if rhr_yellow or sleep_yellow:
        return "yellow", "Caution — your body may be under-recovered tomorrow."
    return "green", "Recovery looks on track."


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=state.artifact["version"] if state.artifact else "unknown",
        n_models=len(state.models),
        samples=list(state.samples.keys()),
    )


@app.get("/sample-subjects")
async def sample_subjects() -> list[SubjectMeta]:
    out = []
    for sid, s in state.samples.items():
        rhr_baseline = float(np.nanmean([v for v in s["rhr_baseline"] if v is not None]))
        sleep_baseline = float(np.nanmean([v for v in s["sleep_efficiency_baseline"] if v is not None]))
        out.append(SubjectMeta(
            subject_id=sid,
            n_days=len(s["dates"]),
            rhr_baseline=round(rhr_baseline, 1),
            sleep_baseline=round(sleep_baseline, 1),
        ))
    return out


@app.get("/sample-subject/{subject_id}", response_model=SubjectHistory)
async def sample_subject(subject_id: str):
    if subject_id not in state.samples:
        raise HTTPException(status_code=404, detail=f"Unknown subject {subject_id}")
    s = state.samples[subject_id]
    return SubjectHistory(
        subject_id=subject_id,
        dates=s["dates"],
        rhr=s["rhr"],
        rhr_baseline=s["rhr_baseline"],
        sleep_efficiency=s["sleep_efficiency"],
        sleep_efficiency_baseline=s["sleep_efficiency_baseline"],
        steps=s["steps"],
        strain_proxy=s["strain_proxy"],
    )


def _predict_with_slider(subject_id: str, slider: float) -> dict:
    """Core prediction logic — used by both /predict and /recommend-workout."""
    if subject_id not in state.samples:
        raise HTTPException(status_code=404, detail=f"Unknown subject {subject_id}")
    s = state.samples[subject_id]

    # Last 14 days of features, last day's strain replaced by the slider value
    window = np.array(s["last_window_features"], dtype=np.float32).copy()
    feat_idx = state.feature_cols.index("strain_proxy")
    window[-1, feat_idx] = _slider_to_strain(slider, s["strain_proxy"])

    # Standardize, run ensemble, denormalize, add baseline back
    win_s = _standardize(window)
    pred_resid_std = _ensemble_predict_residual(win_s)
    pred_resid = (
        pred_resid_std * state.artifact["y_std_residual"][0]
        + state.artifact["y_mean_residual"][0]
    )
    last_day_targets = window[-1, :2]                  # last_day rhr_dev, sleep_eff_dev
    pred_dev = pred_resid + last_day_targets             # absolute deviation prediction

    # Add the user's baseline back to get bpm / pp
    rhr_baseline = float(np.nanmean([v for v in s["rhr_baseline"] if v is not None]))
    sleep_baseline = float(np.nanmean([v for v in s["sleep_efficiency_baseline"] if v is not None]))
    rhr_pred = float(pred_dev[0] + rhr_baseline)
    sleep_pred = float(pred_dev[1] + sleep_baseline)

    # Apply split conformal half-widths
    q = state.artifact["split_conformal_q_hat"]
    rhr_lower, rhr_upper = rhr_pred - float(q[0]), rhr_pred + float(q[0])
    sleep_lower, sleep_upper = sleep_pred - float(q[1]), sleep_pred + float(q[1])

    level, msg = _compute_warning(
        rhr_baseline, rhr_pred, rhr_lower, rhr_upper,
        sleep_baseline, sleep_pred, sleep_lower, sleep_upper,
    )

    return {
        "rhr": {
            "point": round(rhr_pred, 2),
            "lower": round(rhr_lower, 2),
            "upper": round(rhr_upper, 2),
            "baseline": round(rhr_baseline, 2),
        },
        "sleep_efficiency": {
            "point": round(sleep_pred, 2),
            "lower": round(sleep_lower, 2),
            "upper": round(sleep_upper, 2),
            "baseline": round(sleep_baseline, 2),
        },
        "warning_level": level,
        "warning_message": msg,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest):
    s = state.samples.get(req.subject_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Unknown subject {req.subject_id}")
    result = _predict_with_slider(req.subject_id, req.planned_strain_slider)

    # Target date = last sample date + 1 day
    last_date = s["dates"][-1]
    target_date = (np.datetime64(last_date) + 1).astype(str)

    return PredictResponse(
        subject_id=req.subject_id,
        target_date=str(target_date),
        rhr=result["rhr"],
        sleep_efficiency=result["sleep_efficiency"],
        warning_level=result["warning_level"],
        warning_message=result["warning_message"],
    )


@app.post("/recommend-workout", response_model=RecommendResponse)
async def recommend_endpoint(req: RecommendRequest):
    """Sweep slider 1-10, return the highest non-warning value."""
    if req.subject_id not in state.samples:
        raise HTTPException(status_code=404, detail=f"Unknown subject {req.subject_id}")

    sweep = []
    recommended = 1.0
    for slider in range(1, 11):
        result = _predict_with_slider(req.subject_id, float(slider))
        sweep.append({
            "slider": slider,
            "warning_level": result["warning_level"],
            "rhr_point": result["rhr"]["point"],
            "rhr_lower": result["rhr"]["lower"],
            "rhr_upper": result["rhr"]["upper"],
        })
        # The highest slider with green status is the recommendation
        if result["warning_level"] == "green":
            recommended = float(slider)

    return RecommendResponse(
        subject_id=req.subject_id,
        recommended_max_slider=int(recommended),
        sweep=sweep,
    )


# -----------------------------------------------------------------------------
# Manual entry — synthesize a 14-day window from baseline + yesterday values
# -----------------------------------------------------------------------------


class ManualEntryRequest(BaseModel):
    """Inputs for users without wearable data. We synthesize a 14-day window
    from a small number of fields, then run the same prediction pipeline.

    All fields are optional except rhr_baseline (which we need to anchor the
    prediction). Missing values fall back to cohort means.
    """
    rhr_baseline: float = Field(..., ge=35, le=95, description="Your typical resting HR over the last 30 days")
    yesterday_rhr: float | None = Field(None, ge=35, le=95)
    sleep_baseline: float = Field(default=92, ge=50, le=100, description="Your typical sleep efficiency %")
    yesterday_sleep: float | None = Field(None, ge=50, le=100)
    typical_steps: float = Field(default=10000, ge=0, le=50000)
    yesterday_steps: float | None = Field(None, ge=0, le=50000)
    planned_strain_slider: float = Field(default=5.0, ge=1.0, le=10.0)


class ManualEntryResponse(BaseModel):
    rhr: dict
    sleep_efficiency: dict
    warning_level: Literal["green", "yellow", "red"]
    warning_message: str
    note: str = "Prediction based on synthesized history; bands are population-calibrated."


def _synthesize_window(req: ManualEntryRequest) -> np.ndarray:
    """Build a fake 14-day feature window from the manual inputs.

    Strategy: fill 13 days with values matching the user's baseline (zero
    deviation), and the 14th (last) day with `yesterday_*` values. Strain on
    the last day is set by the slider via `_slider_to_strain`.
    """
    n_features = len(state.feature_cols)
    window = np.zeros((14, n_features), dtype=np.float32)

    yest_rhr = req.yesterday_rhr if req.yesterday_rhr is not None else req.rhr_baseline
    yest_sleep = req.yesterday_sleep if req.yesterday_sleep is not None else req.sleep_baseline
    yest_steps = req.yesterday_steps if req.yesterday_steps is not None else req.typical_steps

    # Deviation-from-baseline features. All 0 except today (last row).
    rhr_dev_today = yest_rhr - req.rhr_baseline
    sleep_dev_today = yest_sleep - req.sleep_baseline
    steps_dev_today = yest_steps - req.typical_steps

    # Targets first (positions 0, 1)
    window[-1, state.feature_cols.index("rhr_dev")] = rhr_dev_today
    window[-1, state.feature_cols.index("sleep_efficiency_dev")] = sleep_dev_today
    window[-1, state.feature_cols.index("steps_dev")] = steps_dev_today

    # Trailing means: assume stable -> 0 deviation
    # rhr_ma3, rhr_ma7 stay 0
    # sleep_minutes_dev, calories_dev stay 0

    # Strain: scale slider 1-10 to a synthetic value (we don't know the user's
    # true max, so use a conservative 200 strain-units/day at slider=10).
    window[-1, state.feature_cols.index("strain_proxy")] = (req.planned_strain_slider - 1) / 9.0 * 200
    window[-1, state.feature_cols.index("strain_minutes")] = (req.planned_strain_slider - 1) / 9.0 * 60

    # Day-of-week: assume tomorrow is the day after today's date; encode that.
    # For manual entry we simplify: assume Wednesday.
    window[-1, state.feature_cols.index("dow_wed")] = 1

    # Days since last workout: assume 1 (yesterday was the last)
    window[-1, state.feature_cols.index("days_since_workout")] = 1

    return window


def _predict_from_window(window: np.ndarray, rhr_baseline: float, sleep_baseline: float) -> dict:
    """Same forward pass as _predict_with_slider but for a freely-built window."""
    win_s = _standardize(window)
    pred_resid_std = _ensemble_predict_residual(win_s)
    pred_resid = pred_resid_std * state.artifact["y_std_residual"][0] + state.artifact["y_mean_residual"][0]
    last_day_targets = window[-1, :2]
    pred_dev = pred_resid + last_day_targets

    rhr_pred = float(pred_dev[0] + rhr_baseline)
    sleep_pred = float(pred_dev[1] + sleep_baseline)
    q = state.artifact["split_conformal_q_hat"]
    rhr_lower, rhr_upper = rhr_pred - float(q[0]), rhr_pred + float(q[0])
    sleep_lower, sleep_upper = sleep_pred - float(q[1]), sleep_pred + float(q[1])

    level, msg = _compute_warning(
        rhr_baseline, rhr_pred, rhr_lower, rhr_upper,
        sleep_baseline, sleep_pred, sleep_lower, sleep_upper,
    )
    return {
        "rhr": {"point": round(rhr_pred, 2), "lower": round(rhr_lower, 2),
                 "upper": round(rhr_upper, 2), "baseline": round(rhr_baseline, 2)},
        "sleep_efficiency": {"point": round(sleep_pred, 2), "lower": round(sleep_lower, 2),
                              "upper": round(sleep_upper, 2), "baseline": round(sleep_baseline, 2)},
        "warning_level": level,
        "warning_message": msg,
    }


@app.post("/predict-from-stats", response_model=ManualEntryResponse)
async def predict_from_stats(req: ManualEntryRequest):
    window = _synthesize_window(req)
    out = _predict_from_window(window, req.rhr_baseline, req.sleep_baseline)
    return ManualEntryResponse(
        rhr=out["rhr"],
        sleep_efficiency=out["sleep_efficiency"],
        warning_level=out["warning_level"],
        warning_message=out["warning_message"],
    )


@app.post("/recommend-from-stats")
async def recommend_from_stats(req: ManualEntryRequest):
    """Sweep slider 1-10 for manual-entry users; return highest green slider."""
    sweep = []
    recommended_slider = 1.0
    for slider_val in range(1, 11):
        req_copy = req.model_copy(update={"planned_strain_slider": float(slider_val)})
        window = _synthesize_window(req_copy)
        r = _predict_from_window(window, req.rhr_baseline, req.sleep_baseline)
        sweep.append({
            "slider": slider_val,
            "warning_level": r["warning_level"],
            "rhr_point": r["rhr"]["point"],
            "rhr_lower": r["rhr"]["lower"],
            "rhr_upper": r["rhr"]["upper"],
        })
        if r["warning_level"] == "green":
            recommended_slider = float(slider_val)
    return {
        "subject_id": "manual",
        "recommended_max_slider": int(recommended_slider),
        "sweep": sweep,
    }


# -----------------------------------------------------------------------------
# Apple Health upload
# -----------------------------------------------------------------------------


class AppleHealthResponse(BaseModel):
    rhr: dict | None
    sleep_efficiency: dict | None
    warning_level: Literal["green", "yellow", "red"] | None
    warning_message: str | None
    n_rhr_records: int
    n_sleep_records: int
    has_apple_watch_data: bool
    note: str
    history: dict | None = None
    sweep: list[dict] | None = None
    recommended_max_slider: float | None = None


def _parse_apple_health_xml(xml_bytes: bytes, days_back: int = 60) -> dict:
    """Stream-parse Apple Health export.xml. Extracts the last `days_back` days
    of daily-aggregated RHR, sleep efficiency, and steps.

    Returns a dict suitable for use by the prediction pipeline.
    """
    cutoff = datetime.now() - timedelta(days=days_back)
    rhr_per_day: dict[date, list[float]] = {}
    sleep_records: list[tuple[datetime, datetime, str]] = []
    steps_per_day: dict[date, float] = {}

    # iterparse so we don't hold the whole tree in memory
    for event, elem in ET.iterparse(BytesIO(xml_bytes), events=("end",)):
        if elem.tag != "Record":
            continue
        rtype = elem.attrib.get("type", "")
        try:
            start = datetime.strptime(
                elem.attrib["startDate"][:19], "%Y-%m-%d %H:%M:%S"
            )
        except (KeyError, ValueError):
            elem.clear()
            continue
        if start < cutoff:
            elem.clear()
            continue

        try:
            value = elem.attrib.get("value", "")
            if rtype == "HKQuantityTypeIdentifierRestingHeartRate":
                rhr_per_day.setdefault(start.date(), []).append(float(value))
            elif rtype == "HKQuantityTypeIdentifierStepCount":
                steps_per_day[start.date()] = steps_per_day.get(start.date(), 0.0) + float(value)
            elif rtype == "HKCategoryTypeIdentifierSleepAnalysis":
                end_str = elem.attrib.get("endDate", "")[:19]
                end = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
                sleep_records.append((start, end, value))
        except (ValueError, KeyError):
            pass
        finally:
            elem.clear()

    # Aggregate sleep into per-night efficiency
    sleep_per_night: dict[date, dict[str, float]] = {}
    for start, end, val in sleep_records:
        night_date = end.date()  # the day they woke up
        bucket = sleep_per_night.setdefault(night_date, {"in_bed": 0.0, "asleep": 0.0})
        minutes = (end - start).total_seconds() / 60.0
        if "InBed" in val:
            bucket["in_bed"] += minutes
        elif "Asleep" in val:
            bucket["asleep"] += minutes
            bucket["in_bed"] += minutes  # asleep counts toward in-bed too if the InBed record is missing

    # Daily RHR = mean of recordings that day
    daily_rhr = {d: float(np.mean(vs)) for d, vs in rhr_per_day.items() if vs}

    # Daily sleep efficiency
    daily_sleep_eff: dict[date, float] = {}
    for d, b in sleep_per_night.items():
        if b["in_bed"] > 0:
            daily_sleep_eff[d] = (b["asleep"] / b["in_bed"]) * 100.0

    return {
        "rhr": daily_rhr,
        "sleep_efficiency": daily_sleep_eff,
        "steps": steps_per_day,
    }


@app.post("/upload-apple-health", response_model=AppleHealthResponse)
async def upload_apple_health(file: UploadFile = File(...), planned_strain_slider: float = 5.0):
    """Accept an `export.xml` (or zip containing it) from Apple Health.

    iPhone-only exports lack RHR + sleep stages → we return a friendly note
    rather than a misleading prediction. Apple-Watch exports run the full pipeline.
    """
    raw = await file.read()
    # Handle a zipped export
    xml_bytes = raw
    if raw[:2] == b"PK":
        try:
            with ZipFile(BytesIO(raw)) as zf:
                names = [n for n in zf.namelist() if n.endswith("export.xml")]
                if not names:
                    raise HTTPException(status_code=400, detail="ZIP contained no export.xml")
                xml_bytes = zf.read(names[0])
        except BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")

    parsed = _parse_apple_health_xml(xml_bytes, days_back=60)
    n_rhr = len(parsed["rhr"])
    n_sleep = len(parsed["sleep_efficiency"])

    if n_rhr < 7:
        return AppleHealthResponse(
            rhr=None,
            sleep_efficiency=None,
            warning_level=None,
            warning_message=None,
            n_rhr_records=n_rhr,
            n_sleep_records=n_sleep,
            has_apple_watch_data=False,
            note=(
                f"Only {n_rhr} resting-HR days found in your export. The recovery "
                f"forecaster needs an Apple Watch (or similar) for continuous heart-rate "
                f"and sleep tracking. iPhone-only exports lack this data. Use the "
                f"manual-entry form below or pick a sample subject to explore the demo."
            ),
        )

    # Build a 14-day window from the most recent 14 days that have RHR
    rhr_dates = sorted(parsed["rhr"].keys())[-14:]
    if len(rhr_dates) < 14:
        return AppleHealthResponse(
            rhr=None, sleep_efficiency=None, warning_level=None, warning_message=None,
            n_rhr_records=n_rhr, n_sleep_records=n_sleep, has_apple_watch_data=True,
            note=f"Found only {len(rhr_dates)} consecutive days of RHR data. Need 14+. "
                  f"The forecaster works best with at least 30 days of Apple Watch data.",
        )

    # Compute the user's personal baselines from the full 60-day history
    rhr_baseline = float(np.nanmean(list(parsed["rhr"].values())))
    sleep_baseline = float(np.nanmean(list(parsed["sleep_efficiency"].values()))) if parsed["sleep_efficiency"] else 92.0

    # Build the 14-day feature window
    n_features = len(state.feature_cols)
    window = np.zeros((14, n_features), dtype=np.float32)
    rhr_dev_idx = state.feature_cols.index("rhr_dev")
    sleep_dev_idx = state.feature_cols.index("sleep_efficiency_dev")
    steps_dev_idx = state.feature_cols.index("steps_dev")
    strain_idx = state.feature_cols.index("strain_proxy")
    typical_steps = float(np.mean(list(parsed["steps"].values()))) if parsed["steps"] else 10000.0

    for i, d in enumerate(rhr_dates):
        window[i, rhr_dev_idx] = parsed["rhr"][d] - rhr_baseline
        if d in parsed["sleep_efficiency"]:
            window[i, sleep_dev_idx] = parsed["sleep_efficiency"][d] - sleep_baseline
        if d in parsed["steps"]:
            window[i, steps_dev_idx] = parsed["steps"][d] - typical_steps

    # Apply slider to last day's strain
    window[-1, strain_idx] = (planned_strain_slider - 1) / 9.0 * 200

    out = _predict_from_window(window, rhr_baseline, sleep_baseline)

    # Sweep slider 1-10 to build recommendation
    sweep = []
    recommended_slider = 1.0
    for slider_val in range(1, 11):
        w = window.copy()
        w[-1, strain_idx] = (slider_val - 1) / 9.0 * 200
        r = _predict_from_window(w, rhr_baseline, sleep_baseline)
        sweep.append({
            "slider": slider_val,
            "warning_level": r["warning_level"],
            "rhr_point": r["rhr"]["point"],
            "rhr_lower": r["rhr"]["lower"],
            "rhr_upper": r["rhr"]["upper"],
        })
        if r["warning_level"] == "green":
            recommended_slider = float(slider_val)

    # Build history for chart display (up to 60 days)
    all_dates = sorted(parsed["rhr"].keys())
    history_dates = [d.isoformat() for d in all_dates]
    history_rhr = [parsed["rhr"].get(d) for d in all_dates]
    history_sleep = [parsed["sleep_efficiency"].get(d) for d in all_dates]
    history_rhr_baseline = [rhr_baseline] * len(all_dates)
    history_sleep_baseline = [sleep_baseline] * len(all_dates)

    return AppleHealthResponse(
        rhr=out["rhr"],
        sleep_efficiency=out["sleep_efficiency"],
        warning_level=out["warning_level"],
        warning_message=out["warning_message"],
        n_rhr_records=n_rhr,
        n_sleep_records=n_sleep,
        has_apple_watch_data=True,
        note=f"Parsed {n_rhr} RHR days and {n_sleep} sleep records from your export.",
        sweep=sweep,
        recommended_max_slider=int(recommended_slider),
        history={
            "subject_id": "upload",
            "dates": history_dates,
            "rhr": history_rhr,
            "rhr_baseline": history_rhr_baseline,
            "sleep_efficiency": history_sleep,
            "sleep_efficiency_baseline": history_sleep_baseline,
            "steps": [parsed["steps"].get(d, 0) for d in all_dates],
            "strain_proxy": [0.0] * len(all_dates),
        },
    )


# -----------------------------------------------------------------------------
# Google Fit upload
# -----------------------------------------------------------------------------


def _parse_google_fit_csv_steps(csv_text: str) -> float:
    """Sum the 'Step count' column across all 15-min rows in a daily CSV."""
    total = 0.0
    for row in csv_module.DictReader(StringIO(csv_text)):
        val = row.get("Step count", "").strip()
        if val:
            try:
                total += float(val)
            except ValueError:
                pass
    return total


@app.post("/upload-google-fit")
async def upload_google_fit(
    file: UploadFile = File(...),
    rhr_baseline: float = 60.0,
    yesterday_rhr: float = 62.0,
    planned_strain_slider: float = 5.0,
):
    """Accept a Google Fit Takeout zip (or single daily CSV). Uses the caller-
    supplied RHR values together with real step history to run the recovery
    forecast pipeline.
    """
    raw = await file.read()
    steps_per_day: dict[date, float] = {}

    if raw[:2] == b"PK":
        try:
            with ZipFile(BytesIO(raw)) as zf:
                for name in zf.namelist():
                    basename = name.split("/")[-1]
                    if not basename.endswith(".csv"):
                        continue
                    day_str = basename[:-4]
                    try:
                        day = date.fromisoformat(day_str)
                    except ValueError:
                        continue
                    csv_text = zf.read(name).decode("utf-8", errors="replace")
                    steps_per_day[day] = _parse_google_fit_csv_steps(csv_text)
        except BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")
    else:
        # Single CSV — treat as yesterday
        csv_text = raw.decode("utf-8", errors="replace")
        steps_per_day[date.today() - timedelta(days=1)] = _parse_google_fit_csv_steps(csv_text)

    if not steps_per_day:
        raise HTTPException(status_code=400, detail="No daily step CSVs found in the file.")

    # Use last 60 days for history display
    all_dates = sorted(steps_per_day.keys())[-60:]
    typical_steps = float(np.mean([steps_per_day[d] for d in all_dates]))

    # 14-day prediction window
    last_14 = all_dates[-14:] if len(all_dates) >= 14 else all_dates
    sleep_baseline = 92.0

    n_features = len(state.feature_cols)
    window = np.zeros((14, n_features), dtype=np.float32)
    rhr_dev_idx = state.feature_cols.index("rhr_dev")
    steps_dev_idx = state.feature_cols.index("steps_dev")
    strain_idx = state.feature_cols.index("strain_proxy")

    for i, d in enumerate(last_14):
        window[i, steps_dev_idx] = steps_per_day.get(d, typical_steps) - typical_steps

    window[-1, rhr_dev_idx] = yesterday_rhr - rhr_baseline
    window[-1, strain_idx] = (planned_strain_slider - 1) / 9.0 * 200

    out = _predict_from_window(window, rhr_baseline, sleep_baseline)

    sweep = []
    recommended_slider = 1.0
    for slider_val in range(1, 11):
        w = window.copy()
        w[-1, strain_idx] = (slider_val - 1) / 9.0 * 200
        r = _predict_from_window(w, rhr_baseline, sleep_baseline)
        sweep.append({
            "slider": slider_val,
            "warning_level": r["warning_level"],
            "rhr_point": r["rhr"]["point"],
            "rhr_lower": r["rhr"]["lower"],
            "rhr_upper": r["rhr"]["upper"],
        })
        if r["warning_level"] == "green":
            recommended_slider = float(slider_val)

    history_dates = [d.isoformat() for d in all_dates]
    return {
        "rhr": out["rhr"],
        "sleep_efficiency": out["sleep_efficiency"],
        "warning_level": out["warning_level"],
        "warning_message": out["warning_message"],
        "note": f"Parsed {len(steps_per_day)} days of step data from Google Fit. RHR from your entries.",
        "sweep": sweep,
        "recommended_max_slider": int(recommended_slider),
        "history": {
            "subject_id": "googlefit",
            "dates": history_dates,
            "rhr": [rhr_baseline] * len(all_dates),
            "rhr_baseline": [rhr_baseline] * len(all_dates),
            "sleep_efficiency": [sleep_baseline] * len(all_dates),
            "sleep_efficiency_baseline": [sleep_baseline] * len(all_dates),
            "steps": [steps_per_day.get(d, 0) for d in all_dates],
            "strain_proxy": [0.0] * len(all_dates),
        },
    }
