# Conformal Strain–Recovery Forecaster

Predicting next-day resting heart rate and sleep efficiency with calibrated 90% conformal prediction bounds, using a TCN forecaster trained on Fitbit lifelogging data.

🔗 **Live demo:** _pending Day 17_
📄 **Paper (arXiv):** _pending Day 20_
💾 **Dataset:** PMData (n=16, ~5 months/subject) — [osf.io/3p7dq](https://osf.io/3p7dq/)

## Status

- [ ] Phase 0 — Setup
- [ ] Phase 1 — Foundations + EDA
- [ ] Phase 2 — Core ML (TCN + conformal)
- [ ] Phase 3 — Baselines + Activity model
- [ ] Phase 4 — Backend
- [ ] Phase 5 — Frontend + Demo
- [ ] Phase 6 — Paper + Submit

## Quickstart

```bash
# Create environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify PMData is in place
ls data/pmdata/p01/fitbit/
```

## Project structure

```
.
├── PLAN.md                          # canonical plan + decisions log
├── README.md                        # this file
├── requirements.txt
├── data/
│   ├── pmdata/                      # raw, gitignored
│   └── processed/                   # daily aggregates, gitignored
├── notebooks/                       # exploratory + EDA
├── src/conformal_recovery/
│   ├── data/                        # loaders, features, splits
│   ├── models/                      # TCN, baselines, activity
│   ├── conformal/                   # MAPIE wrappers + custom split conformal
│   └── eval/                        # coverage, width, MAE, calibration plots
└── tests/
```

## Citation

_TBD on arXiv submission._

## License

MIT
