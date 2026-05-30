# Conformal Strain–Recovery Forecaster — Project Plan

Predicting next-day resting heart rate and sleep efficiency with calibrated uncertainty bounds, using split + adaptive conformal prediction over a TCN forecaster trained on Fitbit lifelogging data. Live demo + arXiv paper + IEEE conference submission.

---

## Problem statement

### The everyday decision we help with

Smartwatch users face this question every morning:
> *"My fitness app says I slept okay, but I feel slightly off. If I do a heavy workout today, am I going to crash tomorrow, or is my body actually fine?"*

Existing options are unsatisfying:
1. **Trust your gut** — often wrong, especially for high-performers who push through fatigue
2. **Trust your watch's "recovery score"** — Whoop / Garmin / Oura give a single black-box number with no honest uncertainty

We're building the **open, honest, uncertainty-aware version** of that recovery score.

### Technical problem (precise)

- **Input:** past 7+ days of wearable data (RHR, sleep efficiency, daily steps, workouts, etc.) — Watch-class for recovery model, phone-only for activity model
- **Output:** tomorrow's RHR and sleep efficiency, **with rigorous 90% confidence bounds**
- **Interactive lever:** "tomorrow's planned workout strain" slider (1-10) — drag → predictions update live
- **Decision rule:** two-tier overtraining warning (yellow on point estimate, red on conformal bound)

### What we're NOT solving (out of scope)
- Not detecting illness (Mishra 2020 did COVID; we don't)
- Not predicting injury (different downstream problem)
- Not measuring fitness improvement over time
- Not personalizing to elite athletes — baseline is healthy adults

---

## Novelty & positioning vs. commercial systems

### The reviewer's question
> *"Whoop, Garmin, Oura already produce recovery scores. Why does this paper exist?"*

This is the question we must pre-empt in the paper's intro. Our defense:

### Where Whoop / Garmin / Oura genuinely fall short

| What they do | What they don't do | Why that matters |
|---|---|---|
| Output a single "Recovery Score" 0-100 | Output **calibrated uncertainty bounds** | A single number can't tell you when the model is unsure — safety concern |
| Closed-source, proprietary algorithms | Allow **reproducibility, auditing, retraining** | Academic / clinical users can't validate, extend, or modify them |
| Train on internal user base | Have **published cross-population validation** | Bellenger et al. 2021 and others found Whoop's HRV / recovery metrics inconsistent across users |
| Conflate signals into one number | **Decompose** into interpretable components (RHR vs sleep) | "Score = 67" doesn't tell you *why* you're under-recovered |

### What we ARE NOT claiming (avoid overreach)
- ❌ Conformal prediction is new (Vovk 2005)
- ❌ Recovery forecasting from wearables is new
- ❌ Personalized baselines is new (Mishra 2020)
- ❌ We beat Whoop on accuracy (their data is locked, can't compare)

### What we ARE claiming (paper-defensible)
- ✅ **First open-source, end-to-end, conformally-calibrated recovery forecaster** evaluated with rigorous Leave-One-Subject-Out + temporal hold-out splits on PMData (n=16, 5 months/subject)
- ✅ **Empirical demonstration that 90% conformal bands achieve nominal 90% coverage** on held-out wearable data
- ✅ **Comparative study:** conformal vs bootstrap CIs vs quantile regression vs naive — which uncertainty method actually works on noisy physiological time-series
- ✅ **Reproducible artifact:** code, dataset loaders, evaluation harness — so the next researcher extends instead of re-implements

### One-line contribution
> *"We provide the first openly evaluable, conformally-calibrated recovery forecaster — filling the gap between closed commercial algorithms (Whoop, Garmin, Oura) and academic point-prediction models."*

The novelty is **integration + rigorous evaluation + openness**, not any individual component. This is the right framing for **applied IEEE venues** (J-BHI, EMBC, HEALTHCOM) — they reward useful, well-evaluated systems. It would NOT pass NeurIPS / ICML, which want method novelty — those are not our target venues.

### Pre-written reviewer rebuttal (drop into paper §1)
> *"R1 raises the valid concern that commercial recovery scores (Whoop, Garmin, Oura) already address this problem. We agree these systems exist, but they (i) provide no calibrated uncertainty estimates, (ii) are closed-source with limited reproducibility, and (iii) have shown inconsistent validation across populations [Bellenger 2021]. Our contribution is not a new method, but an open, conformally-calibrated, reproducibly-evaluable system that fills the gap between commercial black boxes and academic point-prediction models."*

---

## Open items — to discuss before/during build

These are *not yet locked*. We need to talk through each before or during the relevant phase.

✅ **Prediction targets** (locked 2026-05-28): next-day resting HR + next-day sleep efficiency. Skipped: HRV (inconsistent in both PMData and Apple Health), step count (behavior, not recovery), mood (not in native Apple Health → would break upload feature).   
✅ **Strain definition** (locked 2026-05-28). Training feature per past day: HR × duration (Banister TRIMP) for Watch users / calories × duration for phone users. Demo slider: 1-10 with labels (1=rest day, 3=light walk, 5=moderate run, 7=hard run, 10=race effort), mapped behind the scenes to a strain estimate calibrated to user's historical workouts.    
✅ **Personalization** (locked 2026-05-28). One global model trained on all PMData subjects. At inference, user features centered on their own rolling-30-day baseline (subtract personal mean) so the model predicts deviation-from-normal, not absolute values. Baseline added back when displaying. Same approach for both recovery and activity models.   
✅ **Interactivity scope** (locked 2026-05-28). One slider only: tomorrow's planned workout strain (1-10 with labels). Optional "Rest Day" preset button that snaps slider to 1. No caffeine / sleep / stress sliders — those features aren't in our model and would be theater. **Workout recommendation:** backend additionally computes the *highest* slider value that doesn't trigger an overtraining warning, exposed as `"Recommended max intensity for tomorrow: X/10"`. Slider snaps to this on initial load; user can drag for "what if" exploration.    
✅ **Overtraining warning** (locked 2026-05-28). Two-tier (Option C):
- 🟡 **Yellow / Caution:** point prediction RHR > baseline + 3 bpm OR point prediction sleep efficiency < baseline - 3 pp. Message: *"Caution — your body may be under-recovered tomorrow."*
- 🔴 **Red / Warning:** lower 90% bound RHR > baseline + 5 bpm OR upper 90% bound sleep efficiency < baseline - 5 pp. Message: *"Strong signs you'll be under-recovered tomorrow. Consider rest or light activity."*
- Trigger logic: **either** condition (RHR or sleep) is sufficient — not "both required". Cite Achten & Jeukendrup 2003 and Plews et al. 2013 for the 5-bpm threshold.    

[x] **Conference target** — arXiv first is locked. IEEE J-BHI (rolling) vs HEALTHCOM 2026 vs BHI 2027 — pick once paper is closer.      
[x] **Lit survey scope** — how many citations? Probably 25-35 for an IEEE paper.

Mark each ✅ as we lock it through discussion.

---

## Locked spec

- **Datasets:** PMData only (16 subjects, ~5 months each, public, no labeling). **Data quality filter applied:** drop p04 (76% RHR missing), p12 (100% RHR missing), p13 (100% RHR missing) → **13 usable subjects, ~1,800 RHR-days total**. Within usable subjects, clip RHR outliers to [35, 95] bpm. Distance values in distance.json are in 10⁻⁵ km units (loader divides by 100,000).
- **Validation strategy:** Leave-One-Subject-Out (LOSO) cross-validation + temporal hold-out (train on first 70% of each subject's days, test on last 30%) + combined LOSO+temporal as the hardest test. Single-dataset is standard practice in wearable-forecasting literature; broader cross-cohort generalization listed in "Limitations and Future Work".
- **Two-track model design (Path 3, locked 2026-05-28):**
  - **Recovery model** (paper's main contribution) — TCN + conformal predicting next-day RHR + sleep efficiency. Requires Watch-class input (RHR, sleep). Trained on full PMData.
  - **Activity model** (deployment fallback) — Gradient-boosting predicting next-day step count + walking steadiness. Trained on PMData with phone-only feature subset (steps, distance, active energy, gait metrics — RHR/sleep excluded).
  - Backend auto-routes uploads: if Watch-class signals present → recovery model; otherwise → activity model.
- **Demo input modes:** Sample subjects (default) + Apple Health upload (auto-detects Watch vs iPhone-only data, routes to correct model) + manual mini-form (fallback).
- **Data-quantity fallback (locked 2026-05-28):**
  - 30+ days uploaded → full personalization (30-day rolling baseline)
  - 7-29 days → predict but show "limited history" badge (use shorter rolling baseline)
  - <7 days → refuse with friendly error: "Need at least 7 days of recent data"
  - Manual form → skip personalization, use population baseline from PMData
  - Sample subjects → trivially fine (months of data)
- **Stack:**
  - Frontend: Next.js + Tailwind + Recharts → Vercel
  - Backend: FastAPI → Railway / Render
  - Model: TCN/LSTM in PyTorch + MAPIE for conformal
- **Timeline:** 20 days build → arXiv day 20 → IEEE conf submission shortly after.
- **Polish:** middle ground.

---

## 20-day phase plan

### Phase 0 — Setup (Day 0, ~2 hours)
- [ ] Create GitHub repo `conformal-recovery-forecaster`
- [ ] Add this PLAN.md
- [ ] Python env: 3.11 + pytorch, mapie, pandas, numpy, matplotlib, fastapi, uvicorn
- [ ] Frontend repo or `/web` subfolder: `npx create-next-app@latest`

### Phase 1 — Foundations (Days 1-3)

| Day | Read | Code |
|---|---|---|
| 1 | Angelopoulos & Bates §1-3 (~3h) | Repo skeleton; download PMData from OSF |
| 2 | PMData paper (~30m) + Mishra et al. (~1h) | EDA notebook: plot RHR / sleep / steps for 1-2 subjects |
| 3 | Stankeviciute et al. TS conformal (~2h) | Build data loader: rolling window features (last 7 days → predict tomorrow) |

**Decision point end of Day 3:** confirm prediction targets.

### Phase 2 — Core ML (Days 4-8)

| Day | Read | Code |
|---|---|---|
| 4 | — | Feature engineering: rolling strain, trailing avg, day-of-week, time-since-last-workout. **Baseline-center features per subject** (subtract their rolling-30-day mean RHR/sleep/etc. so model learns deviations, not absolutes). |
| 5 | Bai et al. TCN (skim, ~1h) | TCN forecaster: last 7 days × ~10 features → next-day RHR + sleep efficiency |
| 6 | — | Train TCN on PMData; sanity-check vs naive baseline. Set up Leave-One-Subject-Out (LOSO) cross-validation harness. |
| 7 | — | LSTM as alternative; pick winner via MAE |
| 8 | Re-skim A&B §4 (split conformal) | Wrap chosen forecaster in MAPIE → 90% prediction intervals |

**End of Phase 2:** point predictions + 90% conformal bands working.

### Phase 3 — Baselines + Activity model (Days 9-11)

| Day | Read | Code |
|---|---|---|
| 9 | Banister fitness-fatigue (~1h) + Foster (skim) | Banister baseline + **Activity model**: gradient-boosting (LightGBM) predicting next-day step count + walking steadiness on PMData phone-only features |
| 10 | Xu & Xie adaptive conformal (~1h) | Adaptive conformal (ACI) on recovery model. Compare vs split conformal. |
| 11 | — | Full eval matrix: all recovery models × all metrics. Activity model evaluated separately (MAE on steps, etc.) |

**Baselines (locked):**
1. Naive persistence — predict tomorrow = today
2. 7-day moving average
3. ARIMA / SARIMA (`pmdarima.auto_arima`)
4. Banister fitness-fatigue model
5. TCN/LSTM point prediction (no conformal)
6. TCN + Bootstrap CI
7. TCN + Quantile Regression
8. **TCN + Split Conformal** ← contribution
9. **TCN + Adaptive Conformal (ACI)** ← contribution

**Metrics:**
- Coverage @ 90% (does the band actually cover 90% of truth?)
- Average width (narrower at same coverage = better)
- MAE / RMSE (point estimates)
- Calibration plot (Q-Q style)

### Phase 4 — Backend (Days 12-14)

| Day | Code |
|---|---|
| 12 | FastAPI: `/predict-recovery` endpoint (RHR + sleep) and `/predict-activity` endpoint (steps + steadiness). Both return point + bounds. |
| 13 | `/upload-apple-health` endpoint: XML parser + **router that detects whether RHR/sleep records exist** → calls correct model |
| 14 | Containerize, deploy to Railway. CORS for Vercel domain. |

### Phase 5 — Frontend + Demo (Days 15-17)

| Day | Code |
|---|---|
| 15 | Subject picker (5 PMData samples), time-series chart with bands (Recharts) |
| 16 | Workout-load slider → live re-render. Overtraining warning UI when lower bound < baseline. **Workout recommendation**: backend sweeps slider 1-10 and returns highest non-warning value; slider snaps there on load. |
| 17 | Apple Health upload + **upload result UI adapts to which model ran** (recovery view vs activity view). Manual mini-form. Polish. Deploy to Vercel. |

### Phase 6 — Paper + Submit (Days 18-20)

| Day | Output |
|---|---|
| 18 | Draft: Abstract, Intro, Related Work, Methodology |
| 19 | Draft: Datasets, Experiments, Results, Discussion, Limitations |
| 20 | IEEE template (Overleaf), polish, **arXiv submission**, identify IEEE conf with open deadline, submit |

---

## Reading list

### Tier 1 — read before code (Days 0-3)
- **Angelopoulos & Bates (2021)** — *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification* — arxiv.org/abs/2107.07511 — read §1, §2, §3
- **Stankeviciute, Alaa, van der Schaar (NeurIPS 2021)** — *Conformal Time-Series Forecasting* — proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html
- **Thambawita et al. (MMSys 2020)** — *PMData: a sports logging dataset* — dl.acm.org/doi/10.1145/3339825.3394926 (data: osf.io/3p7dq)
- **Mishra et al. (Nature BME 2020)** — *Pre-symptomatic detection of COVID-19 from smartwatch data* — nature.com/articles/s41551-020-00640-6

### Tier 2 — read in Week 1 while building
- **Xu & Xie (ICML 2021)** — *Conformal prediction interval for dynamic time-series* — proceedings.mlr.press/v139/xu21h.html
- **Bai, Kolter, Koltun (2018)** — *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* (TCN paper) — arxiv.org/abs/1803.01271
- **Banister, Calvert, Savage, Bach (1975)** — *A systems model of training for athletic performance* — Australian Journal of Sports Medicine. Search title on Google Scholar for PDF.

### Tier 3 — skim for Related Work only
- **Foster (1998)** — *Monitoring training in athletes with reference to overtraining syndrome* — Med Sci Sports Exerc 30(7). Search title on journals.lww.com.
- **Halson (2014)** — *Monitoring training load to understand fatigue in athletes* — Sports Medicine 44(2):139-147 — link.springer.com/article/10.1007/s40279-014-0253-z
- **Quer et al. (Nature Medicine 2021)** — *Wearable sensor data and self-reported symptoms for COVID-19 detection* — nature.com/articles/s41591-020-1123-x

### Tier 4 — for the overtraining-threshold citation (Day 9)
- **Achten & Jeukendrup (2003)** — *Heart rate monitoring: applications and limitations* — Sports Medicine 33(7):517-538
- **Plews et al. (2013)** — *Training adaptation and heart rate variability in elite endurance athletes* — Sports Medicine 43(9):773-781

### Practical docs
- MAPIE library: mapie.readthedocs.io
- PMData: osf.io/3p7dq

---

## Lit survey keywords (find papers, ask Claude yes/no)

**Conformal prediction angle**
- `"conformal prediction" "time series"`
- `"adaptive conformal inference"`
- `"distribution-free uncertainty quantification"`
- `"prediction intervals" deep learning forecasting`

**Wearable / physiology angle**
- `wearable "heart rate" forecasting deep learning`
- `"resting heart rate" recovery training-load`
- `"smartwatch" "machine learning" sleep forecasting`
- `"training load" "fatigue" recovery prediction`
- `personalized physiological prediction`
- `"digital biomarkers" wearable`

**Sports-science / recovery angle**
- `"Banister" "fitness fatigue" machine learning`
- `"acute chronic workload ratio" injury`
- `"overtraining syndrome" detection wearable`
- `"readiness score" wearable algorithm`

Bring ~10-15 candidate titles → Claude filters to 5-7 worth citing.

---

## Live README sketch (will live in repo root)

```markdown
# Conformal Strain–Recovery Forecaster

🔗 Live demo: [vercel-url]
📄 Paper (arXiv): [pending]
💾 Datasets: PMData (n=16), MMASH (n=22)

## Status
- [ ] Phase 1 — Foundations
- [ ] Phase 2 — Core ML
- [ ] Phase 3 — Baselines
- [ ] Phase 4 — Backend
- [ ] Phase 5 — Frontend
- [ ] Phase 6 — Paper

## Results
| Model | Coverage@90 | Avg width | MAE |
|---|---|---|---|
| (fill in) | | | |
```

---

## Notes / changelog
- 2026-05-28: plan drafted; open items above need discussion before Phase 1 starts.
- 2026-05-28: prediction targets locked (RHR + sleep efficiency). Mood/HRV/steps skipped.
- 2026-05-28: Option A locked — Apple Health upload supported only for Apple Watch users; iPhone-only exports lack RHR/sleep so demo will fall back to sample subjects or manual entry. User confirmed she's OK demoing with sample subjects rather than her own data.
- 2026-05-28: **Path 3 locked — two-track model.** Recovery model (TCN + conformal, RHR + sleep, Watch users, paper's main contribution) + Activity model (LightGBM, steps + steadiness, phone-only users, deployment fallback so 90%+ of demo visitors can use their own data). Backend auto-routes uploads. Adds ~1 day of work.
- 2026-05-28: **Strain definition locked.** Watch users: HR × duration (Banister TRIMP, classical sports-science measure). Phone users: calories × duration (proxy). Slider: 1-10 with intuitive labels, calibrated per-user.
- 2026-05-28: **Personalization locked.** Single global model + per-user baseline centering at inference (predict deviation from user's rolling-30-day normal, not absolute values). Cite Mishra et al. 2020 as precedent.
- 2026-05-28: **Interactivity locked.** One slider (workout strain 1-10) + optional Rest Day shortcut. No fake levers (no caffeine / stress / sleep sliders).
- 2026-05-28: **Data-quantity fallback locked.** Tiered: 30+ days = full personalization; 7-29 days = limited-history badge; <7 days = refuse; manual form = population baseline.
- 2026-05-28: **Overtraining warning locked (Option C — two-tier).** Yellow on point estimate, red on conformal bound. RHR threshold ±5 bpm grounded in Achten & Jeukendrup 2003 / Plews et al. 2013.
- 2026-05-28: **Problem statement + novelty positioning written.** Pre-empts the inevitable reviewer question *"why this when Whoop / Garmin / Oura exist?"* with a clear honest answer: open + conformal + reproducible + multi-dataset validated. Contribution framed as *integration + rigorous evaluation + openness*, not method novelty. Targets IEEE applied venues (J-BHI / EMBC), not NeurIPS / ICML.
- 2026-05-28: **MMASH dropped.** Initially planned as cross-cohort validation, but on inspection MMASH only has 1-2 days per subject — insufficient for our 7+ day forecasting window. Cross-cohort RHR-distribution comparison was considered as a fallback use but rejected as too thin. Validation is now PMData-only with rigorous LOSO + temporal hold-out splits. Limitation acknowledged in paper.
- 2026-05-28: **Workout recommendation feature added.** Backend sweeps slider values 1-10 and returns highest value that doesn't trigger overtraining warning. Inverts the slider UX: "tomorrow you can handle up to X/10" instead of just "this slider value will / won't be a warning." ~2-3 hours of work on Day 16.
- 2026-05-28: **Phase 0 setup done + Day 1 EDA started.** Repo skeleton, .gitignore, requirements.txt, README.md, src/conformal_recovery package layout created. PMData moved to data/pmdata/. First loader (loaders.py) written and smoke-tested. EDA notebook (notebooks/01_eda.py) drafted as `# %%`-cell Python file (works as both script and Jupyter notebook).
- 2026-05-28: **Data quality audit done.** All 16 subjects audited. **Drop p04, p12, p13** (severe RHR missingness). Use 13 subjects. **Distance unit confirmed:** raw values are 10⁻⁵ km — loader updated to divide by 100,000. **RHR outlier clipping** to [35, 95] bpm planned for feature pipeline. Sleep-efficiency target has additional missingness on p03 (48%) and p10 (33%); model will train on fewer sleep windows for those subjects.
- 2026-05-29: **Critical bug found and fixed via EDA plots.** Personalization baseline plots revealed Fitbit encodes "no RHR data" as `0` (not NaN). These zero-encoded sensor errors poisoned the rolling 30-day baselines (p05's baseline dropped from 65 → 45 bpm during a sensor-failure week). Filter `[35, 95]` now applied directly in `load_subject()` so all downstream code sees clean RHR. Effect: RHR std for "high-variability" subjects (p03, p10, p11, p14) collapsed by 5-15× (e.g., p10: std 29.7 → 2.0), confirming the variability was sensor error not biology. Means also shifted upward for affected subjects (p10: 50.7 → 68.0 bpm). Net usable RHR-days: 1,818 → 1,673; drop list unchanged.
