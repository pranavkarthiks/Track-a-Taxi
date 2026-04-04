# Project Plan: Spatial Propagation of Game-Day Congestion Using PDFormer

**Course:** COMS30050 / COMSM0055 Applied Data Science, University of Bristol  
**Deadlines:** WIP Presentation — 16 April 2026 | Final Report — 28 April 2026

---

## Research Question

> How far from a major event venue (e.g. Madison Square Garden) does the game-day congestion effect propagate across Manhattan, and at what distance does the ~1.6% travel time impact reported in prior work become statistically negligible?

---

## Narrative Arc

**Prior work** (Jiang et al., 2024 — *Transportation Research Record*) found that NBA game days increase travel times by approximately **1.6%** on average. However, this is a city-wide aggregate figure — it does not tell us *where* in the city the effect is concentrated, how fast it spreads, or how far it reaches.

**Our contribution:** We use NYC taxi trip data as a proxy for urban traffic conditions (following Castro et al., 2012; Li et al., 2022) to spatially decompose this effect:
- Measure congestion (trip speed / travel time) by distance ring from the stadium
- Identify the radius at which the game-day effect drops to statistical noise
- Use PDFormer's propagation-delay attention to model *how* the signal radiates outward across zones over time

**Why PDFormer?** Its core mechanism — modelling the time-lagged spatial propagation of traffic signals — is a natural fit. The learned propagation delay weights directly map to "how fast does the game-day effect travel from MSG into surrounding zones?"

---

## Methodology

### Step 1 — Demand as a Baseline
- Aggregate taxi pickups per zone per hour → demand surface
- Establish game-day vs non-game-day demand baseline around MSG / Barclays Center

### Step 2 — Congestion as a Function of Demand
- Derive congestion proxy: mean trip speed (distance / duration) per zone-hour
- Congestion = f(demand) — model the relationship, validate against demand spikes on game days

### Step 3 — Spatial Propagation Analysis
- Define concentric distance rings from the stadium (e.g., 0–0.5km, 0.5–1km, 1–2km, 2–5km, 5km+)
- For each ring: compute % change in travel time on game days vs matched non-game days
- Statistical significance test per ring → find the radius where the effect vanishes
- Replicate/extend the 1.6% finding spatially

### Step 4 — PDFormer as the Predictive Model
- PDFormer learns to predict congestion across all zones simultaneously
- Its propagation-delay mechanism captures the temporal lag of the effect spreading outward
- Inspect attention weights during game periods — which zones does the model link back to the stadium?
- Ablation: model with vs without event features → quantify the stadium signal's predictive value

---

## Data

| Source | Description | Status |
|--------|-------------|--------|
| NYC TLC trip records | Taxi pickups, dropoffs, speed proxies by zone, 2022–2025 | Available (`taxi_weather_2022_2025.parquet`) |
| Weather data | Hourly weather merged with taxi data | Available |
| Manhattan events | Game schedules, concerts, etc. | Available (`manhattan_events.parquet`) |
| US holidays | Public holidays (control variable) | Available (`us_holidays.csv`) |
| NYC taxi zone shapefile | Zone geometry for distance ring construction | To download — NYC Open Data |

**Key derived fields to create:**
- `mean_trip_speed_kmh` per zone-hour (congestion proxy)
- `distance_to_msg_km` per zone (from zone centroid to MSG)
- `distance_ring` — categorical ring label per zone
- `is_game_day`, `hours_since_game_end` — event timing features

---

## Baselines

| Model | Type | Rationale |
|-------|------|-----------|
| Historical average | Naive | Lower bound |
| SARIMA | Statistical | Standard time-series baseline |
| XGBoost | Gradient boosting | Strong tabular baseline with event features |
| LSTM | Deep learning | Sequential baseline (`07_lstm_model.py` already exists) |
| STGCN or GWNET | Spatial-temporal GNN | Immediate predecessor class to PDFormer |

Metrics: MAE, RMSE, MAPE — consistent with PDFormer paper and comparable to the 1.6% travel time effect size.

---

## PDFormer Implementation

- [ ] Install LibCity framework (PDFormer dependency)
- [ ] Format NYC taxi data into LibCity atomic files (`.dyna`, `.geo`, `.rel`, `.ext`)
- [ ] Add event + distance-to-venue features as external inputs (`.ext`) — novel addition
- [ ] Run PDFormer using existing `NYCTaxi` config as starting point
- [ ] Tune: attention heads, propagation delay window, spatial graph construction
- [ ] Ablation: with vs without event features
- [ ] Extract and visualise learned propagation delay weights during game events

---

## EDA / Analysis Checklist

- [ ] Demand heatmaps: game vs non-game days, by hour
- [ ] Speed/congestion heatmaps: game vs non-game days
- [ ] Distance-ring analysis: % travel time impact by ring, with confidence intervals
- [ ] Replicate the 1.6% aggregate finding on our data → validate dataset
- [ ] Identify the "significance horizon" — radius at which effect is no longer detectable
- [ ] Temporal profile: how many hours before/after the game does the effect persist?

---

## Novel Contributions

1. **Spatial decomposition** of the game-day travel time effect (extending the 1.6% aggregate finding)
2. **Propagation radius** — a quantified "impact horizon" around event venues
3. **PDFormer with event features** — showing that a propagation-aware model better captures the radiating congestion signal than non-spatial baselines

---

## File Structure

```
datasciencev2?/
├── data/                    # Raw and processed datasets
├── scripts/                 # Existing pipeline (01–08)
├── plan/                    # This plan
│   └── PROJECT_PLAN.md
├── pdformer/                # PDFormer repo (to clone)
├── outputs/                 # Results, figures
├── articles/                # Reference papers
└── decisions/               # Key decision notes
```

---

## Timeline

| Period | Dates | Goal |
|--------|-------|------|
| Now | 3–6 Apr | Derive congestion proxy; zone-level aggregation; distance rings |
| Week 2 | 7–10 Apr | Distance-ring analysis + replicate 1.6% finding; baselines running |
| **WIP Presentation** | **16 Apr** | EDA + spatial propagation findings + PDFormer in progress |
| Week 3 | 17–24 Apr | PDFormer tuning + ablation + attention weight analysis |
| Week 4 | 25–27 Apr | Report writing |
| **Final Deadline** | **28 Apr** | Submit report + individual reflective discussion |

---

## Key References

- Jiang et al. (2024). [Game-day traffic impact study — 1.6% travel time effect]. *Transportation Research Record*. https://journals.sagepub.com/doi/full/10.1177/15270025241289400
- Jiang et al. (2023). PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction. *AAAI 2023*.
- Castro et al. (2012). Taxi data as urban traffic proxy.
- Li et al. (2022). Taxi trips for urban traffic conditions.
- LibCity framework: https://github.com/LibCity/Bigscity-LibCity
- NYC TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
