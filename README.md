# Track-a-Taxi

A data science project investigating the **spatial propagation of game-day congestion** across Manhattan, using NYC yellow taxi trip data as a traffic proxy and PDFormer as the predictive model.

**Course:** COMS30050 / COMSM0055 Applied Data Science — University of Bristol  
**Deadlines:** WIP Presentation — 16 April 2026 | Final Report — 28 April 2026

---

## Research Question

> How far from a major event venue (e.g. Madison Square Garden) does the game-day congestion effect propagate across Manhattan, and at what distance does the ~1.3% travel time impact reported in prior work become statistically negligible?

---

## Project Overview

Prior work (Jiang et al., 2024) found that NBA game days increase travel times by approximately **1.3%** city-wide — but this is an aggregate figure. It tells us nothing about *where* in the city the effect is concentrated, how fast it spreads, or how far it reaches.

This project spatially decomposes that effect using NYC taxi trip data as a congestion proxy (following Castro et al., 2012; Li et al., 2022):

- **Demand baseline** — taxi pickups per zone per hour, game vs non-game days
- **Congestion proxy** — mean trip speed (distance / duration) per zone-hour
- **Distance-ring analysis** — % change in travel time by concentric rings from MSG / Barclays Center, with statistical significance per ring
- **PDFormer** — a propagation-delay-aware spatial-temporal transformer that learns how the game-day signal radiates outward across zones over time

The dataset covers **2022–2025**, at **hourly granularity**, across all **263 NYC taxi zones**.

---

## Repository Structure

```
exploration/
├── data_exploration.py       # EDA script — run this first
└── plots/                    # Output figures

scripts/                      # Modelling pipeline (01–08)

plan_xav.md                   # Full project plan (Xavier)
```

---

## Methodology

### 1. Demand Baseline
Aggregate taxi pickups per zone per hour and establish game-day vs non-game-day demand patterns around MSG and Barclays Center.

### 2. Congestion Proxy
Derive mean trip speed (`distance / duration`) per zone-hour. Validate that congestion spikes on game days match the expected demand increase.

### 3. Spatial Propagation Analysis
- Define concentric distance rings from the stadium (0–0.5 km, 0.5–1 km, 1–2 km, 2–5 km, 5 km+)
- For each ring: compute % change in travel time on game days vs matched controls
- Statistical significance per ring → identify the radius where the effect vanishes
- Replicate and spatially extend the 1.6% aggregate finding

### 4. PDFormer Prediction
- Format data into LibCity atomic files (`.dyna`, `.geo`, `.rel`, `.ext`)
- Add event + distance-to-venue features as external inputs — novel contribution
- Run PDFormer; inspect propagation-delay attention weights during game events
- Ablation: with vs without event features

---

## Dataset

| Source | Description | Status |
|--------|-------------|--------|
| NYC TLC trip records | Taxi pickups, dropoffs, speed proxies by zone, 2022–2025 | Available |
| NOAA weather data | Hourly weather merged with taxi data | Available |
| Manhattan events | Game schedules, concerts, etc. | Available |
| US holidays | Public holidays (control variable) | Available |
| NYC taxi zone shapefile | Zone geometry for distance rings | To download |

**Key derived fields:**
- `mean_trip_speed_kmh` per zone-hour (congestion proxy)
- `distance_to_msg_km` per zone (centroid to MSG)
- `distance_ring` — categorical ring label
- `is_game_day`, `hours_since_game_end` — event timing features

> Data files are not included in this repo due to size.

---

## Baselines

| Model | Type |
|-------|------|
| Historical average | Naive lower bound |
| SARIMA | Statistical |
| XGBoost | Gradient boosting with event features |
| LSTM | Deep learning sequential baseline |
| STGCN / GWNET | Spatial-temporal GNN (PDFormer predecessor class) |

Metrics: MAE, RMSE, MAPE.

---

## Novel Contributions

1. **Spatial decomposition** of the game-day travel time effect — extending the 1.3% aggregate finding to a per-ring map
2. **Propagation radius** — a quantified "impact horizon" around event venues
3. **PDFormer with event features** — demonstrating that a propagation-aware model better captures the radiating congestion signal than non-spatial baselines

---

## Key References

- Jiang et al. (2024). Game-day traffic impact — 1.6% travel time effect. *Transportation Research Record*. https://journals.sagepub.com/doi/full/10.1177/15270025241289400
- Jiang et al. (2023). PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction. *AAAI 2023*.
- Castro et al. (2012). Taxi data as urban traffic proxy.
- Li et al. (2022). Taxi trips for urban traffic conditions.
- LibCity framework: https://github.com/LibCity/Bigscity-LibCity
- NYC TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

---

## License

MIT
