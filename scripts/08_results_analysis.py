"""
08_results_analysis.py
=======================
Final results analysis and figure generation for the IEEE paper.

Combines all model results, generates publication-quality figures,
and produces the zone-level analysis for H4.

Output: ../outputs/figures/ (publication figures)
        ../outputs/final_results_summary.txt

References:
- Sample report: table/figure style for IEEE format
- Lab 5: Plotly visualisations
"""

import pandas as pd
import numpy as np
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIG_DIR = os.path.join(OUT_DIR, 'figures')

# ── Load Results ───────────────────────────────────────────────────────
results_file = os.path.join(OUT_DIR, 'model_results_all.csv')
if os.path.exists(results_file):
    results = pd.read_csv(results_file)
else:
    results = pd.read_csv(os.path.join(OUT_DIR, 'model_results.csv'))

print("="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

results = results.sort_values('RMSE')
print(f"\n{'Model':<35s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'MAPE':>8s}")
print("-" * 70)
for _, row in results.iterrows():
    mape = f"{row['MAPE']:.1f}%" if not pd.isna(row['MAPE']) else "N/A"
    print(f"{row['model']:<35s} {row['RMSE']:>8.3f} {row['MAE']:>8.3f} "
          f"{row['R2']:>8.4f} {mape:>8s}")

# ── Publication Figure: Model Comparison ───────────────────────────────
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['RMSE (lower is better)', 'R² (higher is better)'])

sorted_models = results.sort_values('RMSE')

# Colour code: with events = coral, without = steelblue, baseline = grey
colors = []
for m in sorted_models['model']:
    if 'with events' in m.lower():
        colors.append('coral')
    elif 'no events' in m.lower():
        colors.append('steelblue')
    else:
        colors.append('grey')

fig.add_trace(
    go.Bar(y=sorted_models['model'], x=sorted_models['RMSE'],
           orientation='h', marker_color=colors, showlegend=False),
    row=1, col=1
)

fig.add_trace(
    go.Bar(y=sorted_models['model'], x=sorted_models['R2'],
           orientation='h', marker_color=colors, showlegend=False),
    row=1, col=2
)

fig.update_layout(height=500, width=1200,
                  title_text='Model Performance Comparison (Test Set: 2025)')
fig.write_html(os.path.join(FIG_DIR, 'fig_final_model_comparison.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig_final_model_comparison.png'), scale=2)
print("\nSaved fig_final_model_comparison")

# ── H3 Analysis ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("H3: EVENT FEATURE IMPACT ON PREDICTION")
print("="*60)

for model_type in ['Ridge', 'Random Forest', 'XGBoost', 'LSTM']:
    no_evt = results[results['model'].str.contains(f'{model_type}.*no events', case=False)]
    with_evt = results[results['model'].str.contains(f'{model_type}.*with events', case=False)]

    if len(no_evt) > 0 and len(with_evt) > 0:
        rmse_no = no_evt.iloc[0]['RMSE']
        rmse_with = with_evt.iloc[0]['RMSE']
        improvement = 100 * (rmse_no - rmse_with) / rmse_no
        symbol = "+" if improvement > 0 else ""
        print(f"  {model_type:20s}: RMSE {rmse_no:.3f} -> {rmse_with:.3f} "
              f"({symbol}{improvement:.2f}%)")

print("\n  Interpretation:")
print("  The lag features (lag_1, lag_24, lag_168) already capture event-driven")
print("  demand spikes implicitly. By the time an event is happening, its effect")
print("  on demand is reflected in the 1-hour lag. This means explicit event")
print("  features add redundant information when strong autoregressive features")
print("  are present.")
print("\n  However, event features would be valuable for:")
print("  1. Longer-horizon forecasts (>24h) where lag data is unavailable")
print("  2. Scenario planning (what-if an event is scheduled)")
print("  3. Explaining demand patterns (not just predicting)")

# ── Feature Importance Summary ─────────────────────────────────────────
print("\n" + "="*60)
print("FEATURE IMPORTANCE RANKING (consistent across models)")
print("="*60)
print("  1. lag_1 (1-hour lag)           - dominant predictor (~64-80%)")
print("  2. lag_168 (1-week lag)         - weekly seasonality (~16-23%)")
print("  3. lag_24 (1-day lag)           - daily seasonality (~2-3%)")
print("  4. zone_hour_mean              - zone baseline demand (~2-4%)")
print("  5. Hour cyclical features       - time of day (~0.5-1%)")
print("  6. Event features               - marginal (<0.5%)")
print("  7. Weather features             - marginal (<0.5%)")
print("  8. Holiday flag                 - small but notable (~0.5%)")

# ── Zone-Level Error Analysis ──────────────────────────────────────────
print("\n" + "="*60)
print("ZONE-LEVEL ANALYSIS")
print("="*60)

# Load data and best model predictions for zone analysis
df = pd.read_parquet(os.path.join(DATA_DIR, 'manhattan_features.parquet'))
for col in df.select_dtypes(include='bool').columns:
    df[col] = df[col].astype(int)

test = df[df['year'] == 2025].copy()

zone_stats = test.groupby('PULocationID').agg(
    mean_demand=('trip_count', 'mean'),
    std_demand=('trip_count', 'std'),
    event_hours=('has_event', 'sum'),
    event_pct=('has_event', 'mean'),
).reset_index()

zone_stats['event_pct'] = zone_stats['event_pct'] * 100
zone_stats = zone_stats.sort_values('mean_demand', ascending=False)

print(f"\n  {'Zone':>6s} {'Mean Demand':>12s} {'Std':>8s} {'Event Hrs':>10s} {'Event %':>8s}")
print("  " + "-" * 50)
for _, row in zone_stats.head(15).iterrows():
    print(f"  {int(row['PULocationID']):>6d} {row['mean_demand']:>12.1f} {row['std_demand']:>8.1f} "
          f"{int(row['event_hours']):>10d} {row['event_pct']:>7.1f}%")

# Zone demand vs event frequency scatter
fig = px.scatter(zone_stats, x='mean_demand', y='event_pct',
                 size='event_hours', hover_data=['PULocationID'],
                 title='Zone Demand vs Event Frequency',
                 labels={'mean_demand': 'Mean Hourly Demand',
                         'event_pct': '% Hours with Events',
                         'event_hours': 'Total Event Hours'},
                 trendline='ols')
fig.update_layout(width=800, height=500)
fig.write_html(os.path.join(FIG_DIR, 'fig_zone_demand_events.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig_zone_demand_events.png'), scale=2)
print("\nSaved fig_zone_demand_events")

# ── Summary for write-up ──────────────────────────────────────────────
summary = """
FINAL RESULTS SUMMARY FOR IEEE PAPER
=====================================

1. HYPOTHESIS RESULTS:
   H1 (SUPPORTED): Public events significantly increase taxi demand.
     - Mean demand during events: ~120 trips/hr vs ~47 trips/hr (no events)
     - Cohen's d = 0.81 (large effect)
     - p < 0.001 (Welch's t-test and Mann-Whitney U)
     - 67% of tested zones show significant effect after controlling for time

   H2 (SUPPORTED): Weather-event interaction is significant.
     - Interaction coefficient significant (p < 0.001)
     - However, direction is negative: rain REDUCES the event demand boost
       (possibly because events themselves are cancelled/reduced in rain)

   H3 (NOT SUPPORTED): Event features do NOT improve prediction accuracy.
     - Lag features (especially lag_1) dominate prediction
     - Event information is already embedded in recent lag values
     - RMSE unchanged or slightly worse with event features
     - This is a valid negative result with practical implications

   H4 (NOT CONFIRMED): Spatial heterogeneity not statistically significant.
     - Event effects vary (CV = 0.97) but not systematically by exposure level
     - Kruskal-Wallis p = 0.83

2. BEST MODEL:
   Random Forest (no events): RMSE = 14.69, MAE = 7.56, R² = 0.961

3. KEY INSIGHT:
   Events DO cause real demand spikes (H1 confirmed), but these spikes
   are already captured by autoregressive features. For real-time prediction
   where recent demand data is available, event calendars add little value.
   They would matter more for longer-horizon forecasting.

4. PRACTICAL IMPLICATIONS:
   - Real-time dispatch: lag features sufficient, no event calendar needed
   - Day-ahead planning: event features become valuable when lags unavailable
   - Urban planning: events explain ~5% of zone-hour variation in demand
"""

with open(os.path.join(OUT_DIR, 'final_results_summary.txt'), 'w') as f:
    f.write(summary)
print(summary)
print("Done!")
