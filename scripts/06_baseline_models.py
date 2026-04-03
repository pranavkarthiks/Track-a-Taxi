"""
06_baseline_models.py
======================
Trains and evaluates baseline models for taxi demand prediction.

Models:
1. Historical Average (naive baseline)
2. Linear Regression (Ridge)
3. Random Forest
4. XGBoost

Each model is trained twice:
- WITHOUT event features (temporal + weather + lags)
- WITH event features (all features)

This directly tests H3: does adding event features improve predictions?

Train: 2022-2024 (2023-2024 for event-feature models)
Test:  2025

Output: ../outputs/model_results.csv
        ../outputs/figures/model_comparison.png

References:
- Lab 4: scikit-learn methodology
- 04_model_selection.md: rationale for model choices
- Hochmair 2016: regression approach
- Sample report: Random Forest with feature importance
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(os.path.join(DATA_DIR, 'manhattan_features.parquet'))

# Convert booleans to int
for col in df.select_dtypes(include='bool').columns:
    df[col] = df[col].astype(int)

print(f"  Shape: {df.shape}")

# ── Feature Sets ───────────────────────────────────────────────────────
# Base features (no events)
BASE_FEATURES = [
    'hour', 'day_of_week', 'month', 'year',
    'is_weekend', 'is_rush_hour', 'is_holiday',
    'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'sin_month', 'cos_month',
    'precipitation', 'temperature', 'wind_speed', 'visibility', 'rain_category',
    'lag_1', 'lag_2', 'lag_24', 'lag_168',
    'zone_mean_demand', 'zone_hour_mean', 'neighbor_lag1',
]

# Event features
EVENT_FEATURES = [
    'has_event', 'event_count',
    'evt_commercial_shoot', 'evt_film_production', 'evt_other_filming',
    'evt_performance', 'evt_photo_shoot', 'evt_tv_production',
]

# All features
ALL_FEATURES = BASE_FEATURES + EVENT_FEATURES

TARGET = 'trip_count'

# ── Train/Test Split ───────────────────────────────────────────────────
print("\nSplitting data...")
# Use 2023-2024 for training (event data available from 2023)
# Use 2025 for testing
train = df[(df['year'] >= 2023) & (df['year'] <= 2024)].copy()
test = df[df['year'] == 2025].copy()

# Sample training data if too large (for RF/XGBoost speed)
MAX_TRAIN = 200000
if len(train) > MAX_TRAIN:
    train_sampled = train.sample(n=MAX_TRAIN, random_state=42)
    print(f"  Train (full): {len(train):,} rows, sampled to {MAX_TRAIN:,} for tree models")
else:
    train_sampled = train

print(f"  Train: {len(train):,} rows ({train['pickup_hour'].min()} to {train['pickup_hour'].max()})")
print(f"  Test:  {len(test):,} rows ({test['pickup_hour'].min()} to {test['pickup_hour'].max()})")

X_train_base = train_sampled[BASE_FEATURES].values
X_train_all = train_sampled[ALL_FEATURES].values
X_test_base = test[BASE_FEATURES].values
X_test_all = test[ALL_FEATURES].values
y_train = train_sampled[TARGET].values
y_test = test[TARGET].values

# Scale features for Ridge
scaler_base = StandardScaler()
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_test_base_scaled = scaler_base.transform(X_test_base)

scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_test_all_scaled = scaler_all.transform(X_test_all)


# ── Evaluation Function ───────────────────────────────────────────────

def evaluate(y_true, y_pred, model_name):
    """Compute metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE (avoid division by zero)
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    return {'model': model_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


# ══════════════════════════════════════════════════════════════════════
# Model 1: Historical Average (Naive Baseline)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Model 1: Historical Average")
print("="*60)

# Mean trip count for each (zone, hour, day_of_week) in training data (use full train)
historical_avg = train.groupby(['PULocationID', 'hour', 'day_of_week'])[TARGET].mean()

# Predict on test set
test_pred_naive = test.set_index(['PULocationID', 'hour', 'day_of_week']).index.map(
    lambda x: historical_avg.get(x, train[TARGET].mean())
)
test_pred_naive = np.array(test_pred_naive, dtype=float)

result_naive = evaluate(y_test, test_pred_naive, 'Historical Average')
print(f"  RMSE: {result_naive['RMSE']:.3f}")
print(f"  MAE:  {result_naive['MAE']:.3f}")
print(f"  R2:   {result_naive['R2']:.4f}")

results = [result_naive]

# ══════════════════════════════════════════════════════════════════════
# Model 2: Ridge Regression
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Model 2: Ridge Regression")
print("="*60)

for name, X_tr, X_te in [
    ('Ridge (no events)', X_train_base_scaled, X_test_base_scaled),
    ('Ridge (with events)', X_train_all_scaled, X_test_all_scaled)
]:
    t0 = time.time()
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_pred = np.clip(y_pred, 0, None)  # demand can't be negative
    elapsed = time.time() - t0

    result = evaluate(y_test, y_pred, name)
    results.append(result)
    print(f"\n  {name}:")
    print(f"    RMSE: {result['RMSE']:.3f}, MAE: {result['MAE']:.3f}, "
          f"R2: {result['R2']:.4f} ({elapsed:.1f}s)")

    # Print top coefficients for the event-feature model
    if 'with events' in name:
        coef_names = ALL_FEATURES
        coef_importance = sorted(zip(coef_names, model.coef_),
                                 key=lambda x: abs(x[1]), reverse=True)
        print("    Top 10 coefficients:")
        for feat, coef in coef_importance[:10]:
            print(f"      {feat:25s}: {coef:+.3f}")

# ══════════════════════════════════════════════════════════════════════
# Model 3: Random Forest
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Model 3: Random Forest")
print("="*60)

for name, X_tr, X_te, feat_names in [
    ('Random Forest (no events)', X_train_base, X_test_base, BASE_FEATURES),
    ('Random Forest (with events)', X_train_all, X_test_all, ALL_FEATURES)
]:
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    elapsed = time.time() - t0

    result = evaluate(y_test, y_pred, name)
    results.append(result)
    print(f"\n  {name}:")
    print(f"    RMSE: {result['RMSE']:.3f}, MAE: {result['MAE']:.3f}, "
          f"R2: {result['R2']:.4f} ({elapsed:.1f}s)")

    # Feature importance
    importance = sorted(zip(feat_names, model.feature_importances_),
                        key=lambda x: x[1], reverse=True)
    print("    Top 10 features:")
    for feat, imp in importance[:10]:
        print(f"      {feat:25s}: {imp:.4f}")

# ══════════════════════════════════════════════════════════════════════
# Model 4: XGBoost
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Model 4: XGBoost")
print("="*60)

try:
    import xgboost as xgb

    for name, X_tr, X_te, feat_names in [
        ('XGBoost (no events)', X_train_base, X_test_base, BASE_FEATURES),
        ('XGBoost (with events)', X_train_all, X_test_all, ALL_FEATURES)
    ]:
        t0 = time.time()
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            tree_method='hist'
        )
        model.fit(X_tr, y_train,
                  eval_set=[(X_te, y_test)],
                  verbose=False)
        y_pred = model.predict(X_te)
        elapsed = time.time() - t0

        result = evaluate(y_test, y_pred, name)
        results.append(result)
        print(f"\n  {name}:")
        print(f"    RMSE: {result['RMSE']:.3f}, MAE: {result['MAE']:.3f}, "
              f"R2: {result['R2']:.4f} ({elapsed:.1f}s)")

        importance = sorted(zip(feat_names, model.feature_importances_),
                            key=lambda x: x[1], reverse=True)
        print("    Top 10 features:")
        for feat, imp in importance[:10]:
            print(f"      {feat:25s}: {imp:.4f}")

except ImportError:
    print("  XGBoost not installed. Skipping.")
    print("  Install with: pip install xgboost")

# ══════════════════════════════════════════════════════════════════════
# Results Summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')

print(f"\n{'Model':<35s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'MAPE':>8s}")
print("-" * 70)
for _, row in results_df.iterrows():
    print(f"{row['model']:<35s} {row['RMSE']:>8.3f} {row['MAE']:>8.3f} "
          f"{row['R2']:>8.4f} {row['MAPE']:>7.1f}%")

# H3 Test: compare with vs without events for each model type
print("\n\nH3: Event Feature Improvement")
print("-" * 50)
model_types = ['Ridge', 'Random Forest', 'XGBoost']
for mt in model_types:
    no_evt = results_df[results_df['model'] == f'{mt} (no events)']
    with_evt = results_df[results_df['model'] == f'{mt} (with events)']
    if len(no_evt) > 0 and len(with_evt) > 0:
        rmse_no = no_evt.iloc[0]['RMSE']
        rmse_with = with_evt.iloc[0]['RMSE']
        improvement = 100 * (rmse_no - rmse_with) / rmse_no
        print(f"  {mt}: RMSE {rmse_no:.3f} -> {rmse_with:.3f} "
              f"({improvement:+.2f}% {'improvement' if improvement > 0 else 'worse'})")

# Save results
results_df.to_csv(os.path.join(OUT_DIR, 'model_results.csv'), index=False)
print(f"\nResults saved to {os.path.join(OUT_DIR, 'model_results.csv')}")

# ── Visualisation ──────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go

# Model comparison bar chart
fig = go.Figure()

# Group by model type
for metric, color in [('RMSE', 'steelblue'), ('MAE', 'coral')]:
    fig.add_trace(go.Bar(
        name=metric,
        x=results_df['model'],
        y=results_df[metric],
        marker_color=color
    ))

fig.update_layout(
    title='Model Comparison: RMSE and MAE',
    xaxis_title='Model',
    yaxis_title='Error',
    barmode='group',
    width=1100,
    height=500,
    xaxis_tickangle=-30
)
fig.write_html(os.path.join(FIG_DIR, 'fig12_model_comparison.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig12_model_comparison.png'), scale=2)
print("Saved fig12_model_comparison")

print("\nDone!")
