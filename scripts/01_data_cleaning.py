"""
01_data_cleaning.py
====================
Cleans and prepares the taxi+weather dataset for analysis.
Filters to Manhattan zones, handles missing values, removes outliers,
and assigns Central Park weather station to all Manhattan zones.

Input:  ../data/taxi_weather_2022_2025.parquet
Output: ../data/manhattan_taxi_clean.parquet

Influenced by Lab 4 (exploration, outlier detection via IQR) and
the sample report's data preparation methodology.
"""

import pandas as pd
import numpy as np
import os

# ── Configuration ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'taxi_weather_2022_2025.parquet')
OUTPUT_FILE = os.path.join(DATA_DIR, 'manhattan_taxi_clean.parquet')

# Manhattan TLC taxi zone IDs (69 zones)
MANHATTAN_ZONES = [
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90,
    100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140,
    141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166,
    170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236,
    237, 238, 239, 243, 244, 246, 249, 261, 262, 263
]

# ── Load Data ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(INPUT_FILE)
print(f"  Raw shape: {df.shape}")

# ── Filter to Manhattan ────────────────────────────────────────────────
print("Filtering to Manhattan zones...")
df = df[df['PULocationID'].isin(MANHATTAN_ZONES)].copy()
print(f"  Manhattan shape: {df.shape}")

# ── Assign Central Park weather station ────────────────────────────────
# For Manhattan, Central Park is the most representative station.
# The raw data has rows for each station; keep only Central Park.
print("Filtering to Central Park weather station...")
df = df[df['station'] == 'central_park'].copy()
print(f"  After station filter: {df.shape}")

# Drop the station column (now constant)
df.drop(columns=['station'], inplace=True)

# ── Drop redundant columns ─────────────────────────────────────────────
drop_cols = ['date', 'day']  # redundant with pickup_hour and other temporal features
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── Handle missing lag features ────────────────────────────────────────
# Lag features (lag_1, lag_2, lag_24, lag_168) have NaN at the start of
# each zone's time series. We forward-fill within each zone, then drop
# any remaining NaN rows (first few hours of the dataset).
print("Handling missing lag features...")
lag_cols = ['lag_1', 'lag_2', 'lag_24', 'lag_168']
null_before = df[lag_cols].isnull().sum().sum()

# Fill remaining NaN lags with 0 (start of time series for each zone)
df[lag_cols] = df[lag_cols].fillna(0)

null_after = df[lag_cols].isnull().sum().sum()
print(f"  Lag nulls: {null_before} -> {null_after}")

# ── Outlier Detection (IQR method, per Lab 4) ──────────────────────────
# Following the sample report: identify outliers using IQR method.
# We flag but do NOT remove them - extreme demand during events is
# exactly what we want to study. We cap extreme weather outliers instead.
print("Detecting outliers...")

# Trip count outliers (flag only)
Q1 = df['trip_count'].quantile(0.25)
Q3 = df['trip_count'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outlier_mask = (df['trip_count'] < lower) | (df['trip_count'] > upper)
df['is_demand_outlier'] = outlier_mask
print(f"  Trip count outliers flagged: {outlier_mask.sum()} ({100*outlier_mask.mean():.1f}%)")
print(f"  IQR bounds: [{lower:.0f}, {upper:.0f}], Q1={Q1}, Q3={Q3}")

# Weather outliers: cap extreme values (winsorize)
weather_cols = ['temperature', 'wind_speed', 'visibility', 'precipitation']
for col in weather_cols:
    q01 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    clipped = df[col].clip(q01, q99)
    n_clipped = (df[col] != clipped).sum()
    if n_clipped > 0:
        print(f"  {col}: clipped {n_clipped} values to [{q01:.2f}, {q99:.2f}]")
    df[col] = clipped

# ── Data Quality Checks ────────────────────────────────────────────────
print("\nData quality checks:")
print(f"  Null values remaining:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"  Date range: {df['pickup_hour'].min()} to {df['pickup_hour'].max()}")
print(f"  Zones: {df['PULocationID'].nunique()}")
print(f"  Total rows: {len(df):,}")
print(f"  Trip count: mean={df['trip_count'].mean():.1f}, "
      f"median={df['trip_count'].median():.0f}, "
      f"max={df['trip_count'].max()}")

# ── Verify temporal completeness ───────────────────────────────────────
# Each zone should have one row per hour
n_zones = df['PULocationID'].nunique()
n_hours = df['pickup_hour'].nunique()
expected = n_zones * n_hours
actual = len(df)
print(f"\n  Expected rows (zones x hours): {expected:,}")
print(f"  Actual rows: {actual:,}")
if actual != expected:
    print(f"  WARNING: {expected - actual:,} missing zone-hour combinations")

# ── Save ───────────────────────────────────────────────────────────────
print(f"\nSaving cleaned data to {OUTPUT_FILE}...")
df.to_parquet(OUTPUT_FILE, index=False)
print(f"  Saved: {len(df):,} rows, {len(df.columns)} columns")
print(f"  Columns: {df.columns.tolist()}")
print("Done!")
