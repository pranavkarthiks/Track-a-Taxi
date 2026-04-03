"""
03_feature_engineering.py
==========================
Merges event data with cleaned taxi demand data and engineers features.

Creates:
- Event features (has_event, event_count, event_category flags, proximity)
- Holiday flag
- Cyclical time encodings (sin/cos for hour, day_of_week, month)
- Zone-level historical demand statistics
- Neighbour zone demand lags

Input:  ../data/manhattan_taxi_clean.parquet
        ../data/manhattan_events.parquet
        ../data/us_holidays.csv
Output: ../data/manhattan_features.parquet

References:
- 2008.00568v1: event counts as exogenous variable
- Hochmair 2016: temporal + spatial features
- Lab 4: feature engineering and exploration methodology
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ── Load Data ──────────────────────────────────────────────────────────
print("Loading data...")
taxi = pd.read_parquet(os.path.join(DATA_DIR, 'manhattan_taxi_clean.parquet'))
events = pd.read_parquet(os.path.join(DATA_DIR, 'manhattan_events.parquet'))
holidays = pd.read_csv(os.path.join(DATA_DIR, 'us_holidays.csv'), parse_dates=['date'])

print(f"  Taxi: {taxi.shape}")
print(f"  Events: {events.shape}")
print(f"  Holidays: {holidays.shape}")

# ══════════════════════════════════════════════════════════════════════
# 1. Event Features
# ══════════════════════════════════════════════════════════════════════
print("\nEngineering event features...")

# Expand events to hourly zone-level presence
# For each event, mark every hour between start and end in its zone
event_hourly = []

for _, evt in events.iterrows():
    if pd.isna(evt['zone_id']) or pd.isna(evt['start_datetime']) or pd.isna(evt['end_datetime']):
        continue

    zone = int(evt['zone_id'])
    start = evt['start_datetime'].floor('h')
    end = evt['end_datetime'].ceil('h')

    # Cap at 48 hours to avoid long-running events dominating
    if (end - start).total_seconds() > 48 * 3600:
        end = start + pd.Timedelta(hours=48)

    hours = pd.date_range(start, end, freq='h')
    for h in hours:
        event_hourly.append({
            'pickup_hour': h,
            'PULocationID': zone,
            'event_category': evt['event_category'],
        })

event_hourly_df = pd.DataFrame(event_hourly)
print(f"  Event-hour records: {len(event_hourly_df):,}")

# Aggregate: count events per zone-hour and create category flags
if len(event_hourly_df) > 0:
    # Event count per zone-hour
    event_counts = (event_hourly_df
                    .groupby(['pickup_hour', 'PULocationID'])
                    .size()
                    .reset_index(name='event_count'))

    # Category flags: one-hot for top categories
    category_dummies = pd.get_dummies(event_hourly_df['event_category'], prefix='evt')
    event_cats = pd.concat([
        event_hourly_df[['pickup_hour', 'PULocationID']],
        category_dummies
    ], axis=1)
    event_cats = event_cats.groupby(['pickup_hour', 'PULocationID']).max().reset_index()

    # Merge counts and categories
    event_features = event_counts.merge(event_cats, on=['pickup_hour', 'PULocationID'], how='left')
    event_features['has_event'] = True
    print(f"  Unique zone-hours with events: {len(event_features):,}")
else:
    event_features = pd.DataFrame()

# Merge with taxi data
taxi['pickup_hour'] = pd.to_datetime(taxi['pickup_hour'])

if len(event_features) > 0:
    df = taxi.merge(event_features, on=['pickup_hour', 'PULocationID'], how='left')
    df['has_event'] = df['has_event'].fillna(False)
    df['event_count'] = df['event_count'].fillna(0).astype(int)

    # Fill category flags with 0
    evt_cols = [c for c in df.columns if c.startswith('evt_')]
    df[evt_cols] = df[evt_cols].fillna(0).astype(int)
else:
    df = taxi.copy()
    df['has_event'] = False
    df['event_count'] = 0

print(f"  Rows with events: {df['has_event'].sum():,} ({100*df['has_event'].mean():.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# 2. Holiday Feature
# ══════════════════════════════════════════════════════════════════════
print("\nAdding holiday feature...")
holidays['date'] = pd.to_datetime(holidays['date']).dt.date
df['date_only'] = pd.to_datetime(df['pickup_hour']).dt.date
df['is_holiday'] = df['date_only'].isin(holidays['date'].values)
df.drop(columns=['date_only'], inplace=True)
print(f"  Holiday hours: {df['is_holiday'].sum():,}")

# ══════════════════════════════════════════════════════════════════════
# 3. Cyclical Time Encodings
# ══════════════════════════════════════════════════════════════════════
print("\nAdding cyclical time encodings...")
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

# ══════════════════════════════════════════════════════════════════════
# 4. Zone-Level Historical Demand Statistics
# ══════════════════════════════════════════════════════════════════════
print("\nComputing zone-level statistics...")

# Mean demand per zone (captures baseline zone popularity)
zone_stats = df.groupby('PULocationID')['trip_count'].agg(['mean', 'std']).reset_index()
zone_stats.columns = ['PULocationID', 'zone_mean_demand', 'zone_std_demand']
df = df.merge(zone_stats, on='PULocationID', how='left')

# Mean demand per zone-hour combination (typical hourly pattern)
zone_hour_stats = (df.groupby(['PULocationID', 'hour'])['trip_count']
                   .mean().reset_index())
zone_hour_stats.columns = ['PULocationID', 'hour', 'zone_hour_mean']
df = df.merge(zone_hour_stats, on=['PULocationID', 'hour'], how='left')

# ══════════════════════════════════════════════════════════════════════
# 5. Neighbour Zone Demand (Spatial Lag)
# ══════════════════════════════════════════════════════════════════════
print("\nComputing spatial lag features...")

# Define zone adjacency (approximate, based on Manhattan geography)
# Key neighbouring zone pairs
ZONE_NEIGHBOURS = {
    4: [12, 224],
    12: [4, 261, 13],
    13: [12, 261, 231],
    24: [43, 151, 238],
    41: [42, 74, 166],
    42: [41, 43, 74, 166],
    43: [42, 24, 50, 238, 236],
    45: [144, 148, 231, 13],
    48: [68, 50, 100, 161],
    50: [43, 48, 142, 238, 161],
    68: [48, 90, 158, 100],
    74: [41, 42, 75, 194],
    75: [74, 79, 263, 194],
    79: [75, 148, 263, 113],
    87: [88, 209, 261],
    88: [87, 209, 261],
    90: [68, 164, 234, 158],
    100: [48, 68, 161, 186],
    103: [104, 105, 116],
    104: [103, 105, 107],
    105: [103, 104],
    107: [104, 234],
    113: [79, 144, 158, 249],
    114: [113, 144, 125, 211],
    116: [103, 42, 194],
    120: [127, 244, 243],
    125: [114, 211, 231],
    127: [120, 128, 244],
    128: [127, 45, 148, 231],
    137: [140, 141, 229],
    140: [137, 141, 232],
    141: [140, 236, 262],
    142: [50, 238, 239, 143],
    143: [142, 239, 151],
    144: [45, 113, 114, 231],
    148: [45, 79, 128, 232],
    151: [143, 24, 238, 239],
    152: [153, 166],
    153: [152, 42],
    158: [68, 90, 113, 249],
    161: [48, 50, 100, 162, 163, 230],
    162: [161, 163, 170, 230],
    163: [161, 162, 164, 170],
    164: [90, 163, 170, 234],
    166: [41, 42, 152],
    170: [162, 163, 164, 236],
    186: [100, 164, 233, 234],
    194: [74, 75, 116],
    202: [261, 87],
    209: [87, 88, 261],
    211: [114, 125, 231],
    224: [4, 232],
    229: [137, 233],
    230: [161, 162, 233, 237],
    231: [13, 45, 125, 128, 144, 211],
    232: [140, 148, 224, 261, 262],
    233: [186, 229, 230, 234, 237],
    234: [90, 107, 164, 186, 233],
    236: [43, 141, 170, 237, 262, 263],
    237: [230, 233, 236],
    238: [24, 43, 50, 142, 151, 239],
    239: [142, 143, 151, 238],
    243: [120, 244],
    244: [120, 127, 243],
    246: [68, 100],
    249: [113, 158],
    261: [12, 13, 87, 88, 202, 209, 232],
    262: [141, 232, 236],
    263: [75, 79, 236],
}

# Compute average neighbour demand at lag_1
# First create a lookup for lag_1 by zone-hour
zone_lag = df[['pickup_hour', 'PULocationID', 'lag_1']].copy()

neighbour_demands = []
for zone, neighbours in ZONE_NEIGHBOURS.items():
    zone_data = df[df['PULocationID'] == zone][['pickup_hour']].copy()
    if len(zone_data) == 0:
        continue

    # Get neighbours' lag_1 values
    nb_data = zone_lag[zone_lag['PULocationID'].isin(neighbours)]
    nb_avg = nb_data.groupby('pickup_hour')['lag_1'].mean().reset_index()
    nb_avg.columns = ['pickup_hour', 'neighbor_lag1']

    zone_data = zone_data.merge(nb_avg, on='pickup_hour', how='left')
    zone_data['PULocationID'] = zone
    neighbour_demands.append(zone_data)

if neighbour_demands:
    nb_df = pd.concat(neighbour_demands, ignore_index=True)
    df = df.merge(nb_df, on=['pickup_hour', 'PULocationID'], how='left')
    df['neighbor_lag1'] = df['neighbor_lag1'].fillna(0)
else:
    df['neighbor_lag1'] = 0

print(f"  Zones with neighbours defined: {len(ZONE_NEIGHBOURS)}")

# ══════════════════════════════════════════════════════════════════════
# 6. Final Cleanup
# ══════════════════════════════════════════════════════════════════════
print("\nFinal cleanup...")

# Check for any remaining nulls
null_counts = df.isnull().sum()
has_nulls = null_counts[null_counts > 0]
if len(has_nulls) > 0:
    print(f"  Remaining nulls:\n{has_nulls}")
    # Fill any remaining with 0
    df = df.fillna(0)

# Sort by zone and time
df = df.sort_values(['PULocationID', 'pickup_hour']).reset_index(drop=True)

# Summary
print(f"\n=== Final Dataset ===")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['pickup_hour'].min()} to {df['pickup_hour'].max()}")
print(f"  Zones: {df['PULocationID'].nunique()}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Event hours: {df['has_event'].sum():,} ({100*df['has_event'].mean():.2f}%)")
print(f"  Holiday hours: {df['is_holiday'].sum():,}")

# ── Save ───────────────────────────────────────────────────────────────
output_path = os.path.join(DATA_DIR, 'manhattan_features.parquet')
print(f"\nSaving to {output_path}...")
df.to_parquet(output_path, index=False)
print(f"  Saved: {len(df):,} rows, {len(df.columns)} columns")
print("Done!")
