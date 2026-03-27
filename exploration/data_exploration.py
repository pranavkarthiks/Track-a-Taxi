# =============================================================================
# DATA EXPLORATION: NYC Yellow Taxi + Weather Dataset
# =============================================================================
#
# Goal: Understand the structure and patterns in the merged taxi-weather data
#       before building a predictive model for hourly zone-level demand.
#
# Dataset covers 2022–2025, one row per zone per hour.
# Target variable: trip_count (how many yellow taxi pickups in that zone-hour)
#
# Outputs: printed stats + saved plots in a local 'data/' subfolder
# =============================================================================

import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# --- PATHS -------------------------------------------------------------------
# Locate files relative to this script, so it works regardless of where you
# open it from in your IDE
SCRIPT_DIR  = Path(__file__).parent
ZIP_PATH    = SCRIPT_DIR / "OneDrive_1_27-03-2026" / "taxiWeatherFull.zip"
PARQUET_NAME = "taxi_weather_2022_2025.parquet"   # the full 4-year merged file
EXTRACT_DIR = SCRIPT_DIR / "data"                 # where we unzip + save plots

# Global plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (14, 5)


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================
print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

# Create the output directory if it doesn't exist
EXTRACT_DIR.mkdir(exist_ok=True)
parquet_path = EXTRACT_DIR / PARQUET_NAME

# Extract the parquet from the zip only if we haven't done it already
if not parquet_path.exists():
    print(f"Extracting {PARQUET_NAME} from zip (one-time, may take a moment)...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extract(PARQUET_NAME, EXTRACT_DIR)
    print("Extraction complete.")
else:
    print(f"Parquet already extracted, loading from: {parquet_path}")

print("Reading parquet into dataframe...")
# fastparquet handles these particular files correctly (pyarrow has a known issue)
df = pd.read_parquet(parquet_path, engine="fastparquet")
print(f"Loaded {len(df):,} rows x {df.shape[1]} columns.\n")


# =============================================================================
# STEP 2 — BASIC OVERVIEW
# =============================================================================
print("=" * 70)
print("STEP 2: BASIC OVERVIEW")
print("=" * 70)

print("\n--- Column Data Types ---")
print(df.dtypes)

# Check for missing values — lag columns will have NaNs at the very start of
# each zone's time series (nothing to lag back to), which is expected
print("\n--- Missing Values per Column ---")
nulls = df.isnull().sum()
nulls_nonzero = nulls[nulls > 0]
if nulls_nonzero.empty:
    print("None (lag NaNs are handled separately)")
else:
    print(nulls_nonzero)
    print("(Lag NaNs at series start are expected — not a data quality issue)")

print("\n--- First 3 Rows ---")
print(df.head(3).to_string())

# Time coverage
print(f"\n--- Date Range ---")
print(f"  Earliest: {df['pickup_hour'].min()}")
print(f"  Latest:   {df['pickup_hour'].max()}")
print(f"  Years:    {sorted(df['year'].unique().tolist())}")

# Zones and weather stations
print(f"\n--- Spatial / Station Summary ---")
print(f"  Zones (PULocationID): {df['PULocationID'].nunique()} unique zones")
print(f"  Weather stations:     {df['station'].unique().tolist()}")
# Each zone is assigned its nearest weather station, so there's still one row
# per zone per hour (not three rows)


# =============================================================================
# STEP 3 — TARGET VARIABLE: trip_count
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: TRIP COUNT (DEMAND) DISTRIBUTION")
print("=" * 70)

print("\n--- Summary Statistics: trip_count ---")
print(df['trip_count'].describe())

# Many zone-hours will be zero — most zones are quiet most of the time
zero_pct = (df['trip_count'] == 0).mean() * 100
print(f"\n  {zero_pct:.1f}% of zone-hours have ZERO trips (dataset is sparse)")

# Focus on active zone-hours to understand real demand patterns
active = df[df['trip_count'] > 0]
print(f"  Active zone-hours (trip_count > 0): {len(active):,}")
print(f"\n--- Summary Statistics: trip_count (active zone-hours only) ---")
print(active['trip_count'].describe())

# Plot: raw distribution vs log-transformed
# Log transform helps visualise skewed count data
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(active['trip_count'], bins=100, color='steelblue', edgecolor='none')
axes[0].set_title("Hourly Trip Count Distribution\n(active zone-hours only)")
axes[0].set_xlabel("Trips per zone per hour")
axes[0].set_ylabel("Frequency")

axes[1].hist(np.log1p(df['trip_count']), bins=100, color='coral', edgecolor='none')
axes[1].set_title("log(1 + trip_count) Distribution\n(all zone-hours incl. zeros)")
axes[1].set_xlabel("log(1 + trips)")
axes[1].set_ylabel("Frequency")

plt.suptitle("Figure 1: Demand Distribution", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(EXTRACT_DIR / "fig1_demand_distribution.png", dpi=120, bbox_inches='tight')
plt.show()
print("  -> Saved: fig1_demand_distribution.png")


# =============================================================================
# STEP 4 — TEMPORAL PATTERNS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: TEMPORAL PATTERNS")
print("=" * 70)

# --- 4a: Overall daily demand trend ---
# Sum all zones together to see city-wide demand over time
daily = df.groupby('date')['trip_count'].sum().reset_index()
print(f"\n--- Daily Total Demand (all zones summed) ---")
print(daily['trip_count'].describe())

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily['date'], daily['trip_count'], linewidth=0.8, color='steelblue', alpha=0.9)
ax.set_title("Total Daily Yellow Taxi Demand — All Zones (2022–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Total trips")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.savefig(EXTRACT_DIR / "fig2_daily_trend.png", dpi=120, bbox_inches='tight')
plt.show()
print("  -> Saved: fig2_daily_trend.png")

# --- 4b: Demand by hour, day of week, and month ---
hourly_avg  = df.groupby('hour')['trip_count'].mean()
dow_avg     = df.groupby('day_of_week')['trip_count'].mean()
monthly_avg = df.groupby(['year', 'month'])['trip_count'].mean().reset_index()

dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

print(f"\n--- Average Demand by Hour of Day ---")
print(hourly_avg.to_string())

print(f"\n--- Average Demand by Day of Week ---")
for i, val in dow_avg.items():
    print(f"  {dow_labels[i]}: {val:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Hour of day — expect peaks in morning and evening rush hours
axes[0].plot(hourly_avg.index, hourly_avg.values, marker='o', color='steelblue')
axes[0].set_title("Avg Demand by Hour of Day")
axes[0].set_xlabel("Hour (0 = midnight)")
axes[0].set_ylabel("Avg trips per zone-hour")
axes[0].set_xticks(range(0, 24, 2))

# Day of week — expect weekdays vs weekend differences
axes[1].bar(range(7), dow_avg.values, color='coral')
axes[1].set_title("Avg Demand by Day of Week")
axes[1].set_xlabel("Day")
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(dow_labels)
axes[1].set_ylabel("Avg trips per zone-hour")

# Monthly trend — compare same month across years to spot seasonality
for yr in sorted(monthly_avg['year'].unique()):
    subset = monthly_avg[monthly_avg['year'] == yr]
    axes[2].plot(subset['month'], subset['trip_count'], marker='o', label=str(yr))
axes[2].set_title("Avg Demand by Month (each year)")
axes[2].set_xlabel("Month")
axes[2].set_xticks(range(1, 13))
axes[2].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec'], rotation=45)
axes[2].legend(title="Year")
axes[2].set_ylabel("Avg trips per zone-hour")

plt.suptitle("Figure 3: Temporal Demand Patterns", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(EXTRACT_DIR / "fig3_temporal_patterns.png", dpi=120, bbox_inches='tight')
plt.show()
print("  -> Saved: fig3_temporal_patterns.png")


# =============================================================================
# STEP 5 — ZONE-LEVEL ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: ZONE-LEVEL DEMAND ANALYSIS")
print("=" * 70)

# Total demand per zone across all years
zone_demand = df.groupby('PULocationID')['trip_count'].sum().sort_values(ascending=False)

print(f"\n--- Top 10 Busiest Zones (by total trips 2022–2025) ---")
print(zone_demand.head(10).to_string())

print(f"\n--- 10 Quietest Zones ---")
print(zone_demand.tail(10).to_string())

# Zones with no demand at all — may be airports, depots, or out-of-service areas
zero_zones = zone_demand[zone_demand == 0]
print(f"\n--- Zones with ZERO demand across entire dataset: {len(zero_zones)} ---")
if not zero_zones.empty:
    print(zero_zones.to_string())

# Bar chart — all 263 zones sorted by demand
# Shows just how top-heavy the demand distribution is
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(range(len(zone_demand)), zone_demand.values, color='steelblue', width=1.0)
ax.set_title("Total Demand by Zone, Sorted Highest → Lowest (2022–2025)")
ax.set_xlabel("Zone rank (1 = busiest)")
ax.set_ylabel("Total trips")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
plt.tight_layout()
plt.savefig(EXTRACT_DIR / "fig4_zone_demand_ranked.png", dpi=120, bbox_inches='tight')
plt.show()
print("  -> Saved: fig4_zone_demand_ranked.png")


# =============================================================================
# STEP 6 — WEATHER VARIABLE DISTRIBUTIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: WEATHER VARIABLE DISTRIBUTIONS")
print("=" * 70)

weather_cols = ['precipitation', 'temperature', 'wind_speed', 'visibility']

print("\n--- Weather Summary Statistics ---")
print(df[weather_cols].describe())

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
colours = ['steelblue', 'coral', 'mediumpurple', 'seagreen']

for i, col in enumerate(weather_cols):
    axes[i].hist(df[col].dropna(), bins=60, color=colours[i], edgecolor='none')
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    if i == 0:
        axes[i].set_ylabel("Frequency")

plt.suptitle("Figure 5: Weather Variable Distributions", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(EXTRACT_DIR / "fig5_weather_distributions.png", dpi=120, bbox_inches='tight')
plt.show()
print("  -> Saved: fig5_weather_distributions.png")


# =============================================================================
# STEP 7 — WEATHER vs DEMAND CORRELATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: WEATHER–DEMAND CORRELATION")
print("=" * 70)

# Build a correlation matrix across weather, time, and lag features
# This shows which variables are most linearly associated with trip_count
corr_cols = [
    'trip_count',
    'precipitation', 'temperature', 'wind_speed', 'visibility',
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
    'lag_1', 'lag_24', 'lag_168'
]
corr_df = df[corr_cols].copy()

# Convert boolean columns to int so they work in the correlation matrix
corr_df['is_weekend']   = corr_df['is_weekend'].astype(int)
corr_df['is_rush_hour'] = corr_df['is_rush_hour'].astype(int)

corr_matrix = corr_df.corr()

# Extract just the correlations with trip_count, sorted by absolute strength
trip_corr = corr_matrix['trip_count'].drop('trip_count')
trip_corr_sorted = trip_corr.abs().sort_values(ascending=False)

print("\n--- Pearson Correlations with trip_count (sorted by strength) ---")
for feat in trip_corr_sorted.index:
    r = trip_corr[feat]
    bar = '█' * int(abs(r) * 30)
    direction = '+' if r > 0 else '-'
    print(f"  {feat:<15}  r = {r:+.4f}  {direction} {bar}")

# Heatmap of the full correlation matrix
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, square=True, ax=ax,
    cbar_kws={"shrink": 0.8}, linewidths=0.5
)
ax.set_title("Figure 6: Correlation Matrix — Demand, Weather & Time Features",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(EXTRACT_DIR / "fig6_correlation_heatmap.png", dpi=120, bbox_inches='tight')
plt.show()
print("  -> Saved: fig6_correlation_heatmap.png")


# =============================================================================
# STEP 8 — DEMAND BY RAIN CATEGORY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: DEMAND BY RAIN CATEGORY")
print("=" * 70)

# rain_category bins precipitation into intensity levels:
#   0 = no rain, 1 = trace, 2 = light, 3 = moderate, 4 = heavy
rain_labels = {0: 'No rain', 1: 'Trace', 2: 'Light', 3: 'Moderate', 4: 'Heavy'}

rain_demand = df.groupby('rain_category')['trip_count'].agg(['mean', 'median', 'count'])
rain_demand.index = [rain_labels.get(int(i), str(i)) for i in rain_demand.index]

print("\n--- Average & Median trip_count by Rain Category ---")
print(rain_demand.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar: average demand per rain level
axes[0].bar(rain_demand.index, rain_demand['mean'], color='steelblue')
axes[0].set_title("Average Demand by Rain Category")
axes[0].set_xlabel("Rain intensity")
axes[0].set_ylabel("Avg trips per zone-hour")

# Scatter sample: raw precipitation vs trip_count
# Sample 20k active rows to avoid overplotting
sample = df[df['trip_count'] > 0].sample(n=min(20000, len(active)), random_state=42)
axes[1].scatter(sample['precipitation'], sample['trip_count'],
                alpha=0.1, s=4, color='steelblue')
axes[1].set_title("Precipitation vs Trip Count\n(20k active zone-hour sample)")
axes[1].set_xlabel("Precipitation (inches)")
axes[1].set_ylabel("trip_count")

plt.suptitle("Figure 7: Rainfall & Demand", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(EXTRACT_DIR / "fig7_rain_demand.png", dpi=120, bbox_inches='tight')
plt.show()
print("  -> Saved: fig7_rain_demand.png")

# Scatter: temperature vs trip_count
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(sample['temperature'], sample['trip_count'],
           alpha=0.1, s=4, color='coral')
ax.set_title("Temperature vs Trip Count\n(20k active zone-hour sample)")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("trip_count")
plt.tight_layout()
plt.savefig(EXTRACT_DIR / "fig8_temperature_demand.png", dpi=120, bbox_inches='tight')
plt.show()
print("  -> Saved: fig8_temperature_demand.png")


# =============================================================================
# STEP 9 — LAG FEATURE ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: LAG FEATURE ANALYSIS")
print("=" * 70)

# Lag features represent demand N hours ago for the same zone.
# They capture how much autocorrelation exists in the time series —
# i.e. does knowing last hour's demand help predict this hour's demand?
lag_cols = ['lag_1', 'lag_2', 'lag_24', 'lag_168']
lag_corr  = df[lag_cols + ['trip_count']].corr()['trip_count'].drop('trip_count')

print("\n--- Correlation of Lag Features with trip_count ---")
lag_desc = {
    'lag_1':   '1 hour ago (same zone)',
    'lag_2':   '2 hours ago (same zone)',
    'lag_24':  '24 hours ago — same time yesterday',
    'lag_168': '168 hours ago — same time last week',
}
for feat, r in lag_corr.items():
    print(f"  {feat:<10}  r = {r:.4f}  ({lag_desc[feat]})")

print("\n  NOTE: High lag correlations confirm strong temporal autocorrelation.")
print("  These will be some of the most important features in the predictive model.")

# Scatter: lag_1 vs trip_count (shows autocorrelation directly)
sample_lag = df[['lag_1', 'trip_count']].dropna().sample(n=20000, random_state=42)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(sample_lag['lag_1'], sample_lag['trip_count'],
           alpha=0.1, s=4, color='mediumpurple')
ax.set_title("lag_1 (demand 1hr ago) vs Current trip_count\n(20k sample)")
ax.set_xlabel("Trips in previous hour (lag_1)")
ax.set_ylabel("Current trip_count")
plt.tight_layout()
plt.savefig(EXTRACT_DIR / "fig9_lag1_scatter.png", dpi=120, bbox_inches='tight')
plt.show()
print("  -> Saved: fig9_lag1_scatter.png")


# =============================================================================
# SUMMARY — KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 70)
print("EXPLORATION COMPLETE — KEY TAKEAWAYS")
print("=" * 70)

top_zone    = zone_demand.idxmax()
top_trips   = zone_demand.max()
peak_hour   = hourly_avg.idxmax()
peak_dow    = dow_avg.idxmax()

print(f"""
  DATASET
    Rows:          {len(df):,}
    Columns:       {df.shape[1]}
    Coverage:      2022–2025, hourly, 263 zones
    Sparsity:      {zero_pct:.0f}% of zone-hours have zero trips

  DEMAND PATTERNS
    Busiest zone:  LocationID {top_zone}  ({top_trips:,} total trips)
    Peak hour:     {peak_hour}:00  (avg {hourly_avg[peak_hour]:.2f} trips/zone-hour)
    Peak day:      {dow_labels[peak_dow]}

  WEATHER–DEMAND CORRELATIONS (Pearson r with trip_count)
""")
for feat in ['precipitation', 'temperature', 'wind_speed', 'visibility']:
    r = trip_corr[feat]
    print(f"    {feat:<15}  r = {r:+.4f}")

print(f"""
  LAG–DEMAND CORRELATIONS (strongest expected model features)
""")
for feat, r in lag_corr.items():
    print(f"    {feat:<10}  r = {r:+.4f}")

print(f"""
  All figures saved to: {EXTRACT_DIR}/
  Next step: build a predictive model (e.g. XGBoost) using weather +
             time-of-day + lag features to forecast hourly zone demand.
""")
