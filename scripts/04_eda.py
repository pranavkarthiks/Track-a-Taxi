"""
04_eda.py
==========
Exploratory Data Analysis for Manhattan taxi demand.
Produces visualisations using Plotly (per Lab 5) and statistical summaries.
Outputs saved to ../outputs/figures/

Follows Lab 4 methodology: descriptive statistics, outlier analysis,
correlation measures. Follows Lab 5: interactive Plotly visualisations.

References:
- Lab 4: Exploration, descriptive stats, quartiles, outlier detection
- Lab 5: Plotly visualisations
- Sample report: correlation heatmaps, boxplots, geographic maps
"""

import pandas as pd
import numpy as np
import os

# Visualisation imports (Lab 5 style)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Statistical imports (Lab 4 style)
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(os.path.join(DATA_DIR, 'manhattan_features.parquet'))
print(f"  Shape: {df.shape}")

# ══════════════════════════════════════════════════════════════════════
# 1. Descriptive Statistics (Lab 4 style)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. DESCRIPTIVE STATISTICS")
print("="*60)

# Central tendency and spread
key_vars = ['trip_count', 'temperature', 'precipitation', 'wind_speed',
            'visibility', 'event_count']
desc = df[key_vars].describe()
print(desc.to_string())

# Skewness and kurtosis (Lab 4: scipy.stats)
print("\nSkewness and Kurtosis:")
for col in key_vars:
    skew = stats.skew(df[col].dropna())
    kurt = stats.kurtosis(df[col].dropna())
    print(f"  {col}: skew={skew:.3f}, kurtosis={kurt:.3f}")

# Save stats
desc.to_csv(os.path.join(FIG_DIR, 'descriptive_stats.csv'))

# ══════════════════════════════════════════════════════════════════════
# 2. Trip Count Distribution (Sample report style: boxplots)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. DEMAND DISTRIBUTION")
print("="*60)

# Overall distribution
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Trip Count Distribution', 'Trip Count by Zone (Top 15)'])

# Histogram
fig.add_trace(
    go.Histogram(x=df['trip_count'], nbinsx=100, name='All zones',
                 marker_color='steelblue'),
    row=1, col=1
)

# Top 15 zones boxplot
top_zones = df.groupby('PULocationID')['trip_count'].mean().nlargest(15).index
zone_data = df[df['PULocationID'].isin(top_zones)]
for zone in top_zones:
    zd = zone_data[zone_data['PULocationID'] == zone]['trip_count']
    fig.add_trace(
        go.Box(y=zd, name=f'Zone {zone}', showlegend=False),
        row=1, col=2
    )

fig.update_layout(height=500, width=1100, title_text="Trip Count Distribution")
fig.write_html(os.path.join(FIG_DIR, 'fig1_demand_distribution.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig1_demand_distribution.png'), scale=2)
print("  Saved fig1_demand_distribution")

# ══════════════════════════════════════════════════════════════════════
# 3. Temporal Patterns
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. TEMPORAL PATTERNS")
print("="*60)

# Hourly demand: weekday vs weekend
hourly = df.groupby(['hour', 'is_weekend'])['trip_count'].mean().reset_index()
hourly['day_type'] = hourly['is_weekend'].map({True: 'Weekend', False: 'Weekday'})

fig = px.line(hourly, x='hour', y='trip_count', color='day_type',
              title='Average Hourly Taxi Demand: Weekday vs Weekend',
              labels={'hour': 'Hour of Day', 'trip_count': 'Mean Trip Count',
                      'day_type': 'Day Type'},
              color_discrete_map={'Weekday': 'steelblue', 'Weekend': 'coral'})
fig.update_layout(width=800, height=450)
fig.write_html(os.path.join(FIG_DIR, 'fig2_hourly_patterns.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig2_hourly_patterns.png'), scale=2)
print("  Saved fig2_hourly_patterns")

# Day of week pattern
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow = df.groupby('day_of_week')['trip_count'].mean().reset_index()
dow['day_name'] = dow['day_of_week'].map(dict(enumerate(dow_labels)))

fig = px.bar(dow, x='day_name', y='trip_count',
             title='Average Daily Taxi Demand by Day of Week',
             labels={'day_name': 'Day', 'trip_count': 'Mean Trip Count'},
             color='trip_count', color_continuous_scale='Blues')
fig.update_layout(width=700, height=400)
fig.write_html(os.path.join(FIG_DIR, 'fig3_daily_patterns.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig3_daily_patterns.png'), scale=2)
print("  Saved fig3_daily_patterns")

# Monthly trend
monthly = df.groupby(['year', 'month'])['trip_count'].mean().reset_index()
monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))

fig = px.line(monthly, x='date', y='trip_count',
              title='Monthly Average Taxi Demand (2022-2025)',
              labels={'date': 'Date', 'trip_count': 'Mean Trip Count'})
fig.update_layout(width=900, height=400)
fig.write_html(os.path.join(FIG_DIR, 'fig4_monthly_trend.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig4_monthly_trend.png'), scale=2)
print("  Saved fig4_monthly_trend")

# ══════════════════════════════════════════════════════════════════════
# 4. Spatial Demand Heatmap
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. SPATIAL PATTERNS")
print("="*60)

# Zone-level average demand
zone_demand = df.groupby('PULocationID')['trip_count'].mean().reset_index()
zone_demand.columns = ['zone', 'mean_demand']
zone_demand = zone_demand.sort_values('mean_demand', ascending=False)

print("  Top 10 demand zones:")
for _, row in zone_demand.head(10).iterrows():
    print(f"    Zone {int(row['zone'])}: {row['mean_demand']:.1f} trips/hour")

fig = px.bar(zone_demand.head(20), x='zone', y='mean_demand',
             title='Top 20 Manhattan Zones by Average Hourly Demand',
             labels={'zone': 'Zone ID', 'mean_demand': 'Mean Trips/Hour'},
             color='mean_demand', color_continuous_scale='Viridis')
fig.update_layout(width=900, height=450)
fig.write_html(os.path.join(FIG_DIR, 'fig5_zone_demand.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig5_zone_demand.png'), scale=2)
print("  Saved fig5_zone_demand")

# Zone-hour heatmap (top 15 zones)
top15 = zone_demand.head(15)['zone'].values
zh = df[df['PULocationID'].isin(top15)].groupby(['PULocationID', 'hour'])['trip_count'].mean().reset_index()
zh_pivot = zh.pivot(index='PULocationID', columns='hour', values='trip_count')

fig = px.imshow(zh_pivot,
                title='Hourly Demand Heatmap (Top 15 Zones)',
                labels={'x': 'Hour of Day', 'y': 'Zone ID', 'color': 'Mean Trips'},
                color_continuous_scale='YlOrRd',
                aspect='auto')
fig.update_layout(width=900, height=500)
fig.write_html(os.path.join(FIG_DIR, 'fig6_zone_hour_heatmap.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig6_zone_hour_heatmap.png'), scale=2)
print("  Saved fig6_zone_hour_heatmap")

# ══════════════════════════════════════════════════════════════════════
# 5. Event Impact Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. EVENT IMPACT ANALYSIS")
print("="*60)

# Compare demand: event vs no-event (2023-2025 only, where we have event data)
df_evt_period = df[df['year'] >= 2023].copy()

event_comparison = df_evt_period.groupby('has_event')['trip_count'].agg(['mean', 'median', 'std', 'count'])
print("\n  Demand comparison (2023-2025):")
print(f"    No event: mean={event_comparison.loc[False, 'mean']:.1f}, "
      f"median={event_comparison.loc[False, 'median']:.0f}, "
      f"n={event_comparison.loc[False, 'count']:,}")
if True in event_comparison.index:
    print(f"    Event:    mean={event_comparison.loc[True, 'mean']:.1f}, "
          f"median={event_comparison.loc[True, 'median']:.0f}, "
          f"n={event_comparison.loc[True, 'count']:,}")

    # Statistical test (H1)
    no_event = df_evt_period[~df_evt_period['has_event']]['trip_count']
    with_event = df_evt_period[df_evt_period['has_event']]['trip_count']
    t_stat, p_value = stats.ttest_ind(with_event, no_event, equal_var=False)
    u_stat, u_pvalue = stats.mannwhitneyu(with_event, no_event, alternative='greater')
    print(f"\n  Welch's t-test: t={t_stat:.3f}, p={p_value:.2e}")
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_pvalue:.2e}")

# Boxplot: event vs no event
fig = px.box(df_evt_period, x='has_event', y='trip_count',
             title='Trip Count Distribution: Event vs No Event (2023-2025)',
             labels={'has_event': 'Event Present', 'trip_count': 'Trip Count'},
             color='has_event',
             color_discrete_map={True: 'coral', False: 'steelblue'})
fig.update_layout(width=600, height=450, yaxis_range=[0, 300])
fig.write_html(os.path.join(FIG_DIR, 'fig7_event_vs_noevent.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig7_event_vs_noevent.png'), scale=2)
print("  Saved fig7_event_vs_noevent")

# Event impact by zone (which zones are most event-sensitive?)
if True in event_comparison.index:
    zone_event_impact = []
    for zone in df_evt_period['PULocationID'].unique():
        zd = df_evt_period[df_evt_period['PULocationID'] == zone]
        no_evt = zd[~zd['has_event']]['trip_count'].mean()
        evt = zd[zd['has_event']]['trip_count'].mean()
        n_evt = zd['has_event'].sum()
        if n_evt > 0 and no_evt > 0:
            pct_change = 100 * (evt - no_evt) / no_evt
            zone_event_impact.append({
                'zone': zone, 'no_event_mean': no_evt,
                'event_mean': evt, 'pct_change': pct_change,
                'event_hours': n_evt
            })

    impact_df = pd.DataFrame(zone_event_impact).sort_values('pct_change', ascending=False)
    print("\n  Top 10 event-sensitive zones (% demand increase):")
    for _, row in impact_df.head(10).iterrows():
        print(f"    Zone {int(row['zone'])}: +{row['pct_change']:.1f}% "
              f"({row['no_event_mean']:.0f} -> {row['event_mean']:.0f} trips/hr, "
              f"n_events={int(row['event_hours'])})")

    fig = px.bar(impact_df.head(20), x='zone', y='pct_change',
                 title='Top 20 Zones: % Demand Change During Events',
                 labels={'zone': 'Zone ID', 'pct_change': '% Change in Demand'},
                 color='pct_change', color_continuous_scale='RdYlGn')
    fig.update_layout(width=900, height=450)
    fig.write_html(os.path.join(FIG_DIR, 'fig8_zone_event_sensitivity.html'))
    fig.write_image(os.path.join(FIG_DIR, 'fig8_zone_event_sensitivity.png'), scale=2)
    print("  Saved fig8_zone_event_sensitivity")

# ══════════════════════════════════════════════════════════════════════
# 6. Weather Impact
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. WEATHER IMPACT")
print("="*60)

# Temperature vs demand
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Temperature vs Demand', 'Precipitation vs Demand'])

# Bin temperature and get mean demand
temp_bins = pd.cut(df['temperature'], bins=20)
temp_demand = df.groupby(temp_bins, observed=True)['trip_count'].mean().reset_index()
temp_demand['temperature'] = temp_demand['temperature'].astype(str)

fig.add_trace(
    go.Scatter(x=temp_demand['temperature'], y=temp_demand['trip_count'],
               mode='markers+lines', name='Temperature',
               marker=dict(color='firebrick')),
    row=1, col=1
)

# Rain vs demand
rain_demand = df.groupby('rain_category')['trip_count'].mean().reset_index()
rain_labels = {0: 'None', 1: 'Light', 2: 'Moderate', 3: 'Heavy'}
rain_demand['label'] = rain_demand['rain_category'].map(rain_labels)

fig.add_trace(
    go.Bar(x=rain_demand['label'], y=rain_demand['trip_count'],
           name='Precipitation', marker_color='steelblue'),
    row=1, col=2
)

fig.update_layout(height=400, width=1000, title_text="Weather Impact on Demand",
                  showlegend=False)
fig.write_html(os.path.join(FIG_DIR, 'fig9_weather_impact.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig9_weather_impact.png'), scale=2)
print("  Saved fig9_weather_impact")

# Weather-Event interaction (H2)
if True in event_comparison.index:
    df_evt_period['is_rainy'] = df_evt_period['precipitation'] > 0
    interaction = df_evt_period.groupby(['has_event', 'is_rainy'])['trip_count'].mean().reset_index()
    interaction['condition'] = interaction.apply(
        lambda r: f"{'Event' if r['has_event'] else 'No Event'}, {'Rain' if r['is_rainy'] else 'No Rain'}", axis=1)

    fig = px.bar(interaction, x='condition', y='trip_count',
                 title='Weather-Event Interaction on Demand',
                 labels={'condition': 'Condition', 'trip_count': 'Mean Trip Count'},
                 color='condition')
    fig.update_layout(width=700, height=400, showlegend=False)
    fig.write_html(os.path.join(FIG_DIR, 'fig10_weather_event_interaction.html'))
    fig.write_image(os.path.join(FIG_DIR, 'fig10_weather_event_interaction.png'), scale=2)
    print("  Saved fig10_weather_event_interaction")

# ══════════════════════════════════════════════════════════════════════
# 7. Correlation Analysis (Sample report style)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7. CORRELATION ANALYSIS")
print("="*60)

# Select numeric features for correlation
corr_cols = ['trip_count', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
             'precipitation', 'temperature', 'wind_speed', 'visibility',
             'event_count', 'has_event', 'is_holiday',
             'lag_1', 'lag_24', 'lag_168', 'zone_mean_demand']

# Convert booleans to int for correlation
corr_df = df[corr_cols].copy()
for col in corr_df.select_dtypes(include='bool').columns:
    corr_df[col] = corr_df[col].astype(int)

corr_matrix = corr_df.corr()

# Print correlations with trip_count
print("\n  Correlations with trip_count:")
trip_corr = corr_matrix['trip_count'].drop('trip_count').sort_values(key=abs, ascending=False)
for feat, val in trip_corr.items():
    print(f"    {feat:25s}: {val:+.4f}")

# Heatmap (sample report Fig 4 style)
fig = px.imshow(corr_matrix,
                title='Feature Correlation Matrix',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                aspect='auto')
fig.update_layout(width=900, height=800)
fig.write_html(os.path.join(FIG_DIR, 'fig11_correlation_heatmap.html'))
fig.write_image(os.path.join(FIG_DIR, 'fig11_correlation_heatmap.png'), scale=2)
print("  Saved fig11_correlation_heatmap")

# ══════════════════════════════════════════════════════════════════════
# 8. Holiday Impact
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("8. HOLIDAY IMPACT")
print("="*60)

hol_comparison = df.groupby('is_holiday')['trip_count'].agg(['mean', 'median'])
print(f"  Non-holiday: mean={hol_comparison.loc[False, 'mean']:.1f}")
print(f"  Holiday:     mean={hol_comparison.loc[True, 'mean']:.1f}")
pct = 100 * (hol_comparison.loc[True, 'mean'] - hol_comparison.loc[False, 'mean']) / hol_comparison.loc[False, 'mean']
print(f"  Change: {pct:+.1f}%")

print("\n" + "="*60)
print("EDA COMPLETE - All figures saved to outputs/figures/")
print("="*60)
