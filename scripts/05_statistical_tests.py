"""
05_statistical_tests.py
========================
Formal hypothesis testing for the research questions.

H1: Events significantly affect taxi demand at zone level
H2: Weather-event interaction effects
H3: (Tested in model scripts via paired comparison)
H4: Spatial heterogeneity of event effects

Output: ../outputs/statistical_results.txt

References:
- Lab 4: scipy.stats for statistical testing
- Sample report: correlation measures (PMCC), outlier analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

results = []

def log(msg):
    print(msg)
    results.append(msg)

# ── Load Data ──────────────────────────────────────────────────────────
df = pd.read_parquet(os.path.join(DATA_DIR, 'manhattan_features.parquet'))
# Focus on 2023-2025 where we have event data
df = df[df['year'] >= 2023].copy()
log(f"Data: {len(df):,} rows, 2023-2025, {df['PULocationID'].nunique()} zones")

# ══════════════════════════════════════════════════════════════════════
# H1: Event-Demand Correlation
# ══════════════════════════════════════════════════════════════════════
log("\n" + "="*70)
log("HYPOTHESIS 1: Public events significantly affect taxi demand")
log("="*70)

no_event = df[~df['has_event']]['trip_count']
with_event = df[df['has_event']]['trip_count']

log(f"\n  No event: n={len(no_event):,}, mean={no_event.mean():.2f}, std={no_event.std():.2f}")
log(f"  Event:    n={len(with_event):,}, mean={with_event.mean():.2f}, std={with_event.std():.2f}")
log(f"  Difference: +{with_event.mean() - no_event.mean():.2f} trips/hr "
    f"(+{100*(with_event.mean()/no_event.mean()-1):.1f}%)")

# Welch's t-test (unequal variances)
t_stat, p_value = stats.ttest_ind(with_event, no_event, equal_var=False)
log(f"\n  Welch's t-test: t={t_stat:.4f}, p={p_value:.2e}")

# Mann-Whitney U (non-parametric, one-sided: events > no events)
u_stat, u_pvalue = stats.mannwhitneyu(with_event, no_event, alternative='greater')
log(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_pvalue:.2e}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((no_event.std()**2 + with_event.std()**2) / 2)
cohens_d = (with_event.mean() - no_event.mean()) / pooled_std
log(f"  Cohen's d: {cohens_d:.4f}")

if p_value < 0.05:
    log("\n  >> REJECT H0: Events significantly affect demand (p < 0.05)")
else:
    log("\n  >> FAIL TO REJECT H0")

# ══════════════════════════════════════════════════════════════════════
# H1 (Controlled): Control for time-of-day and day-of-week
# ══════════════════════════════════════════════════════════════════════
log("\n" + "-"*70)
log("H1 CONTROLLED: Controlling for hour and day-of-week")
log("-"*70)

# Match event and non-event observations by hour and day_of_week
significant_zones = 0
total_zones_tested = 0

for zone in sorted(df['PULocationID'].unique()):
    zd = df[df['PULocationID'] == zone]
    n_evt = zd['has_event'].sum()

    if n_evt < 50:  # need sufficient event observations
        continue

    total_zones_tested += 1

    # For each (hour, day_of_week) combo, compare event vs non-event
    residuals_event = []
    residuals_no_event = []

    for (h, dow), group in zd.groupby(['hour', 'day_of_week']):
        baseline = group[~group['has_event']]['trip_count'].mean()
        if np.isnan(baseline):
            continue
        for _, row in group.iterrows():
            residual = row['trip_count'] - baseline
            if row['has_event']:
                residuals_event.append(residual)
            else:
                residuals_no_event.append(residual)

    if len(residuals_event) > 10:
        _, p = stats.mannwhitneyu(residuals_event, residuals_no_event, alternative='greater')
        if p < 0.05:
            significant_zones += 1

log(f"\n  Zones tested: {total_zones_tested}")
log(f"  Zones with significant event effect (p<0.05): {significant_zones} "
    f"({100*significant_zones/max(total_zones_tested,1):.0f}%)")

# ══════════════════════════════════════════════════════════════════════
# H2: Weather-Event Interaction
# ══════════════════════════════════════════════════════════════════════
log("\n" + "="*70)
log("HYPOTHESIS 2: Weather-Event interaction")
log("="*70)

from sklearn.linear_model import LinearRegression

# Simple OLS with interaction term
df['is_rainy'] = (df['precipitation'] > 0).astype(int)
df['has_event_int'] = df['has_event'].astype(int)
df['event_x_rain'] = df['has_event_int'] * df['is_rainy']

X = df[['has_event_int', 'is_rainy', 'event_x_rain', 'hour',
        'day_of_week', 'temperature']].values
y = df['trip_count'].values

# Use statsmodels for coefficient p-values
try:
    import statsmodels.api as sm
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    log(f"\n  OLS Regression with interaction term:")
    log(f"  {'Variable':<20s} {'Coef':>10s} {'Std Err':>10s} {'t':>10s} {'P>|t|':>10s}")
    log(f"  {'-'*60}")
    var_names = ['const', 'has_event', 'is_rainy', 'event_x_rain',
                 'hour', 'day_of_week', 'temperature']
    for name, coef, se, t, p in zip(var_names, model.params, model.bse, model.tvalues, model.pvalues):
        log(f"  {name:<20s} {coef:>10.3f} {se:>10.3f} {t:>10.3f} {p:>10.2e}")

    log(f"\n  R-squared: {model.rsquared:.4f}")
    interaction_p = model.pvalues[3]  # event_x_rain
    if interaction_p < 0.05:
        log(f"  >> REJECT H0: Significant weather-event interaction (p={interaction_p:.2e})")
    else:
        log(f"  >> FAIL TO REJECT H0: No significant interaction (p={interaction_p:.2e})")
except ImportError:
    log("  statsmodels not available, skipping detailed regression")

# ══════════════════════════════════════════════════════════════════════
# H4: Spatial Heterogeneity
# ══════════════════════════════════════════════════════════════════════
log("\n" + "="*70)
log("HYPOTHESIS 4: Spatial heterogeneity of event effects")
log("="*70)

zone_effects = []
for zone in sorted(df['PULocationID'].unique()):
    zd = df[df['PULocationID'] == zone]
    n_evt = zd['has_event'].sum()
    if n_evt < 50:
        continue

    evt_mean = zd[zd['has_event']]['trip_count'].mean()
    no_evt_mean = zd[~zd['has_event']]['trip_count'].mean()

    if no_evt_mean > 0:
        pct_change = 100 * (evt_mean - no_evt_mean) / no_evt_mean
    else:
        pct_change = 0

    zone_effects.append({
        'zone': zone,
        'pct_change': pct_change,
        'abs_change': evt_mean - no_evt_mean,
        'n_events': n_evt
    })

effects_df = pd.DataFrame(zone_effects)

log(f"\n  Zone-level event effect statistics:")
log(f"  Mean % change: {effects_df['pct_change'].mean():.1f}%")
log(f"  Std % change:  {effects_df['pct_change'].std():.1f}%")
log(f"  Min % change:  {effects_df['pct_change'].min():.1f}% (Zone {effects_df.loc[effects_df['pct_change'].idxmin(), 'zone']})")
log(f"  Max % change:  {effects_df['pct_change'].max():.1f}% (Zone {effects_df.loc[effects_df['pct_change'].idxmax(), 'zone']})")

# Coefficient of variation
cv = effects_df['pct_change'].std() / abs(effects_df['pct_change'].mean())
log(f"  Coefficient of variation: {cv:.3f}")

# Levene's test: are the variances of event effects different across zones?
# Group zones into high/medium/low event exposure
effects_df['event_exposure'] = pd.qcut(effects_df['n_events'], q=3, labels=['low', 'med', 'high'])
groups = [g['pct_change'].values for _, g in effects_df.groupby('event_exposure')]
if len(groups) >= 2:
    levene_stat, levene_p = stats.levene(*groups)
    log(f"\n  Levene's test for heterogeneity: F={levene_stat:.3f}, p={levene_p:.4f}")

kruskal_stat, kruskal_p = stats.kruskal(*groups)
log(f"  Kruskal-Wallis test: H={kruskal_stat:.3f}, p={kruskal_p:.4f}")

if kruskal_p < 0.05:
    log("  >> REJECT H0: Event effects are spatially heterogeneous")
else:
    log("  >> FAIL TO REJECT H0: Cannot confirm spatial heterogeneity")

# ── Save Results ───────────────────────────────────────────────────────
output_file = os.path.join(OUT_DIR, 'statistical_results.txt')
with open(output_file, 'w') as f:
    f.write('\n'.join(results))
print(f"\nResults saved to {output_file}")
