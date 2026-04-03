"""
02_event_scraping.py
=====================
Collects public event data for Manhattan (2022-2025) from multiple sources:

1. NYC Open Data - Film Permits (Socrata API) - 2023-2025 coverage
   ~8,000 Manhattan events including major TV/film shoots, street closures
2. NYC Open Data - Permitted Events (Socrata API) - limited historical data
3. US Federal Holidays

Following Lab 1 methodology: REST API consumption with structured JSON responses.

Output: ../data/manhattan_events.parquet
        ../data/manhattan_events.csv
        ../data/us_holidays.csv

References:
- Lab 1 (Data Ingress): API-based data collection approach
- 2008.00568v1: Used NYC permitted event data as exogenous variable
"""

import pandas as pd
import numpy as np
import requests
import time
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ══════════════════════════════════════════════════════════════════════
# Helper: Socrata API fetcher
# ══════════════════════════════════════════════════════════════════════

def fetch_socrata(endpoint, where_clause, limit=50000):
    """Generic Socrata API paginated fetcher (Lab 1 style)."""
    all_records = []
    offset = 0

    while True:
        params = {
            "$where": where_clause,
            "$limit": limit,
            "$offset": offset,
            "$order": ":id"
        }

        print(f"  Requesting offset={offset}...")
        response = requests.get(endpoint, params=params)

        if response.status_code != 200:
            print(f"  ERROR: {response.status_code} - {response.text[:300]}")
            break

        data = response.json()
        if not data:
            break

        all_records.extend(data)
        print(f"  Got {len(data)} records (total: {len(all_records)})")

        if len(data) < limit:
            break

        offset += limit
        time.sleep(1)

    return all_records


# ══════════════════════════════════════════════════════════════════════
# 1. Film Permits (best historical coverage: 2023-2025)
# ══════════════════════════════════════════════════════════════════════

def fetch_film_permits():
    """Fetch film/TV shoot permits from NYC Open Data.
    These represent significant street-level activity that impacts traffic patterns."""

    print("\n=== Fetching Film Permits (2022-2025, Manhattan) ===")
    endpoint = "https://data.cityofnewyork.us/resource/tg4x-b46p.json"
    where = "borough = 'Manhattan' AND startdatetime >= '2022-01-01' AND startdatetime <= '2025-12-31'"

    raw = fetch_socrata(endpoint, where)

    if not raw:
        return pd.DataFrame()

    records = []
    for r in raw:
        try:
            records.append({
                'event_id': f"FILM_{r.get('eventid', '')}",
                'event_name': f"{r.get('category', 'Film')} - {r.get('subcategoryname', 'Unknown')}",
                'event_type': 'film_permit',
                'event_category': categorise_film(r.get('category', ''), r.get('subcategoryname', '')),
                'start_datetime': pd.to_datetime(r.get('startdatetime')),
                'end_datetime': pd.to_datetime(r.get('enddatetime')),
                'event_location': r.get('parkingheld', ''),
                'event_borough': 'Manhattan',
                'zipcode': r.get('zipcode_s', ''),
            })
        except:
            continue

    df = pd.DataFrame(records)
    df = df.dropna(subset=['start_datetime'])
    print(f"  Parsed: {len(df)} film permits")
    print(f"  Date range: {df['start_datetime'].min()} to {df['start_datetime'].max()}")
    print(f"  Categories:\n{df['event_category'].value_counts()}")
    return df


def categorise_film(category, subcategory):
    """Categorise film permits by impact level."""
    cat = str(category).lower()
    sub = str(subcategory).lower()

    if 'commercial' in cat:
        return 'commercial_shoot'
    elif 'television' in cat:
        return 'tv_production'
    elif 'film' in cat or 'feature' in cat:
        return 'film_production'
    elif 'still' in cat or 'photo' in cat:
        return 'photo_shoot'
    elif 'theater' in cat or 'music' in cat:
        return 'performance'
    else:
        return 'other_filming'


# ══════════════════════════════════════════════════════════════════════
# 2. NYC Permitted Events (limited: mid-2024 onwards)
# ══════════════════════════════════════════════════════════════════════

def fetch_permitted_events():
    """Fetch NYC permitted events (parades, street fairs, sports, etc.)."""

    print("\n=== Fetching NYC Permitted Events (Manhattan) ===")
    endpoint = "https://data.cityofnewyork.us/resource/tvpp-9vvx.json"
    where = "event_borough = 'Manhattan' AND start_date_time >= '2022-01-01' AND start_date_time <= '2025-12-31'"

    raw = fetch_socrata(endpoint, where)

    if not raw:
        return pd.DataFrame()

    records = []
    for r in raw:
        try:
            event_type = r.get('event_type', 'Unknown')
            records.append({
                'event_id': f"EVT_{r.get('event_id', '')}",
                'event_name': r.get('event_name', ''),
                'event_type': 'permitted_event',
                'event_category': categorise_permitted(event_type),
                'start_datetime': pd.to_datetime(r.get('start_date_time')),
                'end_datetime': pd.to_datetime(r.get('end_date_time')),
                'event_location': r.get('event_location', ''),
                'event_borough': 'Manhattan',
                'zipcode': '',
            })
        except:
            continue

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.dropna(subset=['start_datetime'])
        print(f"  Parsed: {len(df)} permitted events")
    else:
        print("  No permitted events found in date range")
    return df


def categorise_permitted(event_type):
    """Map permitted event types to categories."""
    if pd.isna(event_type):
        return 'other_event'
    et = str(event_type).lower()

    if any(w in et for w in ['street', 'block', 'fair', 'festival', 'market']):
        return 'street_event'
    elif any(w in et for w in ['parade', 'march', 'rally', 'protest']):
        return 'parade_rally'
    elif any(w in et for w in ['sport', 'race', 'run', 'marathon']):
        return 'sports_event'
    elif any(w in et for w in ['concert', 'music', 'performance']):
        return 'concert'
    elif any(w in et for w in ['ceremony', 'celebration']):
        return 'ceremony'
    elif any(w in et for w in ['farmers', 'greenmarket']):
        return 'market'
    else:
        return 'other_event'


# ══════════════════════════════════════════════════════════════════════
# 3. US Federal Holidays
# ══════════════════════════════════════════════════════════════════════

def get_us_holidays(start_year=2022, end_year=2025):
    """Get US federal holidays."""
    try:
        import holidays as hol_lib
        us_holidays = hol_lib.US(years=range(start_year, end_year + 1))
        holiday_df = pd.DataFrame([
            {'date': date, 'holiday_name': name}
            for date, name in sorted(us_holidays.items())
        ])
    except ImportError:
        print("  holidays library not found, using manual list...")
        dates = []
        for y in range(start_year, end_year + 1):
            dates.extend([
                (f"{y}-01-01", "New Year's Day"),
                (f"{y}-07-04", "Independence Day"),
                (f"{y}-12-25", "Christmas Day"),
                (f"{y}-11-11", "Veterans Day"),
            ])
            # Approximate floating holidays
            import datetime
            # MLK Day: 3rd Monday of January
            jan1 = datetime.date(y, 1, 1)
            mlk = jan1 + datetime.timedelta(days=(7 - jan1.weekday()) % 7 + 14)
            dates.append((str(mlk), "MLK Day"))
            # Presidents Day: 3rd Monday of February
            feb1 = datetime.date(y, 2, 1)
            pres = feb1 + datetime.timedelta(days=(7 - feb1.weekday()) % 7 + 14)
            dates.append((str(pres), "Presidents Day"))
            # Memorial Day: last Monday of May
            may31 = datetime.date(y, 5, 31)
            mem = may31 - datetime.timedelta(days=(may31.weekday()) % 7)
            dates.append((str(mem), "Memorial Day"))
            # Labor Day: 1st Monday of September
            sep1 = datetime.date(y, 9, 1)
            labor = sep1 + datetime.timedelta(days=(7 - sep1.weekday()) % 7)
            dates.append((str(labor), "Labor Day"))
            # Thanksgiving: 4th Thursday of November
            nov1 = datetime.date(y, 11, 1)
            thanks = nov1 + datetime.timedelta(days=(3 - nov1.weekday()) % 7 + 21)
            dates.append((str(thanks), "Thanksgiving"))

        holiday_df = pd.DataFrame(dates, columns=['date', 'holiday_name'])
        holiday_df['date'] = pd.to_datetime(holiday_df['date']).dt.date

    print(f"  Holidays: {len(holiday_df)} dates")
    return holiday_df


# ══════════════════════════════════════════════════════════════════════
# 4. Zone Mapping
# ══════════════════════════════════════════════════════════════════════

LOCATION_TO_ZONE = {
    'madison square garden': 186, 'msg': 186, 'penn station': 186,
    'times square': 230, 'central park': 43, 'lincoln center': 142,
    'carnegie hall': 230, 'radio city': 161, 'rockefeller': 161,
    'union square': 234, 'washington square': 113, 'battery park': 12,
    'world trade': 261, 'wall street': 261, 'chinatown': 128,
    'little italy': 128, 'soho': 144, 'tribeca': 231,
    'greenwich village': 113, 'east village': 79, 'west village': 158,
    'chelsea': 68, 'midtown': 161, 'harlem': 42,
    'upper east': 236, 'upper west': 238, 'lower east side': 148,
    'gramercy': 164, 'murray hill': 170, 'flatiron': 90,
    'herald square': 164, 'columbus circle': 50, 'bryant park': 161,
    'broadway': 230, 'park ave': 170, 'lexington': 236,
    'madison ave': 164, 'houston': 144, 'canal st': 128,
    '14th st': 234, '23rd st': 68, '34th st': 186,
    '42nd st': 161, '57th st': 50, '72nd st': 238,
    '96th st': 42, '110th st': 42, '125th st': 42,
}

# Manhattan zip code to approximate zone mapping
ZIP_TO_ZONE = {
    '10001': 186, '10002': 148, '10003': 234, '10004': 261, '10005': 261,
    '10006': 261, '10007': 261, '10009': 79, '10010': 164, '10011': 158,
    '10012': 144, '10013': 231, '10014': 158, '10016': 170, '10017': 170,
    '10018': 186, '10019': 161, '10020': 161, '10021': 236, '10022': 161,
    '10023': 238, '10024': 238, '10025': 238, '10026': 42, '10027': 42,
    '10028': 236, '10029': 116, '10030': 42, '10031': 42, '10032': 42,
    '10033': 42, '10034': 42, '10035': 116, '10036': 230, '10037': 42,
    '10038': 261, '10039': 42, '10040': 42, '10044': 194, '10065': 236,
    '10069': 238, '10075': 236, '10103': 161, '10105': 161, '10106': 161,
    '10110': 161, '10111': 161, '10112': 161, '10115': 238, '10119': 186,
    '10128': 236, '10280': 12, '10282': 261,
}


def map_to_zone(row):
    """Map an event to a TLC zone using location text and zip code."""
    # Try location text first
    loc = str(row.get('event_location', '')).lower()
    for keyword, zone in LOCATION_TO_ZONE.items():
        if keyword in loc:
            return zone

    # Try zip code
    zips = str(row.get('zipcode', ''))
    for zc in zips.split(','):
        zc = zc.strip()[:5]
        if zc in ZIP_TO_ZONE:
            return ZIP_TO_ZONE[zc]

    return None


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Fetch from all sources
    film_df = fetch_film_permits()
    events_df = fetch_permitted_events()

    # Combine all event sources
    all_frames = [df for df in [film_df, events_df] if len(df) > 0]

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
    else:
        print("\nWARNING: No events fetched!")
        combined = pd.DataFrame()

    if len(combined) > 0:
        # Map to zones
        combined['zone_id'] = combined.apply(map_to_zone, axis=1)
        mapped = combined['zone_id'].notna().sum()
        print(f"\n=== Combined Results ===")
        print(f"  Total events: {len(combined)}")
        print(f"  Mapped to zones: {mapped} ({100*mapped/len(combined):.1f}%)")
        print(f"  Date range: {combined['start_datetime'].min()} to {combined['start_datetime'].max()}")
        print(f"  By source type:\n{combined['event_type'].value_counts()}")
        print(f"  By category:\n{combined['event_category'].value_counts()}")
        print(f"  By year:\n{combined['start_datetime'].dt.year.value_counts().sort_index()}")

        # Derive extra columns
        combined['start_date'] = combined['start_datetime'].dt.date
        combined['start_hour'] = combined['start_datetime'].dt.hour
        combined['end_hour'] = combined['end_datetime'].dt.hour
        combined['duration_hours'] = (
            (combined['end_datetime'] - combined['start_datetime']).dt.total_seconds() / 3600
        ).clip(0, 720)

        # Save
        out_parquet = os.path.join(DATA_DIR, 'manhattan_events.parquet')
        out_csv = os.path.join(DATA_DIR, 'manhattan_events.csv')
        combined.to_parquet(out_parquet, index=False)
        combined.to_csv(out_csv, index=False)
        print(f"\n  Saved {len(combined)} events to {out_parquet}")

    # Holidays
    holidays_df = get_us_holidays(2022, 2025)
    holidays_df.to_csv(os.path.join(DATA_DIR, 'us_holidays.csv'), index=False)
    print(f"  Saved holidays to us_holidays.csv")

    print("\nDone!")
