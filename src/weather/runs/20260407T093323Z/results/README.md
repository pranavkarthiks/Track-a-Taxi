# Weather Data Exploration

Source dataset: `weather.nc`

Reproduce with: `./.venv/bin/python explore_weather.py --input weather.nc --results-dir runs/20260407T093323Z/results --graphs-dir graphs`

## What This Includes

- Schema and coverage checks
- Missingness, duplicates, and non-finite checks
- Distribution and sparsity analysis
- Interval-level and day-level precipitation summaries
- Entity-level ranking for the wettest zones
- CSV exports for downstream analysis

## Dataset Checks

- Shape: `{'time': 48, 'zone_id': 259}`
- Dimensions: `['time', 'zone_id']`
- Observation key columns: `['time', 'zone_id']`
- Variable: `precipitation_mm_30min` (`float32`)
- Time coverage: `2026-01-01 00:00:00` to `2026-01-01 23:30:00`
- Time deltas in minutes: `[30.0]`
- Records: `12432`
- Duplicate keys: `0`
- Missing values: `0`
- Non-finite values: `0`
- Negative values: `0`
- Zero values: `11899`
- Positive values: `533`
- Wet share: `4.29%`

## Key Findings

- The dataset contains 48 time steps and 259.
- Coverage is 2026-01-01 00:00:00 through 2026-01-01 23:30:00 at [30.0] minute spacing.
- No missing values or duplicate observation keys were found.
- 533 of 12432 observations are non-zero (4.29% wet observations).
- Maximum 30-minute precipitation is 0.7000; mean is 0.0082.
- The wettest entity is zone 84 with total precipitation 1.2000 across 3 wet intervals.
- The file metadata does not provide a reliable precipitation unit (`units=unknown`).

## Distribution Summary

- Min: `0.0000`
- 25th percentile: `0.0000`
- Median: `0.0000`
- 75th percentile: `0.0000`
- 95th percentile: `0.0000`
- 99th percentile: `0.2000`
- Max: `0.7000`
- Mean: `0.0082`
- Std: `0.0452`

## Non-Zero Distribution

- Min non-zero: `0.1000`
- Median non-zero: `0.2000`
- 95th percentile non-zero: `0.4000`
- Max non-zero: `0.7000`

## Sample Interval Summary

| time | count | sum | mean | median | max | std | active_entities | pct_active_entities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-01 00:00:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 00:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 01:00:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 01:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 02:00:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 02:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 03:00:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 03:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 04:00:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 04:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 05:00:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 05:30:00 | 259 | 0.5 | 0.0019000000320374966 | 0.0 | 0.10000000149011612 | 0.013799999840557575 | 5 | 0.0193 |

## Daily Summary

| date | daily_precipitation_total | daily_active_entity_count |
| --- | --- | --- |
| 2026-01-01 | 102.19999694824219 | 533 |

## Top Rain Intervals

| time | count | total_precipitation | mean | median | max | std | active_entities | pct_active_entities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-01 11:30:00 | 259 | 64.80000305175781 | 0.2502000033855438 | 0.20000000298023224 | 0.699999988079071 | 0.12460000067949295 | 255 | 0.9846 |
| 2026-01-01 11:00:00 | 259 | 31.399999618530273 | 0.12120000272989273 | 0.10000000149011612 | 0.30000001192092896 | 0.07450000196695328 | 225 | 0.8687 |
| 2026-01-01 12:00:00 | 259 | 4.699999809265137 | 0.01810000091791153 | 0.0 | 0.20000000298023224 | 0.045099999755620956 | 40 | 0.1544 |
| 2026-01-01 06:00:00 | 259 | 0.699999988079071 | 0.0027000000700354576 | 0.0 | 0.10000000149011612 | 0.016200000420212746 | 7 | 0.027 |
| 2026-01-01 05:30:00 | 259 | 0.5 | 0.0019000000320374966 | 0.0 | 0.10000000149011612 | 0.013799999840557575 | 5 | 0.0193 |
| 2026-01-01 12:30:00 | 259 | 0.10000000149011612 | 0.00039999998989515007 | 0.0 | 0.10000000149011612 | 0.006200000178068876 | 1 | 0.0039 |
| 2026-01-01 00:00:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 00:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 01:00:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-01-01 01:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |

## Top Rainy Entities

| zone_id | observation_count | total_precipitation | mean_precipitation | max_precipitation | wet_intervals | pct_wet_intervals | grid_lat | grid_lon | latitude | longitude |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 84 | 48 | 1.2000000476837158 | 0.02500000037252903 | 0.699999988079071 | 3 | 0.0625 | 40.535 | -74.175 | 40.532 | -74.1739 |
| 5 | 48 | 1.2000000476837158 | 0.02500000037252903 | 0.6000000238418579 | 4 | 0.0833 | 40.555 | -74.185 | 40.5503 | -74.1899 |
| 99 | 48 | 1.2000000476837158 | 0.02500000037252903 | 0.5 | 5 | 0.1042 | 40.575 | -74.185 | 40.5796 | -74.1877 |
| 109 | 48 | 1.100000023841858 | 0.02290000021457672 | 0.6000000238418579 | 3 | 0.0625 | 40.545 | -74.155 | 40.5488 | -74.1527 |
| 44 | 48 | 1.100000023841858 | 0.02290000021457672 | 0.5 | 5 | 0.1042 | 40.525 | -74.225 | 40.5273 | -74.2295 |
| 204 | 48 | 1.100000023841858 | 0.02290000021457672 | 0.5 | 5 | 0.1042 | 40.545 | -74.205 | 40.5407 | -74.207 |
| 110 | 48 | 1.0 | 0.020800000056624413 | 0.699999988079071 | 3 | 0.0625 | 40.545 | -74.125 | 40.5433 | -74.1258 |
| 176 | 48 | 1.0 | 0.020800000056624413 | 0.699999988079071 | 3 | 0.0625 | 40.565 | -74.115 | 40.5621 | -74.1196 |
| 64 | 48 | 0.8999999761581421 | 0.018799999728798866 | 0.6000000238418579 | 4 | 0.0833 | 40.765 | -73.735 | 40.7606 | -73.7314 |
| 118 | 48 | 0.8999999761581421 | 0.018699999898672104 | 0.699999988079071 | 2 | 0.0417 | 40.585 | -74.135 | 40.5856 | -74.1371 |

## Coordinate Outliers

These rows fall outside a typical NYC bounding box and should be treated as lookup-data issues until validated.

| zone_id | observation_count | total_precipitation | mean_precipitation | max_precipitation | wet_intervals | pct_wet_intervals | grid_lat | grid_lon | latitude | longitude |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## CSV Outputs

- `csv/per_time_summary.csv`
- `csv/daily_summary.csv`
- `csv/top_intervals.csv`
- `csv/entity_summary.csv`
- `csv/top_entities.csv`
- `csv/coordinate_outliers.csv`
- `csv/weather_data_flat.csv`

## Figures

![interval_totals.png](./figures/interval_totals.png)
![daily_totals.png](./figures/daily_totals.png)
![non_zero_distribution.png](./figures/non_zero_distribution.png)
![top_entities.png](./figures/top_entities.png)
![zone_total_precipitation_choropleth.png](./figures/zone_total_precipitation_choropleth.png)
![zone_peak_precipitation_choropleth.png](./figures/zone_peak_precipitation_choropleth.png)

## Metadata Notes

- Global attrs: `{'source': 'NOAA MRMS noaa-mrms-pds', 'product': 'CONUS/RadarOnly_QPE_15M_00.00', 'interval_minutes': '30'}`
- Variable attrs: `{'GRIB_paramId': '0', 'GRIB_dataType': 'ra', 'GRIB_numberOfPoints': '24500000', 'GRIB_typeOfLevel': 'heightAboveSea', 'GRIB_stepUnits': '1', 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_uvRelativeToGrid': '0', 'GRIB_NV': '0', 'GRIB_Nx': '7000', 'GRIB_Ny': '3500', 'GRIB_cfName': 'unknown', 'GRIB_cfVarName': 'unknown', 'GRIB_gridDefinitionDescription': 'Latitude/longitude', 'GRIB_iDirectionIncrementInDegrees': '0.01', 'GRIB_iScansNegatively': '0', 'GRIB_jDirectionIncrementInDegrees': '0.01', 'GRIB_jPointsAreConsecutive': '0', 'GRIB_jScansPositively': '0', 'GRIB_latitudeOfFirstGridPointInDegrees': '54.995', 'GRIB_latitudeOfLastGridPointInDegrees': '20.005001', 'GRIB_longitudeOfFirstGridPointInDegrees': '230.005', 'GRIB_longitudeOfLastGridPointInDegrees': '299.994998', 'GRIB_missingValue': '3.4028234663852886e+38', 'GRIB_name': 'unknown', 'GRIB_shortName': 'unknown', 'GRIB_units': 'unknown', 'long_name': '30-minute precipitation accumulation', 'units': 'unknown', 'standard_name': 'unknown', 'source_product': 'NOAA MRMS RadarOnly_QPE_15M_00.00'}`
