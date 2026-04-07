# Weather Data Exploration

Source dataset: `weather.nc`

Reproduce with: `./.venv/bin/python explore_weather.py --input weather.nc --results-dir runs/20260407T121644Z/results --graphs-dir graphs`

## What This Includes

- Schema and coverage checks
- Missingness, duplicates, and non-finite checks
- Distribution and sparsity analysis
- Interval-level and day-level precipitation summaries
- Entity-level ranking for the wettest zones
- CSV exports for downstream analysis

## Dataset Checks

- Shape: `{'time': 24, 'zone_id': 259}`
- Dimensions: `['time', 'zone_id']`
- Observation key columns: `['time', 'zone_id']`
- Variable: `precipitation_mm` (`float32`)
- Time coverage: `2026-03-06 08:00:00` to `2026-03-06 09:55:00`
- Time deltas in minutes: `[5.0]`
- Records: `6216`
- Duplicate keys: `0`
- Missing values: `0`
- Non-finite values: `0`
- Negative values: `0`
- Zero values: `6188`
- Positive values: `28`
- Wet share: `0.45%`

## Key Findings

- The dataset contains 24 time steps and 259.
- Coverage is 2026-03-06 08:00:00 through 2026-03-06 09:55:00 at [5.0] minute spacing.
- No missing values or duplicate observation keys were found.
- 28 of 6216 observations are non-zero (0.45% wet observations).
- Maximum 5-minute precipitation is 0.0283; mean is 0.0001.
- The wettest entity is zone 134 with total precipitation 0.0400 across 2 wet intervals.

## Distribution Summary

- Min: `0.0000`
- 25th percentile: `0.0000`
- Median: `0.0000`
- 75th percentile: `0.0000`
- 95th percentile: `0.0000`
- 99th percentile: `0.0000`
- Max: `0.0283`
- Mean: `0.0001`
- Std: `0.0011`

## Non-Zero Distribution

- Min non-zero: `0.0050`
- Median non-zero: `0.0100`
- 95th percentile non-zero: `0.0250`
- Max non-zero: `0.0283`

## Sample Interval Summary

| time | count | sum | mean | median | max | std | active_entities | pct_active_entities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-06 08:00:00 | 259 | 0.13830000162124634 | 0.0005000000237487257 | 0.0 | 0.028300000354647636 | 0.0032999999821186066 | 8 | 0.0309 |
| 2026-03-06 08:05:00 | 259 | 0.13169999420642853 | 0.0005000000237487257 | 0.0 | 0.019999999552965164 | 0.0024999999441206455 | 12 | 0.0463 |
| 2026-03-06 08:10:00 | 259 | 0.10999999940395355 | 0.00039999998989515007 | 0.0 | 0.02500000037252903 | 0.002899999963119626 | 6 | 0.0232 |
| 2026-03-06 08:15:00 | 259 | 0.009999999776482582 | 0.0 | 0.0 | 0.004999999888241291 | 0.00039999998989515007 | 2 | 0.0077 |
| 2026-03-06 08:20:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:25:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:35:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:40:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:45:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:50:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:55:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |

## Daily Summary

| date | daily_precipitation_total | daily_active_entity_count |
| --- | --- | --- |
| 2026-03-06 | 0.38999998569488525 | 28 |

## Top Rain Intervals

| time | count | total_precipitation | mean | median | max | std | active_entities | pct_active_entities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-06 08:00:00 | 259 | 0.13830000162124634 | 0.0005000000237487257 | 0.0 | 0.028300000354647636 | 0.0032999999821186066 | 8 | 0.0309 |
| 2026-03-06 08:05:00 | 259 | 0.13169999420642853 | 0.0005000000237487257 | 0.0 | 0.019999999552965164 | 0.0024999999441206455 | 12 | 0.0463 |
| 2026-03-06 08:10:00 | 259 | 0.10999999940395355 | 0.00039999998989515007 | 0.0 | 0.02500000037252903 | 0.002899999963119626 | 6 | 0.0232 |
| 2026-03-06 08:15:00 | 259 | 0.009999999776482582 | 0.0 | 0.0 | 0.004999999888241291 | 0.00039999998989515007 | 2 | 0.0077 |
| 2026-03-06 08:20:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:25:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:35:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:40:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-06 08:45:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |

## Top Rainy Entities

| zone_id | observation_count | total_precipitation | mean_precipitation | max_precipitation | wet_intervals | pct_wet_intervals | grid_lat | grid_lon | latitude | longitude |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 134 | 24 | 0.03999999910593033 | 0.0017000000225380063 | 0.019999999552965164 | 2 | 0.0833 | 40.705 | -73.825 | 40.7088 | -73.83 |
| 28 | 24 | 0.0333000011742115 | 0.00139999995008111 | 0.019999999552965164 | 2 | 0.0833 | 40.715 | -73.805 | 40.7109 | -73.8073 |
| 189 | 24 | 0.0333000011742115 | 0.00139999995008111 | 0.028300000354647636 | 2 | 0.0833 | 40.675 | -73.965 | 40.6772 | -73.9683 |
| 40 | 24 | 0.029999999329447746 | 0.0013000000035390258 | 0.02500000037252903 | 2 | 0.0833 | 40.675 | -73.995 | 40.6785 | -73.9958 |
| 106 | 24 | 0.029999999329447746 | 0.0013000000035390258 | 0.02500000037252903 | 2 | 0.0833 | 40.675 | -73.995 | 40.6734 | -73.9918 |
| 205 | 24 | 0.029999999329447746 | 0.0013000000035390258 | 0.02500000037252903 | 2 | 0.0833 | 40.695 | -73.765 | 40.6922 | -73.7626 |
| 131 | 24 | 0.029999999329447746 | 0.0013000000035390258 | 0.019999999552965164 | 2 | 0.0833 | 40.725 | -73.775 | 40.7204 | -73.7714 |
| 98 | 24 | 0.019999999552965164 | 0.0007999999797903001 | 0.019999999552965164 | 1 | 0.0417 | 40.735 | -73.775 | 40.7338 | -73.7796 |
| 135 | 24 | 0.019999999552965164 | 0.0007999999797903001 | 0.019999999552965164 | 1 | 0.0417 | 40.725 | -73.825 | 40.7294 | -73.824 |
| 160 | 24 | 0.019999999552965164 | 0.0007999999797903001 | 0.019999999552965164 | 1 | 0.0417 | 40.715 | -73.885 | 40.7185 | -73.8807 |

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

- Global attrs: `{'source': 'NOAA MRMS noaa-mrms-pds', 'product': 'CONUS/PrecipRate_00.00', 'source_mode': 'precip_rate', 'interval_minutes': '5'}`
- Variable attrs: `{'GRIB_paramId': '0', 'GRIB_dataType': 'ra', 'GRIB_numberOfPoints': '24500000', 'GRIB_typeOfLevel': 'heightAboveSea', 'GRIB_stepUnits': '1', 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_uvRelativeToGrid': '0', 'GRIB_NV': '0', 'GRIB_Nx': '7000', 'GRIB_Ny': '3500', 'GRIB_cfName': 'unknown', 'GRIB_cfVarName': 'unknown', 'GRIB_gridDefinitionDescription': 'Latitude/longitude', 'GRIB_iDirectionIncrementInDegrees': '0.01', 'GRIB_iScansNegatively': '0', 'GRIB_jDirectionIncrementInDegrees': '0.01', 'GRIB_jPointsAreConsecutive': '0', 'GRIB_jScansPositively': '0', 'GRIB_latitudeOfFirstGridPointInDegrees': '54.995', 'GRIB_latitudeOfLastGridPointInDegrees': '20.005001', 'GRIB_longitudeOfFirstGridPointInDegrees': '230.005', 'GRIB_longitudeOfLastGridPointInDegrees': '299.994998', 'GRIB_missingValue': '3.4028234663852886e+38', 'GRIB_name': 'unknown', 'GRIB_shortName': 'unknown', 'GRIB_units': 'unknown', 'long_name': '5-minute precipitation accumulation', 'units': 'mm', 'standard_name': 'unknown', 'source_product': 'NOAA MRMS PrecipRate'}`
