# Weather Data Exploration

Source dataset: `weather.nc`

Reproduce with: `./.venv/bin/python explore_weather.py --input weather.nc --results-dir runs/20260407T121251Z/results --graphs-dir graphs`

## What This Includes

- Schema and coverage checks
- Missingness, duplicates, and non-finite checks
- Distribution and sparsity analysis
- Interval-level and day-level precipitation summaries
- Entity-level ranking for the wettest zones
- CSV exports for downstream analysis

## Dataset Checks

- Shape: `{'time': 36, 'zone_id': 259}`
- Dimensions: `['time', 'zone_id']`
- Observation key columns: `['time', 'zone_id']`
- Variable: `precipitation_mm` (`float32`)
- Time coverage: `2026-03-01 07:00:00` to `2026-03-01 09:55:00`
- Time deltas in minutes: `[5.0]`
- Records: `9324`
- Duplicate keys: `0`
- Missing values: `0`
- Non-finite values: `0`
- Negative values: `0`
- Zero values: `9324`
- Positive values: `0`
- Wet share: `0.00%`

## Key Findings

- The dataset contains 36 time steps and 259.
- Coverage is 2026-03-01 07:00:00 through 2026-03-01 09:55:00 at [5.0] minute spacing.
- No missing values or duplicate observation keys were found.
- 0 of 9324 observations are non-zero (0.00% wet observations).
- Maximum 5-minute precipitation is 0.0000; mean is 0.0000.
- The wettest entity is zone 2 with total precipitation 0.0000 across 0 wet intervals.

## Distribution Summary

- Min: `0.0000`
- 25th percentile: `0.0000`
- Median: `0.0000`
- 75th percentile: `0.0000`
- 95th percentile: `0.0000`
- 99th percentile: `0.0000`
- Max: `0.0000`
- Mean: `0.0000`
- Std: `0.0000`

## Sample Interval Summary

| time | count | sum | mean | median | max | std | active_entities | pct_active_entities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-01 07:00:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:05:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:10:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:15:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:20:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:25:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:35:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:40:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:45:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:50:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:55:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |

## Daily Summary

| date | daily_precipitation_total | daily_active_entity_count |
| --- | --- | --- |
| 2026-03-01 | 0.0 | 0 |

## Top Rain Intervals

| time | count | total_precipitation | mean | median | max | std | active_entities | pct_active_entities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-01 07:00:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:05:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:10:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:15:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:20:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:25:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:30:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:35:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:40:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| 2026-03-01 07:45:00 | 259 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |

## Top Rainy Entities

| zone_id | observation_count | total_precipitation | mean_precipitation | max_precipitation | wet_intervals | pct_wet_intervals | grid_lat | grid_lon | latitude | longitude |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 36 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 40.625 | -73.825 | 40.6257 | -73.8261 |
| 3 | 36 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 40.865 | -73.845 | 40.8659 | -73.8495 |
| 4 | 36 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 40.725 | -73.975 | 40.7242 | -73.977 |
| 5 | 36 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 40.555 | -74.185 | 40.5503 | -74.1899 |
| 6 | 36 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 40.595 | -74.065 | 40.5991 | -74.0678 |
| 7 | 36 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 40.765 | -73.925 | 40.7611 | -73.9215 |
| 8 | 36 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 40.775 | -73.925 | 40.7786 | -73.9232 |
| 9 | 36 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 40.755 | -73.785 | 40.7544 | -73.788 |
| 10 | 36 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 40.675 | -73.795 | 40.6781 | -73.7917 |
| 11 | 36 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 40.605 | -74.015 | 40.604 | -74.0106 |

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
