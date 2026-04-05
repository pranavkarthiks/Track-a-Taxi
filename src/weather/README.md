`noaa_mrms_30min_precip.py` downloads official NOAA MRMS `RadarOnly_QPE_15M_00.00`
files from `noaa-mrms-pds`, aggregates them into 30-minute precipitation totals,
samples the taxi zone coordinate lookup, and by default writes NetCDF or CSV
output into `src/weather`.

Dependencies:

- `xarray`
- `cfgrib`
- `pandas`
- `eccodes` runtime for `cfgrib`

Example:

`  python ./Track-a-Taxi/src/weather/noaa_mrms_30min_precip.py \
    --start 2026-03-27T00:00:00Z \
    --end 2026-03-27T06:00:00Z \
    --locations-csv ./Track-a-Taxi/src/zones/data/taxi_zones/taxi_zone_lookup_coordinates.csv \
    --output /tmp/nyc_mrms_30min.nc \
    --format netcdf`
