`noaa_mrms_30min_precip.py` downloads official NOAA MRMS precipitation products
from `noaa-mrms-pds`, using `RadarOnly_QPE_15M_00.00` for 15-minute multiples
and `PrecipRate` for other intervals such as 5 minutes, then aggregates them into
n-minute precipitation totals,
samples the taxi zone coordinate lookup, and by default writes NetCDF or CSV
output into `src/weather`.

Dependencies:

- `xarray`
- `cfgrib`
- `pandas`
- `eccodes` runtime for `cfgrib`

Example:

`  python ./Track-a-Taxi/src/weather/noaa_mrms_30min_precip.py \
    --start 2026-01-01T00:00:00Z \
    --end 2026-01-02T00:00:00Z \
    --interval-minutes 30 \
    --locations-csv ./Track-a-Taxi/src/zones/data/taxi_zones/taxi_zone_lookup_coordinates.csv \
    --output /tmp/nyc_mrms_30min.nc \
    --format netcdf`


### Data source:
https://registry.opendata.aws/noaa-mrms-pds/
