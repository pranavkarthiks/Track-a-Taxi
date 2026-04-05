import argparse
import csv
from datetime import datetime, timedelta, timezone
import gzip
import http.client
from pathlib import Path
import shutil
import tempfile
import time
import urllib.error
import urllib.request

import pandas as pd
import xarray as xr


BASE_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT = "CONUS/RadarOnly_QPE_15M_00.00"
FILE_TEMPLATE = (
    "{product}/{date}/MRMS_RadarOnly_QPE_15M_00.00_{date}-{time}.grib2.gz"
)
FETCH_RETRIES = 4
FETCH_BACKOFF_SECONDS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download NOAA MRMS 15-minute precipitation, aggregate to 30-minute "
            "intervals, sample taxi zone coordinates, and save to NetCDF or CSV."
        )
    )
    parser.add_argument("--start", required=True, help="UTC start time in ISO format.")
    parser.add_argument(
        "--end",
        required=True,
        help="UTC end time in ISO format. This bound is exclusive.",
    )
    parser.add_argument(
        "--locations-csv",
        default=str(default_locations_csv()),
        help=(
            "CSV file containing taxi zone coordinates. Expected columns: "
            "LocationID, latitude, longitude."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Output file path. Defaults to mrms_30min_precip.<ext> in the weather "
            "script directory."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("netcdf", "csv"),
        default="netcdf",
        help="Output format.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate decompressed GRIB2 files for debugging.",
    )
    return parser.parse_args()


def default_locations_csv():
    return (
        Path(__file__).resolve().parent.parent
        / "zones"
        / "data"
        / "taxi_zones"
        / "taxi_zone_lookup_coordinates.csv"
    )


def default_output_path(fmt):
    suffix = ".nc" if fmt == "netcdf" else ".csv"
    return Path(__file__).resolve().parent / f"mrms_30min_precip{suffix}"


def parse_utc(value):
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid datetime '{value}'. Use ISO format like 2026-03-27T00:00:00Z."
        ) from exc

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp


def validate_args(args):
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if start >= end:
        raise ValueError("--start must be earlier than --end.")
    if start.minute not in (0, 30) or start.second != 0:
        raise ValueError("--start must align to a 30-minute boundary.")
    if end.minute not in (0, 30) or end.second != 0:
        raise ValueError("--end must align to a 30-minute boundary.")
    if not Path(args.locations_csv).expanduser().exists():
        raise ValueError(f"--locations-csv not found: {args.locations_csv}")
    return start, end


def expected_keys(start, end):
    interval = timedelta(minutes=30)
    quarter_hour = timedelta(minutes=15)
    current = start
    windows = []
    while current < end:
        first = current
        second = current + quarter_hour
        windows.append((current, [build_key(first), build_key(second)]))
        current += interval
    return windows


def build_key(timestamp):
    date = timestamp.strftime("%Y%m%d")
    time = timestamp.strftime("%H%M%S")
    return FILE_TEMPLATE.format(product=PRODUCT, date=date, time=time)


def fetch_grib2_file(key, workdir):
    url = f"{BASE_URL}/{key}"
    gz_path = workdir / Path(key).name
    grib2_path = gz_path.with_suffix("")
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                with gz_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)

            with gzip.open(gz_path, "rb") as source:
                with grib2_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
            break
        except (
            TimeoutError,
            ConnectionResetError,
            http.client.RemoteDisconnected,
            urllib.error.URLError,
        ):
            gz_path.unlink(missing_ok=True)
            grib2_path.unlink(missing_ok=True)
            if attempt == FETCH_RETRIES:
                raise
            time.sleep(FETCH_BACKOFF_SECONDS * attempt)
    return grib2_path


def open_precip_dataset(grib2_path):
    dataset = xr.open_dataset(
        grib2_path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )
    variable_name = next(iter(dataset.data_vars))
    data_array = dataset[variable_name].load()
    dataset.close()
    return data_array


def normalize_coords(data_array):
    rename_map = {}
    if "latitude" in data_array.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in data_array.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        data_array = data_array.rename(rename_map)

    if "lon" not in data_array.coords or "lat" not in data_array.coords:
        raise ValueError("Expected lon/lat coordinates in the MRMS GRIB2 file.")

    lon = data_array["lon"]
    if float(lon.max()) > 180:
        adjusted_lon = ((lon + 180) % 360) - 180
        data_array = data_array.assign_coords(lon=adjusted_lon)

    data_array = data_array.sortby("lat")
    data_array = data_array.sortby("lon")
    return data_array


def load_zone_locations(path):
    csv_path = Path(path).expanduser()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"LocationID", "latitude", "longitude"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                f"Location CSV is missing required columns: {missing_list}"
            )

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("Location CSV did not contain any rows.")

    zone_locations = frame.rename(columns={"LocationID": "zone_id"}).copy()
    zone_locations["zone_id"] = zone_locations["zone_id"].astype(int)
    zone_locations["latitude"] = pd.to_numeric(zone_locations["latitude"], errors="coerce")
    zone_locations["longitude"] = pd.to_numeric(zone_locations["longitude"], errors="coerce")
    zone_locations = zone_locations.dropna(subset=["latitude", "longitude"]).copy()
    zone_locations = zone_locations[
        zone_locations["latitude"].between(-90, 90)
        & zone_locations["longitude"].between(-180, 180)
    ].copy()
    if zone_locations.empty:
        raise ValueError("No valid zone coordinates remained after filtering invalid rows.")
    return zone_locations


def select_zone_points(data_array, zone_locations):
    zone_ids = zone_locations["zone_id"].tolist()
    latitudes = xr.DataArray(
        zone_locations["latitude"].to_numpy(),
        dims="zone_id",
        coords={"zone_id": zone_ids},
    )
    longitudes = xr.DataArray(
        zone_locations["longitude"].to_numpy(),
        dims="zone_id",
        coords={"zone_id": zone_ids},
    )

    selected = data_array.sel(lat=latitudes, lon=longitudes, method="nearest")
    if selected.size == 0:
        raise ValueError("No MRMS grid cells were found for the requested zone points.")

    selected = selected.rename({"lat": "grid_lat", "lon": "grid_lon"})
    selected = selected.assign_coords(
        latitude=("zone_id", zone_locations["latitude"].to_numpy()),
        longitude=("zone_id", zone_locations["longitude"].to_numpy()),
    )
    return selected


def build_30min_dataset(args, start, end):
    zone_locations = load_zone_locations(args.locations_csv)
    outputs = []
    with tempfile.TemporaryDirectory(prefix="mrms_30min_") as temp_dir:
        workdir = Path(temp_dir)
        for window_start, keys in expected_keys(start, end):
            arrays = []
            temp_paths = []
            for key in keys:
                grib2_path = fetch_grib2_file(key, workdir)
                temp_paths.append(grib2_path)
                arrays.append(
                    select_zone_points(
                        normalize_coords(open_precip_dataset(grib2_path)),
                        zone_locations,
                    )
                )

            total = arrays[0] + arrays[1]
            total = total.expand_dims(time=[window_start.replace(tzinfo=None)])
            total.name = "precipitation_mm_30min"
            total.attrs["long_name"] = "30-minute precipitation accumulation"
            total.attrs["source_product"] = "NOAA MRMS RadarOnly_QPE_15M_00.00"
            total.attrs["units"] = arrays[0].attrs.get("units", "mm")
            outputs.append(total)

            if args.keep_temp:
                for path in temp_paths:
                    shutil.copy2(path, Path.cwd() / path.name)

    dataset = xr.concat(outputs, dim="time").to_dataset(name="precipitation_mm_30min")
    dataset.attrs["source"] = "NOAA MRMS noaa-mrms-pds"
    dataset.attrs["product"] = "CONUS/RadarOnly_QPE_15M_00.00"
    dataset.attrs["interval_minutes"] = 30
    return dataset


def write_output(dataset, output_path, fmt):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "netcdf":
        dataset.to_netcdf(output_path)
        return

    frame = dataset["precipitation_mm_30min"].to_dataframe().reset_index()
    if "zone_id" in frame.columns:
        frame = frame.sort_values(["time", "zone_id"]).reset_index(drop=True)
    frame.to_csv(output_path, index=False)


def main():
    args = parse_args()
    start, end = validate_args(args)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = default_output_path(args.format)

    dataset = build_30min_dataset(args, start, end)
    write_output(dataset, output_path, args.format)
    print(f"Wrote {args.format} output to {output_path}")


if __name__ == "__main__":
    main()
