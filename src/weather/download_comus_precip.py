import argparse
import csv
from datetime import datetime, timedelta, timezone
import gzip
import http.client
import numpy as np
from pathlib import Path
import shutil
import tempfile
import time
import urllib.error
import urllib.request

import pandas as pd
import xarray as xr


BASE_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
QPE_PRODUCT = "CONUS/RadarOnly_QPE_15M_00.00"
QPE_FILE_TEMPLATE = (
    "{product}/{date}/MRMS_RadarOnly_QPE_15M_00.00_{date}-{time}.grib2.gz"
)
PRECIP_RATE_PRODUCT = "CONUS/PrecipRate_00.00"
PRECIP_RATE_FILE_TEMPLATE = (
    "{product}/{date}/MRMS_PrecipRate_00.00_{date}-{time}.grib2.gz"
)
FETCH_RETRIES = 4
FETCH_BACKOFF_SECONDS = 3
QPE_STEP_MINUTES = 15
PRECIP_RATE_STEP_MINUTES = 2


class MissingRemoteFileError(FileNotFoundError):
    def __init__(self, key):
        super().__init__(f"Remote NOAA MRMS object not found: {key}")
        self.key = key


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download NOAA MRMS 15-minute precipitation, aggregate to n-minute "
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
            "Output file path. Defaults to mrms_<interval>min_precip.<ext> in the "
            "weather script directory."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("netcdf", "csv"),
        default="netcdf",
        help="Output format.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=30,
        help=(
            "Aggregation interval in minutes. Multiples of 15 use the 15-minute QPE "
            "product; other intervals use the 2-minute PrecipRate product."
        ),
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


def default_output_path(fmt, interval_minutes):
    suffix = ".nc" if fmt == "netcdf" else ".csv"
    return Path(__file__).resolve().parent / f"mrms_{interval_minutes}min_precip{suffix}"


def failed_points_output_path(output_path):
    return output_path.with_name(f"{output_path.stem}_failed_points.csv")


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
    if args.interval_minutes <= 0:
        raise ValueError("--interval-minutes must be positive.")
    mode = source_mode(args.interval_minutes)
    if mode == "precip_rate" and args.interval_minutes < QPE_STEP_MINUTES and args.interval_minutes % 5 != 0:
        raise ValueError(
            "--interval-minutes below 15 must be a multiple of 5 when using "
            "the 2-minute PrecipRate source."
        )
    start_seconds = int(start.timestamp())
    end_seconds = int(end.timestamp())
    if mode == "qpe":
        source_seconds = QPE_STEP_MINUTES * 60
        if start.second != 0 or start_seconds % source_seconds != 0:
            raise ValueError(
                f"--start must align to a {QPE_STEP_MINUTES}-minute boundary when "
                "using the 15-minute QPE source."
            )
        if end.second != 0 or end_seconds % source_seconds != 0:
            raise ValueError(
                f"--end must align to a {QPE_STEP_MINUTES}-minute boundary when "
                "using the 15-minute QPE source."
            )
    else:
        if start.second != 0 or end.second != 0:
            raise ValueError("--start and --end must be aligned to a whole minute.")
    if not Path(args.locations_csv).expanduser().exists():
        raise ValueError(f"--locations-csv not found: {args.locations_csv}")
    return start, end


def source_mode(interval_minutes):
    return "qpe" if interval_minutes % QPE_STEP_MINUTES == 0 else "precip_rate"


def expected_qpe_keys(start, end, interval_minutes):
    interval = timedelta(minutes=interval_minutes)
    quarter_hour = timedelta(minutes=QPE_STEP_MINUTES)
    current = start
    windows = []
    while current < end:
        window_end = min(current + interval, end)
        window_keys = []
        step = current
        while step < window_end:
            window_keys.append(build_key(step, "qpe"))
            step += quarter_hour
        windows.append((current, window_keys))
        current = window_end
    return windows


def floor_to_step(timestamp, step_minutes):
    step_seconds = step_minutes * 60
    floored = int(timestamp.timestamp()) // step_seconds * step_seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def expected_precip_rate_keys(start, end, interval_minutes):
    interval = timedelta(minutes=interval_minutes)
    rate_step = timedelta(minutes=PRECIP_RATE_STEP_MINUTES)
    current = start
    windows = []
    while current < end:
        window_end = min(current + interval, end)
        source_time = floor_to_step(current, PRECIP_RATE_STEP_MINUTES)
        segments = []
        while source_time < window_end:
            segment_start = max(current, source_time)
            segment_end = min(window_end, source_time + rate_step)
            if segment_end > segment_start:
                segments.append((source_time, segment_end - segment_start))
            source_time += rate_step
        windows.append((current, segments))
        current = window_end
    return windows


def build_key(timestamp, mode):
    date = timestamp.strftime("%Y%m%d")
    time = timestamp.strftime("%H%M%S")
    if mode == "qpe":
        return QPE_FILE_TEMPLATE.format(product=QPE_PRODUCT, date=date, time=time)
    return PRECIP_RATE_FILE_TEMPLATE.format(
        product=PRECIP_RATE_PRODUCT,
        date=date,
        time=time,
    )


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
        except urllib.error.HTTPError as exc:
            gz_path.unlink(missing_ok=True)
            grib2_path.unlink(missing_ok=True)
            if exc.code == 404:
                raise MissingRemoteFileError(key) from exc
            if attempt == FETCH_RETRIES:
                raise
            time.sleep(FETCH_BACKOFF_SECONDS * attempt)
        except (
            EOFError,
            OSError,
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
    variable_name = next(iter(dataset.data_vars), None)
    if variable_name is None:
        dataset.close()
        raise ValueError(f"No data variables found in GRIB2 file: {grib2_path}")
    data_array = dataset[variable_name].load()
    dataset.close()
    return data_array


def load_cached_or_fetch_array(key, workdir, zone_locations, cached_arrays):
    if key in cached_arrays:
        return cached_arrays[key], None

    gz_path = workdir / Path(key).name
    grib2_path = gz_path.with_suffix("")
    for attempt in range(1, FETCH_RETRIES + 1):
        grib2_path = fetch_grib2_file(key, workdir)
        try:
            selected = select_zone_points(
                normalize_coords(open_precip_dataset(grib2_path)),
                zone_locations,
            )
            cached_arrays[key] = selected
            return selected, grib2_path
        except (EOFError, OSError, ValueError):
            gz_path.unlink(missing_ok=True)
            grib2_path.unlink(missing_ok=True)
            if attempt == FETCH_RETRIES:
                raise
            time.sleep(FETCH_BACKOFF_SECONDS * attempt)


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


def accumulation_from_rate(rate_array, duration):
    duration_hours = duration.total_seconds() / 3600.0
    units = str(rate_array.attrs.get("units", "")).strip().lower()
    if units in {"", "none", "unknown"}:
        units = "mm/hr"
    if units in {"mm/h", "mm hr-1", "mm h-1", "mm/hr", "kg m-2 h-1", "kg m^-2 h^-1"}:
        accumulation = rate_array * duration_hours
    elif units in {"mm/s", "mm s-1", "mm sec-1", "kg m-2 s-1", "kg m^-2 s^-1"}:
        accumulation = rate_array * duration.total_seconds()
    else:
        raise ValueError(
            f"Unsupported PrecipRate units '{rate_array.attrs.get('units')}'."
        )
    accumulation.attrs = dict(rate_array.attrs)
    accumulation.attrs["units"] = "mm"
    return accumulation


def empty_precipitation_dataset(zone_locations, interval_minutes, mode):
    zone_ids = zone_locations["zone_id"].to_numpy()
    dataset = xr.Dataset(
        data_vars={
            "precipitation_mm": xr.DataArray(
                np.empty((0, len(zone_ids))),
                dims=("time", "zone_id"),
                coords={
                    "time": [],
                    "zone_id": zone_ids,
                    "latitude": ("zone_id", zone_locations["latitude"].to_numpy()),
                    "longitude": ("zone_id", zone_locations["longitude"].to_numpy()),
                },
            )
        }
    )
    dataset["precipitation_mm"].attrs["long_name"] = (
        f"{interval_minutes}-minute precipitation accumulation"
    )
    dataset["precipitation_mm"].attrs["source_product"] = (
        "NOAA MRMS RadarOnly_QPE_15M_00.00"
        if mode == "qpe"
        else "NOAA MRMS PrecipRate"
    )
    dataset["precipitation_mm"].attrs["units"] = "mm"
    return dataset


def build_dataset(args, start, end):
    zone_locations = load_zone_locations(args.locations_csv)
    outputs = []
    failed_windows = []
    mode = source_mode(args.interval_minutes)
    with tempfile.TemporaryDirectory(prefix=f"mrms_{args.interval_minutes}min_") as temp_dir:
        workdir = Path(temp_dir)
        cached_arrays = {}
        missing_keys = set()
        if mode == "qpe":
            windows = expected_qpe_keys(start, end, args.interval_minutes)
        else:
            windows = expected_precip_rate_keys(start, end, args.interval_minutes)

        for window_start, window_inputs in windows:
            arrays = []
            temp_paths = []
            missing_key = None
            for item in window_inputs:
                if mode == "qpe":
                    key = build_key(item, mode)
                    duration = None
                else:
                    source_time, duration = item
                    key = build_key(source_time, mode)

                if key in missing_keys:
                    missing_key = key
                    break

                try:
                    source_array, grib2_path = load_cached_or_fetch_array(
                        key, workdir, zone_locations, cached_arrays
                    )
                except MissingRemoteFileError:
                    missing_keys.add(key)
                    missing_key = key
                    break
                except (EOFError, OSError, ValueError):
                    missing_keys.add(key)
                    missing_key = key
                    break
                if grib2_path is not None:
                    temp_paths.append(grib2_path)
                arrays.append(
                    source_array if duration is None else accumulation_from_rate(source_array, duration)
                )

            if missing_key is not None:
                failed_windows.append((window_start, missing_key))
                continue

            total = arrays[0]
            for array in arrays[1:]:
                total = total + array
            total = total.expand_dims(time=[window_start.replace(tzinfo=None)])
            total.name = "precipitation_mm"
            total.attrs["long_name"] = (
                f"{args.interval_minutes}-minute precipitation accumulation"
            )
            total.attrs["source_product"] = (
                "NOAA MRMS RadarOnly_QPE_15M_00.00"
                if mode == "qpe"
                else "NOAA MRMS PrecipRate"
            )
            total.attrs["units"] = "mm"
            outputs.append(total)

            if args.keep_temp:
                for path in temp_paths:
                    shutil.copy2(path, Path.cwd() / path.name)

    if outputs:
        dataset = xr.concat(outputs, dim="time").to_dataset(name="precipitation_mm")
    else:
        dataset = empty_precipitation_dataset(zone_locations, args.interval_minutes, mode)
    dataset.attrs["source"] = "NOAA MRMS noaa-mrms-pds"
    dataset.attrs["product"] = (
        QPE_PRODUCT if mode == "qpe" else PRECIP_RATE_PRODUCT
    )
    dataset.attrs["source_mode"] = mode
    dataset.attrs["interval_minutes"] = args.interval_minutes
    return dataset, zone_locations, failed_windows


def write_output(dataset, output_path, fmt):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "netcdf":
        dataset.to_netcdf(output_path)
        return

    variable = next(iter(dataset.data_vars))
    frame = dataset[variable].to_dataframe().reset_index()
    if "zone_id" in frame.columns:
        frame = frame.sort_values(["time", "zone_id"]).reset_index(drop=True)
    frame.to_csv(output_path, index=False)


def write_failed_points(zone_locations, failed_windows, output_path):
    failure_path = failed_points_output_path(output_path)
    failure_path.parent.mkdir(parents=True, exist_ok=True)

    if not failed_windows:
        if failure_path.exists():
            failure_path.unlink()
        return None

    frames = []
    for window_start, missing_key in failed_windows:
        frame = zone_locations[["zone_id"]].copy()
        frame["key"] = missing_key
        frame["datetime"] = window_start.replace(tzinfo=None)
        frames.append(frame)

    pd.concat(frames, ignore_index=True).sort_values(["datetime", "zone_id"]).to_csv(
        failure_path,
        index=False,
    )
    return failure_path


def main():
    args = parse_args()
    start, end = validate_args(args)
    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        output_path = default_output_path(args.format, args.interval_minutes)

    dataset, zone_locations, failed_windows = build_dataset(args, start, end)
    write_output(dataset, output_path, args.format)
    failure_path = write_failed_points(zone_locations, failed_windows, output_path)
    print(f"Wrote {args.format} output to {output_path}")
    if failure_path is not None:
        print(f"Wrote failed points log to {failure_path}")


if __name__ == "__main__":
    main()
