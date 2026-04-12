from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from download_comus_precip import failed_points_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill weather rows listed in a failed-points CSV by inserting the "
            "mean precipitation between the nearest surrounding observations "
            "for the same zone."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the extracted weather file (.nc or .csv).",
    )
    return parser.parse_args()


def _normalize_failed_points(failed_points: pd.DataFrame) -> pd.DataFrame:
    normalized = failed_points.copy()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"])
    normalized = normalized.drop_duplicates(subset=["datetime", "zone_id"])
    normalized = normalized.sort_values(["datetime", "zone_id"]).reset_index(drop=True)
    return normalized


def _time_blocks(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    if frame.empty:
        return []
    return list(frame.groupby("time", sort=True))


def _merge_sorted_frames(existing: pd.DataFrame, insertions: list[pd.DataFrame]) -> pd.DataFrame:
    inserted = pd.concat(insertions, ignore_index=True)
    existing_records = existing.to_dict("records")
    inserted_records = inserted.to_dict("records")

    merged: list[dict[str, object]] = []
    existing_index = 0
    inserted_index = 0

    while existing_index < len(existing_records) and inserted_index < len(inserted_records):
        existing_key = (
            existing_records[existing_index]["time"],
            existing_records[existing_index]["zone_id"],
        )
        inserted_key = (
            inserted_records[inserted_index]["time"],
            inserted_records[inserted_index]["zone_id"],
        )
        if existing_key <= inserted_key:
            merged.append(existing_records[existing_index])
            existing_index += 1
        else:
            merged.append(inserted_records[inserted_index])
            inserted_index += 1

    if existing_index < len(existing_records):
        merged.extend(existing_records[existing_index:])
    if inserted_index < len(inserted_records):
        merged.extend(inserted_records[inserted_index:])

    return pd.DataFrame.from_records(merged, columns=existing.columns)


def repair_weather_frame(frame: pd.DataFrame, failed_points: pd.DataFrame) -> pd.DataFrame:
    repaired = frame.copy()
    repaired["time"] = pd.to_datetime(repaired["time"])
    if "valid_time" in repaired.columns:
        repaired["valid_time"] = pd.to_datetime(repaired["valid_time"])
    repaired = repaired.sort_values(["time", "zone_id"]).reset_index(drop=True)

    failures = _normalize_failed_points(failed_points)
    if failures.empty:
        return repaired

    time_blocks = _time_blocks(repaired)
    if not time_blocks:
        raise ValueError("Cannot repair an empty weather frame.")

    insertions: list[pd.DataFrame] = []
    prior_block: pd.DataFrame | None = None
    block_index = 0

    for missing_time, failure_group in failures.groupby("datetime", sort=True):
        while block_index < len(time_blocks) and time_blocks[block_index][0] < missing_time:
            prior_block = time_blocks[block_index][1]
            block_index += 1

        if prior_block is None or block_index >= len(time_blocks):
            raise ValueError(
                f"Cannot interpolate {missing_time}: missing a surrounding time slice."
            )

        next_time, next_block = time_blocks[block_index]
        if next_time == missing_time:
            continue

        previous_lookup = prior_block.set_index("zone_id")
        next_lookup = next_block.set_index("zone_id")
        required_zones = set(failure_group["zone_id"])
        missing_zones = sorted(
            (required_zones - set(previous_lookup.index))
            | (required_zones - set(next_lookup.index))
        )
        if missing_zones:
            raise ValueError(
                f"Cannot interpolate {missing_time}: missing surrounding rows for zone(s) "
                f"{missing_zones[:10]}"
            )

        inserted_rows = previous_lookup.loc[failure_group["zone_id"]].copy()
        inserted_rows["time"] = missing_time
        if "valid_time" in inserted_rows.columns:
            previous_valid = pd.to_datetime(inserted_rows["valid_time"])
            next_valid = pd.to_datetime(next_lookup.loc[failure_group["zone_id"], "valid_time"].to_numpy())
            inserted_rows["valid_time"] = previous_valid + ((next_valid - previous_valid) / 2)
        inserted_rows["precipitation_mm"] = (
            inserted_rows["precipitation_mm"].to_numpy()
            + next_lookup.loc[failure_group["zone_id"], "precipitation_mm"].to_numpy()
        ) / 2
        insertions.append(inserted_rows.reset_index())

    if not insertions:
        return repaired

    return _merge_sorted_frames(repaired, insertions)


def repair_csv(path: Path, failed_path: Path) -> int:
    frame = pd.read_csv(path)
    failed_points = pd.read_csv(failed_path)
    repaired = repair_weather_frame(frame, failed_points)
    inserted_rows = len(repaired) - len(frame)
    repaired.to_csv(path, index=False)
    return inserted_rows


def dataframe_to_dataset(frame: pd.DataFrame, template: xr.Dataset, variable: str) -> xr.Dataset:
    repaired = frame.copy()
    repaired["time"] = pd.to_datetime(repaired["time"])
    if "valid_time" in repaired.columns:
        repaired["valid_time"] = pd.to_datetime(repaired["valid_time"])

    time_index = pd.Index(sorted(repaired["time"].unique()), name="time")
    zone_index = pd.Index(sorted(repaired["zone_id"].unique()), name="zone_id")

    precipitation = (
        repaired.pivot(index="time", columns="zone_id", values=variable)
        .reindex(index=time_index, columns=zone_index)
    )
    zone_meta_columns = [column for column in ("grid_lat", "grid_lon", "latitude", "longitude") if column in repaired.columns]
    time_meta_columns = [column for column in ("valid_time",) if column in repaired.columns]

    zone_meta = repaired.drop_duplicates("zone_id").set_index("zone_id").reindex(zone_index)
    time_meta = repaired.drop_duplicates("time").set_index("time").reindex(time_index)

    coords: dict[str, object] = {
        "time": time_index.to_numpy(),
        "zone_id": zone_index.to_numpy(dtype=template["zone_id"].dtype),
    }
    for column in zone_meta_columns:
        coords[column] = ("zone_id", zone_meta[column].to_numpy(dtype=template[column].dtype))
    for column in time_meta_columns:
        coords[column] = ("time", time_meta[column].to_numpy(dtype=template[column].dtype))
    if "step" in template.coords:
        coords["step"] = template["step"].values
    if "heightAboveSea" in template.coords:
        coords["heightAboveSea"] = template["heightAboveSea"].values

    repaired_dataset = xr.Dataset(
        {
            variable: (
                ("time", "zone_id"),
                precipitation.to_numpy(dtype=template[variable].dtype),
            )
        },
        coords=coords,
        attrs=template.attrs.copy(),
    )
    repaired_dataset[variable].attrs = template[variable].attrs.copy()
    return repaired_dataset


def repair_netcdf(path: Path, failed_path: Path) -> int:
    with xr.open_dataset(path) as dataset:
        dataset.load()
        variable = next(iter(dataset.data_vars))
        frame = dataset[variable].to_dataframe().reset_index()
        repaired_template = dataset.copy(deep=True)
    failed_points = pd.read_csv(failed_path)
    repaired_frame = repair_weather_frame(frame, failed_points)
    inserted_rows = len(repaired_frame) - len(frame)
    repaired_dataset = dataframe_to_dataset(repaired_frame, repaired_template, variable)
    repaired_dataset.to_netcdf(path)
    return inserted_rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    failed_path = failed_points_output_path(input_path)

    if not failed_path.exists():
        print(f"No failed points file found for {input_path}; nothing to repair.")
        return

    if input_path.suffix == ".csv":
        inserted_rows = repair_csv(input_path, failed_path)
    elif input_path.suffix == ".nc":
        inserted_rows = repair_netcdf(input_path, failed_path)
    else:
        raise ValueError(f"Unsupported weather file format: {input_path.suffix}")

    print(f"Inserted {inserted_rows} repaired rows into {input_path}")


if __name__ == "__main__":
    main()
