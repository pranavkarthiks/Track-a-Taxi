import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge precipitation data into an existing NYCTLC .dyna file."
    )
    parser.add_argument(
        "--weather-csv",
        required=True,
        help="Weather CSV with columns: time, zone_id, precipitation_mm",
    )
    parser.add_argument("--dataset", default="NYCTLC")
    parser.add_argument(
        "--pdformer-root",
        default=str(Path(__file__).resolve().parent.parent / "PDFormer"),
    )
    return parser.parse_args()


def load_weather(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(path, usecols=["time", "zone_id", "precipitation_mm"])
    weather["time"] = pd.to_datetime(weather["time"], utc=True)
    weather = weather.rename(columns={"zone_id": "entity_id"})
    weather["entity_id"] = weather["entity_id"].astype(int)
    # Aggregate to one value per (time, zone) in case of duplicates
    weather = (
        weather.groupby(["time", "entity_id"], as_index=False)["precipitation_mm"]
        .mean()
    )
    return weather


def main():
    args = parse_args()
    dataset_dir = Path(args.pdformer_root).resolve() / "raw_data" / args.dataset
    dyna_path = dataset_dir / f"{args.dataset}.dyna"

    if not dyna_path.exists():
        raise FileNotFoundError(f".dyna file not found: {dyna_path}")

    # ── Load .dyna ──
    dyna = pd.read_csv(dyna_path)
    dyna["time"] = pd.to_datetime(dyna["time"], utc=True)
    dyna = dyna.drop(columns=["precipitation_mm"], errors="ignore")

    # ── Load weather ──
    weather = load_weather(Path(args.weather_csv))

    # ── Left-join: keep every .dyna row, fill missing precip with per-zone median ──
    merged = dyna.merge(weather, on=["time", "entity_id"], how="left")

    zone_medians = weather.groupby("entity_id")["precipitation_mm"].median()
    global_median = weather["precipitation_mm"].median()

    missing_mask = merged["precipitation_mm"].isna()
    if missing_mask.any():
        merged["precipitation_mm"] = merged["precipitation_mm"].fillna(
            merged["entity_id"].map(zone_medians)
        )
        # Zones with no weather data at all get the global median
        merged["precipitation_mm"] = merged["precipitation_mm"].fillna(global_median)

    n_total = len(merged)
    n_filled = int(missing_mask.sum())
    print(
        f"Merged precipitation: {n_total} rows, "
        f"{n_filled} ({100 * n_filled / n_total:.1f}%) filled with median"
    )

    # ── Write back ──
    merged["time"] = merged["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    merged[
        ["dyna_id", "type", "time", "entity_id", "inflow", "outflow", "precipitation_mm"]
    ].to_csv(dyna_path, index=False)
    print(f"Wrote {dyna_path}")

    # ── Update config.json to include precipitation_mm in data_col ──
    config_path = dataset_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    data_col = config["info"]["data_col"]
    if "precipitation_mm" not in data_col:
        data_col.append("precipitation_mm")
        config["info"]["data_col"] = data_col
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"Updated {config_path}: data_col = {data_col}")


if __name__ == "__main__":
    main()
