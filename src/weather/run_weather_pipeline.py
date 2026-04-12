from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from download_comus_precip import failed_points_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weather extraction and EDA into a timestamped run directory."
    )
    parser.add_argument("--start", required=True, help="UTC start time in ISO format.")
    parser.add_argument("--end", required=True, help="UTC end time in ISO format.")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Parent directory for timestamped run outputs.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Optional run directory name. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--format",
        choices=("netcdf", "csv"),
        default="netcdf",
        help="Primary extracted weather format.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=30,
        help="Aggregation interval in minutes passed to the extractor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.runs_dir) / run_name
    data_dir = run_dir / "data"
    results_dir = run_dir / "results"
    graphs_dir = run_dir / "graphs"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".nc" if args.format == "netcdf" else ".csv"
    extracted_path = data_dir / f"weather{suffix}"

    python = sys.executable

    subprocess.run(
        [
            python,
            "download_comus_precip.py",
            "--start",
            args.start,
            "--end",
            args.end,
            "--interval-minutes",
            str(args.interval_minutes),
            "--output",
            str(extracted_path),
            "--format",
            args.format,
        ],
        check=True,
    )

    failure_path = failed_points_output_path(extracted_path)
    if failure_path.exists():
        subprocess.run(
            [
                python,
                "repair_failed_weather_points.py",
                "--input",
                str(extracted_path),
            ],
            check=True,
        )

    if args.format == "netcdf":
        subprocess.run(
            [
                python,
                "explore_weather.py",
                "--input",
                str(extracted_path),
                "--results-dir",
                str(results_dir),
                "--graphs-dir",
                str(graphs_dir),
                "--export-raw-csv",
            ],
            check=True,
        )

    print(f"Wrote pipeline run to {run_dir}")


if __name__ == "__main__":
    main()
