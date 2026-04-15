import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


HF_DATASET_REPO = "xshah-123/taxi_x_pdformer_folder"
HF_EXPORT_DIR = "pdformer_export 2"
HF_SOURCE_DATASET = "NYCTaxi_sample"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a zone-graph NYCTLC dataset for PDFormer and optionally run training."
    )
    parser.add_argument("--timeseries")
    parser.add_argument("--adjacency")
    parser.add_argument("--weather-csv", default=None, help="Weather CSV with time, zone_id, precipitation_mm columns.")
    parser.add_argument("--no-dist-rel", action="store_true", help="Skip centroid distance and edge adjacency generation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config-file", default=None, help="PDFormer JSON config name (defaults to dataset name).")
    parser.add_argument(
        "--pdformer-root",
        default=str(Path(__file__).resolve().parent.parent / "PDFormer"),
    )
    parser.add_argument("--run", action="store_true", help="Run PDFormer after preparing files.")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a trained model without retraining.")
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment ID for eval-only (default: latest).")
    parser.add_argument(
        "--debug-cuda",
        action="store_true",
        help="Force synchronous CUDA errors and richer stack traces for debugging.",
    )
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--zone-col", default="LocationID")
    parser.add_argument("--inflow-col", default="inflow")
    parser.add_argument("--outflow-col", default="outflow")
    parser.add_argument("--freq-minutes", type=int, default=30)
    parser.add_argument("--input-window", type=int, default=6)
    parser.add_argument("--output-window", type=int, default=1)
    return parser.parse_args()


def required_dataset_files(pdformer_root, dataset):
    dataset_dir = pdformer_root / "raw_data" / dataset
    return [
        dataset_dir / f"{dataset}.dyna",
    ]


def ensure_dataset_from_huggingface(pdformer_root, dataset):
    missing_files = [path for path in required_dataset_files(pdformer_root, dataset) if not path.exists()]
    if not missing_files:
        return

    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=[
                f"{HF_EXPORT_DIR}/raw_data/{HF_SOURCE_DATASET}/*",
                f"{HF_EXPORT_DIR}/{dataset}.json",
            ],
        )
    )

    source_dir = snapshot_dir / HF_EXPORT_DIR / "raw_data" / HF_SOURCE_DATASET
    destination_dir = pdformer_root / "raw_data" / dataset
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / f"{HF_SOURCE_DATASET}.dyna", destination_dir / f"{dataset}.dyna")

    missing_files = [path for path in required_dataset_files(pdformer_root, dataset) if not path.exists()]
    if missing_files:
        missing_list = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            f"Failed to assemble the prepared {dataset} dataset from Hugging Face repo {HF_DATASET_REPO}.\n"
            f"Missing files after download:\n{missing_list}"
        )


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    taxiformer_root = script_dir.parent
    pdformer_root = Path(args.pdformer_root).resolve()
    dataset_dir = pdformer_root / "raw_data" / args.dataset
    python = sys.executable

    if not pdformer_root.exists():
        raise FileNotFoundError(f"PDFormer root not found: {pdformer_root}")
    if not (taxiformer_root / "run_model.py").exists():
        raise FileNotFoundError(f"run_model.py not found in {taxiformer_root}")

    if args.timeseries:
        timeseries = Path(args.timeseries).expanduser().resolve()
    else:
        timeseries = next(
            (
                path for pattern in ("*.parquet", "*.pq", "*.csv")
                for path in sorted(dataset_dir.glob(pattern))
                if "adj" not in path.stem.lower()
            ),
            None,
        )

    adjacency = (
        Path(args.adjacency).expanduser().resolve()
        if args.adjacency
        else dataset_dir / "taxi_zones_adjacency_matrix.csv"
    )

    if timeseries is not None:
        if not timeseries.exists():
            raise FileNotFoundError(f"Timeseries file not found: {timeseries}")
        if not adjacency.exists():
            raise FileNotFoundError(f"Adjacency file not found: {adjacency}")

        prepare_cmd = [
            python,
            str(script_dir / "prepare_dataset.py"),
            "--timeseries",
            str(timeseries),
            "--adjacency",
            str(adjacency),
            "--dataset",
            args.dataset,
            "--pdformer-root",
            str(pdformer_root),
            "--time-col",
            args.time_col,
            "--zone-col",
            args.zone_col,
            "--inflow-col",
            args.inflow_col,
            "--outflow-col",
            args.outflow_col,
            "--freq-minutes",
            str(args.freq_minutes),
            "--input-window",
            str(args.input_window),
            "--output-window",
            str(args.output_window),
        ]
        if args.no_dist_rel:
            prepare_cmd.append("--no-dist-rel")
        subprocess.run(prepare_cmd, check=True, cwd=taxiformer_root)

    if args.weather_csv:
        weather_csv = Path(args.weather_csv).expanduser().resolve()
        if not weather_csv.exists():
            raise FileNotFoundError(f"Weather CSV not found: {weather_csv}")
        merge_cmd = [
            python,
            str(script_dir / "merge_precip.py"),
            "--weather-csv",
            str(weather_csv),
            "--dataset",
            args.dataset,
            "--pdformer-root",
            str(pdformer_root),
        ]
        subprocess.run(merge_cmd, check=True, cwd=taxiformer_root)

    if not args.timeseries:
        ensure_dataset_from_huggingface(pdformer_root, args.dataset)
        missing_files = [path for path in required_dataset_files(pdformer_root, args.dataset) if not path.exists()]
        if missing_files:
            missing_list = "\n".join(str(path) for path in missing_files)
            raise FileNotFoundError(
                "No input timeseries was provided and the prepared PDFormer dataset is incomplete.\n"
                f"Missing files:\n{missing_list}\n"
                "Pass --timeseries and --adjacency, or generate the dataset files first."
            )

    if not args.run:
        print(
            "Run:\n"
            f"cd {pdformer_root}\n"
            f"PYTHONPATH={pdformer_root} {python} {taxiformer_root / 'run_model.py'} "
            f"--task traffic_state_pred --model PDFormer --dataset {args.dataset}"
        )
        return

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{pdformer_root}:{existing_pythonpath}" if existing_pythonpath else str(pdformer_root)
    if args.debug_cuda:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    run_cmd = [
        python,
        str(taxiformer_root / "run_model.py"),
        "--task",
        "traffic_state_pred",
        "--model",
        "PDFormer",
        "--dataset",
        args.dataset,
    ]
    if args.config_file:
        run_cmd += ["--config_file", args.config_file]
    if args.no_dist_rel:
        run_cmd += ["--type_short_path", "hop"]
    if args.eval_only:
        run_cmd += ["--train", "false"]
        exp_id = args.exp_id
        if exp_id is None:
            cache_dir = pdformer_root / "libcity" / "cache"
            if cache_dir.exists():
                exp_dirs = sorted(cache_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                if exp_dirs:
                    exp_id = exp_dirs[-1].name
        if exp_id is None:
            raise FileNotFoundError("No experiment found in cache. Pass --exp-id explicitly.")
        run_cmd += ["--exp_id", exp_id]
    subprocess.run(run_cmd, check=True, cwd=pdformer_root, env=env)


if __name__ == "__main__":
    main()
