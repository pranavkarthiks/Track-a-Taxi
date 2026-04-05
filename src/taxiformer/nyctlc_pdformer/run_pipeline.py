import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a zone-graph NYCTLC dataset for PDFormer and optionally run training."
    )
    parser.add_argument("--timeseries", required=True)
    parser.add_argument("--adjacency", required=True)
    parser.add_argument("--dataset", default="NYCTLC")
    parser.add_argument(
        "--pdformer-root",
        default=str(Path(__file__).resolve().parents[1] / "PDFormer"),
    )
    parser.add_argument("--run", action="store_true", help="Run PDFormer after preparing files.")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--zone-col", default="LocationID")
    parser.add_argument("--inflow-col", default="inflow")
    parser.add_argument("--outflow-col", default="outflow")
    parser.add_argument("--freq-minutes", type=int, default=30)
    parser.add_argument("--input-window", type=int, default=6)
    parser.add_argument("--output-window", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    taxiformer_root = script_dir.parents[1]
    pdformer_root = Path(args.pdformer_root).resolve()
    workspace_root = taxiformer_root.parents[2]
    venv_python = workspace_root / ".venv" / "bin" / "python"
    python = str(venv_python if venv_python.exists() else sys.executable)

    if not Path(args.timeseries).exists():
        raise FileNotFoundError(f"Timeseries file not found: {args.timeseries}")
    if not Path(args.adjacency).exists():
        raise FileNotFoundError(f"Adjacency file not found: {args.adjacency}")
    if not pdformer_root.exists():
        raise FileNotFoundError(f"PDFormer root not found: {pdformer_root}")
    if not (taxiformer_root / "run_model.py").exists():
        raise FileNotFoundError(f"run_model.py not found in {taxiformer_root}")

    prepare_cmd = [
        python,
        str(script_dir / "prepare_dataset.py"),
        "--timeseries",
        args.timeseries,
        "--adjacency",
        args.adjacency,
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
    subprocess.run(prepare_cmd, check=True, cwd=taxiformer_root)

    if not args.run:
        print(
            "Preparation complete. Run:\n"
            f"cd {pdformer_root}\n"
            f"PYTHONPATH={pdformer_root} {python} {taxiformer_root / 'run_model.py'} "
            f"--task traffic_state_pred --model PDFormer --dataset {args.dataset} --config_file {args.dataset}"
        )
        return

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{pdformer_root}:{existing_pythonpath}" if existing_pythonpath else str(pdformer_root)
    run_cmd = [
        python,
        str(taxiformer_root / "run_model.py"),
        "--task",
        "traffic_state_pred",
        "--model",
        "PDFormer",
        "--dataset",
        args.dataset,
        "--config_file",
        args.dataset,
    ]
    subprocess.run(run_cmd, check=True, cwd=pdformer_root, env=env)


if __name__ == "__main__":
    main()
