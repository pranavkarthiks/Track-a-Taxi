import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert zone-level NYCTLC time series into LibCity atomic files for PDFormer."
    )
    parser.add_argument("--timeseries", required=True, help="Input long-format CSV/Parquet file.")
    parser.add_argument("--adjacency", required=True, help="Zone adjacency matrix CSV.")
    parser.add_argument("--dataset", default="NYCTLC", help="Output dataset name.")
    parser.add_argument(
        "--pdformer-root",
        default=str(Path(__file__).resolve().parent.parent / "PDFormer"),
        help="PDFormer root containing raw_data/ and dataset JSON configs.",
    )
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--zone-col", default="LocationID")
    parser.add_argument("--inflow-col", default="inflow")
    parser.add_argument("--outflow-col", default="outflow")
    parser.add_argument("--freq-minutes", type=int, default=30)
    parser.add_argument("--input-window", type=int, default=6)
    parser.add_argument("--output-window", type=int, default=1)
    parser.add_argument("--train-rate", type=float, default=0.7)
    parser.add_argument("--eval-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bidir", action="store_true", default=True)
    parser.add_argument("--no-bidir", dest="bidir", action="store_false")
    parser.add_argument("--far-mask-delta", type=float, default=5.0)
    parser.add_argument("--geo-num-heads", type=int, default=2)
    parser.add_argument("--sem-num-heads", type=int, default=2)
    parser.add_argument("--t-num-heads", type=int, default=4)
    parser.add_argument("--cluster-method", default="kshape")
    parser.add_argument("--cand-key-days", type=int, default=14)
    parser.add_argument("--type-ln", default="pre")
    parser.add_argument("--set-loss", default="huber")
    parser.add_argument("--huber-delta", type=int, default=2)
    parser.add_argument("--mode", default="average")
    parser.add_argument("--no-dist-rel", action="store_true", help="Skip centroid distance and edge adjacency generation.")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type for {path}. Use CSV or Parquet.")


def normalize_zone_ids(values) -> pd.Index:
    return pd.Index(pd.Series(values).astype(int), dtype="int64")


def load_adjacency(path: Path) -> pd.DataFrame:
    adjacency = pd.read_csv(path, index_col=0)
    adjacency.index = normalize_zone_ids(adjacency.index)
    adjacency.columns = normalize_zone_ids(adjacency.columns)
    adjacency = adjacency.sort_index().sort_index(axis=1)
    if not adjacency.index.equals(adjacency.columns):
        raise ValueError("Adjacency matrix must have matching row/column zone IDs.")
    return adjacency


def load_timeseries(args, zone_ids: pd.Index) -> pd.DataFrame:
    ts = read_table(Path(args.timeseries))
    required = [args.time_col, args.zone_col, args.inflow_col, args.outflow_col]
    missing = [col for col in required if col not in ts.columns]
    if missing:
        raise ValueError(f"Timeseries file is missing required columns: {missing}")

    ts = ts[required].copy()
    ts.columns = ["time", "entity_id", "inflow", "outflow"]
    ts["entity_id"] = ts["entity_id"].astype(int)
    ts = ts[ts["entity_id"].isin(zone_ids)].copy()
    if ts.empty:
        raise ValueError("No rows remain after filtering timeseries to adjacency zone IDs.")

    ts["time"] = pd.to_datetime(ts["time"], utc=True)
    ts = ts.groupby(["entity_id", "time"], as_index=False)[["inflow", "outflow"]].sum()

    times = pd.Index(sorted(ts["time"].unique()))
    full_index = pd.MultiIndex.from_product([zone_ids, times], names=["entity_id", "time"])
    ts = ts.set_index(["entity_id", "time"]).reindex(full_index, fill_value=0).reset_index()
    ts["type"] = "state"
    ts["time"] = ts["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ts.insert(0, "dyna_id", np.arange(len(ts), dtype=np.int64))
    return ts[["dyna_id", "type", "time", "entity_id", "inflow", "outflow"]]


def build_rel(adjacency: pd.DataFrame) -> pd.DataFrame:
    edges = adjacency.stack().reset_index()
    edges.columns = ["origin_id", "destination_id", "weight"]
    edges = edges[(edges["weight"] > 0) & (edges["origin_id"] != edges["destination_id"])].copy()
    edges.insert(0, "type", "geo")
    edges.insert(0, "rel_id", np.arange(len(edges), dtype=np.int64))
    return edges[["rel_id", "type", "origin_id", "destination_id", "weight"]]


def build_raw_config(args) -> dict:
    return {
        "info": {
            "data_col": ["inflow", "outflow"],
            "data_files": [args.dataset],
            "geo_file": args.dataset,
            "rel_file": args.dataset if args.no_dist_rel else f"{args.dataset}_dist",
            "output_dim": 2,
            "time_intervals": args.freq_minutes * 60,
            "init_weight_inf_or_zero": "zero" if args.no_dist_rel else "inf",
            "set_weight_link_or_dist": "link" if args.no_dist_rel else "dist",
            "calculate_weight_adj": not args.no_dist_rel,
            "weight_adj_epsilon": 0,
            "weight_col": "weight",
        }
    }


def build_model_config(args) -> dict:
    return {
        "dataset_class": "PDFormerDataset",
        "input_window": args.input_window,
        "output_window": args.output_window,
        "train_rate": args.train_rate,
        "eval_rate": args.eval_rate,
        "batch_size": args.batch_size,
        "add_time_in_day": True,
        "add_day_in_week": True,
        "bidir": args.bidir,
        "far_mask_delta": args.far_mask_delta,
        "geo_num_heads": args.geo_num_heads,
        "sem_num_heads": args.sem_num_heads,
        "t_num_heads": args.t_num_heads,
        "cluster_method": args.cluster_method,
        "cand_key_days": args.cand_key_days,
        **({"type_short_path": "centroid_dist",
            "short_path_distance_file": f"{args.dataset}_full_dist.npy"} if not args.no_dist_rel else {}),
        "type_ln": args.type_ln,
        "set_loss": args.set_loss,
        "huber_delta": args.huber_delta,
        "mode": args.mode,
    }


def clear_dataset_caches(pdformer_root: Path, dataset: str) -> None:
    cache_dir = pdformer_root / "libcity" / "cache" / "dataset_cache"
    if not cache_dir.exists():
        return

    patterns = [
        f"traffic_state_{dataset}_*.npz",
        f"point_based_{dataset}_*.npz",
        f"pdformer_point_based_{dataset}_*.npz",
        f"dtw_{dataset}.npy",
        f"pattern_keys_*_{dataset}_*.npy",
    ]
    for pattern in patterns:
        for path in cache_dir.glob(pattern):
            path.unlink()


def main():
    args = parse_args()
    pdformer_root = Path(args.pdformer_root).resolve()
    python = sys.executable
    script_dir = Path(__file__).resolve().parent
    dataset_dir = pdformer_root / "raw_data" / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    adjacency = load_adjacency(Path(args.adjacency))
    zone_ids = adjacency.index

    geo = pd.DataFrame({"geo_id": zone_ids})
    rel = build_rel(adjacency)
    dyna = load_timeseries(args, zone_ids)

    geo.to_csv(dataset_dir / f"{args.dataset}.geo", index=False)
    rel.to_csv(dataset_dir / f"{args.dataset}.rel", index=False)
    dyna.to_csv(dataset_dir / f"{args.dataset}.dyna", index=False)

    with open(dataset_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(build_raw_config(args), f, indent=2)
        f.write("\n")

    with open(pdformer_root / f"{args.dataset}.json", "w", encoding="utf-8") as f:
        json.dump(build_model_config(args), f, indent=2)
        f.write("\n")

    clear_dataset_caches(pdformer_root, args.dataset)

    if not args.no_dist_rel:
        subprocess.run(
            [
                python,
                str(pdformer_root / "generate_nyctlc_dist_rel.py"),
                "--adjacency",
                str(Path(args.adjacency).resolve()),
                "--output-rel",
                str(dataset_dir / f"{args.dataset}_dist.rel"),
                "--output-full-dist",
                str(dataset_dir / f"{args.dataset}_full_dist.npy"),
            ],
            check=True,
            cwd=script_dir.parent,
        )

    print(f"Wrote dataset files to {dataset_dir}")
    print(f"Wrote model config to {pdformer_root / f'{args.dataset}.json'}")


if __name__ == "__main__":
    main()
