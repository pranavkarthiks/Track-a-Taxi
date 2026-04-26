import argparse
import csv
import os
import sys
import ast
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PDFormer with geographic, semantic, and temporal attention heads ablated."
    )
    parser.add_argument("--dataset", default="NYCTLC")
    parser.add_argument("--config-file", default=None, help="PDFormer config name without .json, e.g. NYCTLC.")
    parser.add_argument("--log-file", default=None, help="Training log containing the exact LibCity config dict for this run.")
    parser.add_argument("--exp-id", required=True, help="Experiment id under PDFormer/libcity/cache.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to model_cache/PDFormer_DATASET.m.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for faster diagnostics.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Verify config/checkpoint compatibility without loading data.")
    return parser.parse_args()


def load_run_config(cache_dir):
    config_path = cache_dir / "run_config.json"
    if not config_path.exists():
        return {}, None
    import json

    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle), config_path


def load_log_config(log_file):
    if log_file is None:
        return {}, None
    log_path = Path(log_file)
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = " - INFO - {"
        if marker not in line:
            continue
        raw = "{" + line.split(marker, 1)[1]
        raw = raw.replace("device(type='cuda', index=0)", "'cuda:0'")
        cfg = ast.literal_eval(raw)
        cfg.pop("device", None)
        cfg.pop("exp_id", None)
        cfg.pop("local_rank", None)
        cfg.pop("distributed", None)
        return cfg, log_path
    raise ValueError("No LibCity config dict found in {}".format(log_path))


def set_all_head_ablations(model, groups=None, heads=None):
    for block in model.encoder_blocks:
        block.st_attn.set_head_ablation(groups=groups, heads=heads)


def load_checkpoint(executor, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state, optimizer_state = split_checkpoint(checkpoint, checkpoint_path)
    executor.model.load_state_dict(model_state)
    if optimizer_state is not None:
        executor.optimizer.load_state_dict(optimizer_state)


def split_checkpoint(checkpoint, checkpoint_path):
    if isinstance(checkpoint, tuple):
        model_state, optimizer_state = checkpoint
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        optimizer_state = checkpoint.get("optimizer_state_dict")
    else:
        raise ValueError("Unsupported checkpoint format: {}".format(checkpoint_path))
    return model_state, optimizer_state


def check_checkpoint_config(config, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state, _ = split_checkpoint(checkpoint, checkpoint_path)

    embed_dim = config.get("embed_dim", 64)
    geo_heads = config.get("geo_num_heads", 4)
    sem_heads = config.get("sem_num_heads", 2)
    t_heads = config.get("t_num_heads", 2)
    total_heads = geo_heads + sem_heads + t_heads
    if embed_dim % total_heads != 0:
        raise ValueError("embed_dim {} is not divisible by total heads {}".format(embed_dim, total_heads))
    head_dim = embed_dim // total_heads

    expected_shapes = {
        "encoder_blocks.0.st_attn.geo_q_conv.weight": (geo_heads * head_dim, embed_dim, 1, 1),
        "encoder_blocks.0.st_attn.sem_q_conv.weight": (sem_heads * head_dim, embed_dim, 1, 1),
        "encoder_blocks.0.st_attn.t_q_conv.weight": (t_heads * head_dim, embed_dim, 1, 1),
        "end_conv1.weight": (config.get("output_window", 12), config.get("input_window", 12), 1, 1),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(state[key].shape)
        if actual != expected:
            raise ValueError("{} shape mismatch: checkpoint {} != config {}".format(key, actual, expected))

    print("Checkpoint/config OK")
    print("checkpoint={}".format(checkpoint_path))
    print("input_window={} output_window={}".format(config.get("input_window"), config.get("output_window")))
    print("heads geo={} sem={} temporal={} head_dim={}".format(geo_heads, sem_heads, t_heads, head_dim))


def evaluate_variant(executor, dataloader, variant_name, max_batches=None):
    model = executor.model
    model.eval()
    losses = []
    y_truths = []
    y_preds = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch.to_tensor(executor.device)
            y_true = batch["y"]
            y_predicted = model(batch, executor.lap_mx)
            loss = model.calculate_loss_without_predict(
                y_true, y_predicted, batches_seen=0, set_loss=executor.set_loss
            )
            losses.append(loss.item())
            y_true_inv = executor._scaler.inverse_transform(y_true[..., :executor.output_dim])
            y_pred_inv = executor._scaler.inverse_transform(y_predicted[..., :executor.output_dim])
            y_truths.append(y_true_inv.cpu().numpy())
            y_preds.append(y_pred_inv.cpu().numpy())

    if not y_truths:
        raise ValueError("No batches evaluated for {}".format(variant_name))

    y_truths = np.concatenate(y_truths, axis=0)
    y_preds = np.concatenate(y_preds, axis=0)
    executor.evaluator.clear()
    executor.evaluator.collect({"y_true": torch.tensor(y_truths), "y_pred": torch.tensor(y_preds)})
    result = executor.evaluator.evaluate()
    row = {"variant": variant_name, "val_loss": float(np.mean(losses))}
    for key, value in sorted(result.items()):
        row[key] = float(value)
    return row


def add_deltas(rows):
    baseline = rows[0]
    numeric_keys = [key for key in baseline if key not in {"variant"}]
    out = []
    for row in rows:
        row = dict(row)
        for key in numeric_keys:
            row["delta_" + key] = row[key] - baseline[key]
        out.append(row)
    return out


def write_csv(rows, path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, metric, path, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping plot {}".format(path))
        return
    variants = [row["variant"] for row in rows if row["variant"] != "full"]
    values = [row.get("delta_" + metric, float("nan")) for row in rows if row["variant"] != "full"]
    plt.figure(figsize=(max(7, len(variants) * 0.55), 4))
    colours = ["#b2182b" if value > 0 else "#2166ac" for value in values]
    plt.bar(variants, values, color=colours)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Delta " + metric)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()

    global np
    global torch
    import numpy as np
    import torch

    taxiformer_root = Path(__file__).resolve().parent.parent
    pdformer_root = taxiformer_root / "PDFormer"
    os.chdir(pdformer_root)
    sys.path.insert(0, str(pdformer_root))

    from libcity.config import ConfigParser
    from libcity.data import get_dataset
    from libcity.utils import get_executor, get_model, get_logger

    run_cache_dir = pdformer_root / "libcity" / "cache" / args.exp_id
    run_config, run_config_path = load_log_config(args.log_file)
    if run_config_path is None:
        run_config, run_config_path = load_run_config(run_cache_dir)
    if run_config_path is None and args.config_file is None:
        raise FileNotFoundError(
            "No log config or run_config.json found. Pass --log-file, --config-file, or add run_config.json at {}.".format(
                run_cache_dir / "run_config.json"
            )
        )
    other_args = {"exp_id": args.exp_id}
    other_args.update(run_config)
    if args.cpu or not torch.cuda.is_available():
        other_args["gpu"] = False
    if run_config_path is not None:
        print("Using config from {}".format(run_config_path))
    config = ConfigParser(
        "traffic_state_pred",
        "PDFormer",
        args.dataset,
        config_file=args.config_file,
        saved_model=True,
        train=False,
        other_args=other_args,
    )
    config["exp_id"] = args.exp_id
    get_logger(config)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = run_cache_dir / "model_cache" / f"PDFormer_{args.dataset}.m"
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    if args.check_only:
        check_checkpoint_config(config, checkpoint)
        return

    dataset = get_dataset(config)
    _, val_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    model = get_model(config, data_feature)
    executor = get_executor(config, model)
    load_checkpoint(executor, checkpoint)

    dataloader = val_data if args.split == "val" else test_data
    attn = executor.model.encoder_blocks[0].st_attn
    variants = [("full", [], {})]
    variants.extend([
        ("no_geo_group", ["geo"], {}),
        ("no_sem_group", ["sem"], {}),
        ("no_temporal_group", ["t"], {}),
    ])
    variants.extend((f"no_geo_head_{idx}", [], {"geo": [idx]}) for idx in range(attn.geo_num_heads))
    variants.extend((f"no_sem_head_{idx}", [], {"sem": [idx]}) for idx in range(attn.sem_num_heads))
    variants.extend((f"no_temporal_head_{idx}", [], {"t": [idx]}) for idx in range(attn.t_num_heads))

    rows = []
    for name, groups, heads in variants:
        set_all_head_ablations(executor.model, groups=groups, heads=heads)
        rows.append(evaluate_variant(executor, dataloader, name, max_batches=args.max_batches))
        print(name, rows[-1])
    set_all_head_ablations(executor.model)

    rows = add_deltas(rows)
    output_dir = Path(args.output_dir) if args.output_dir else taxiformer_root / "results" / "pdformer_head_ablation" / args.exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.dataset}_{args.split}_head_ablation.csv"
    write_csv(rows, csv_path)

    metric_keys = [key for key in rows[0] if key.startswith("MAE@") or key.startswith("masked_MAE@")]
    plot_metrics = ["val_loss"]
    if "MAE@1" in rows[0]:
        plot_metrics.append("MAE@1")
    if "masked_MAE@1" in rows[0]:
        plot_metrics.append("masked_MAE@1")
    elif metric_keys:
        plot_metrics.append(metric_keys[0])
    for metric in plot_metrics:
        plot_metric(
            rows,
            metric,
            output_dir / f"{args.dataset}_{args.split}_delta_{metric.replace('@', '_at_')}.png",
            f"PDFormer head ablation: delta {metric}",
        )
    print("Saved", csv_path)
    print("Saved plots in", output_dir)


if __name__ == "__main__":
    main()
