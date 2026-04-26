#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path


LOG_FILES = [
    "log-job-mask-eval-3m.out",
    "log-job-mask-eval-3m-dist.out",
    "log-job-3m-base-cfg2.out",
    "log-job-3m-dist-fix.out",
]


def find_log_root(root: Path) -> Path:
    roots = []
    for first_log in sorted(root.rglob(LOG_FILES[0])):
        candidate = first_log.parent
        if all((candidate / name).is_file() for name in LOG_FILES):
            roots.append(candidate.resolve())

    if len(roots) != 1:
        print(
            f"Expected exactly one directory containing the four required logs, found {len(roots)}.",
            file=sys.stderr,
        )
        for path in roots:
            print(f"  {path}", file=sys.stderr)
        raise SystemExit(1)

    return roots[0]


def parse_exp_id(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Begin pipeline,.*exp_id=([0-9]+)", text)
    if not matches:
        print(f"Could not read exp_id from log: {log_path}", file=sys.stderr)
        raise SystemExit(1)
    return matches[-1]


def require_completed_model_line(log_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if not re.findall(r"Saved model at .*PDFormer_NYCTLC\.m", text):
        print(f"Could not read completed saved model from log: {log_path}", file=sys.stderr)
        raise SystemExit(1)


def match_checkpoint(log_root: Path, exp_id: str, log_path: Path) -> Path:
    matches = []
    for checkpoint in sorted(log_root.rglob("model_cache/PDFormer_NYCTLC.m")):
        run_dir = checkpoint.parent.parent.name
        if run_dir == exp_id:
            matches.append(checkpoint.resolve())

    if len(matches) == 1:
        return matches[0]

    if not matches:
        print(
            f"No real checkpoint under {log_root} matches exp_id={exp_id} from log: {log_path}",
            file=sys.stderr,
        )
    else:
        print(
            f"Multiple checkpoints under {log_root} match exp_id={exp_id} from log: {log_path}",
            file=sys.stderr,
        )
        for path in matches:
            print(f"  {path}", file=sys.stderr)
    raise SystemExit(1)


def run_ablation(exp_id: str, log_path: Path, checkpoint: Path) -> None:
    print(f"Running PDFormer head ablation for exp_id={exp_id}", flush=True)
    print(f"  output={log_path}", flush=True)
    print(f"  checkpoint={checkpoint}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "nyctlc_pdformer/pdformer_head_ablation.py",
            "--dataset",
            "NYCTLC",
            "--exp-id",
            exp_id,
            "--split",
            "val",
            "--log-file",
            str(log_path),
            "--checkpoint",
            str(checkpoint),
        ],
        check=True,
    )


def main() -> None:
    if len(sys.argv) != 1:
        print(f"Usage: {Path(sys.argv[0]).name}", file=sys.stderr)
        raise SystemExit(1)

    root = Path.cwd()
    log_root = find_log_root(root)

    for log_name in LOG_FILES:
        log_path = log_root / log_name
        exp_id = parse_exp_id(log_path)
        require_completed_model_line(log_path)
        checkpoint = match_checkpoint(log_root, exp_id, log_path)
        run_ablation(exp_id, log_path, checkpoint)


if __name__ == "__main__":
    main()
