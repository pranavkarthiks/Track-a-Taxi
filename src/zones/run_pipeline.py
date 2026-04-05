import argparse
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--zones", default="src/zones/data/taxi_zones/taxi_zones.shp")
parser.add_argument("--adjacency-out", default="src/zones/data/taxi_zones_adjacency_matrix.csv")
parser.add_argument("--map-out", default="src/zones/data/taxi_zones_adjacency_map.png")
parser.add_argument("--graphml", default="")
parser.add_argument("--include-islands", action="store_true")
args = parser.parse_args()

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
venv_python = repo_root / ".venv" / "bin" / "python"
python = str(venv_python if venv_python.exists() else sys.executable)

generate_command = [
    str(python),
    "-m",
    "src.zones.generate_adj_grid",
    "--zones",
    args.zones,
    "--adjacency-out",
    args.adjacency_out,
]
if args.graphml:
    generate_command.extend(["--graphml", args.graphml])
if args.include_islands:
    generate_command.append("--include-islands")

try:
    subprocess.run(generate_command, check=True, cwd=repo_root)
    visualise_command = [
        str(python),
        "-m",
        "src.zones.visualise_adj_map",
        "--zones",
        args.zones,
        "--adjacency",
        args.adjacency_out,
        "--output",
        args.map_out,
    ]
    if args.include_islands:
        visualise_command.append("--include-islands")

    subprocess.run(visualise_command, check=True, cwd=repo_root)
except subprocess.CalledProcessError as exc:
    raise SystemExit(f"Zone pipeline failed while running: {' '.join(exc.cmd)}")
