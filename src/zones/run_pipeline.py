import argparse
from pathlib import Path
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--zones", default="data/taxi_zones/taxi_zones.shp")
parser.add_argument("--adjacency-out", default="data/taxi_zones_adjacency_matrix.csv")
parser.add_argument("--map-out", default="data/taxi_zones_adjacency_map.png")
parser.add_argument("--graphml", default="")
args = parser.parse_args()

python = Path(".venv/bin/python")

generate_command = [
    str(python),
    "generate_adj_grid.py",
    "--zones",
    args.zones,
    "--adjacency-out",
    args.adjacency_out,
]
if args.graphml:
    generate_command.extend(["--graphml", args.graphml])

subprocess.run(generate_command, check=True)
subprocess.run(
    [
        str(python),
        "visualise_adj_map.py",
        "--zones",
        args.zones,
        "--adjacency",
        args.adjacency_out,
        "--output",
        args.map_out,
    ],
    check=True,
)
