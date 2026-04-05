import argparse

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt

from src.zones.zone_utils import load_zones


parser = argparse.ArgumentParser()
parser.add_argument("--zones", default="src/zones/data/taxi_zones/taxi_zones.shp")
parser.add_argument(
    "--adjacency",
    default="src/zones/data/taxi_zones_adjacency_matrix.csv",
)
parser.add_argument(
    "--coordinates",
    default="src/zones/data/taxi_zones/taxi_zone_lookup_coordinates.csv",
)
parser.add_argument(
    "--output",
    default="src/zones/data/taxi_zones_centroids_map.png",
)
parser.add_argument("--include-islands", action="store_true")
args = parser.parse_args()


zones = load_zones(
    args.zones,
    adjacency_path=args.adjacency,
    include_islands=args.include_islands,
).copy()
coords = pd.read_csv(args.coordinates)

points = gpd.GeoDataFrame(
    coords,
    geometry=gpd.points_from_xy(coords["longitude"], coords["latitude"]),
    crs="EPSG:4326",
).to_crs(zones.crs)

fig, ax = plt.subplots(figsize=(14, 14))
zones.plot(ax=ax, facecolor="#f3f1ea", edgecolor="#555555", linewidth=0.35)
points.plot(ax=ax, color="#d62828", markersize=8, alpha=0.8)

ax.set_axis_off()
ax.set_title("NYC Taxi Zone Polygons with Derived Coordinate Points", fontsize=16)
fig.savefig(args.output, dpi=300, bbox_inches="tight")
print(f"Wrote centroid map to {args.output}")
