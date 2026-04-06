import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.zones.zone_utils import load_zones


parser = argparse.ArgumentParser()
parser.add_argument("--zones", default="src/zones/data/taxi_zones/taxi_zones.shp")
parser.add_argument(
    "--adjacency",
    default="src/zones/data/taxi_zones_adjacency_matrix.csv",
)
parser.add_argument(
    "--lookup",
    default="src/zones/data/taxi_zones/taxi_zone_lookup.csv",
)
parser.add_argument(
    "--output",
    default="src/zones/data/taxi_zones/taxi_zone_lookup_coordinates.csv",
)
parser.add_argument("--include-islands", action="store_true")
args = parser.parse_args()


zones = load_zones(
    args.zones,
    adjacency_path=args.adjacency,
    include_islands=args.include_islands,
).copy()

# Use an interior point so the coordinate is guaranteed to lie within the zone polygon.
points = zones.geometry.representative_point()
points_wgs84 = gpd.GeoSeries(points, crs=zones.crs).to_crs(epsg=4326)

lookup = pd.read_csv(args.lookup)
lookup["LocationID"] = lookup["LocationID"].astype(int)
lookup = lookup[
    ["LocationID", "Borough", "Zone", "service_zone"]
].drop_duplicates(subset=["LocationID"])

output = zones[["LocationID"]].copy().merge(lookup, on="LocationID", how="left")
output["latitude"] = points_wgs84.y
output["longitude"] = points_wgs84.x

output.to_csv(args.output, index=False)
print(f"Wrote zone coordinate lookup to {args.output}")
