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

output = zones[["LocationID", "borough", "zone"]].copy()
output = output.rename(columns={"borough": "Borough", "zone": "Zone"})
output["service_zone"] = ""
output["latitude"] = points_wgs84.y
output["longitude"] = points_wgs84.x

source_lookup = Path(args.output).with_name("taxi_zone_lookup.csv")
if source_lookup.exists():
    lookup = pd.read_csv(source_lookup)
    lookup = lookup[["LocationID", "service_zone"]].drop_duplicates(subset=["LocationID"])
    output = output.drop(columns=["service_zone"]).merge(lookup, on="LocationID", how="left")

output.to_csv(args.output, index=False)
print(f"Wrote zone coordinate lookup to {args.output}")
