import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import kagglehub
import pandas as pd
from libpysal.weights import Queen
from shapely.geometry import Point


def is_bridge_or_tunnel(edge_data):
    bridge = str(edge_data.get("bridge", "")).strip().lower()
    tunnel = str(edge_data.get("tunnel", "")).strip().lower()
    return bridge not in {"", "0", "false", "no", "none"} or tunnel not in {"", "0", "false", "no", "none"}


parser = argparse.ArgumentParser()
parser.add_argument("--zones", default="data/taxi_zones/taxi_zones.shp")
parser.add_argument("--adjacency-out", default="data/taxi_zones_adjacency_matrix.csv")
parser.add_argument("--graphml", default="")
args = parser.parse_args()


def add_bridge_tunnel_adjacency(adjacency, zones, graphml_path):
    root = ET.parse(graphml_path).getroot()
    namespace = {"graphml": "http://graphml.graphdrawing.org/xmlns"}
    key_map = {
        key.attrib["id"]: key.attrib.get("attr.name", "")
        for key in root.findall("graphml:key", namespace)
    }

    node_points = {}
    for node in root.findall(".//graphml:node", namespace):
        node_data = {
            key_map[data.attrib["key"]]: (data.text or "")
            for data in node.findall("graphml:data", namespace)
        }
        if "x" in node_data and "y" in node_data:
            node_points[node.attrib["id"]] = Point(float(node_data["x"]), float(node_data["y"]))

    bridge_tunnel_edges = []
    endpoint_ids = set()

    for edge in root.findall(".//graphml:edge", namespace):
        edge_data = {
            key_map[data.attrib["key"]]: (data.text or "")
            for data in edge.findall("graphml:data", namespace)
        }
        if not is_bridge_or_tunnel(edge_data):
            continue

        source = edge.attrib["source"]
        target = edge.attrib["target"]
        bridge_tunnel_edges.append((source, target))
        endpoint_ids.add(source)
        endpoint_ids.add(target)

    # GraphML node coordinates are lon/lat, so reproject them into the taxi zone CRS
    # before checking whether each bridge/tunnel endpoint falls within a zone polygon.
    endpoint_points = gpd.GeoDataFrame(
        {"node_id": sorted(endpoint_ids)},
        geometry=[node_points[node_id] for node_id in sorted(endpoint_ids)],
        crs="EPSG:4326",
    ).to_crs(zones.crs)

    # Only keep endpoints that directly intersect a taxi zone polygon.
    joined = gpd.sjoin(
        endpoint_points,
        zones[["LocationID", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    node_zones = (
        joined.groupby("node_id")["LocationID"]
        .apply(lambda values: sorted(set(values.astype(int))))
        .to_dict()
    )

    for source, target in bridge_tunnel_edges:
        for source_zone in node_zones.get(source, []):
            for target_zone in node_zones.get(target, []):
                if source_zone != target_zone:
                    adjacency.loc[source_zone, target_zone] = 1
                    adjacency.loc[target_zone, source_zone] = 1


graphml_path = Path(args.graphml) if args.graphml else Path(kagglehub.dataset_download("crailtap/street-network-of-new-york-in-graphml")) / "newyork.graphml"
zones = gpd.read_file(args.zones).sort_values("LocationID")
weights = Queen.from_dataframe(zones, ids=zones["LocationID"].astype(int).tolist())
matrix, ids = weights.full()
adjacency = pd.DataFrame(matrix.astype(int), index=ids, columns=ids)

add_bridge_tunnel_adjacency(adjacency, zones, graphml_path)

#  island/bridge corrections:
# Islands: 202 -> 193, 46 -> 184.

# Remove & disconnected islands (ie.  liberty island, Newark Airport, etc.)
connected = adjacency.index[adjacency.sum(axis=1) > 0]
adjacency = adjacency.loc[connected, connected]

adjacency.index.name = "LocationID"
adjacency.to_csv(args.adjacency_out)
