from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_zones(zones_path, adjacency_path=None, include_islands=False):
    zones = gpd.read_file(zones_path).sort_values("LocationID").copy()
    zones["LocationID"] = zones["LocationID"].astype(int)

    if include_islands or not adjacency_path:
        return zones

    adjacency = pd.read_csv(Path(adjacency_path), index_col=0)
    zone_ids = adjacency.index.astype(int).tolist()
    return zones.set_index("LocationID").loc[zone_ids].reset_index()
