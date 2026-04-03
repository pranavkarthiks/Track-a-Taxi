import geopandas as gpd


def reproject_lon_lat_points(point_ids, point_lookup, target_crs):
    return gpd.GeoDataFrame(
        {"node_id": list(point_ids)},
        geometry=[point_lookup[point_id] for point_id in point_ids],
        crs="EPSG:4326",
    ).to_crs(target_crs)
