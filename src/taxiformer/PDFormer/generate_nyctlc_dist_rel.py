import argparse
import csv
import math
import struct
from pathlib import Path


def build_parser():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate centroid-distance relation files for the NYCTLC PDFormer dataset."
    )
    parser.add_argument(
        "--coords",
        default=str(root.parent.parent / "zones" / "data" / "taxi_zones" / "taxi_zone_lookup_coordinates.csv"),
        help="CSV containing LocationID, latitude, and longitude columns.",
    )
    parser.add_argument(
        "--adjacency",
        default=str(root / "raw_data" / "NYCTLC" / "taxi_zones_adjacency_matrix.csv"),
        help="Adjacency matrix CSV for the NYCTLC taxi zones.",
    )
    parser.add_argument(
        "--output-rel",
        default=str(root / "raw_data" / "NYCTLC" / "NYCTLC_dist.rel"),
        help="Output .rel file with centroid distances on adjacent edges.",
    )
    parser.add_argument(
        "--output-full-dist",
        default=str(root / "raw_data" / "NYCTLC" / "NYCTLC_full_dist.npy"),
        help="Output .npy file with the full pairwise centroid distance matrix.",
    )
    return parser


def load_coordinates(path):
    coords = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            zone_id = int(row["LocationID"])
            coords[zone_id] = (float(row["latitude"]), float(row["longitude"]))
    if not coords:
        raise ValueError(f"No coordinates loaded from {path}")
    return coords


def load_adjacency(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        zone_ids = [int(value) for value in header[1:]]
        matrix = []
        for row in reader:
            row_zone = int(row[0])
            values = [float(value) for value in row[1:]]
            matrix.append((row_zone, values))
    if [row_zone for row_zone, _ in matrix] != zone_ids:
        raise ValueError("Adjacency matrix rows/columns do not share the same ordered zone IDs.")
    return zone_ids, matrix


def haversine_km(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    term = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(term))


def build_full_distance_matrix(zone_ids, coords):
    matrix = [[0.0 for _ in zone_ids] for _ in zone_ids]
    for i, origin_id in enumerate(zone_ids):
        if origin_id not in coords:
            raise KeyError(f"Missing coordinates for zone {origin_id}")
        for j in range(i + 1, len(zone_ids)):
            destination_id = zone_ids[j]
            if destination_id not in coords:
                raise KeyError(f"Missing coordinates for zone {destination_id}")
            distance = haversine_km(coords[origin_id], coords[destination_id])
            matrix[i][j] = distance
            matrix[j][i] = distance
    return matrix


def write_rel(path, zone_ids, adjacency_rows, full_dist_mx):
    zone_to_ind = {zone_id: idx for idx, zone_id in enumerate(zone_ids)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rel_id", "type", "origin_id", "destination_id", "weight"])
        rel_id = 0
        for origin_id, adjacency_values in adjacency_rows:
            for destination_id, connected in zip(zone_ids, adjacency_values):
                if origin_id == destination_id or connected <= 0:
                    continue
                distance = full_dist_mx[zone_to_ind[origin_id]][zone_to_ind[destination_id]]
                writer.writerow([rel_id, "geo", origin_id, destination_id, f"{distance:.6f}"])
                rel_id += 1


def write_npy(path, matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    header = (
        "{'descr': '<f4', 'fortran_order': False, 'shape': (" + str(rows) + ", " + str(cols) + "), }"
    ).encode("latin1")
    preamble_len = 10
    padding = (16 - ((preamble_len + len(header) + 1) % 16)) % 16
    header += b" " * padding + b"\n"
    with open(path, "wb") as handle:
        handle.write(b"\x93NUMPY")
        handle.write(bytes([1, 0]))
        handle.write(struct.pack("<H", len(header)))
        handle.write(header)
        for row in matrix:
            for value in row:
                handle.write(struct.pack("<f", value))


def main():
    args = build_parser().parse_args()
    coords = load_coordinates(Path(args.coords))
    zone_ids, adjacency_rows = load_adjacency(Path(args.adjacency))
    full_dist_mx = build_full_distance_matrix(zone_ids, coords)
    zone_to_ind = {zone_id: idx for idx, zone_id in enumerate(zone_ids)}

    output_rel = Path(args.output_rel)
    output_full_dist = Path(args.output_full_dist)
    write_rel(output_rel, zone_ids, adjacency_rows, full_dist_mx)
    output_full_dist.parent.mkdir(parents=True, exist_ok=True)
    write_npy(output_full_dist, full_dist_mx)

    adjacent_distances = []
    for origin_id, adjacency_values in adjacency_rows:
        origin_index = zone_to_ind[origin_id]
        for destination_id, connected in zip(zone_ids, adjacency_values):
            if origin_id >= destination_id or connected <= 0:
                continue
            destination_index = zone_to_ind[destination_id]
            adjacent_distances.append(full_dist_mx[origin_index][destination_index])

    print(f"Wrote {output_rel}")
    print(f"Wrote {output_full_dist}")
    print("Adjacent edge distances (km): "
          f"min={min(adjacent_distances):.3f}, "
          f"median={sorted(adjacent_distances)[len(adjacent_distances) // 2]:.3f}, "
          f"max={max(adjacent_distances):.3f}")


if __name__ == "__main__":
    main()
