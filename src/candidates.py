from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree


def load_trip(trip_dir: str | Path) -> dict[str, np.ndarray | str]:
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    track = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
    ).to_numpy(dtype=np.float64)
    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
    ).to_numpy(dtype=np.float64)
    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
    ).to_numpy(dtype=np.int64)
    route_path = trip_dir / f"{trip_id}.route"
    route = (
        pd.read_csv(
            route_path,
            sep=r"\s+",
            header=None,
            names=["arc_id"],
        ).to_numpy(dtype=np.int64).ravel()
        if route_path.exists()
        else None
    )
    return {"trip_id": trip_id, "track": track, "nodes": nodes, "arcs": arcs, "route": route}


def project_trip(
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    northern = float(np.mean(lat)) >= 0.0
    zone = int((float(np.mean(lon)) + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    epsg = (32600 if northern else 32700) + zone
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y), epsg


def point_to_segment(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ab = b - a
    ab_len2 = np.einsum("ij,ij->i", ab, ab)
    ap = points[:, None, :] - a[None, :, :]
    t = np.einsum("bij,ij->bi", ap, ab) / np.where(ab_len2 > 1e-12, ab_len2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    projected = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    differences = points[:, None, :] - projected
    distances = np.linalg.norm(differences, axis=2)
    return distances, t


def nearest_candidate_arcs(
    track_xy: np.ndarray,
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    k: int = 12,
    endpoint_k: int = 60,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    arc_a = nodes_xy[arcs[:, 0]]
    arc_b = nodes_xy[arcs[:, 1]]
    arc_midpoints = (arc_a + arc_b) * 0.5
    tree = cKDTree(arc_midpoints)
    midpoint_neighbors = tree.query(track_xy, k=min(endpoint_k, len(arcs)))[1]

    candidates: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for i, point in enumerate(track_xy):
        neighbor_ids = np.atleast_1d(midpoint_neighbors[i]).astype(np.int64)
        neighbor_ids = np.unique(neighbor_ids)
        dist, projection = point_to_segment(
            point[None, :],
            arc_a[neighbor_ids],
            arc_b[neighbor_ids],
        )
        dist = dist.ravel()
        order = np.argsort(dist)[:k]
        candidates.append(neighbor_ids[order])
        distances.append(dist[order])
    return candidates, distances


def align_route_labels(
    route: np.ndarray,
    candidate_ids: list[np.ndarray],
    candidate_distances: list[np.ndarray],
) -> np.ndarray:
    labels = np.full(len(candidate_ids), -1, dtype=np.int64)
    route_set = set(map(int, route))
    route_positions = {int(arc_id): position for position, arc_id in enumerate(route)}
    search_start = 0

    for i, (ids, dists) in enumerate(zip(candidate_ids, candidate_distances)):
        matching = [j for j, arc_id in enumerate(ids) if int(arc_id) in route_set]
        if not matching:
            continue

        best = matching[0]
        best_score = float("inf")
        for j in matching:
            arc_id = int(ids[j])
            position = route_positions[arc_id]
            backward_penalty = max(0, search_start - position) * 1e6
            forward_penalty = abs(position - search_start) * 0.01
            score = float(dists[j]) + backward_penalty + forward_penalty
            if score < best_score:
                best_score = score
                best = j

        labels[i] = int(ids[best])
        search_start = max(search_start, route_positions[int(ids[best])])
    return labels


def summarize_trip(trip_dir: Path, k: int = 12) -> dict[str, float | int | str]:
    data = load_trip(trip_dir)
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    track_x, track_y, epsg = project_trip(track[:, 0], track[:, 1])
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    nodes_x, nodes_y = transformer.transform(nodes[:, 0], nodes[:, 1])
    track_xy = np.column_stack((track_x, track_y))
    nodes_xy = np.column_stack((nodes_x, nodes_y))

    ids, dists = nearest_candidate_arcs(track_xy, nodes_xy, arcs, k=k)
    if route is None:
        return {
            "trip_id": str(data["trip_id"]),
            "epsg": epsg,
            "observations": len(track),
            "candidate_miss_rate": 0.0,
            "topk_recall": 0.0,
        }

    route_set = set(map(int, route))
    candidate_hits = [int(bool(route_set.intersection(map(int, cands)))) for cands in ids]
    candidate_miss_rate = 1.0 - float(np.mean(candidate_hits))
    labels = align_route_labels(route, ids, dists)
    topk_recall = float(np.mean(labels >= 0))
    return {
        "trip_id": str(data["trip_id"]),
        "epsg": epsg,
        "observations": len(track),
        "candidate_miss_rate": candidate_miss_rate,
        "topk_recall": topk_recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trip_dir", type=Path)
    parser.add_argument("--k", type=int, default=12)
    args = parser.parse_args()
    print(summarize_trip(args.trip_dir, k=args.k))


if __name__ == "__main__":
    main()
