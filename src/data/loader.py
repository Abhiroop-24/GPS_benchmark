from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix

from src.data.projector import UTMProjector


@dataclass
class TripData:
    trip_id: str
    partition: str
    track: np.ndarray        # shape (N, 3): lon, lat, timestamp (float64)
    nodes: np.ndarray        # shape (M, 2): lon, lat (float64)
    arcs: np.ndarray         # shape (A, 2): from_node, to_node (int64)
    route: Optional[np.ndarray] = None  # shape (R,): arc_ids (int64)
    
    # Projected coordinates (UTM in meters)
    projector: Optional[UTMProjector] = None
    track_xy: Optional[np.ndarray] = None  # shape (N, 2): x, y
    nodes_xy: Optional[np.ndarray] = None  # shape (M, 2): x, y
    epsg: Optional[int] = None
    
    validation_report: Dict[str, Any] = field(default_factory=dict)


def load_raw_trip(
    trip_dir: str | Path,
    partition: str = "train",
    require_route: bool = False,
) -> TripData:
    """Load and validate raw trip files from a trip directory."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    missing = []
    for f, name in [(track_file, ".track"), (nodes_file, ".nodes"), (arcs_file, ".arcs")]:
        if not f.exists():
            missing.append(name)
    if require_route and not route_file.exists():
        missing.append(".route")

    if missing:
        raise FileNotFoundError(f"Trip {trip_id} is missing required files: {missing}")

    # Read .track
    track_df = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype={"lon": np.float64, "lat": np.float64, "timestamp": np.float64},
    )
    track = track_df.to_numpy(dtype=np.float64)

    # Read .nodes
    nodes_df = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64},
    )
    nodes = nodes_df.to_numpy(dtype=np.float64)

    # Read .arcs
    arcs_df = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={"from_node": np.int64, "to_node": np.int64},
    )
    arcs = arcs_df.to_numpy(dtype=np.int64)

    # Read .route if available
    route = None
    if route_file.exists():
        route_df = pd.read_csv(
            route_file,
            sep=r"\s+",
            header=None,
            names=["arc_id"],
            dtype={"arc_id": np.int64},
        )
        route = route_df["arc_id"].to_numpy(dtype=np.int64)

    # Perform validation
    validation = validate_trip_data(trip_id, track, nodes, arcs, route)

    # Compute UTM projection
    projector = UTMProjector.from_coordinates(track[:, 0], track[:, 1])
    tx, ty = projector.forward(track[:, 0], track[:, 1])
    nx, ny = projector.forward(nodes[:, 0], nodes[:, 1])
    track_xy = np.column_stack((tx, ty))
    nodes_xy = np.column_stack((nx, ny))

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
        projector=projector,
        track_xy=track_xy,
        nodes_xy=nodes_xy,
        epsg=projector.epsg,
        validation_report=validation,
    )


def validate_trip_data(
    trip_id: str,
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    route: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Perform comprehensive data integrity checks on a trip."""
    report: Dict[str, Any] = {"trip_id": trip_id, "issues": []}

    # 1. Coordinate validity & NaN / Inf
    for arr, name in [(track, "track"), (nodes, "nodes")]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            report["issues"].append(f"{name} contains NaN or Inf values")

    if not (-180.0 <= track[:, 0].min() and track[:, 0].max() <= 180.0):
        report["issues"].append("track longitude out of bounds [-180, 180]")
    if not (-90.0 <= track[:, 1].min() and track[:, 1].max() <= 90.0):
        report["issues"].append("track latitude out of bounds [-90, 90]")
    if not (-180.0 <= nodes[:, 0].min() and nodes[:, 0].max() <= 180.0):
        report["issues"].append("nodes longitude out of bounds [-180, 180]")
    if not (-90.0 <= nodes[:, 1].min() and nodes[:, 1].max() <= 90.0):
        report["issues"].append("nodes latitude out of bounds [-90, 90]")

    # 2. Timestamp ordering & intervals
    timestamps = track[:, 2]
    dt = np.diff(timestamps)
    ordered = bool((dt >= 0).all())
    strictly_ordered = bool((dt > 0).all())
    duplicate_timestamps = int((dt == 0).sum())
    irregular_timestamps = int((np.abs(dt - 1.0) > 1e-3).sum())
    large_gaps = int((dt > 5.0).sum())
    max_gap_s = float(dt.max()) if len(dt) > 0 else 0.0

    report["timestamps_ordered"] = ordered
    report["duplicate_timestamps"] = duplicate_timestamps
    report["irregular_timestamps"] = irregular_timestamps
    report["large_gaps_gt_5s"] = large_gaps
    report["max_gap_seconds"] = max_gap_s

    # 3. Arcs and Nodes validity
    num_nodes = len(nodes)
    num_arcs = len(arcs)
    invalid_from = (arcs[:, 0] < 0) | (arcs[:, 0] >= num_nodes)
    invalid_to = (arcs[:, 1] < 0) | (arcs[:, 1] >= num_nodes)
    if invalid_from.any() or invalid_to.any():
        report["issues"].append("arcs reference invalid node IDs outside [0, num_nodes)")

    # Zero-length arcs
    zero_len_arcs = int((arcs[:, 0] == arcs[:, 1]).sum())
    report["zero_length_arcs"] = zero_len_arcs

    # 4. Route validity if present
    if route is not None:
        invalid_route = (route < 0) | (route >= num_arcs)
        if invalid_route.any():
            report["issues"].append("route references invalid arc IDs outside [0, num_arcs)")
        
        # Check repeated arcs in route
        unique_arcs = len(np.unique(route))
        report["route_length"] = len(route)
        report["route_unique_arcs"] = unique_arcs
        report["route_repeated_arcs"] = len(route) - unique_arcs
        
        # Check connectivity of consecutive route arcs
        # Consecutive arcs: arc[i].to_node == arc[i+1].from_node
        disconnected_route_steps = 0
        for i in range(len(route) - 1):
            curr_to = arcs[route[i], 1]
            next_from = arcs[route[i + 1], 0]
            if curr_to != next_from:
                disconnected_route_steps += 1
        report["disconnected_route_steps"] = disconnected_route_steps

    # 5. Network topology analysis
    in_degrees = np.bincount(arcs[:, 1], minlength=num_nodes)
    out_degrees = np.bincount(arcs[:, 0], minlength=num_nodes)
    isolated_nodes = int(((in_degrees == 0) & (out_degrees == 0)).sum())
    source_nodes = int(((in_degrees == 0) & (out_degrees > 0)).sum())
    sink_nodes = int(((in_degrees > 0) & (out_degrees == 0)).sum())

    # Weakly connected components
    adj = csr_matrix((np.ones(num_arcs), (arcs[:, 0], arcs[:, 1])), shape=(num_nodes, num_nodes))
    n_components, labels = csgraph.connected_components(adj, directed=False)

    report["num_observations"] = len(track)
    report["num_nodes"] = num_nodes
    report["num_arcs"] = num_arcs
    report["isolated_nodes"] = isolated_nodes
    report["source_nodes"] = source_nodes
    report["sink_nodes"] = sink_nodes
    report["weak_components"] = int(n_components)

    # 6. Extents
    report["track_lon_min"] = float(track[:, 0].min())
    report["track_lon_max"] = float(track[:, 0].max())
    report["track_lat_min"] = float(track[:, 1].min())
    report["track_lat_max"] = float(track[:, 1].max())

    return report
