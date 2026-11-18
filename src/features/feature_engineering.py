from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.loader import TripData
from src.features.candidates import CandidateSet
from src.features.alignment import AlignmentResult

FEATURE_SCHEMA_VERSION = "v1.0"

FEATURE_COLUMNS = [
    "perp_distance",
    "dist_to_start",
    "dist_to_end",
    "proj_fraction",
    "segment_len",
    "seg_bearing",
    "gps_bearing",
    "bearing_diff",
    "sin_bearing_diff",
    "cos_bearing_diff",
    "gps_speed",
    "timestamp_gap",
    "dist_from_prev_gps",
    "dist_to_next_gps",
    "candidate_rank",
    "candidate_count",
    "nearest_to_arc_score",
    "dist_diff_from_nearest",
    "trip_mean_speed",
    "trip_total_obs",
]


def compute_bearing(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compute bearing in radians [0, 2*pi)."""
    bearing = np.arctan2(dx, dy)
    return np.where(bearing < 0, bearing + 2 * np.pi, bearing)


def extract_trip_features(
    trip: TripData,
    candidates: CandidateSet,
    alignment: Optional[AlignmentResult] = None,
) -> pd.DataFrame:
    """
    Extract candidate-level features for a trip.
    Generates one row per (observation, candidate) pair.
    """
    track_xy = trip.track_xy
    N = len(track_xy)
    timestamps = trip.track[:, 2]

    # Precompute GPS movement features
    dt = np.zeros(N, dtype=np.float64)
    dt[1:] = np.maximum(timestamps[1:] - timestamps[:-1], 1e-4)

    disp_prev = np.zeros(N, dtype=np.float64)
    disp_prev[1:] = np.linalg.norm(track_xy[1:] - track_xy[:-1], axis=1)

    disp_next = np.zeros(N, dtype=np.float64)
    disp_next[:-1] = np.linalg.norm(track_xy[1:] - track_xy[:-1], axis=1)

    speed = np.zeros(N, dtype=np.float64)
    speed[1:] = disp_prev[1:] / dt[1:]
    if N > 1:
        speed[0] = speed[1]

    trip_mean_speed = float(np.mean(speed))

    # GPS bearings
    gps_bearing = np.zeros(N, dtype=np.float64)
    if N > 1:
        dx_fwd = track_xy[1:, 0] - track_xy[:-1, 0]
        dy_fwd = track_xy[1:, 1] - track_xy[:-1, 1]
        b_fwd = compute_bearing(dx_fwd, dy_fwd)
        gps_bearing[:-1] = b_fwd
        gps_bearing[-1] = b_fwd[-1]

    # Segments geometry
    seg_a = trip.nodes_xy[trip.arcs[:, 0]]
    seg_b = trip.nodes_xy[trip.arcs[:, 1]]
    ab = seg_b - seg_a
    seg_lens = np.linalg.norm(ab, axis=1)
    seg_bearings = compute_bearing(ab[:, 0], ab[:, 1])

    rows = []
    for i in range(N):
        pt = track_xy[i]
        cands = candidates.candidate_arc_ids[i]
        dists = candidates.perpendicular_dists[i]
        ts = candidates.projection_ts[i]
        num_cands = len(cands)
        min_dist = dists[0] if num_cands > 0 else 0.0

        for rank, arc_id in enumerate(cands):
            arc_id = int(arc_id)
            d_start = float(np.linalg.norm(pt - seg_a[arc_id]))
            d_end = float(np.linalg.norm(pt - seg_b[arc_id]))
            s_len = float(seg_lens[arc_id])
            s_bearing = float(seg_bearings[arc_id])
            g_bearing = float(gps_bearing[i])

            # Bearing difference in [0, pi]
            b_diff = abs(s_bearing - g_bearing)
            if b_diff > np.pi:
                b_diff = 2 * np.pi - b_diff

            row = {
                "trip_id": trip.trip_id,
                "observation_idx": i,
                "timestamp": float(timestamps[i]),
                "candidate_arc_id": arc_id,
                "perp_distance": float(dists[rank]),
                "dist_to_start": d_start,
                "dist_to_end": d_end,
                "proj_fraction": float(ts[rank]),
                "segment_len": s_len,
                "seg_bearing": s_bearing,
                "gps_bearing": g_bearing,
                "bearing_diff": b_diff,
                "sin_bearing_diff": float(np.sin(b_diff)),
                "cos_bearing_diff": float(np.cos(b_diff)),
                "gps_speed": float(speed[i]),
                "timestamp_gap": float(dt[i]),
                "dist_from_prev_gps": float(disp_prev[i]),
                "dist_to_next_gps": float(disp_next[i]),
                "candidate_rank": rank,
                "candidate_count": num_cands,
                "nearest_to_arc_score": float(np.exp(-dists[rank] / 20.0)),
                "dist_diff_from_nearest": float(dists[rank] - min_dist),
                "trip_mean_speed": trip_mean_speed,
                "trip_total_obs": N,
            }

            if alignment is not None:
                row["label"] = int(alignment.labels_per_candidate[i][rank])

            rows.append(row)

    df = pd.DataFrame(rows)
    return df
