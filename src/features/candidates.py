from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.spatial import cKDTree

from src.data.loader import TripData


@dataclass
class CandidateSet:
    trip_id: str
    k: int
    candidate_arc_ids: List[np.ndarray]     # List of length N, each array of length <= K (int64)
    perpendicular_dists: List[np.ndarray]   # List of length N, each array of length <= K (float64)
    projection_ts: List[np.ndarray]         # List of length N, each array of length <= K (float64 in [0, 1])
    projected_points: List[np.ndarray]      # List of length N, each array of shape (K, 2) (float64)
    candidate_counts: np.ndarray            # Array of length N (int32)


def point_to_segment_vectorized(
    point: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact perpendicular distance, projection fraction t, and projected point
    from a single 2D point (x, y) to multiple segments [A_i, B_i].
    """
    ab = seg_b - seg_a  # (S, 2)
    ab_len2 = np.sum(ab * ab, axis=1)  # (S,)
    ap = point - seg_a  # (S, 2)
    
    safe_len2 = np.where(ab_len2 > 1e-12, ab_len2, 1.0)
    t_raw = np.sum(ap * ab, axis=1) / safe_len2
    t = np.clip(t_raw, 0.0, 1.0)
    
    projected = seg_a + t[:, None] * ab
    diff = point - projected
    dist = np.linalg.norm(diff, axis=1)
    return dist, t, projected


class SpatialCandidateGenerator:
    """
    Generates spatial candidate road arcs for GPS observations with adaptive search radius,
    dual endpoint/midpoint spatial indexing, and exact nearest fallback.
    """

    def __init__(
        self,
        nodes_xy: np.ndarray,
        arcs: np.ndarray,
        radii: Tuple[float, ...] = (50.0, 100.0, 150.0),
        min_pool_size: int = 60,
    ):
        self.nodes_xy = np.asarray(nodes_xy, dtype=np.float64)
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.num_arcs = len(self.arcs)
        self.radii = sorted(radii)
        self.min_pool_size = min(min_pool_size, self.num_arcs)

        self.seg_a = self.nodes_xy[self.arcs[:, 0]]
        self.seg_b = self.nodes_xy[self.arcs[:, 1]]
        self.midpoints = (self.seg_a + self.seg_b) * 0.5

        # Midpoint spatial index
        self.midpoint_tree = cKDTree(self.midpoints)
        # Node spatial index
        self.node_tree = cKDTree(self.nodes_xy)

        # Mapping from node ID to arc IDs connected to it
        self.node_to_arcs: Dict[int, List[int]] = {i: [] for i in range(len(self.nodes_xy))}
        for arc_id, (u, v) in enumerate(self.arcs):
            self.node_to_arcs[int(u)].append(arc_id)
            self.node_to_arcs[int(v)].append(arc_id)

    def generate_candidates(
        self,
        track_xy: np.ndarray,
        k: int = 12,
    ) -> CandidateSet:
        track_xy = np.asarray(track_xy, dtype=np.float64)
        num_obs = len(track_xy)
        pool_target = max(self.min_pool_size, k * 4)

        candidate_arc_ids: List[np.ndarray] = []
        perpendicular_dists: List[np.ndarray] = []
        projection_ts: List[np.ndarray] = []
        projected_points: List[np.ndarray] = []
        counts = np.zeros(num_obs, dtype=np.int32)

        # Fast batched query of nearest midpoints to ensure high recall
        k_mid = min(pool_target, self.num_arcs)
        _, all_mid_neighbors = self.midpoint_tree.query(track_xy, k=k_mid)
        all_mid_neighbors = np.atleast_2d(all_mid_neighbors)

        for i in range(num_obs):
            pt = track_xy[i]
            matched_arc_set = set(all_mid_neighbors[i].tolist())

            # Also check adaptive radius for any additional nearby nodes/arcs
            for radius in self.radii:
                node_indices = self.node_tree.query_ball_point(pt, r=radius)
                for nid in node_indices:
                    matched_arc_set.update(self.node_to_arcs[nid])
                if len(matched_arc_set) >= pool_target:
                    break

            cand_pool = np.array(list(matched_arc_set), dtype=np.int64)
            if len(cand_pool) == 0:
                cand_pool = np.arange(min(k, self.num_arcs), dtype=np.int64)

            # Compute exact finite-segment distance for all candidate arcs in pool
            sub_a = self.seg_a[cand_pool]
            sub_b = self.seg_b[cand_pool]
            dists, ts, projs = point_to_segment_vectorized(pt, sub_a, sub_b)

            # Sort by perpendicular distance and take top K
            sort_idx = np.argsort(dists)[:k]
            top_cand_ids = cand_pool[sort_idx]
            top_dists = dists[sort_idx]
            top_ts = ts[sort_idx]
            top_projs = projs[sort_idx]

            candidate_arc_ids.append(top_cand_ids)
            perpendicular_dists.append(top_dists)
            projection_ts.append(top_ts)
            projected_points.append(top_projs)
            counts[i] = len(top_cand_ids)

        return CandidateSet(
            trip_id="",
            k=k,
            candidate_arc_ids=candidate_arc_ids,
            perpendicular_dists=perpendicular_dists,
            projection_ts=projection_ts,
            projected_points=projected_points,
            candidate_counts=counts,
        )
