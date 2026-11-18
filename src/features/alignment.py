from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.candidates import CandidateSet


@dataclass
class AlignmentResult:
    trip_id: str
    aligned_arc_ids: np.ndarray          # shape (N,), arc ID or -1
    aligned_route_indices: np.ndarray    # shape (N,), index in route or -1
    is_matched: np.ndarray               # shape (N,), bool
    is_ambiguous: np.ndarray             # shape (N,), bool
    alignment_confidence: np.ndarray     # shape (N,), float
    labels_per_candidate: List[np.ndarray] # List of length N, binary 0/1 array for each candidate
    candidate_miss_rate: float
    topk_recall: float


def align_trip_candidates(
    route: np.ndarray,
    candidates: CandidateSet,
    miss_penalty: float = 40.0,
    backward_penalty: float = 1e6,
) -> AlignmentResult:
    """
    Monotonically aligns GPS observation candidates to reference route arcs using dynamic programming.
    Preserves chronological order, handles repeated arcs and route loops gracefully,
    and identifies candidate misses without false-positive leakage.
    """
    route = np.asarray(route, dtype=np.int64)
    R = len(route)
    N = len(candidates.candidate_arc_ids)

    if R == 0 or N == 0:
        empty_aligned = np.full(N, -1, dtype=np.int64)
        return AlignmentResult(
            trip_id=candidates.trip_id,
            aligned_arc_ids=empty_aligned,
            aligned_route_indices=empty_aligned,
            is_matched=np.zeros(N, dtype=bool),
            is_ambiguous=np.zeros(N, dtype=bool),
            alignment_confidence=np.zeros(N, dtype=np.float64),
            labels_per_candidate=[np.zeros(len(c), dtype=np.int64) for c in candidates.candidate_arc_ids],
            candidate_miss_rate=1.0,
            topk_recall=0.0,
        )

    # Inverted index: arc_id -> list of route indices where this arc appears
    arc_to_route_positions: Dict[int, List[int]] = {}
    for pos, arc_id in enumerate(route):
        arc_to_route_positions.setdefault(int(arc_id), []).append(pos)

    # States at step i: list of tuples (route_idx, cand_idx, emission_cost)
    # Plus the MISS state (-1, -1, miss_penalty)
    step_states: List[List[Tuple[int, int, float]]] = []

    for i in range(N):
        cands = candidates.candidate_arc_ids[i]
        dists = candidates.perpendicular_dists[i]
        states_i: List[Tuple[int, int, float]] = []

        for cand_idx, arc_id in enumerate(cands):
            arc_int = int(arc_id)
            if arc_int in arc_to_route_positions:
                for r_pos in arc_to_route_positions[arc_int]:
                    states_i.append((r_pos, cand_idx, float(dists[cand_idx])))

        # Always include MISS state
        states_i.append((-1, -1, float(miss_penalty)))
        step_states.append(states_i)

    # DP matrices
    # cost_dp[i][s_idx]
    # backpointer[i][s_idx]
    dp_costs: List[np.ndarray] = []
    backpointers: List[np.ndarray] = []

    # Step 0 initialization
    init_costs = np.array([s[2] + (0.0 if s[0] == -1 else s[0] * 0.1) for s in step_states[0]], dtype=np.float64)
    dp_costs.append(init_costs)
    backpointers.append(np.full(len(step_states[0]), -1, dtype=np.int32))

    for i in range(1, N):
        prev_states = step_states[i - 1]
        curr_states = step_states[i]
        prev_cost = dp_costs[i - 1]

        n_prev = len(prev_states)
        n_curr = len(curr_states)

        curr_cost_arr = np.empty(n_curr, dtype=np.float64)
        curr_bp_arr = np.empty(n_curr, dtype=np.int32)

        for curr_idx, (curr_r, curr_c, curr_emit) in enumerate(curr_states):
            best_val = float("inf")
            best_prev_idx = -1

            for prev_idx, (prev_r, prev_c, _) in enumerate(prev_states):
                # Calculate transition penalty
                if prev_r == -1:  # Coming from MISS
                    trans_cost = 15.0 if curr_r != -1 else 0.0
                elif curr_r == -1:  # Going to MISS
                    trans_cost = 25.0
                else:  # Both matched
                    step_diff = curr_r - prev_r
                    if step_diff == 0:
                        trans_cost = 0.0
                    elif 0 < step_diff <= 2:
                        trans_cost = step_diff * 1.5
                    elif step_diff > 2:
                        trans_cost = 5.0 + step_diff * 0.8
                    else:  # Backward movement: step_diff < 0
                        trans_cost = backward_penalty + abs(step_diff) * 1000.0

                total_cost = prev_cost[prev_idx] + trans_cost
                if total_cost < best_val:
                    best_val = total_cost
                    best_prev_idx = prev_idx

            curr_cost_arr[curr_idx] = best_val + curr_emit
            curr_bp_arr[curr_idx] = best_prev_idx

        dp_costs.append(curr_cost_arr)
        backpointers.append(curr_bp_arr)

    # Backtracking
    aligned_arc_ids = np.full(N, -1, dtype=np.int64)
    aligned_route_indices = np.full(N, -1, dtype=np.int64)
    is_matched = np.zeros(N, dtype=bool)
    is_ambiguous = np.zeros(N, dtype=bool)
    confidence = np.zeros(N, dtype=np.float64)
    labels_per_cand: List[np.ndarray] = []

    # Find minimum cost at final step
    best_last_idx = int(np.argmin(dp_costs[-1]))
    curr_bp = best_last_idx

    path: List[Tuple[int, int]] = []  # list of (r_pos, cand_idx)
    for i in range(N - 1, -1, -1):
        r_pos, c_idx, _ = step_states[i][curr_bp]
        path.append((r_pos, c_idx))
        curr_bp = backpointers[i][curr_bp]
    path.reverse()

    for i in range(N):
        r_pos, c_idx = path[i]
        num_cands = len(candidates.candidate_arc_ids[i])
        cand_labels = np.zeros(num_cands, dtype=np.int64)

        if r_pos != -1 and c_idx != -1:
            matched_arc = int(candidates.candidate_arc_ids[i][c_idx])
            aligned_arc_ids[i] = matched_arc
            aligned_route_indices[i] = r_pos
            is_matched[i] = True
            cand_labels[c_idx] = 1

            # Check ambiguity: another candidate close in distance that also matches near r_pos
            best_dist = candidates.perpendicular_dists[i][c_idx]
            ambig = False
            for j in range(num_cands):
                if j != c_idx:
                    other_arc = int(candidates.candidate_arc_ids[i][j])
                    other_positions = arc_to_route_positions.get(other_arc, [])
                    if any(abs(p - r_pos) <= 2 for p in other_positions):
                        other_dist = candidates.perpendicular_dists[i][j]
                        if abs(other_dist - best_dist) < 15.0:
                            ambig = True
                            break
            is_ambiguous[i] = ambig
            confidence[i] = max(0.1, 1.0 / (1.0 + best_dist / 10.0))
        else:
            confidence[i] = 0.0

        labels_per_cand.append(cand_labels)

    matched_count = int(is_matched.sum())
    topk_recall = float(matched_count / max(N, 1))
    candidate_miss_rate = 1.0 - topk_recall

    return AlignmentResult(
        trip_id=candidates.trip_id,
        aligned_arc_ids=aligned_arc_ids,
        aligned_route_indices=aligned_route_indices,
        is_matched=is_matched,
        is_ambiguous=is_ambiguous,
        alignment_confidence=confidence,
        labels_per_candidate=labels_per_cand,
        candidate_miss_rate=candidate_miss_rate,
        topk_recall=topk_recall,
    )


def save_alignment_audit(
    trip_id: str,
    candidates: CandidateSet,
    alignment: AlignmentResult,
    output_path: str | Path,
) -> pd.DataFrame:
    """Save an alignment audit CSV detailing each observation match."""
    rows = []
    N = len(candidates.candidate_arc_ids)
    for i in range(N):
        rows.append({
            "observation_idx": i,
            "is_matched": bool(alignment.is_matched[i]),
            "aligned_arc_id": int(alignment.aligned_arc_ids[i]),
            "aligned_route_idx": int(alignment.aligned_route_indices[i]),
            "is_ambiguous": bool(alignment.is_ambiguous[i]),
            "confidence": float(alignment.alignment_confidence[i]),
            "num_candidates": len(candidates.candidate_arc_ids[i]),
            "nearest_cand_id": int(candidates.candidate_arc_ids[i][0]) if len(candidates.candidate_arc_ids[i]) > 0 else -1,
            "nearest_cand_dist": float(candidates.perpendicular_dists[i][0]) if len(candidates.perpendicular_dists[i]) > 0 else 999.0,
        })
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
