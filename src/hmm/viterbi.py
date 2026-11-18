from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.features.candidates import CandidateSet
from src.hmm.emission import compute_log_emissions
from src.hmm.transitions import RoadGraph, TransitionModel
from src.evaluation.metrics import calculate_route_continuity


@dataclass
class ViterbiResult:
    trip_id: str
    predicted_arcs: np.ndarray      # shape (N,), int64
    classifier_arcs: np.ndarray     # shape (N,), int64 (Top-1 without HMM)
    log_likelihood: float
    route_continuity_hmm: float
    route_continuity_classifier: float
    disconnected_steps_hmm: int
    disconnected_steps_classifier: int


class ViterbiDecoder:
    """Log-space Viterbi sequence decoder for map-matching."""

    def __init__(
        self,
        nodes_xy: np.ndarray,
        arcs: np.ndarray,
        beta: float = 15.0,
        disconnected_penalty: float = -45.0,
    ):
        self.road_graph = RoadGraph(nodes_xy, arcs)
        self.transition_model = TransitionModel(
            self.road_graph, beta=beta, disconnected_penalty=disconnected_penalty
        )
        self.arcs = np.asarray(arcs, dtype=np.int64)

    def decode(
        self,
        trip_id: str,
        track_xy: np.ndarray,
        timestamps: np.ndarray,
        candidates: CandidateSet,
        candidate_probabilities: List[np.ndarray],
    ) -> ViterbiResult:
        N = len(track_xy)
        if N == 0:
            return ViterbiResult(
                trip_id=trip_id,
                predicted_arcs=np.array([], dtype=np.int64),
                classifier_arcs=np.array([], dtype=np.int64),
                log_likelihood=0.0,
                route_continuity_hmm=1.0,
                route_continuity_classifier=1.0,
                disconnected_steps_hmm=0,
                disconnected_steps_classifier=0,
            )

        # 1. Compute log emissions
        log_emissions = compute_log_emissions(candidate_probabilities)

        # 2. Extract classifier-only Top-1 predictions
        classifier_arcs = np.empty(N, dtype=np.int64)
        for i in range(N):
            cands = candidates.candidate_arc_ids[i]
            probs = candidate_probabilities[i]
            if len(cands) > 0 and len(probs) > 0:
                best_k = int(np.argmax(probs))
                classifier_arcs[i] = int(cands[best_k])
            elif len(cands) > 0:
                classifier_arcs[i] = int(cands[0])
            else:
                classifier_arcs[i] = -1

        # 3. Log-space Viterbi forward pass
        viterbi_costs: List[np.ndarray] = []
        backpointers: List[np.ndarray] = []

        # Step 0
        init_v = log_emissions[0].copy()
        viterbi_costs.append(init_v)
        backpointers.append(np.full(len(init_v), -1, dtype=np.int32))

        # Steps 1 to N-1
        for i in range(1, N):
            prev_v = viterbi_costs[i - 1]
            curr_emit = log_emissions[i]
            prev_cands = candidates.candidate_arc_ids[i - 1]
            curr_cands = candidates.candidate_arc_ids[i]

            dt = max(float(timestamps[i] - timestamps[i - 1]), 0.1)
            prev_pt = track_xy[i - 1]
            curr_pt = track_xy[i]

            # Transition matrix shape (K_prev, K_curr)
            log_trans = self.transition_model.compute_transition_matrix(
                prev_cands, curr_cands, prev_pt, curr_pt, dt
            )

            # For each current candidate k, find max_j [prev_v[j] + log_trans[j, k]]
            # Broadcast: prev_v[:, None] + log_trans -> shape (K_prev, K_curr)
            accum_costs = prev_v[:, None] + log_trans
            best_prev_idx = np.argmax(accum_costs, axis=0)  # shape (K_curr,)
            max_accum = np.max(accum_costs, axis=0)

            curr_v = max_accum + curr_emit
            viterbi_costs.append(curr_v)
            backpointers.append(best_prev_idx)

        # 4. Backtracking
        best_last_idx = int(np.argmax(viterbi_costs[-1]))
        total_ll = float(viterbi_costs[-1][best_last_idx])

        opt_path_indices = np.empty(N, dtype=np.int32)
        curr_idx = best_last_idx
        for i in range(N - 1, -1, -1):
            opt_path_indices[i] = curr_idx
            curr_idx = backpointers[i][curr_idx]

        # Extract optimal arc IDs
        predicted_arcs = np.empty(N, dtype=np.int64)
        for i in range(N):
            cands = candidates.candidate_arc_ids[i]
            k_idx = opt_path_indices[i]
            if 0 <= k_idx < len(cands):
                predicted_arcs[i] = int(cands[k_idx])
            else:
                predicted_arcs[i] = classifier_arcs[i]

        # 5. Calculate continuity and disconnected steps
        cont_hmm = calculate_route_continuity(predicted_arcs, self.arcs)
        cont_clf = calculate_route_continuity(classifier_arcs, self.arcs)

        # Count disconnected steps
        disc_hmm = sum(
            1 for i in range(N - 1)
            if predicted_arcs[i] != predicted_arcs[i + 1]
            and self.arcs[predicted_arcs[i], 1] != self.arcs[predicted_arcs[i + 1], 0]
        )
        disc_clf = sum(
            1 for i in range(N - 1)
            if classifier_arcs[i] != classifier_arcs[i + 1]
            and self.arcs[classifier_arcs[i], 1] != self.arcs[classifier_arcs[i + 1], 0]
        )

        return ViterbiResult(
            trip_id=trip_id,
            predicted_arcs=predicted_arcs,
            classifier_arcs=classifier_arcs,
            log_likelihood=total_ll,
            route_continuity_hmm=cont_hmm,
            route_continuity_classifier=cont_clf,
            disconnected_steps_hmm=disc_hmm,
            disconnected_steps_classifier=disc_clf,
        )
