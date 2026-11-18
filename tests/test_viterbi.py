import numpy as np
import pytest
from src.features.candidates import CandidateSet
from src.hmm.viterbi import ViterbiDecoder


def test_viterbi_synthetic_chain():
    # 3 nodes: (0,0), (10,0), (20,0)
    # 2 connected arcs: arc 0 = 0->1, arc 1 = 1->2
    # 1 disconnected arc: arc 2 = 2->0 (backward/loop)
    nodes_xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    arcs = np.array([[0, 1], [1, 2], [2, 0]])

    decoder = ViterbiDecoder(nodes_xy, arcs)

    track_xy = np.array([[5.0, 0.0], [15.0, 0.0]])
    timestamps = np.array([0.0, 1.0])

    # Observation 0: candidates [0, 2], probs [0.8, 0.2]
    # Observation 1: candidates [1, 2], probs [0.7, 0.3]
    cands = CandidateSet(
        trip_id="test",
        k=2,
        candidate_arc_ids=[np.array([0, 2]), np.array([1, 2])],
        perpendicular_dists=[np.array([0.0, 10.0]), np.array([0.0, 10.0])],
        projection_ts=[np.array([0.5, 0.5]), np.array([0.5, 0.5])],
        projected_points=[np.zeros((2, 2)), np.zeros((2, 2))],
        candidate_counts=np.array([2, 2]),
    )
    probs = [np.array([0.8, 0.2]), np.array([0.7, 0.3])]

    result = decoder.decode("test", track_xy, timestamps, cands, probs)
    np.testing.assert_array_equal(result.predicted_arcs, [0, 1])
    assert result.route_continuity_hmm == 1.0
    assert result.disconnected_steps_hmm == 0


def test_viterbi_single_candidate():
    nodes_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    arcs = np.array([[0, 1]])

    decoder = ViterbiDecoder(nodes_xy, arcs)
    track_xy = np.array([[5.0, 0.0]])
    timestamps = np.array([0.0])
    cands = CandidateSet(
        trip_id="single",
        k=1,
        candidate_arc_ids=[np.array([0])],
        perpendicular_dists=[np.array([0.0])],
        projection_ts=[np.array([0.5])],
        projected_points=[np.zeros((1, 2))],
        candidate_counts=np.array([1]),
    )
    probs = [np.array([1.0])]

    result = decoder.decode("single", track_xy, timestamps, cands, probs)
    assert len(result.predicted_arcs) == 1
    assert result.predicted_arcs[0] == 0
