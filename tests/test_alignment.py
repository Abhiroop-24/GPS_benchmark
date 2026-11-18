import numpy as np
import pytest
from src.features.candidates import CandidateSet
from src.features.alignment import align_trip_candidates


def test_alignment_monotonic_and_repeated_arcs():
    # Route: arc 10, then 20, then 10 again (loop/repeat)
    route = np.array([10, 20, 10])

    # 3 observations
    # Obs 0 candidates: [10, 99], dists: [2.0, 50.0]
    # Obs 1 candidates: [20, 99], dists: [1.5, 60.0]
    # Obs 2 candidates: [10, 99], dists: [3.0, 70.0]
    cands = CandidateSet(
        trip_id="synthetic",
        k=2,
        candidate_arc_ids=[np.array([10, 99]), np.array([20, 99]), np.array([10, 99])],
        perpendicular_dists=[np.array([2.0, 50.0]), np.array([1.5, 60.0]), np.array([3.0, 70.0])],
        projection_ts=[np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.5])],
        projected_points=[np.zeros((2, 2)) for _ in range(3)],
        candidate_counts=np.array([2, 2, 2]),
    )

    result = align_trip_candidates(route, cands, miss_penalty=40.0)
    assert result.topk_recall == 1.0
    assert result.candidate_miss_rate == 0.0
    np.testing.assert_array_equal(result.aligned_arc_ids, [10, 20, 10])
    # Route indices should be [0, 1, 2]
    np.testing.assert_array_equal(result.aligned_route_indices, [0, 1, 2])
    # Candidate labels: candidate 0 should be 1, candidate 1 should be 0
    for labels in result.labels_per_candidate:
        assert labels[0] == 1
        assert labels[1] == 0


def test_alignment_miss_handling():
    # Route: arc 10, 20
    route = np.array([10, 20])
    # Obs 0 has [10], Obs 1 has [999] (miss!)
    cands = CandidateSet(
        trip_id="synthetic_miss",
        k=1,
        candidate_arc_ids=[np.array([10]), np.array([999])],
        perpendicular_dists=[np.array([2.0]), np.array([10.0])],
        projection_ts=[np.array([0.5]), np.array([0.5])],
        projected_points=[np.zeros((1, 2)), np.zeros((1, 2))],
        candidate_counts=np.array([1, 1]),
    )

    result = align_trip_candidates(route, cands, miss_penalty=40.0)
    assert result.aligned_arc_ids[0] == 10
    assert result.aligned_arc_ids[1] == -1  # Miss!
    assert result.labels_per_candidate[1][0] == 0  # Not silently marked positive!
    assert result.topk_recall == 0.5
    assert result.candidate_miss_rate == 0.5
