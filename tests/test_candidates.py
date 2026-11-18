import numpy as np
import pytest
from src.features.candidates import point_to_segment_vectorized, SpatialCandidateGenerator


def test_point_to_segment_exact_geometry():
    # Segment from (0, 0) to (2, 0)
    seg_a = np.array([[0.0, 0.0]])
    seg_b = np.array([[2.0, 0.0]])

    # 1. Point projecting onto interior
    pt_mid = np.array([1.0, 1.0])
    dists, ts, projs = point_to_segment_vectorized(pt_mid, seg_a, seg_b)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(ts[0], 0.5)
    np.testing.assert_allclose(projs[0], [1.0, 0.0])

    # 2. Point projecting before start
    pt_before = np.array([-1.0, 1.0])
    dists, ts, projs = point_to_segment_vectorized(pt_before, seg_a, seg_b)
    assert np.isclose(dists[0], np.sqrt(2.0))
    assert np.isclose(ts[0], 0.0)
    np.testing.assert_allclose(projs[0], [0.0, 0.0])

    # 3. Point projecting after end
    pt_after = np.array([3.0, 1.0])
    dists, ts, projs = point_to_segment_vectorized(pt_after, seg_a, seg_b)
    assert np.isclose(dists[0], np.sqrt(2.0))
    assert np.isclose(ts[0], 1.0)
    np.testing.assert_allclose(projs[0], [2.0, 0.0])


def test_point_to_segment_degenerate():
    # Segment with length 0
    seg_a = np.array([[1.0, 1.0]])
    seg_b = np.array([[1.0, 1.0]])
    pt = np.array([1.0, 4.0])
    dists, ts, projs = point_to_segment_vectorized(pt, seg_a, seg_b)
    assert np.isclose(dists[0], 3.0)
    assert np.isclose(ts[0], 0.0)
    np.testing.assert_allclose(projs[0], [1.0, 1.0])


def test_spatial_candidate_generator_fallback():
    # 2 nodes, 1 arc
    nodes = np.array([[0.0, 0.0], [10.0, 0.0]])
    arcs = np.array([[0, 1]])
    gen = SpatialCandidateGenerator(nodes, arcs, radii=(5.0,))
    # Query point far away (1000m away)
    track = np.array([[1000.0, 1000.0]])
    cands = gen.generate_candidates(track, k=1)
    # Should still find the arc via nearest fallback
    assert len(cands.candidate_arc_ids[0]) == 1
    assert cands.candidate_arc_ids[0][0] == 0
