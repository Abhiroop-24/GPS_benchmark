from pathlib import Path
import numpy as np
import pytest
from src.data.loader import load_raw_trip
from src.features.candidates import SpatialCandidateGenerator
from src.features.alignment import align_trip_candidates
from src.features.feature_engineering import extract_trip_features, FEATURE_COLUMNS


def test_extract_features_integrity():
    trip_path = Path("data/train/00000026")
    if not trip_path.exists():
        pytest.skip("Dataset trip 00000026 not found")

    trip = load_raw_trip(trip_path, partition="train")
    gen = SpatialCandidateGenerator(trip.nodes_xy, trip.arcs)
    cands = gen.generate_candidates(trip.track_xy, k=8)
    cands.trip_id = trip.trip_id

    align = align_trip_candidates(trip.route, cands)
    df = extract_trip_features(trip, cands, align)

    assert len(df) == len(trip.track) * 8
    for col in FEATURE_COLUMNS:
        assert col in df.columns
        assert not df[col].isna().any()
        assert not np.isinf(df[col]).any()

    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({0, 1})
    # Label 1 should exist (matched candidates)
    assert (df["label"] == 1).sum() > 0
