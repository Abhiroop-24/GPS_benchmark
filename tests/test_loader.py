from pathlib import Path
import pytest
import numpy as np
from src.data.loader import load_raw_trip, validate_trip_data


def test_load_real_small_trip():
    # Use trip 00000026 (smallest training trip)
    trip_path = Path("data/train/00000026")
    if not trip_path.exists():
        pytest.skip("Local dataset not found at data/train/00000026")
    trip_data = load_raw_trip(trip_path, partition="train")
    assert trip_data.trip_id == "00000026"
    assert len(trip_data.track) == 361
    assert trip_data.track.shape[1] == 3
    assert trip_data.nodes.shape[1] == 2
    assert trip_data.arcs.shape[1] == 2
    assert trip_data.route is not None
    assert len(trip_data.route) > 0
    assert trip_data.track_xy.shape == (361, 2)
    assert trip_data.nodes_xy.shape == (len(trip_data.nodes), 2)
    assert trip_data.epsg is not None


def test_load_missing_files(tmp_path):
    trip_dir = tmp_path / "fake_trip"
    trip_dir.mkdir()
    # Missing .track, .nodes, .arcs
    with pytest.raises(FileNotFoundError):
        load_raw_trip(trip_dir, partition="train")


def test_validation_detects_nan_and_bounds():
    track = np.array([[7.0, 50.0, 0.0], [np.nan, 50.1, 1.0]])
    nodes = np.array([[7.0, 50.0], [7.1, 50.1]])
    arcs = np.array([[0, 1]])
    report = validate_trip_data("test_trip", track, nodes, arcs, route=None)
    assert any("NaN or Inf" in issue for issue in report["issues"])

    # Out of bounds lat
    track_bad_lat = np.array([[7.0, 95.0, 0.0]])
    report2 = validate_trip_data("test_trip", track_bad_lat, nodes, arcs, route=None)
    assert any("latitude out of bounds" in issue for issue in report2["issues"])
