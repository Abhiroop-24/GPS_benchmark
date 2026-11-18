import numpy as np
import pytest
from src.data.projector import UTMProjector


def test_utm_projector_zone_and_epsg():
    # Cologne/Bonn area, Germany (~7.1 E, 50.7 N) -> Zone 32N -> EPSG 32632
    projector = UTMProjector(mean_lon=7.1, mean_lat=50.7)
    assert projector.zone == 32
    assert projector.epsg == 32632
    assert projector.northern is True

    # Southern hemisphere test (-46.6 W, -23.5 S) -> Zone 23S -> EPSG 32723
    proj_south = UTMProjector(mean_lon=-46.6, mean_lat=-23.5)
    assert proj_south.zone == 23
    assert proj_south.epsg == 32723
    assert proj_south.northern is False


def test_utm_projector_roundtrip():
    projector = UTMProjector(mean_lon=7.1, mean_lat=50.7)
    lons = np.array([7.0, 7.1, 7.2])
    lats = np.array([50.6, 50.7, 50.8])
    x, y = projector.forward(lons, lats)
    assert len(x) == 3
    assert len(y) == 3
    lon_back, lat_back = projector.inverse(x, y)
    np.testing.assert_allclose(lons, lon_back, atol=1e-6)
    np.testing.assert_allclose(lats, lat_back, atol=1e-6)


def test_utm_geodesic_verification():
    projector = UTMProjector(mean_lon=7.1, mean_lat=50.7)
    res = projector.verify_against_geodesic(7.100, 50.700, 7.105, 50.705)
    # UTM vs Geodesic distance over a 600m segment should differ by < 0.1% (< 0.001 relative diff)
    assert res["relative_diff"] < 0.002
    assert res["geodesic_distance_m"] > 0
    assert res["utm_distance_m"] > 0
