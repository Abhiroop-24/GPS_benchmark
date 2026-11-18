from __future__ import annotations

from typing import Tuple
import numpy as np
import pyproj


class UTMProjector:
    """Handles dynamic WGS84 to local UTM projection and geodesic verification."""

    def __init__(self, mean_lon: float, mean_lat: float):
        self.mean_lon = float(mean_lon)
        self.mean_lat = float(mean_lat)
        self.northern = self.mean_lat >= 0.0
        zone = int((self.mean_lon + 180.0) // 6.0) + 1
        self.zone = max(1, min(60, zone))
        self.epsg = (32600 if self.northern else 32700) + self.zone
        self.transformer_to_utm = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{self.epsg}", always_xy=True
        )
        self.transformer_to_wgs84 = pyproj.Transformer.from_crs(
            f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True
        )
        self.geod = pyproj.Geod(ellps="WGS84")

    @classmethod
    def from_coordinates(cls, lon: np.ndarray, lat: np.ndarray) -> UTMProjector:
        mean_lon = float(np.nanmean(lon))
        mean_lat = float(np.nanmean(lat))
        return cls(mean_lon=mean_lon, mean_lat=mean_lat)

    def forward(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project (lon, lat) to UTM (x, y) in meters."""
        x, y = self.transformer_to_utm.transform(lon, lat)
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def inverse(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project UTM (x, y) back to (lon, lat)."""
        lon, lat = self.transformer_to_wgs84.transform(x, y)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    def verify_against_geodesic(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> dict[str, float]:
        """Compare Euclidean distance in UTM to WGS84 geodesic ellipsoidal distance."""
        _, _, geodesic_dist = self.geod.inv(lon1, lat1, lon2, lat2)
        (x1, x2), (y1, y2) = self.forward(np.array([lon1, lon2]), np.array([lat1, lat2]))
        utm_dist = float(np.hypot(x2 - x1, y2 - y1))
        abs_diff = abs(utm_dist - geodesic_dist)
        rel_diff = abs_diff / max(geodesic_dist, 1e-6)
        return {
            "geodesic_distance_m": float(geodesic_dist),
            "utm_distance_m": float(utm_dist),
            "absolute_diff_m": float(abs_diff),
            "relative_diff": float(rel_diff),
            "epsg": self.epsg,
            "zone": self.zone,
        }
