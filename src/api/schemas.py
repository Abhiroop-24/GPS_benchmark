from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "PathSnap-Laptop1-Backend"
    model_family: str = "xgboost"
    version: str = "1.0.0"


class TripListResponse(BaseModel):
    partitions: List[str]
    trips: Dict[str, List[str]]


class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="8-digit zero-padded trip identifier, e.g. '00000000'")
    partition: str = Field("student_test", description="Dataset partition ('student_test' or 'train')")


class PredictResponse(BaseModel):
    trip_id: str
    partition: str
    status: str
    observation_count: int
    candidate_count: int
    predicted_arc_ids: List[int]
    classifier_only_route: List[int]
    route_nodes: List[List[int]]
    gps_coordinates: List[List[float]]              # [lat, lon] for Leaflet/Folium compatibility
    predicted_route_coordinates: List[List[float]]   # [lat, lon] of road node sequence
    prediction_confidence: float
    route_continuity: float
    execution_time_seconds: float
    metrics: Optional[Dict[str, Any]] = None


class MetricsResponse(BaseModel):
    trip_id: str
    partition: str
    classifier_top1_accuracy: Optional[float] = None
    hmm_top1_accuracy: Optional[float] = None
    route_continuity_classifier: Optional[float] = None
    route_continuity_hmm: Optional[float] = None
    disconnected_steps_classifier: Optional[int] = None
    disconnected_steps_hmm: Optional[int] = None
