from __future__ import annotations

import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    TripListResponse,
    PredictRequest,
    PredictResponse,
    MetricsResponse,
)
from src.api.evaluation import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Laptop 1 backend serving trained classifier, HMM/Viterbi sequential decoder, and evaluation metrics.",
    version="1.0.0",
)

# Enable CORS for Laptop 2 communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
DATA_DIR = Path("data")
MODEL_PATH = Path("artifacts/model.joblib")
engine: Optional[InferenceEngine] = None


@app.on_event("startup")
def startup_event():
    global engine
    logger.info("Initializing InferenceEngine on Laptop 1...")
    engine = InferenceEngine(model_path=MODEL_PATH, data_base_dir=DATA_DIR)
    logger.info("InferenceEngine loaded successfully.")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        service="PathSnap-Laptop1-Backend",
        model_family="xgboost",
        version="1.0.0",
    )


@app.get("/trips", response_model=TripListResponse)
def list_trips():
    train_trips = sorted([d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir()]) if (DATA_DIR / "train").exists() else []
    test_trips = sorted([d.name for d in (DATA_DIR / "student_test").iterdir() if d.is_dir()]) if (DATA_DIR / "student_test").exists() else []
    return TripListResponse(
        partitions=["student_test", "train"],
        trips={
            "student_test": test_trips,
            "train": train_trips,
        },
    )


@app.post("/predict", response_model=PredictResponse)
def predict_trip(req: PredictRequest):
    logger.info(f"Received inference request for trip {req.trip_id} (partition={req.partition})")
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready")

    try:
        response = engine.predict_trip(req.trip_id, partition=req.partition)
        logger.info(f"Processed trip {req.trip_id}: {response.observation_count} obs, continuity={response.route_continuity:.4f}, time={response.execution_time_seconds:.2f}s")
        return response
    except FileNotFoundError as e:
        logger.warning(f"Trip not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed for trip {req.trip_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")


@app.get("/metrics/{trip_id}", response_model=MetricsResponse)
def get_trip_metrics(
    trip_id: str,
    partition: str = Query("student_test", description="Partition name"),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready")

    try:
        resp = engine.predict_trip(trip_id, partition=partition)
        dev_m = resp.metrics or {}
        return MetricsResponse(
            trip_id=trip_id,
            partition=partition,
            classifier_top1_accuracy=dev_m.get("classifier_top1_accuracy"),
            hmm_top1_accuracy=dev_m.get("hmm_top1_accuracy"),
            route_continuity_classifier=None,
            route_continuity_hmm=resp.route_continuity,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/route/{trip_id}")
def get_trip_route(
    trip_id: str,
    partition: str = Query("student_test", description="Partition name"),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready")

    try:
        resp = engine.predict_trip(trip_id, partition=partition)
        return {
            "trip_id": trip_id,
            "partition": partition,
            "predicted_arc_ids": resp.predicted_arc_ids,
            "route_nodes": resp.route_nodes,
            "route_continuity": resp.route_continuity,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
