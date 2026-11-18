from fastapi.testclient import TestClient
import pytest
from src.api.main import app, startup_event


@pytest.fixture(scope="module")
def client():
    startup_event()
    with TestClient(app) as c:
        yield c


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_family"] == "xgboost"


def test_api_list_trips(client):
    response = client.get("/trips")
    assert response.status_code == 200
    data = response.json()
    assert "student_test" in data["trips"]
    assert len(data["trips"]["student_test"]) == 10
    assert "00000052" in data["trips"]["student_test"]


def test_api_predict_student_test_trip(client):
    req_body = {
        "trip_id": "00000052",
        "partition": "student_test",
    }
    response = client.post("/predict", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["trip_id"] == "00000052"
    assert data["status"] == "success"
    assert data["observation_count"] == 262
    assert len(data["predicted_arc_ids"]) == 262
    assert len(data["gps_coordinates"]) == 262
    assert data["route_continuity"] > 0.90
    assert data["execution_time_seconds"] > 0.0


def test_api_predict_invalid_trip(client):
    req_body = {
        "trip_id": "99999999",
        "partition": "student_test",
    }
    response = client.post("/predict", json=req_body)
    assert response.status_code == 404
