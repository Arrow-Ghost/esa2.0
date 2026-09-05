from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_assessment_endpoint():
    payload = {
        "ml_prediction": {
            "event": "fall",
            "confidence": 0.9,
            "timestamp": "2026-09-05T12:30:00Z"
        },
        "context": {
            "heart_rate": 115,
            "temperature_c": 37.0,
            "inactivity_seconds": 35,
            "user_response": "no_response"
        }
    }

    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "assessment" in body
    assert "risk_score" in body["assessment"]
