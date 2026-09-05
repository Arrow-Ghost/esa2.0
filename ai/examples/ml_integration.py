"""
Example for the ML teammate.

After their model predicts a fall/no-fall probability, they can call
the AI service using this helper.
"""

from datetime import datetime, timezone
import requests


def send_ml_prediction(event: str, confidence: float):
    payload = {
        "ml_prediction": {
            "event": event,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "context": {}
    }

    response = requests.post(
        "http://127.0.0.1:8000/api/v1/assess",
        json=payload,
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print(send_ml_prediction("fall", 0.94))
