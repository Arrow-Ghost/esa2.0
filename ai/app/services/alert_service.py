import os
import requests
from app.models.schemas import AssessmentRequest, AssessmentResponse


def send_alert(
    request_data: AssessmentRequest,
    assessment: AssessmentResponse
) -> dict:
    """
    Prototype alert adapter.

    If ALERT_WEBHOOK_URL is configured, this posts JSON to it.
    Otherwise it safely returns a simulated alert response.
    """
    payload = {
        "event": assessment.event,
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "timestamp": assessment.timestamp.isoformat(),
        "recommended_action": assessment.recommended_action,
        "location": {
            "latitude": request_data.context.latitude,
            "longitude": request_data.context.longitude,
        },
    }

    webhook = os.getenv("ALERT_WEBHOOK_URL")

    if not webhook:
        return {
            "sent": False,
            "mode": "simulation",
            "message": "No ALERT_WEBHOOK_URL configured.",
            "payload": payload,
        }

    response = requests.post(webhook, json=payload, timeout=5)
    response.raise_for_status()

    return {
        "sent": True,
        "mode": "webhook",
        "status_code": response.status_code,
    }
