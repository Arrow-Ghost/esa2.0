from datetime import datetime, timezone

from app.models.schemas import AssessmentRequest, MLPrediction, ContextData
from app.services.risk_engine import assess_risk


def test_high_risk_fall():
    payload = AssessmentRequest(
        ml_prediction=MLPrediction(
            event="fall",
            confidence=0.95,
            timestamp=datetime.now(timezone.utc)
        ),
        context=ContextData(
            heart_rate=120,
            temperature_c=38.2,
            inactivity_seconds=60,
            user_response="no_response"
        )
    )

    result = assess_risk(payload)

    assert result.alert is True
    assert result.risk_level in {"HIGH", "CRITICAL"}


def test_low_risk_no_fall():
    payload = AssessmentRequest(
        ml_prediction=MLPrediction(
            event="no_fall",
            confidence=0.95,
            timestamp=datetime.now(timezone.utc)
        ),
        context=ContextData(
            heart_rate=75,
            temperature_c=36.8,
            inactivity_seconds=0,
            user_response="okay"
        )
    )

    result = assess_risk(payload)

    assert result.alert is False
    assert result.risk_level == "LOW"
