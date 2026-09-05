from fastapi import APIRouter
from app.models.schemas import AssessmentRequest
from app.services.risk_engine import assess_risk
from app.services.alert_service import send_alert

router = APIRouter()


@router.post("/assess")
def assess(payload: AssessmentRequest):
    assessment = assess_risk(payload)

    alert_result = None
    if assessment.alert:
        alert_result = send_alert(payload, assessment)

    return {
        "assessment": assessment,
        "alert_result": alert_result
    }
