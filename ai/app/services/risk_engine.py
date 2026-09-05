from app.models.schemas import AssessmentRequest, AssessmentResponse


def _risk_level(score: int) -> str:
    if score >= 85:
        return "CRITICAL"
    if score >= 65:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


def assess_risk(payload: AssessmentRequest) -> AssessmentResponse:
    pred = payload.ml_prediction
    ctx = payload.context

    score = 0
    reasons = []

    # ML contribution
    if pred.event == "fall":
        fall_points = round(pred.confidence * 60)
        score += fall_points
        reasons.append(
            f"ML detected a fall with {pred.confidence:.0%} confidence."
        )
    else:
        no_fall_points = round(pred.confidence * 5)
        score += no_fall_points
        reasons.append(
            f"ML classified the event as no-fall with {pred.confidence:.0%} confidence."
        )

    # Heart-rate context.
    # These are demo thresholds for a student prototype, not medical diagnosis.
    if ctx.heart_rate is not None:
        if ctx.heart_rate >= 130:
            score += 18
            reasons.append("Very high heart-rate reading increased risk.")
        elif ctx.heart_rate >= 110:
            score += 10
            reasons.append("Elevated heart-rate reading increased risk.")
        elif ctx.heart_rate < 45:
            score += 18
            reasons.append("Very low heart-rate reading increased risk.")

    # Temperature context.
    if ctx.temperature_c is not None:
        if ctx.temperature_c >= 39.0:
            score += 12
            reasons.append("High temperature reading increased risk.")
        elif ctx.temperature_c >= 38.0:
            score += 6
            reasons.append("Elevated temperature reading increased risk.")

    # Inactivity after possible impact.
    if ctx.inactivity_seconds is not None:
        if ctx.inactivity_seconds >= 60:
            score += 18
            reasons.append("Long inactivity after the event increased risk.")
        elif ctx.inactivity_seconds >= 30:
            score += 10
            reasons.append("Post-event inactivity increased risk.")

    # User acknowledgement.
    if ctx.user_response == "needs_help":
        score += 30
        reasons.append("User explicitly requested help.")
    elif ctx.user_response == "no_response":
        score += 15
        reasons.append("User did not respond to the safety check.")
    elif ctx.user_response == "okay":
        score -= 15
        reasons.append("User reported that they are okay.")

    score = max(0, min(100, score))
    level = _risk_level(score)

    # Trigger if risk is high, or explicit help is requested.
    alert = level in {"HIGH", "CRITICAL"} or ctx.user_response == "needs_help"

    if level == "CRITICAL":
        action = "Trigger emergency alert and send latest location/context immediately."
    elif level == "HIGH":
        action = "Notify caregiver and request immediate user confirmation."
    elif level == "MEDIUM":
        action = "Ask user for confirmation and continue short-term monitoring."
    else:
        action = "Log event and continue monitoring."

    return AssessmentResponse(
        event=pred.event,
        ml_confidence=pred.confidence,
        risk_score=score,
        risk_level=level,
        alert=alert,
        reasons=reasons,
        recommended_action=action,
        timestamp=pred.timestamp,
    )
