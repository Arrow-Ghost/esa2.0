from datetime import datetime
from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class MLPrediction(BaseModel):
    event: Literal["fall", "no_fall"]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime


class ContextData(BaseModel):
    heart_rate: Optional[float] = Field(default=None, ge=0)
    temperature_c: Optional[float] = None
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    inactivity_seconds: Optional[int] = Field(default=None, ge=0)
    user_response: Optional[
        Literal["okay", "needs_help", "no_response", "unknown"]
    ] = "unknown"


class AssessmentRequest(BaseModel):
    ml_prediction: MLPrediction
    context: ContextData = ContextData()


class AssessmentResponse(BaseModel):
    event: str
    ml_confidence: float
    risk_score: int
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    alert: bool
    reasons: List[str]
    recommended_action: str
    timestamp: datetime
