# ML -> AI Integration Contract

The ML module should send one prediction at a time.

## Request

```json
{
  "ml_prediction": {
    "event": "fall",
    "confidence": 0.94,
    "timestamp": "2026-09-05T12:30:00Z"
  },
  "context": {
    "heart_rate": 118,
    "temperature_c": 38.1,
    "latitude": 12.9716,
    "longitude": 77.5946,
    "inactivity_seconds": 45,
    "user_response": "no_response"
  }
}
```

Only `ml_prediction` is essential to the ML teammate.
The context can be added by the IoT/cloud layer.

## Response

```json
{
  "assessment": {
    "event": "fall",
    "ml_confidence": 0.94,
    "risk_score": 97,
    "risk_level": "CRITICAL",
    "alert": true,
    "reasons": [
      "ML detected a fall with 94% confidence."
    ],
    "recommended_action": "Trigger emergency alert and send latest location/context immediately.",
    "timestamp": "2026-09-05T12:30:00Z"
  },
  "alert_result": {
    "sent": false,
    "mode": "simulation"
  }
}
```

## Important

Do not tightly couple the AI service to the CNN-LSTM implementation.
The AI side should only depend on this standardized prediction contract.
