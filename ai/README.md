# Smart Belt AI Service

AI/context layer for a wearable fall-detection belt.

The ML teammate sends:

```json
{
  "event": "fall",
  "confidence": 0.94,
  "timestamp": "2026-09-05T12:30:00Z"
}
```

This service combines the ML output with optional contextual data such as heart rate,
temperature, GPS, inactivity and user response, then produces:

- risk score
- risk level
- explanation
- recommended action
- alert decision

## Architecture

```text
ESP32 / Sensors
      |
      v
ML Fall Detector
      |
      | event + confidence + timestamp
      v
AI Context API  <---- heart rate / temperature / GPS / inactivity
      |
      v
Risk Assessment
      |
      +----> Alert Service
      |
      +----> Dashboard API
```

## Quick start

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn app.main:app --reload
```

Open:

- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

Run dashboard:

```bash
streamlit run dashboard.py
```

## Example request

POST `/api/v1/assess`

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

## Suggested Git branches

```text
main
develop
feature/api
feature/risk-engine
feature/alert-service
feature/dashboard
feature/ml-integration
feature/tests
```

Recommended workflow:

```text
feature/* -> develop -> main
```

Do not merge unfinished code directly into `main`.
