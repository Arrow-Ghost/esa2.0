# Branch Plan

## main
Stable/demo-ready code only.

## develop
Integration branch. Feature branches merge here first.

## feature/api
FastAPI app, schemas, endpoints.

Files:
- app/main.py
- app/models/schemas.py
- app/routes/assessment.py

## feature/risk-engine
Context/risk-assessment logic.

Files:
- app/services/risk_engine.py

## feature/alert-service
Caregiver/emergency alert adapter.

Files:
- app/services/alert_service.py
- .env.example

## feature/dashboard
Streamlit dashboard.

Files:
- dashboard.py

## feature/ml-integration
Contract between ML and AI.

Files:
- examples/sample_request.json
- examples/ml_integration.py
- docs/ML_AI_CONTRACT.md

## feature/tests
Automated tests.

Files:
- tests/test_api.py
- tests/test_risk_engine.py
