from fastapi import FastAPI
from app.routes.assessment import router as assessment_router

app = FastAPI(
    title="Smart Belt AI Context Service",
    version="1.0.0",
    description="Combines ML fall predictions with health/context data."
)

app.include_router(assessment_router, prefix="/api/v1", tags=["assessment"])


@app.get("/")
def root():
    return {
        "service": "smart-belt-ai",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}
