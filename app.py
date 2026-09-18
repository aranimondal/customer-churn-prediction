"""FastAPI service exposing the churn model.

Run from the project root:

    uvicorn app:app --reload

The persisted pipeline is loaded once at startup rather than per request, and it
is the same object produced by ``python -m src.train`` — preprocessing is never
reimplemented here.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src import config
from src.persistence import load_pipeline
from src.schemas import CustomerFeatures, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("churn-api")

# Populated at startup; kept in a dict so the handlers see updates.
state: dict[str, object] = {"pipeline": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model artifact once, at process start."""
    try:
        state["pipeline"] = load_pipeline()
        logger.info("Loaded model artifact from %s", config.MODEL_FILE)
    except FileNotFoundError as exc:
        # Start anyway so /health can report the problem instead of the
        # container crash-looping with no diagnostics.
        logger.error("%s", exc)
    yield
    state.clear()


app = FastAPI(
    title="Telco Customer Churn API",
    description="Decision Tree churn scoring for the retention team.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return field-level detail so callers can fix the payload."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Invalid request payload.",
            "detail": [
                {
                    "field": ".".join(str(p) for p in err["loc"] if p != "body"),
                    "message": err["msg"],
                }
                for err in exc.errors()
            ],
        },
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.get("pipeline") is not None else "degraded",
        model_loaded=state.get("pipeline") is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Score one customer and return the label plus churn probability."""
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact unavailable. Run: python -m src.train",
        )

    try:
        frame = customer.to_frame()
        probability = float(pipeline.predict_proba(frame)[0][1])
    except Exception:  # unexpected inference failure
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        )

    # Threshold at 0.5, matching pipeline.predict(). The trained model already
    # uses class_weight to shift the decision boundary toward recall, so a
    # second manual threshold here would double-count that adjustment.
    label = (
        config.POSITIVE_CLASS_LABEL
        if probability >= 0.5
        else config.NEGATIVE_CLASS_LABEL
    )
    return PredictionResponse(
        prediction=label, churn_probability=round(probability, 4)
    )
