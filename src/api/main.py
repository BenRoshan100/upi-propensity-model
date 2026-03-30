"""
FastAPI application for the UPI Propensity Score API.

Model is loaded via the lifespan context manager (not at import time)
and stored in app.state.predictor.
"""

import io
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse

from src.api.schemas import UserFeatures, ScoreResponse, BatchScoreResponse
from src.model.predict import load_predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, release on shutdown."""
    logger.info("Loading propensity model...")
    app.state.predictor = load_predictor()
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="UPI Propensity Score API",
    description=(
        "Predict which dormant UPI users will transact in the next 30 days if nudged.\n\n"
        "Built with:\n"
        "- XGBoost classifier trained on UPI behavioral features\n"
        "- Per-prediction SHAP explanations (top 3 drivers)\n"
        "- Precision@K evaluation for campaign targeting\n"
        "- 30 features across transactions, engagement, rewards, and risk"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "name": "UPI Propensity Score API",
        "version": "1.0.0",
        "endpoints": {
            "/score": "Score a single user",
            "/batch_score": "Score multiple users from JSON",
            "/batch_score/csv": "Upload CSV and get scored CSV back",
            "/health": "Health check",
            "/docs": "Interactive API documentation",
        },
    }


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.predictor is not None,
    }


@app.post("/score", response_model=ScoreResponse)
def score_single_user(features: UserFeatures, request: Request):
    """
    Score a single user and return propensity score + top 3 SHAP drivers.

    The top_drivers explain *why* the user got this score — useful for
    deciding which reward or message to send them.
    """
    try:
        predictor = request.app.state.predictor
        result = predictor.score_single(features.model_dump())
        result["user_id"] = features.user_id
        return result
    except Exception as e:
        logger.error("Scoring failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal scoring error")


@app.post("/batch_score", response_model=BatchScoreResponse)
def score_batch_json(
    users: list[UserFeatures],
    request: Request,
    include_shap: bool = True,
):
    """
    Score a list of users from a JSON array. Returns sorted by score descending.

    For batches over 500 users, set `include_shap=false` to avoid timeouts.
    """
    if len(users) > 5000:
        raise HTTPException(status_code=400, detail="Batch size limit is 5000 users")

    # Auto-disable SHAP for large batches to prevent timeouts
    if len(users) > 500 and include_shap:
        logger.warning(
            "Large batch (%d users) with SHAP enabled — consider include_shap=false",
            len(users),
        )

    try:
        predictor = request.app.state.predictor
        records = [u.model_dump() for u in users]
        results = predictor.score_batch(records, include_shap=include_shap)
        high_count = sum(1 for r in results if r["segment"] == "high")
        return {
            "total_users": len(results),
            "high_propensity_count": high_count,
            "results": results,
        }
    except Exception as e:
        logger.error("Batch scoring failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal scoring error")


@app.post("/batch_score/csv")
async def score_batch_csv(request: Request, file: UploadFile = File(...)):
    """
    Upload a CSV of users, get back a scored and ranked CSV.

    CSV must contain the standard UPI feature columns (see /docs for field list).
    """
    try:
        predictor = request.app.state.predictor
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        records = df.to_dict(orient="records")
        results = predictor.score_batch(records, include_shap=True)

        result_df = pd.DataFrame([
            {
                "user_id": r["user_id"],
                "propensity_score": r["propensity_score"],
                "segment": r["segment"],
                "top_driver_1": r["top_drivers"][0]["feature"] if r["top_drivers"] else "",
                "top_driver_2": r["top_drivers"][1]["feature"] if len(r["top_drivers"]) > 1 else "",
                "top_driver_3": r["top_drivers"][2]["feature"] if len(r["top_drivers"]) > 2 else "",
            }
            for r in results
        ])

        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=scored_users.csv"},
        )
    except Exception as e:
        logger.error("CSV scoring failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal scoring error")
