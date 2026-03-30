"""
Inference module — PropensityPredictor class with lazy loading.

No module-level instantiation: the model is loaded via load_predictor()
and attached to FastAPI's app.state during the lifespan event.
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.model.features import FEATURE_COLS, engineer_features, get_top_shap_features

logger = logging.getLogger(__name__)


class PropensityPredictor:
    def __init__(self, model_path: str | Path):
        load_start = time.time()

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_cols = FEATURE_COLS

        load_time = time.time() - load_start
        logger.info("Model loaded from %s in %.2fs", model_path, load_time)

    def score_single(self, features: dict) -> dict:
        """Score a single user. Returns score, segment, and top 3 SHAP drivers."""
        df = pd.DataFrame([features])
        df = engineer_features(df)
        X = df[self.feature_cols]

        prob = float(self.model.predict_proba(X)[0][1])
        shap_vals = self.explainer.shap_values(X)[0]
        top_features = get_top_shap_features(shap_vals, self.feature_cols, top_n=3)

        return {
            "propensity_score": round(prob, 4),
            "segment": _score_to_segment(prob),
            "top_drivers": top_features,
        }

    def score_batch(self, records: list[dict], include_shap: bool = True) -> list[dict]:
        """Score a batch of users. Returns results sorted by score descending."""
        df = pd.DataFrame(records)
        df = engineer_features(df)
        X = df[self.feature_cols]

        probs = self.model.predict_proba(X)[:, 1]

        if include_shap:
            shap_vals = self.explainer.shap_values(X)
        else:
            shap_vals = None

        results = []
        for i, (record, prob) in enumerate(zip(records, probs)):
            if shap_vals is not None:
                top_features = get_top_shap_features(
                    shap_vals[i], self.feature_cols, top_n=3
                )
            else:
                top_features = []

            results.append({
                "user_id": record.get("user_id", str(i)),
                "propensity_score": round(float(prob), 4),
                "segment": _score_to_segment(prob),
                "top_drivers": top_features,
            })

        results.sort(key=lambda x: x["propensity_score"], reverse=True)
        return results


def _score_to_segment(score: float) -> str:
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    else:
        return "low"


def load_predictor(model_path: str | Path | None = None) -> PropensityPredictor:
    """Factory function for creating a predictor. Used in FastAPI lifespan."""
    if model_path is None:
        from config import MODEL_PATH
        model_path = MODEL_PATH
    return PropensityPredictor(model_path)
