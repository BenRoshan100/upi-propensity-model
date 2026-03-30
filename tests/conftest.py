"""
Shared test fixtures.

Creates a dummy XGBoost model if the real model doesn't exist,
so tests can run without training first.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from src.model.features import FEATURE_COLS

MODELS_DIR = Path(__file__).parent.parent / "models"


def _create_dummy_model():
    """Create a small dummy XGBoost model for testing."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = len(FEATURE_COLS)

    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 2, n_samples)

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifacts():
    """Ensure model artifacts exist for tests. Creates dummy ones if needed."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "xgb_model.pkl"
    if not model_path.exists():
        model = _create_dummy_model()
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    feature_names_path = MODELS_DIR / "feature_names.json"
    if not feature_names_path.exists():
        with open(feature_names_path, "w") as f:
            json.dump(FEATURE_COLS, f)

    defaults_path = MODELS_DIR / "feature_defaults.json"
    if not defaults_path.exists():
        defaults = {col: 0.0 for col in FEATURE_COLS}
        with open(defaults_path, "w") as f:
            json.dump(defaults, f)

    yield
