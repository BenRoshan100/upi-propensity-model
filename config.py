"""
Centralized configuration for the propensity-api project.
All settings are read from environment variables with sensible defaults.
"""

import os
import logging
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODELS_DIR / "xgb_model.pkl"))
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
FEATURE_DEFAULTS_PATH = MODELS_DIR / "feature_defaults.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Synthetic data
SYNTHETIC_DATA_SIZE = 200_000
SYNTHETIC_DATA_SEED = 42
