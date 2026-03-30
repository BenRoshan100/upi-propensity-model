"""
Feature engineering module — shared between training notebooks and API inference.

Applies log-transforms to skewed monetary features and fills nulls with
training-time medians from feature_defaults.json.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    # User demographics
    "months_on_book", "age_band", "affluence_segment", "num_products_held",
    # UPI transaction history
    "txn_count_30d", "txn_count_90d", "avg_txn_amount_log",
    "total_txn_value_30d_log", "days_since_last_txn", "txn_frequency_trend",
    # Value share
    "value_share_pct", "primary_app_flag",
    # Balance / financial
    "avg_monthly_balance_log", "balance_trend", "low_balance_months_6m",
    # App engagement
    "app_opens_30d", "app_opens_trend", "days_since_last_app_open",
    "session_duration_avg", "notification_ctr",
    # Rewards
    "cashback_received_90d", "cashback_redeemed_pct",
    "milestones_completed", "reward_tier",
    # Fraud / risk
    "fraud_flag_l1", "fraud_flag_l2", "chargeback_count_12m",
    # Digital behavior
    "has_upi_autopay", "peer_txn_ratio", "bill_pay_active",
]

_feature_defaults = None


def load_feature_defaults() -> dict:
    """Load median feature values saved during training. Used as API defaults."""
    global _feature_defaults
    if _feature_defaults is not None:
        return _feature_defaults

    defaults_path = Path(__file__).parent.parent.parent / "models" / "feature_defaults.json"
    if defaults_path.exists():
        with open(defaults_path) as f:
            _feature_defaults = json.load(f)
        logger.info("Loaded feature defaults from %s", defaults_path)
    else:
        logger.warning("feature_defaults.json not found at %s — using zeros", defaults_path)
        _feature_defaults = {col: 0.0 for col in FEATURE_COLS}

    return _feature_defaults


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for model input.

    - Log-transforms skewed monetary columns (avg_txn_amount, avg_monthly_balance, total_txn_value_30d)
    - Fills nulls (balance_trend → 0.0, notification_ctr → median from defaults)
    - Fills any missing feature columns with training-time defaults
    """
    df = df.copy()
    defaults = load_feature_defaults()

    # --- Log-transform skewed monetary features ---
    for raw_col, log_col in [
        ("avg_txn_amount", "avg_txn_amount_log"),
        ("avg_monthly_balance", "avg_monthly_balance_log"),
        ("total_txn_value_30d", "total_txn_value_30d_log"),
    ]:
        if raw_col in df.columns:
            df[log_col] = np.log1p(pd.to_numeric(df[raw_col], errors="coerce").fillna(0))
        elif log_col in df.columns:
            pass  # Already log-transformed (e.g., from processed parquet)
        else:
            df[log_col] = defaults.get(log_col, 0.0)

    # --- Fill nullable columns ---
    if "balance_trend" in df.columns:
        df["balance_trend"] = pd.to_numeric(df["balance_trend"], errors="coerce").fillna(0.0)
    if "notification_ctr" in df.columns:
        df["notification_ctr"] = pd.to_numeric(
            df["notification_ctr"], errors="coerce"
        ).fillna(defaults.get("notification_ctr", 0.0))

    # --- Fill any missing feature columns with defaults ---
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = defaults.get(col, 0.0)

    # Ensure all feature columns are numeric (Pydantic None → object dtype fix)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
            defaults.get(col, 0.0)
        )

    return df


def get_top_shap_features(shap_values_row, feature_names, top_n=3):
    """Returns top N features driving a single prediction, sorted by absolute impact."""
    impacts = list(zip(feature_names, shap_values_row))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature": name,
            "impact": round(float(val), 4),
            "direction": "increases_propensity" if val > 0 else "decreases_propensity",
        }
        for name, val in impacts[:top_n]
    ]
