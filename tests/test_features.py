"""Tests for the feature engineering module."""

import pandas as pd
import numpy as np
from src.model.features import engineer_features, FEATURE_COLS, get_top_shap_features


SAMPLE_INPUT = {
    "months_on_book": 24,
    "age_band": 2,
    "affluence_segment": 2,
    "num_products_held": 3,
    "txn_count_30d": 12,
    "txn_count_90d": 35,
    "avg_txn_amount": 850.0,
    "total_txn_value_30d": 10200.0,
    "days_since_last_txn": 3,
    "txn_frequency_trend": 0.15,
    "value_share_pct": 0.35,
    "primary_app_flag": 0,
    "avg_monthly_balance": 22000.0,
    "balance_trend": 0.1,
    "low_balance_months_6m": 1,
    "app_opens_30d": 18,
    "app_opens_trend": 0.2,
    "days_since_last_app_open": 1,
    "session_duration_avg": 55.0,
    "notification_ctr": 0.08,
    "cashback_received_90d": 120.0,
    "cashback_redeemed_pct": 0.75,
    "milestones_completed": 3,
    "reward_tier": 2,
    "fraud_flag_l1": 0,
    "fraud_flag_l2": 0,
    "chargeback_count_12m": 0,
    "has_upi_autopay": 1,
    "peer_txn_ratio": 0.45,
    "bill_pay_active": 1,
}


def test_engineer_features_produces_all_columns():
    """Output must contain every column in FEATURE_COLS."""
    df = pd.DataFrame([SAMPLE_INPUT])
    result = engineer_features(df)
    for col in FEATURE_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_engineer_features_no_nulls():
    """Output feature columns must have no nulls."""
    df = pd.DataFrame([SAMPLE_INPUT])
    result = engineer_features(df)
    assert result[FEATURE_COLS].isnull().sum().sum() == 0


def test_engineer_features_handles_missing_columns():
    """Should fill missing columns with defaults."""
    # Provide only a few fields
    df = pd.DataFrame([{"txn_count_30d": 5, "avg_txn_amount": 500.0}])
    result = engineer_features(df)
    assert len(result) == 1
    for col in FEATURE_COLS:
        assert col in result.columns


def test_engineer_features_log_transforms():
    """Log-transformed columns should equal log1p of raw values."""
    df = pd.DataFrame([SAMPLE_INPUT])
    result = engineer_features(df)
    assert abs(result["avg_txn_amount_log"].iloc[0] - np.log1p(850.0)) < 0.01
    assert abs(result["avg_monthly_balance_log"].iloc[0] - np.log1p(22000.0)) < 0.01
    assert abs(result["total_txn_value_30d_log"].iloc[0] - np.log1p(10200.0)) < 0.01


def test_engineer_features_fills_balance_trend_null():
    """Null balance_trend should be filled with 0.0."""
    data = {**SAMPLE_INPUT, "balance_trend": None}
    df = pd.DataFrame([data])
    result = engineer_features(df)
    assert result["balance_trend"].iloc[0] == 0.0


def test_engineer_features_fills_notification_ctr_null():
    """Null notification_ctr should be filled with a default."""
    data = {**SAMPLE_INPUT, "notification_ctr": None}
    df = pd.DataFrame([data])
    result = engineer_features(df)
    assert not pd.isna(result["notification_ctr"].iloc[0])


def test_get_top_shap_features_returns_correct_count():
    shap_row = [0.1, -0.3, 0.5, 0.02, -0.15]
    names = ["f1", "f2", "f3", "f4", "f5"]
    result = get_top_shap_features(shap_row, names, top_n=3)
    assert len(result) == 3


def test_get_top_shap_features_sorted_by_absolute_impact():
    shap_row = [0.1, -0.5, 0.3]
    names = ["f1", "f2", "f3"]
    result = get_top_shap_features(shap_row, names, top_n=3)
    assert result[0]["feature"] == "f2"  # highest absolute impact


def test_get_top_shap_features_direction():
    shap_row = [0.5, -0.3]
    names = ["f1", "f2"]
    result = get_top_shap_features(shap_row, names, top_n=2)
    assert result[0]["direction"] == "increases_propensity"
    assert result[1]["direction"] == "decreases_propensity"
