"""Tests for the prediction module."""

import pytest
from src.model.predict import load_predictor


@pytest.fixture(scope="module")
def predictor(ensure_model_artifacts):
    return load_predictor()


SAMPLE_FEATURES = {
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


def test_score_single_returns_valid_score(predictor):
    result = predictor.score_single(SAMPLE_FEATURES)
    assert 0.0 <= result["propensity_score"] <= 1.0


def test_score_single_returns_valid_segment(predictor):
    result = predictor.score_single(SAMPLE_FEATURES)
    assert result["segment"] in ["high", "medium", "low"]


def test_score_single_returns_shap_drivers(predictor):
    result = predictor.score_single(SAMPLE_FEATURES)
    assert len(result["top_drivers"]) == 3
    for driver in result["top_drivers"]:
        assert "feature" in driver
        assert "impact" in driver
        assert driver["direction"] in ["increases_propensity", "decreases_propensity"]


def test_score_batch_sorted_descending(predictor):
    records = [
        {**SAMPLE_FEATURES, "user_id": "a", "txn_count_30d": 0},
        {**SAMPLE_FEATURES, "user_id": "b", "txn_count_30d": 50},
        {**SAMPLE_FEATURES, "user_id": "c", "txn_count_30d": 12},
    ]
    results = predictor.score_batch(records)
    scores = [r["propensity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_score_batch_without_shap(predictor):
    records = [SAMPLE_FEATURES]
    results = predictor.score_batch(records, include_shap=False)
    assert len(results) == 1
    assert results[0]["top_drivers"] == []
