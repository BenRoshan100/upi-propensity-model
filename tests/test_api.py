"""Tests for the FastAPI endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client(ensure_model_artifacts):
    with TestClient(app) as c:
        yield c


SAMPLE_USER = {
    "user_id": "test_001",
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


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_score_single(client):
    r = client.post("/score", json=SAMPLE_USER)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["propensity_score"] <= 1.0
    assert data["segment"] in ["high", "medium", "low"]
    assert len(data["top_drivers"]) == 3


def test_score_returns_shap_directions(client):
    r = client.post("/score", json=SAMPLE_USER)
    data = r.json()
    for driver in data["top_drivers"]:
        assert driver["direction"] in ["increases_propensity", "decreases_propensity"]


def test_batch_score(client):
    users = [SAMPLE_USER, {**SAMPLE_USER, "user_id": "test_002", "txn_count_30d": 0}]
    r = client.post("/batch_score", json=users)
    assert r.status_code == 200
    data = r.json()
    assert data["total_users"] == 2
    scores = [u["propensity_score"] for u in data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_batch_score_without_shap(client):
    users = [SAMPLE_USER]
    r = client.post("/batch_score?include_shap=false", json=users)
    assert r.status_code == 200
    assert r.json()["results"][0]["top_drivers"] == []


def test_batch_limit(client):
    users = [SAMPLE_USER] * 5001
    r = client.post("/batch_score", json=users)
    assert r.status_code == 400


def test_batch_csv(client):
    csv_cols = ",".join([
        "user_id", "months_on_book", "age_band", "affluence_segment", "num_products_held",
        "txn_count_30d", "txn_count_90d", "avg_txn_amount", "total_txn_value_30d",
        "days_since_last_txn", "txn_frequency_trend", "value_share_pct", "primary_app_flag",
        "avg_monthly_balance", "balance_trend", "low_balance_months_6m",
        "app_opens_30d", "app_opens_trend", "days_since_last_app_open",
        "session_duration_avg", "notification_ctr",
        "cashback_received_90d", "cashback_redeemed_pct", "milestones_completed", "reward_tier",
        "fraud_flag_l1", "fraud_flag_l2", "chargeback_count_12m",
        "has_upi_autopay", "peer_txn_ratio", "bill_pay_active",
    ])
    csv_row = "user_a,24,2,2,3,12,35,850.0,10200.0,3,0.15,0.35,0,22000.0,0.1,1,18,0.2,1,55.0,0.08,120.0,0.75,3,2,0,0,0,1,0.45,1"
    csv_content = f"{csv_cols}\n{csv_row}\n"

    r = client.post(
        "/batch_score/csv",
        files={"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")},
    )
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "propensity_score" in r.text
