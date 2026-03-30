"""Pydantic v2 request/response schemas for the propensity API."""

from pydantic import BaseModel, ConfigDict, Field


class UserFeatures(BaseModel):
    """Input features for scoring a single UPI user's propensity to transact."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "user_id": "user_001",
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
            ]
        }
    )

    # Identity
    user_id: str | None = None

    # User demographics
    months_on_book: int = Field(..., ge=1, le=120, description="Months since account opening")
    age_band: int = Field(..., ge=1, le=5, description="1=18-25, 2=26-35, 3=36-45, 4=46-55, 5=56+")
    affluence_segment: int = Field(..., ge=1, le=4, description="1=mass, 2=mass-affluent, 3=affluent, 4=HNI")
    num_products_held: int = Field(..., ge=1, le=6, description="Number of bank products held")

    # UPI transaction history
    txn_count_30d: int = Field(..., ge=0, description="UPI transactions in last 30 days")
    txn_count_90d: int = Field(..., ge=0, description="UPI transactions in last 90 days")
    avg_txn_amount: float = Field(..., gt=0, description="Average transaction amount in INR")
    total_txn_value_30d: float = Field(..., ge=0, description="Total transaction value in last 30 days (INR)")
    days_since_last_txn: int = Field(..., ge=0, description="Days since last UPI transaction")
    txn_frequency_trend: float = Field(..., ge=-1.0, le=1.0, description="Trend in weekly txn count (-1 to +1)")

    # Value share
    value_share_pct: float = Field(..., ge=0.0, le=1.0, description="Our app's share of user's total UPI value")
    primary_app_flag: int = Field(..., ge=0, le=1, description="1 if value_share > 50%")

    # Balance / financial
    avg_monthly_balance: float = Field(..., ge=0, description="Average monthly account balance in INR")
    balance_trend: float | None = Field(None, ge=-1.0, le=1.0, description="Balance trend (-1 to +1), null for new accounts")
    low_balance_months_6m: int = Field(..., ge=0, le=6, description="Months with below-threshold balance in last 6")

    # App engagement
    app_opens_30d: int = Field(..., ge=0, description="App opens in last 30 days")
    app_opens_trend: float = Field(..., ge=-1.0, le=1.0, description="Trend in weekly app opens")
    days_since_last_app_open: int = Field(..., ge=0, description="Days since last app open")
    session_duration_avg: float = Field(..., ge=0, description="Average session duration in seconds")
    notification_ctr: float | None = Field(None, ge=0.0, le=0.5, description="Notification click-through rate, null if disabled")

    # Rewards
    cashback_received_90d: float = Field(..., ge=0, description="Cashback received in last 90 days (INR)")
    cashback_redeemed_pct: float = Field(..., ge=0.0, le=1.0, description="Fraction of cashback redeemed")
    milestones_completed: int = Field(..., ge=0, description="Gamification milestones completed")
    reward_tier: int = Field(..., ge=1, le=4, description="1=bronze, 2=silver, 3=gold, 4=platinum")

    # Fraud / risk
    fraud_flag_l1: int = Field(..., ge=0, le=1, description="L1 fraud flag (suspected)")
    fraud_flag_l2: int = Field(..., ge=0, le=1, description="L2 fraud flag (confirmed)")
    chargeback_count_12m: int = Field(..., ge=0, description="Chargebacks in last 12 months")

    # Digital behavior
    has_upi_autopay: int = Field(..., ge=0, le=1, description="Has active UPI autopay mandate")
    peer_txn_ratio: float = Field(..., ge=0.0, le=1.0, description="P2P transactions as fraction of total")
    bill_pay_active: int = Field(..., ge=0, le=1, description="Has paid bills via UPI")


class FeatureImpact(BaseModel):
    feature: str
    impact: float
    direction: str


class ScoreResponse(BaseModel):
    user_id: str | None = None
    propensity_score: float = Field(
        ..., description="Score between 0-1. Higher = more likely to transact."
    )
    segment: str = Field(..., description="high (>=0.7), medium (0.4-0.7), low (<0.4)")
    top_drivers: list[FeatureImpact]


class BatchScoreResponse(BaseModel):
    total_users: int
    high_propensity_count: int
    results: list[ScoreResponse]
