# UPI Propensity Score API

**Predicting which dormant UPI users will transact in the next 30 days if nudged** — using behavioral signals from transaction history, app engagement, rewards, and risk profiles.

Built as a production-grade ML API demonstrating:
- XGBoost classifier with `scale_pos_weight` for imbalanced targeting problems
- Per-prediction SHAP explanations (top 3 drivers per user)
- Precision@K evaluation — the right metric for campaign targeting
- FastAPI service with single-user and batch scoring endpoints
- 30 domain-realistic UPI features across 8 categories

---

## The Problem

In fintech, not all dormant users are equally likely to re-engage. Sending the same campaign to 1 million users wastes reward budget on users who wouldn't convert anyway. A propensity model ranks users by conversion likelihood so you spend your budget on the top 5-10% most likely to respond.

## Dataset

This project uses a **synthetic UPI propensity dataset** (200K users) with realistic statistical properties:
- ~5% positive rate (matching real dormant-to-active conversion rates)
- 30 features across transactions, engagement, rewards, balance, and risk
- Correlated features driven by a latent propensity score
- Realistic nulls (balance_trend: 15%, notification_ctr: 5%)

In production, this trains on real customer data from a CDP. The model architecture, evaluation framework, and API design transfer directly — only the data source changes.

---

## Architecture

```
UPI user features → Feature engineering → XGBoost classifier → Propensity score (0-1)
                                                              ↘ SHAP explainer → Top 3 drivers
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/score` | POST | Score a single user, return score + SHAP drivers |
| `/batch_score` | POST | Score list of users, return ranked targeting list |
| `/batch_score/csv` | POST | Upload CSV, download scored + ranked CSV |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

## Feature Categories

| Category | Count | Key Features |
|----------|-------|-------------|
| UPI Transactions | 6 | txn_count_30d, days_since_last_txn, avg_txn_amount |
| App Engagement | 5 | app_opens_30d, days_since_last_app_open, notification_ctr |
| Value Share | 2 | value_share_pct, primary_app_flag |
| Balance / Financial | 3 | avg_monthly_balance, balance_trend |
| Rewards | 4 | cashback_received_90d, milestones_completed, reward_tier |
| User Demographics | 4 | months_on_book, affluence_segment, num_products_held |
| Fraud / Risk | 3 | fraud_flag_l1, fraud_flag_l2, chargeback_count_12m |
| Digital Behavior | 3 | has_upi_autopay, peer_txn_ratio, bill_pay_active |

## Example Response

```json
{
  "user_id": "user_001",
  "propensity_score": 0.7821,
  "segment": "high",
  "top_drivers": [
    {"feature": "txn_count_30d", "impact": 0.42, "direction": "increases_propensity"},
    {"feature": "days_since_last_txn", "impact": -0.28, "direction": "decreases_propensity"},
    {"feature": "app_opens_30d", "impact": 0.18, "direction": "increases_propensity"}
  ]
}
```

---

## Run Locally

```bash
git clone https://github.com/BenRoshan100/propensity-api
cd propensity-api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data + train model
jupyter notebook notebooks/01_eda.ipynb    # generates data/processed/features.parquet
jupyter notebook notebooks/02_model.ipynb  # generates models/xgb_model.pkl

# Or train from CLI
python -m src.model.train

# Start the API
uvicorn src.api.main:app --reload
# Open http://localhost:8000/docs
```

## Run Tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t propensity-api .
docker run -p 8000:8000 propensity-api
```

---

## Project Structure

```
propensity-api/
├── data/
│   └── processed/              # Generated feature set
├── notebooks/
│   ├── 01_eda.ipynb            # Synthetic data generation + EDA
│   └── 02_model.ipynb          # Training + SHAP analysis
├── src/
│   ├── model/
│   │   ├── features.py         # Feature engineering (30 UPI features)
│   │   ├── predict.py          # Inference with SHAP
│   │   └── train.py            # CLI training script
│   └── api/
│       ├── schemas.py          # Pydantic v2 models
│       └── main.py             # FastAPI endpoints
├── models/
│   ├── xgb_model.pkl           # Trained model
│   ├── feature_names.json      # Feature list
│   ├── feature_defaults.json   # Training medians for inference
│   └── model_metadata.json     # Model versioning info
├── tests/                      # pytest suite (23 tests)
├── Dockerfile
├── Procfile                    # Render deployment
└── requirements.txt
```

## Production Considerations

If deploying this at scale, the following improvements apply:

- **Feature store:** Pre-computed user-level features (Feast/Tecton) instead of computing on the fly
- **Model versioning:** MLflow for experiment tracking, model registry, and A/B testing
- **Monitoring:** Track score distribution drift over time — dormant-to-active conversion rates shift seasonally
- **Batch inference:** For large-scale campaigns, run offline batch scoring via Spark/Dask rather than real-time API calls
- **A/B testing:** Compare propensity-targeted campaigns against random targeting to measure incremental lift

## Stack

XGBoost | SHAP | FastAPI | Pydantic v2 | scikit-learn | Docker
