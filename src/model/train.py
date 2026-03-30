"""
CLI training script — reproducible model training from the command line.

Usage:
    python -m src.model.train
    python -m src.model.train --data-path data/processed/features.parquet --n-estimators 500
"""

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.model.features import FEATURE_COLS


def precision_at_k(y_true, y_scores, k):
    """Out of top K ranked users, what fraction are truly positive?"""
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return y_true.iloc[top_k_idx].mean()


def train(
    data_path: str = "data/processed/features.parquet",
    output_dir: str = "models",
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    test_size: float = 0.2,
    random_state: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    X = df[FEATURE_COLS]
    y = df["will_transact"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.1f}")

    # Train
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=random_state,
        n_jobs=-1,
    )

    train_start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    train_time = time.time() - train_start

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nAUC-ROC: {auc:.4f}")

    results = []
    n_test = len(y_test)
    baseline = y_test.mean()
    for k_pct in [0.01, 0.05, 0.10, 0.20]:
        k = int(n_test * k_pct)
        p_at_k = precision_at_k(y_test, y_prob, k)
        lift = p_at_k / baseline
        label = f"Precision@{int(k_pct*100)}%"
        print(f"{label}: {p_at_k:.3f} (lift: {lift:.1f}x)")
        results.append({"metric": label, "value": round(p_at_k, 4), "lift": round(lift, 1)})

    # Save model
    model_path = output_dir / "xgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

    # Save feature defaults (medians)
    feature_defaults = {}
    for col in FEATURE_COLS:
        feature_defaults[col] = float(X_train[col].median())

    with open(output_dir / "feature_defaults.json", "w") as f:
        json.dump(feature_defaults, f, indent=2)

    # Save metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_features": len(FEATURE_COLS),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "auc_roc": round(auc, 4),
        "best_iteration": model.best_iteration,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "xgboost_version": xgb.__version__,
        "precision_at_k": results,
        "training_time_seconds": round(train_time, 1),
    }

    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts saved to {output_dir}/")
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Train UPI Propensity Model")
    parser.add_argument("--data-path", default="data/processed/features.parquet")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
