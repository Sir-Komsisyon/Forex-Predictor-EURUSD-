"""
Phase 7 — Proper Backtesting (Walk-Forward Validation)
Never use a random shuffled train/test split on time series data — it leaks
future information into training. This script retrains on an expanding
window and tests on the immediately following block, repeated across the
whole dataset, then reports honest out-of-sample metrics + a calibration
curve.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from db_config import get_engine
from phase6_train_model import load_dataset, FEATURE_COLUMNS

N_FOLDS = 5


def walk_forward_folds(df: pd.DataFrame, n_folds: int):
    """
    Splits the dataframe into n_folds chronological blocks.
    Fold i trains on everything before the block, tests on the block itself.
    """
    fold_size = len(df) // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]
        if len(test) == 0:
            continue
        folds.append((train, test))
    return folds


def run_walk_forward(df: pd.DataFrame):
    label_map = {-1: 0, 0: 1, 1: 2}
    inverse_map = {v: k for k, v in label_map.items()}

    all_true, all_pred, all_confidence = [], [], []

    folds = walk_forward_folds(df, N_FOLDS)

    for i, (train, test) in enumerate(folds, start=1):
        X_train, y_train = train[FEATURE_COLUMNS], train["direction"].map(label_map)
        X_test, y_test = test[FEATURE_COLUMNS], test["direction"].map(label_map)

        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        confidence = probs.max(axis=1)

        acc = accuracy_score(y_test, preds)
        print(f"\nFold {i}: train={len(train)} rows, test={len(test)} rows, accuracy={acc:.4f}")

        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())
        all_confidence.extend(confidence.tolist())

    return np.array(all_true), np.array(all_pred), np.array(all_confidence), inverse_map


def report_results(y_true, y_pred, confidence, inverse_map):
    print("\n=== OVERALL WALK-FORWARD RESULTS ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion Matrix (rows=actual, cols=predicted) [-1, 0, 1]:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["down(-1)", "flat(0)", "up(1)"]))

    # Calibration: is a "70% confident" prediction actually right 70% of the time?
    # We treat "correct" as a binary outcome and bin by confidence level.
    correct = (y_true == y_pred).astype(int)
    prob_true, prob_pred = calibration_curve(correct, confidence, n_bins=10, strategy="quantile")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title("Calibration Curve — Confidence vs Actual Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("calibration_curve.png")
    print("\nSaved calibration_curve.png")


def main():
    engine = get_engine()
    df = load_dataset(engine)
    print(f"Loaded {len(df)} rows for walk-forward backtest.")

    y_true, y_pred, confidence, inverse_map = run_walk_forward(df)
    report_results(y_true, y_pred, confidence, inverse_map)


if __name__ == "__main__":
    main()
