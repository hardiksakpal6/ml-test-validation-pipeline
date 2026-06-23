import numpy as np
from src.data import load_data, split_data


def test_no_major_mean_drift_between_splits():
    df = load_data()
    X_train, X_test, _, _ = split_data(df, seed=42)
    for col in X_train.columns:
        train_mean = X_train[col].mean()
        test_mean = X_test[col].mean()
        denom = abs(train_mean) + 1e-9
        rel_shift = abs(train_mean - test_mean) / denom
        # flag if a feature's mean shifts > 35% between splits
        assert rel_shift < 0.35, f"Possible drift in {col}: {rel_shift:.2f}"
