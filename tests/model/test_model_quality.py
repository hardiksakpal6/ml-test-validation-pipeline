import numpy as np
import pandas as pd
from src.data import load_data, split_data

ACCURACY_GATE = 0.90
F1_GATE = 0.90


def test_accuracy_above_gate(trained):
    assert trained["metrics"]["accuracy"] >= ACCURACY_GATE


def test_f1_above_gate(trained):
    assert trained["metrics"]["f1"] >= F1_GATE


def test_predictions_are_valid_classes(trained):
    df = load_data()
    _, X_test, _, _ = split_data(df)
    preds = trained["model"].predict(X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_invariance_to_row_order(trained):
    df = load_data()
    _, X_test, _, _ = split_data(df)
    model = trained["model"]
    p1 = model.predict(X_test)
    shuffled = X_test.sample(frac=1.0, random_state=1)
    p2 = model.predict(shuffled)
    s1 = pd.Series(p1, index=X_test.index)
    s2 = pd.Series(p2, index=shuffled.index)
    assert (s1.sort_index().values == s2.sort_index().values).all()