import pandas as pd
from src.data import load_data, split_data


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_label_column_exists_and_binary():
    df = load_data()
    assert "label" in df.columns
    assert set(df["label"].unique()) == {0, 1}


def test_no_missing_values():
    df = load_data()
    assert df.isnull().sum().sum() == 0


def test_split_shapes_match():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert abs(len(X_test) / len(df) - 0.2) < 0.02


def test_split_is_reproducible():
    df = load_data()
    a = split_data(df, seed=42)[0].index.tolist()
    b = split_data(df, seed=42)[0].index.tolist()
    assert a == b  