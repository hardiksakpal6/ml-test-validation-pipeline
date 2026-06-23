import pandera as pa
from pandera import Column, Check, DataFrameSchema
from src.data import load_data

def build_schema(df):
    feature_cols = [c for c in df.columns if c != "label"]
    schema = DataFrameSchema(
        {c: Column(float, Check.ge(0), nullable=False) for c in feature_cols}
        | {"label": Column(int, Check.isin([0, 1]), nullable=False)}
    )
    return schema

def test_data_matches_schema():
    df = load_data()
    schema = build_schema(df)
    schema.validate(df)  

def test_no_duplicate_rows():
    df = load_data()
    assert df.duplicated().sum() == 0

def test_class_balance_reasonable():
    df = load_data()
    share = df["label"].mean()
    assert 0.2 < share < 0.8