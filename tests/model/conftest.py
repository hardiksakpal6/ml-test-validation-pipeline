import joblib
import pytest
from src.train import train, MODEL_PATH


@pytest.fixture(scope="session")
def trained():
    metrics = train()
    model = joblib.load(MODEL_PATH)
    return {"model": model, "metrics": metrics}