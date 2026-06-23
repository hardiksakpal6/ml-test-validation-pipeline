import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_data, split_data

MODEL_PATH = Path("model.joblib")
METRICS_PATH=Path("metrics.json")

def train():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("Saved model and metrics:", metrics)
    return metrics


if __name__ == "__main__":
    train()