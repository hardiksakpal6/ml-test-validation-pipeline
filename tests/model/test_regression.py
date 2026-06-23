import json
from pathlib import Path

TOLERANCE = 0.01  # allow 1% noise


def test_no_accuracy_regression(trained):
    baseline = json.loads(Path("metrics/baseline.json").read_text())
    current = trained["metrics"]["accuracy"]
    assert current >= baseline["accuracy"] - TOLERANCE, (
        f"Accuracy regressed: {current} < {baseline['accuracy']} - {TOLERANCE}"
    )