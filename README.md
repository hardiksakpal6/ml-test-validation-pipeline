# ML Test &amp; Validation Pipeline

**An end-to-end quality-assurance pipeline for a machine-learning system — automated unit tests, data validation, model quality gates, regression testing, and CI that blocks regressions before merge.**

![CI](https://github.com/hardiksakpal6/ml-test-validation-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)

---

## Overview

This project applies **software-testing discipline to the machine-learning lifecycle**. Instead of only training a model, it treats the model and its data as software that must be continuously tested: every change runs an automated suite of unit tests, data-quality checks, model-performance gates, and regression tests through a CI pipeline that **fails the build if quality drops**.

It is built as a portfolio piece for **MLOps / AI Quality / Test Automation** work.

The model under test is a `RandomForestClassifier` trained on the scikit-learn breast-cancer dataset — deliberately simple, so the focus stays on the **testing and quality engineering**, not the model.

---

## Quality strategy

| Layer | What is tested | Tooling |
|---|---|---|
| **Unit tests** | Data-loading and split logic behave as specified | `pytest` |
| **Data validation** | Schema, types, value ranges, nulls, duplicates, class balance | `pandera` |
| **Model quality gates** | Accuracy / F1 stay above threshold; valid classes; order-invariance | `pytest` |
| **Regression testing** | New changes do not degrade accuracy vs. a stored baseline | baseline metrics + assertion |
| **CI / continuous testing** | Every push & PR runs the full suite; failure blocks merge | GitHub Actions |
| **Reproducibility** | Fixed seeds; deterministic, repeatable runs | `scikit-learn`, seeds |

---

## Architecture

```
   push / pull request
            │
            ▼
   ┌──────────────────────────── GitHub Actions (Ubuntu) ───────────────────────────┐
   │  1. Install dependencies                                                        │
   │  2. Train model & write metrics      (src/train.py -> metrics/metrics.json)     │
   │  3. Unit & data-validation tests     (tests/unit, tests/data)                   │
   │  4. Model quality gates & regression (tests/model)                              │
   │  5. Report metrics                                                              │
   └────────────────────────────────────────────────────────────────────────────────┘
            │  all pass → green / merge allowed
            │  any fail → red / merge blocked
```

---

## Tests implemented

### Unit tests — `tests/unit/test_data.py`
- Data loads as a non-empty DataFrame
- Label column exists and is binary (0/1)
- No missing values
- Train/test split shapes match and respect the split ratio
- Split is reproducible for a fixed seed

### Data-validation tests — `tests/data/test_data_validation.py`
- **Schema** validation with `pandera` (all features numeric, non-negative, non-null; label in {0,1})
- No duplicate rows
- Class balance is reasonable (not degenerate)

### Model quality gates — `tests/model/test_model_quality.py`
- **Accuracy gate:** build fails if accuracy &lt; 0.90
- **F1 gate:** build fails if F1 &lt; 0.90
- Predictions are valid classes only
- **Invariance:** shuffling row order does not change predictions

### Regression test — `tests/model/test_regression.py`
- Current accuracy must not drop more than 1% below the committed baseline (`metrics/baseline.json`)

---

## Tech stack

- **Testing:** pytest, pandera
- **CI:** GitHub Actions
- **ML:** Python, scikit-learn, pandas, numpy, joblib

---

## Repository structure

```
.
├── src/
│   ├── __init__.py
│   ├── data.py          # load & split the dataset
│   └── train.py         # train model, save model + metrics
├── tests/
│   ├── unit/            # test_data.py
│   ├── data/            # test_data_validation.py
│   └── model/           # conftest.py, test_model_quality.py, test_regression.py
├── metrics/
│   ├── metrics.json     # latest run metrics (generated)
│   └── baseline.json    # committed baseline for regression checks
├── .github/workflows/ci.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## How to run locally

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates metrics/metrics.json)
python -m src.train

# 4. Run the full test suite
pytest -v
```

The same suite runs automatically in CI on every push and pull request.

---

## Current results

| Metric | Value | Gate |
|---|---|---|
| Accuracy | 0.9561 | ≥ 0.90 ✅ |
| F1 score | 0.9655 | ≥ 0.90 ✅ |
| Unit + data + model tests | all passing | ✅ |
| CI pipeline | green | ✅ |

---

## What this project demonstrates

- Designing a **test strategy for an ML system**, not just training a model
- Writing **automated tests**: unit, data-validation, model-quality, and regression
- Enforcing **CI quality gates** that block regressions before merge
- Ensuring **reproducibility** with fixed seeds and committed baselines

---

## Author

**Hardik Sakpal** — M.Sc. Computational Sciences in Engineering, TU Braunschweig
[GitHub](https://github.com/hardiksakpal6) · [LinkedIn](https://linkedin.com/in/hardikrsakpal)
