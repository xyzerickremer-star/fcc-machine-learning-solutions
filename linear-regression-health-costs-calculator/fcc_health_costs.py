"""freeCodeCamp Machine Learning with Python
Linear Regression Health Costs Calculator

This script solves the insurance-cost regression project using a reproducible
NumPy linear/ridge regression pipeline. It downloads the public insurance.csv
dataset, engineers tabular features, trains against actual dollar charges, and
verifies that the final test MAE is below the freeCodeCamp threshold of 3500.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
RANDOM_SEED = 0
TEST_FRACTION = 0.20
FCC_MAE_THRESHOLD = 3500.0
RIDGE_ALPHA = 0.1


@dataclass(frozen=True)
class PreparedData:
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray


class LinearRegressionModel:
    """Small Keras-like wrapper around closed-form ridge regression."""

    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features) @ self.weights

    def evaluate(self, features: np.ndarray, labels: np.ndarray, verbose: int = 0) -> tuple[float, float]:
        del verbose
        predictions = self.predict(features)
        errors = predictions - labels
        mse = float(np.mean(errors**2))
        mae = float(np.mean(np.abs(errors)))
        return mse, mae


def _read_rows(csv_text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(csv_text)))


def download_dataset(cache_path: Path = Path("insurance.csv")) -> list[dict[str, str]]:
    """Download insurance.csv, caching it locally for repeatable offline reruns."""
    if cache_path.exists():
        return _read_rows(cache_path.read_text(encoding="utf-8"))

    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        csv_text = response.read().decode("utf-8")
    cache_path.write_text(csv_text, encoding="utf-8")
    return _read_rows(csv_text)


def rows_to_features(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw CSV rows to numeric features and dollar-cost labels.

    The engineered interaction terms are intentionally simple and interpretable:
    smoking status has a large multiplicative effect in this dataset, especially
    together with age and BMI, so cross-features help a linear model meet the FCC
    MAE target while remaining explainable.
    """
    sexes = sorted({row["sex"] for row in rows})
    regions = sorted({row["region"] for row in rows})

    features: list[list[float]] = []
    labels: list[float] = []
    for row in rows:
        age = float(row["age"])
        bmi = float(row["bmi"])
        children = float(row["children"])
        smoker = 1.0 if row["smoker"] == "yes" else 0.0
        obese = 1.0 if bmi >= 30.0 else 0.0

        engineered = [
            1.0,  # intercept
            age,
            bmi,
            children,
            age * age,
            bmi * bmi,
            children * children,
            smoker,
            obese,
            smoker * age,
            smoker * bmi,
            smoker * children,
            smoker * obese,
            smoker * age * bmi,
            smoker * age * obese,
            age * children,
            bmi * children,
        ]
        one_hot = [1.0 if row["sex"] == value else 0.0 for value in sexes]
        one_hot += [1.0 if row["region"] == value else 0.0 for value in regions]

        features.append(engineered + one_hot)
        labels.append(float(row["charges"]))

    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def prepare_data(rows: list[dict[str, str]]) -> PreparedData:
    features, labels = rows_to_features(rows)

    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(features))
    rng.shuffle(indices)
    split = int(len(indices) * (1.0 - TEST_FRACTION))
    train_idx, test_idx = indices[:split], indices[split:]

    train_features = features[train_idx]
    test_features = features[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std == 0.0] = 1.0

    train_features = (train_features - feature_mean) / feature_std
    test_features = (test_features - feature_mean) / feature_std
    train_features[:, 0] = 1.0  # keep intercept unscaled
    test_features[:, 0] = 1.0

    return PreparedData(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def train_model(data: PreparedData, ridge_alpha: float = RIDGE_ALPHA) -> LinearRegressionModel:
    """Fit ridge regression with a closed-form normal-equation solution."""
    x_train = data.train_features
    y_train = data.train_labels
    regularizer = ridge_alpha * np.eye(x_train.shape[1])
    regularizer[0, 0] = 0.0  # do not regularize intercept

    weights = np.linalg.pinv(x_train.T @ x_train + regularizer) @ x_train.T @ y_train
    return LinearRegressionModel(weights)


def main() -> None:
    rows = download_dataset()
    data = prepare_data(rows)
    model = train_model(data)
    loss, mae = model.evaluate(data.test_features, data.test_labels, verbose=0)
    print(f"Test loss (MSE): {loss:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"FCC threshold: < {FCC_MAE_THRESHOLD:.0f}")

    sample_predictions = model.predict(data.test_features[:5]).reshape(-1)
    print("Sample predictions vs actual charges:")
    for predicted, actual in zip(sample_predictions, data.test_labels[:5]):
        print(f"  predicted=${predicted:,.2f}  actual=${actual:,.2f}")

    if mae >= FCC_MAE_THRESHOLD:
        raise SystemExit(f"FAILED: MAE {mae:.2f} is not below {FCC_MAE_THRESHOLD:.0f}")
    print("PASSED: final MAE is below the freeCodeCamp requirement.")


if __name__ == "__main__":
    main()
