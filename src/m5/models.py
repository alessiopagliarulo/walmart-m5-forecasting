"""Forecasting models for the 28-day direct forecast, all behind one small interface.

A model is fitted on training rows (days <= cutoff, target known) and predicts test
rows (the next 28 days) from their features alone: the harness drops the target from
the test rows before calling `predict`, so no model can read what it forecasts.

* seasonal_naive - every day repeats the same weekday of the last training week.
* linear_regression - ordinary least squares on five sales-history features.
* ridge - L2-regularised linear regression on the full feature pipeline.
* hurdle_logistic_ridge - LogisticRegression for "does the item sell at all today"
  (most item-days are zero sales) times a Ridge for how much it sells when it does.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from m5.data import EVENT_COLS
from m5.features import TARGET, feature_columns

SEED = 0

# The simple regression baseline: recent level, weekly pattern and the last known day.
LINEAR_BASELINE_FEATURES = ["lag_28", "lag_35", "roll_mean_7", "roll_mean_28", "same_dow_mean_4"]
# Low-cardinality categoricals are one-hot encoded for the linear models. item_id
# (3,049 levels on one store) is left out: an item's own rolling sales already carry
# its level, and one coefficient per item would mostly memorise the training period.
LINEAR_CATEGORICALS = ["dept_id", "cat_id", *EVENT_COLS]
RIDGE_ALPHA = 1.0
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 500


class Model(Protocol):
    name: str
    description: str

    def fit(self, train: pd.DataFrame) -> Model: ...

    def predict(self, test: pd.DataFrame) -> np.ndarray: ...


class SeasonalNaive:
    """Forecast day cutoff + k with the sales of day cutoff - 7 + ((k - 1) mod 7) + 1."""

    name = "seasonal_naive"
    description = "Repeats the last training week, weekday by weekday."

    def fit(self, train: pd.DataFrame) -> SeasonalNaive:
        self.cutoff_ = int(train["d"].max())
        last_week = train[train["d"] > self.cutoff_ - 7]
        self.last_week_ = last_week.set_index([last_week["id"].astype(str), "d"])[TARGET]
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        offset = (test["d"].to_numpy() - self.cutoff_ - 1) % 7
        source_day = self.cutoff_ - 6 + offset
        key = pd.MultiIndex.from_arrays([test["id"].astype(str).to_numpy(), source_day])
        # An item released after its source day had no sales then.
        values = self.last_week_.reindex(key).fillna(0.0)
        return values.to_numpy(dtype="float64")


def _numeric_features() -> list[str]:
    return [c for c in feature_columns() if c not in {"item_id", *LINEAR_CATEGORICALS}]


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    parts: list[tuple[str, object, list[str]]] = [
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        parts.append(("cat", encoder, categorical))
    return ColumnTransformer(parts, sparse_threshold=0.0)


def _as_input(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    out = frame[numeric].astype("float32")
    for col in categorical:
        out[col] = frame[col].astype("object").where(frame[col].notna(), "none")
    return out


class _Regression:
    """Shared fit/predict for the sklearn regressors; predictions are clipped at 0."""

    name: str
    description: str
    numeric: list[str]
    categorical: list[str]

    def _estimator(self) -> object:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame) -> _Regression:
        self.pipeline_ = Pipeline(
            [("prep", _preprocessor(self.numeric, self.categorical)), ("model", self._estimator())]
        )
        x = _as_input(train, self.numeric, self.categorical)
        self.pipeline_.fit(x, train[TARGET].to_numpy(dtype="float64"))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        x = _as_input(test, self.numeric, self.categorical)
        return np.asarray(np.clip(self.pipeline_.predict(x), 0.0, None))


class LinearBaseline(_Regression):
    name = "linear_regression"
    description = "OLS on lag_28, lag_35, roll_mean_7, roll_mean_28, same_dow_mean_4."
    numeric = LINEAR_BASELINE_FEATURES
    categorical: list[str] = []

    def _estimator(self) -> object:
        return LinearRegression()


class RidgeModel(_Regression):
    name = "ridge"
    description = f"Ridge (alpha={RIDGE_ALPHA}) on the feature pipeline, item_id excluded."

    def __init__(self) -> None:
        self.numeric = _numeric_features()
        self.categorical = LINEAR_CATEGORICALS

    def _estimator(self) -> object:
        return Ridge(alpha=RIDGE_ALPHA, solver="cholesky")


class HurdleLogisticRidge:
    """E[sales] = P(sales > 0) x E[sales | sales > 0].

    The zero/non-zero part is a LogisticRegression, the only place in this project where
    a classification framing fits the problem: on one store about half of all item-days
    sell nothing. The size part is a Ridge fitted on selling days only, floored at 1
    unit because a day that sells sells at least one.
    """

    name = "hurdle_logistic_ridge"
    description = (
        f"LogisticRegression (C={LOGISTIC_C}) for P(sales > 0) times "
        f"Ridge (alpha={RIDGE_ALPHA}) fitted on selling days."
    )

    def __init__(self) -> None:
        self.numeric = _numeric_features()
        self.categorical = LINEAR_CATEGORICALS

    def fit(self, train: pd.DataFrame) -> HurdleLogisticRidge:
        prep = _preprocessor(self.numeric, self.categorical)
        x = prep.fit_transform(_as_input(train, self.numeric, self.categorical))
        y = train[TARGET].to_numpy(dtype="float64")
        sold = y > 0
        self.prep_ = prep
        self.classifier_ = LogisticRegression(
            C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER, random_state=SEED
        ).fit(x, sold)
        self.size_ = Ridge(alpha=RIDGE_ALPHA, solver="cholesky").fit(x[sold], y[sold])
        return self

    def predict_proba_sold(self, test: pd.DataFrame) -> np.ndarray:
        x = self.prep_.transform(_as_input(test, self.numeric, self.categorical))
        return np.asarray(self.classifier_.predict_proba(x)[:, 1])

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        x = self.prep_.transform(_as_input(test, self.numeric, self.categorical))
        p_sold = self.classifier_.predict_proba(x)[:, 1]
        size = np.clip(self.size_.predict(x), 1.0, None)
        return np.asarray(p_sold * size)


MODELS: dict[str, type[Model]] = {
    m.name: m for m in (SeasonalNaive, LinearBaseline, RidgeModel, HurdleLogisticRidge)
}
