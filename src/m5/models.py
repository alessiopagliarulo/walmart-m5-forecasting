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

from collections.abc import Iterator
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

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


class LinearDesign:
    """Design matrix for the linear models, built column by column into one float32 array.

    Numeric features: missing values filled with 0 plus a 0/1 "was missing" column for
    every feature that has gaps in training (with that column, the fill value makes no
    difference to a linear model), then standardised with training statistics.
    Categoricals: one-hot over the categories seen in training ("none" for no event);
    unseen categories get all zeros.

    Written by hand rather than with a scikit-learn ColumnTransformer because on the
    4.5 million training rows the latter's intermediate copies needed about 10 GB.
    """

    def __init__(self, numeric: list[str], categorical: list[str]) -> None:
        self.numeric = numeric
        self.categorical = categorical

    def _raw_columns(self, frame: pd.DataFrame) -> Iterator[tuple[str, np.ndarray]]:
        """Yields one column at a time so only the output matrix is ever held whole."""
        for name in self.numeric:
            values = frame[name].to_numpy(dtype="float32", na_value=np.nan)
            yield name, np.nan_to_num(values, nan=0.0)
            if name in self.missing_:
                yield f"{name}_missing", np.isnan(values).astype("float32")
        for name in self.categorical:
            labels = frame[name].astype("object").where(frame[name].notna(), "none").to_numpy()
            for level in self.levels_[name]:
                yield f"{name}={level}", (labels == level).astype("float32")

    def fit(self, frame: pd.DataFrame) -> LinearDesign:
        self.missing_ = {n for n in self.numeric if frame[n].isna().any()}
        self.levels_ = {
            n: sorted(frame[n].astype("object").where(frame[n].notna(), "none").unique())
            for n in self.categorical
        }
        self.names_: list[str] = []
        self.mean_: list[float] = []
        self.std_: list[float] = []
        for name, col in self._raw_columns(frame):
            self.names_.append(name)
            self.mean_.append(float(col.mean(dtype="float64")))
            std = float(col.std(dtype="float64"))
            self.std_.append(std if std > 0 else 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(frame), len(self.names_)), dtype="float32")
        for j, (_, col) in enumerate(self._raw_columns(frame)):
            out[:, j] = (col - self.mean_[j]) / self.std_[j]
        return out


class _Regression:
    """Shared fit/predict for the sklearn regressors; predictions are clipped at 0."""

    name: str
    description: str
    numeric: list[str]
    categorical: list[str]

    def _estimator(self) -> LinearRegression | Ridge:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame) -> _Regression:
        self.design_ = LinearDesign(self.numeric, self.categorical).fit(train)
        x = self.design_.transform(train)
        self.model_ = self._estimator().fit(x, train[TARGET].to_numpy(dtype="float32"))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        prediction = self.model_.predict(self.design_.transform(test))
        return np.asarray(np.clip(prediction, 0.0, None), dtype="float64")


class LinearBaseline(_Regression):
    name = "linear_regression"
    description = "OLS on lag_28, lag_35, roll_mean_7, roll_mean_28, same_dow_mean_4."
    numeric = LINEAR_BASELINE_FEATURES
    categorical: list[str] = []

    def _estimator(self) -> LinearRegression:
        return LinearRegression(copy_X=False)


class RidgeModel(_Regression):
    name = "ridge"
    description = f"Ridge (alpha={RIDGE_ALPHA}) on the feature pipeline, item_id excluded."

    def __init__(self) -> None:
        self.numeric = _numeric_features()
        self.categorical = LINEAR_CATEGORICALS

    def _estimator(self) -> Ridge:
        return Ridge(alpha=RIDGE_ALPHA, solver="cholesky", copy_X=False)


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
        self.design_ = LinearDesign(self.numeric, self.categorical).fit(train)
        x = self.design_.transform(train)
        y = train[TARGET].to_numpy(dtype="float32")
        sold = y > 0
        self.classifier_ = LogisticRegression(
            C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER, random_state=SEED
        ).fit(x, sold)
        self.size_ = Ridge(alpha=RIDGE_ALPHA, solver="cholesky", copy_X=False)
        self.size_.fit(x[sold], y[sold])
        return self

    def predict_proba_sold(self, test: pd.DataFrame) -> np.ndarray:
        proba = self.classifier_.predict_proba(self.design_.transform(test))[:, 1]
        return np.asarray(proba, dtype="float64")

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        x = self.design_.transform(test)
        p_sold = self.classifier_.predict_proba(x)[:, 1]
        size = np.clip(self.size_.predict(x), 1.0, None)
        return np.asarray(p_sold * size, dtype="float64")


MODELS: dict[str, type[Model]] = {
    m.name: m for m in (SeasonalNaive, LinearBaseline, RidgeModel, HurdleLogisticRidge)
}
