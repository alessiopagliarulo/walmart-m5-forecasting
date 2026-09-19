"""Forecasting models for the 28-day direct forecast, all behind one small interface.

A model is fitted on training rows (days <= cutoff, target known) and predicts test
rows (the next 28 days) from their features alone: the harness drops the target from
the test rows before calling `predict`, so no model can read what it forecasts.

* seasonal_naive - every day repeats the same weekday of the last training week.
* linear_regression - ordinary least squares on five sales-history features.
* ridge - L2-regularised linear regression on the full feature pipeline.
* hurdle_logistic_ridge - LogisticRegression for "does the item sell at all today"
  (most item-days are zero sales) times a Ridge for how much it sells when it does.
* lightgbm, xgboost - gradient-boosted trees with Tweedie loss on the full feature
  pipeline, each tuned inside its own training window (see `BoostedModel`).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from m5.config import HORIZON
from m5.data import EVENT_COLS
from m5.evaluation import Fold, evaluator_for, hierarchy_of, to_matrix
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
# Gradient boosting: boosting rounds are chosen by early stopping on the inner
# validation window, capped at BOOST_MAX_ROUNDS.
BOOST_LEARNING_RATE = 0.05
BOOST_MAX_ROUNDS = 3000
BOOST_EARLY_STOPPING = 50
TWEEDIE_POWERS = [1.1, 1.3, 1.5]


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


class BoostedModel:
    """Gradient-boosted trees with Tweedie loss, tuned by a nested walk-forward split.

    Tweedie suits daily unit sales: non-negative, many exact zeros and a long right tail.
    Every categorical (item, department, category, events) goes in as a native
    categorical. `fit` gets only the fold's training rows and tunes inside them:

    1. Inner split: train on days <= cutoff - 28, validate on the last 28 training days.
       This is the outer fold's setup moved back one horizon, so the inner validation
       rows see exactly the information a real forecast would (features use sales at
       least 28 days old, all inside the inner training window).
    2. For every candidate in the grid `search_space`, train with early stopping on the
       inner window's Tweedie deviance, then score the inner forecast with WRMSSE.
    3. Refit the candidate with the lowest inner WRMSSE on every training day, with the
       number of rounds early stopping chose for it.

    The outer fold's test window is never seen by the search. `tuning_` records every
    candidate's inner score and the choice.
    """

    name: str
    description: str
    fixed_params: dict[str, Any]
    search_space: dict[str, list[Any]]

    def __init__(self) -> None:
        self.features = feature_columns()

    def _train(
        self, params: dict[str, Any], train: pd.DataFrame, valid: pd.DataFrame | None, rounds: int
    ) -> tuple[Any, int]:
        """Trains a booster; with `valid`, early-stops on it. Returns it and its rounds."""
        raise NotImplementedError

    def _predict(self, booster: Any, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def candidates(self) -> list[dict[str, Any]]:
        keys = list(self.search_space)
        grid = itertools.product(*(self.search_space[k] for k in keys))
        return [dict(zip(keys, values, strict=True)) for values in grid]

    def fit(self, train: pd.DataFrame) -> BoostedModel:
        cutoff = int(train["d"].max())
        inner = Fold(0, cutoff - HORIZON, cutoff - HORIZON + 1, cutoff)
        inner_train = train[train["d"] <= inner.train_end]
        inner_valid = train[train["d"] >= inner.test_start]
        hierarchy = hierarchy_of(train)
        ids = pd.Index(hierarchy["id"])
        evaluator = evaluator_for(train, hierarchy, inner)
        actual = to_matrix(inner_valid, inner_valid[TARGET].to_numpy(dtype="float64"), ids, inner)

        tried: list[dict[str, Any]] = []
        for candidate in self.candidates():
            params = {**self.fixed_params, **candidate}
            booster, rounds = self._train(params, inner_train, inner_valid, BOOST_MAX_ROUNDS)
            forecast = to_matrix(inner_valid, self._predict(booster, inner_valid), ids, inner)
            score = evaluator.score(actual, forecast)
            tried.append({"params": candidate, "rounds": rounds, "inner_wrmsse": score.wrmsse})
        best = min(tried, key=lambda c: c["inner_wrmsse"])
        self.params_ = {**self.fixed_params, **best["params"]}
        self.booster_, _ = self._train(self.params_, train, None, best["rounds"])
        self.tuning_ = {
            "inner_train_end_d": inner.train_end,
            "inner_valid_d": [inner.test_start, inner.test_end],
            "candidates": tried,
            "chosen": best["params"],
            "chosen_rounds": best["rounds"],
            "chosen_inner_wrmsse": best["inner_wrmsse"],
        }
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        prediction = self._predict(self.booster_, test)
        return np.asarray(np.clip(prediction, 0.0, None), dtype="float64")


class LightGBMModel(BoostedModel):
    name = "lightgbm"
    description = (
        "LightGBM, Tweedie loss, all 52 features, grid-searched per fold on the "
        "last 28 training days."
    )
    fixed_params = {
        "objective": "tweedie",
        "learning_rate": BOOST_LEARNING_RATE,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    search_space = {"tweedie_variance_power": TWEEDIE_POWERS, "num_leaves": [31, 255]}

    def _train(
        self, params: dict[str, Any], train: pd.DataFrame, valid: pd.DataFrame | None, rounds: int
    ) -> tuple[lgb.Booster, int]:
        data = lgb.Dataset(train[self.features], train[TARGET], free_raw_data=True)
        if valid is None:
            return lgb.train(params, data, num_boost_round=rounds), rounds
        booster = lgb.train(
            params,
            data,
            num_boost_round=rounds,
            valid_sets=[lgb.Dataset(valid[self.features], valid[TARGET], reference=data)],
            callbacks=[lgb.early_stopping(BOOST_EARLY_STOPPING, verbose=False)],
        )
        return booster, int(booster.best_iteration)

    def _predict(self, booster: lgb.Booster, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            booster.predict(frame[self.features], num_iteration=booster.best_iteration or None)
        )


class XGBoostModel(BoostedModel):
    name = "xgboost"
    description = (
        "XGBoost (hist), Tweedie loss, all 52 features, grid-searched per fold on the "
        "last 28 training days."
    )
    fixed_params = {
        "objective": "reg:tweedie",
        "tree_method": "hist",
        "learning_rate": BOOST_LEARNING_RATE,
        "min_child_weight": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": SEED,
    }
    search_space = {"tweedie_variance_power": TWEEDIE_POWERS, "max_depth": [6, 10]}

    def _matrix(self, frame: pd.DataFrame, with_target: bool) -> xgb.DMatrix:
        label = frame[TARGET] if with_target else None
        return xgb.DMatrix(frame[self.features], label=label, enable_categorical=True)

    def _train(
        self, params: dict[str, Any], train: pd.DataFrame, valid: pd.DataFrame | None, rounds: int
    ) -> tuple[xgb.Booster, int]:
        data = self._matrix(train, with_target=True)
        if valid is None:
            return xgb.train(params, data, num_boost_round=rounds), rounds
        booster = xgb.train(
            params,
            data,
            num_boost_round=rounds,
            evals=[(self._matrix(valid, with_target=True), "valid")],
            early_stopping_rounds=BOOST_EARLY_STOPPING,
            verbose_eval=False,
        )
        return booster, int(booster.best_iteration) + 1

    def _predict(self, booster: xgb.Booster, frame: pd.DataFrame) -> np.ndarray:
        # After early stopping, predict with the best iteration's trees only.
        rounds = int(booster.best_iteration) + 1 if "best_iteration" in booster.attributes() else 0
        matrix = self._matrix(frame, with_target=False)
        return np.asarray(booster.predict(matrix, iteration_range=(0, rounds)))


MODELS: dict[str, type[Model]] = {
    m.name: m
    for m in (
        SeasonalNaive,
        LinearBaseline,
        RidgeModel,
        HurdleLogisticRidge,
        LightGBMModel,
        XGBoostModel,
    )
}
