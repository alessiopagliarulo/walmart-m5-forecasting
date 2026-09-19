"""Walk-forward folds and the M5 accuracy metric (WRMSSE), plus RMSSE and MAE.

WRMSSE follows the competition guidelines (Makridakis, Spiliotis & Assimakopoulos,
"The M5 competition", section on the accuracy metric):

* Every series of the 12-level hierarchy is scored with RMSSE: the forecast's RMSE
  divided by the in-sample RMSE of the one-step naive forecast, computed on the
  training period from the series' first non-zero sale onward.
* Each series is weighted by its dollar sales (units x sell price) over the last 28
  days of the training window. Weights sum to 1 within a level, and the 12 levels
  count equally.

On a one-store subset the store, state and item-by-location levels coincide with the
total, category, department and item levels, so the 12 levels reduce to 4 distinct
ones (total, category, department, item), each counted three times. The formula is
unchanged, which keeps the number comparable in spirit to the competition's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from m5.config import HORIZON
from m5.features import TARGET

HIERARCHY_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
# The 12 M5 aggregation levels, as the columns each one groups by.
M5_LEVELS: dict[str, list[str]] = {
    "total": [],
    "state": ["state_id"],
    "store": ["store_id"],
    "cat": ["cat_id"],
    "dept": ["dept_id"],
    "state_cat": ["state_id", "cat_id"],
    "state_dept": ["state_id", "dept_id"],
    "store_cat": ["store_id", "cat_id"],
    "store_dept": ["store_id", "dept_id"],
    "item": ["item_id"],
    "item_state": ["item_id", "state_id"],
    "item_store": ["item_id", "store_id"],
}
WEIGHT_DAYS = 28
N_FOLDS = 3


@dataclass(frozen=True)
class Fold:
    """Train on days d <= train_end, forecast days test_start..test_end."""

    index: int
    train_end: int
    test_start: int
    test_end: int


def make_folds(last_day: int, n_folds: int = N_FOLDS, horizon: int = HORIZON) -> list[Fold]:
    """Back-to-back walk-forward folds whose last test window ends on `last_day`.

    Fold 1 is the oldest. Each fold trains on every day before its cutoff and
    forecasts the next `horizon` days, so the test windows never overlap.
    """
    folds = []
    for i in range(n_folds):
        train_end = last_day - (n_folds - i) * horizon
        if train_end < 1:
            raise ValueError(f"{n_folds} folds of {horizon} days need more than {last_day} days")
        folds.append(Fold(i + 1, train_end, train_end + 1, train_end + horizon))
    return folds


def naive_scale(train: np.ndarray) -> np.ndarray:
    """Per row: mean squared one-step change from the first non-zero value on.

    NaN where fewer than two observations follow the first non-zero value (or there is
    none), because RMSSE is then undefined.
    """
    nonzero = train != 0
    first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), train.shape[1])
    diffs = np.diff(train, axis=1)
    valid = np.arange(diffs.shape[1])[None, :] >= first[:, None]
    count = valid.sum(axis=1)
    total = np.where(valid, diffs**2, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def rmsse(actual: np.ndarray, forecast: np.ndarray, scale: np.ndarray) -> np.ndarray:
    mse = ((actual - forecast) ** 2).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.sqrt(mse / scale))


def _aggregator(hierarchy: pd.DataFrame, keys: list[str]) -> sparse.csr_matrix:
    """Sparse 0/1 matrix mapping bottom series (columns) to the level's groups (rows)."""
    n = len(hierarchy)
    if keys:
        codes = hierarchy.groupby(keys, observed=True, sort=True).ngroup().to_numpy()
    else:
        codes = np.zeros(n, dtype=np.int64)
    return sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(int(codes.max()) + 1, n))


@dataclass
class Score:
    wrmsse: float
    by_level: dict[str, float]
    # Bottom-level (one row per series) diagnostics.
    rmsse: np.ndarray
    mae: float
    rmse: float
    # Dollar-weight share of series whose RMSSE is undefined (no usable history); these
    # are left out and the rest of their level re-normalised. Reported, never hidden.
    undefined_weight_share: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        finite = self.rmsse[np.isfinite(self.rmsse)]
        return {
            "wrmsse": self.wrmsse,
            "wrmsse_by_level": self.by_level,
            "rmsse_mean": float(finite.mean()),
            "rmsse_median": float(np.median(finite)),
            "mae": self.mae,
            "rmse": self.rmse,
            "undefined_weight_share": self.undefined_weight_share,
        }


class WRMSSEEvaluator:
    """Scores 28-day forecasts for one fold.

    hierarchy: one row per bottom series with HIERARCHY_COLS (row order defines the
        order of every matrix below).
    train_sales: series x training days, zeros before release (as in the raw file).
    train_prices: same shape, NaN where there is no price (only possible with no sales).
    """

    def __init__(
        self,
        hierarchy: pd.DataFrame,
        train_sales: np.ndarray,
        train_prices: np.ndarray,
        levels: dict[str, list[str]] | None = None,
        weight_days: int = WEIGHT_DAYS,
    ) -> None:
        if train_sales.shape != train_prices.shape or train_sales.shape[0] != len(hierarchy):
            raise ValueError("hierarchy, train_sales and train_prices do not line up")
        self.levels = M5_LEVELS if levels is None else levels
        recent = train_sales[:, -weight_days:]
        dollars = np.nansum(recent * train_prices[:, -weight_days:], axis=1)
        if np.any(np.isnan(train_prices[:, -weight_days:]) & (recent > 0)):
            raise ValueError("sales without a sell price in the weighting window")

        self._bottom_scale = naive_scale(train_sales)
        self._agg: dict[str, sparse.csr_matrix] = {}
        self._scale: dict[str, np.ndarray] = {}
        self._weight: dict[str, np.ndarray] = {}
        self.undefined_weight_share: dict[str, float] = {}
        for name, keys in self.levels.items():
            agg = _aggregator(hierarchy, keys)
            scale = naive_scale(np.asarray(agg @ train_sales))
            weight = np.asarray(agg @ dollars)
            defined = np.isfinite(scale) & (scale > 0)
            total = weight.sum()
            if total <= 0:
                raise ValueError(f"level {name!r} has no dollar sales in the weighting window")
            self.undefined_weight_share[name] = float(weight[~defined].sum() / total)
            weight = np.where(defined, weight, 0.0)
            self._agg[name], self._scale[name] = agg, scale
            self._weight[name] = weight / weight.sum()

    def score(self, actual: np.ndarray, forecast: np.ndarray) -> Score:
        if actual.shape != forecast.shape or not np.isfinite(forecast).all():
            raise ValueError("forecast must be finite and shaped like actual")
        by_level = {}
        for name, agg in self._agg.items():
            errors = rmsse(np.asarray(agg @ actual), np.asarray(agg @ forecast), self._scale[name])
            weight = self._weight[name]
            by_level[name] = float(np.sum(weight[weight > 0] * errors[weight > 0]))
        bottom = rmsse(actual, forecast, self._bottom_scale)
        err = actual - forecast
        return Score(
            wrmsse=float(np.mean(list(by_level.values()))),
            by_level=by_level,
            rmsse=bottom,
            mae=float(np.abs(err).mean()),
            rmse=float(np.sqrt((err**2).mean())),
            undefined_weight_share=self.undefined_weight_share,
        )


# Panel (one row per series-day) to the series x days matrices the evaluator scores.


def _series_index(rows: pd.DataFrame, ids: pd.Index) -> np.ndarray:
    ids_col = rows["id"]
    if isinstance(ids_col.dtype, pd.CategoricalDtype):
        lookup = ids.get_indexer(ids_col.cat.categories.astype(str))
        r = lookup[ids_col.cat.codes.to_numpy()]
    else:
        r = ids.get_indexer(pd.Index(ids_col.astype(str)))
    if (r < 0).any():
        raise ValueError("rows for series outside the hierarchy")
    return np.asarray(r)


def _wide(panel: pd.DataFrame, ids: pd.Index, days: range, column: str) -> np.ndarray:
    """series x days matrix of `column`; NaN where the panel has no row (not released)."""
    frame = panel[(panel["d"] >= days.start) & (panel["d"] < days.stop)]
    out = np.full((len(ids), len(days)), np.nan)
    out[_series_index(frame, ids), frame["d"].to_numpy() - days.start] = frame[column].to_numpy()
    return out


def hierarchy_of(panel: pd.DataFrame) -> pd.DataFrame:
    first = panel.drop_duplicates("id")[HIERARCHY_COLS].copy()
    for col in HIERARCHY_COLS:
        first[col] = first[col].astype(str)
    return first.sort_values("id").reset_index(drop=True)


def evaluator_for(panel: pd.DataFrame, hierarchy: pd.DataFrame, fold: Fold) -> WRMSSEEvaluator:
    ids = pd.Index(hierarchy["id"])
    days = range(1, fold.train_end + 1)
    sales = np.nan_to_num(_wide(panel, ids, days, TARGET), nan=0.0)
    prices = _wide(panel, ids, days, "sell_price")
    return WRMSSEEvaluator(hierarchy, sales, prices, M5_LEVELS, WEIGHT_DAYS)


def to_matrix(rows: pd.DataFrame, values: np.ndarray, ids: pd.Index, fold: Fold) -> np.ndarray:
    """Lay row-wise values out as series x horizon; days before release are 0."""
    out = np.zeros((len(ids), fold.test_end - fold.test_start + 1))
    out[_series_index(rows, ids), rows["d"].to_numpy() - fold.test_start] = values
    return out
