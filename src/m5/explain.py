"""SHAP explanations of the boosted models and the SNAP-day lift measured from the data.

Usage:
    m5-explain                          # after m5-backtest

For every boosted model and walk-forward fold, rebuilds the exact model the backtest
chose (the parameters and boosting rounds recorded in results/metrics.json, trained on
the fold's training days) and stops unless its forecast reproduces the recorded WRMSSE.
It then computes exact TreeSHAP values, with the boosting library's own
implementation, for every row of the fold's 28-day test window: no sampling. The
values are in the model's raw output space, log(expected units), because the models
use a Tweedie loss with a log link.

Separately, and without any model, it measures how much more each category sells on
SNAP days (see `m5.snap`).

Writes, all generated here and never edited by hand:
  results/shap_summary.json         global importance per model and fold, feature groups,
                                    series-level readings, error by segment
  results/snap_lift.json            SNAP-day lift from the raw sales
  results/explanations.md           the same as readable tables
  results/shap_summary_<model>_<store>.png   beeswarm of the latest fold
  results/shap_importance_<store>.png        mean |SHAP| of both models, all folds
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from m5 import backtest, config  # noqa: E402
from m5.data import EVENT_COLS, load_calendar, load_state_sales, load_store_sales  # noqa: E402
from m5.evaluation import Fold, evaluator_for, hierarchy_of, make_folds, to_matrix  # noqa: E402
from m5.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    PRICE_FEATURES,
    TARGET,
    feature_columns,
)
from m5.models import MODELS, BoostedModel  # noqa: E402
from m5.pipeline import check_hashes  # noqa: E402
from m5.snap import (  # noqa: E402
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    daily_category_units,
    snap_lift,
    snap_method,
)
from m5.verify import DataVerificationError  # noqa: E402

# The beeswarm draws a random subset of the test rows so the points stay readable; every
# number in the JSON uses all rows.
PLOT_SAMPLE_SIZE = 5_000
PLOT_SEED = 0
PLOT_TOP_FEATURES = 20
TOP_FEATURES = 10
# Forecasts must reproduce the backtest's recorded WRMSSE to this tolerance.
REPRODUCE_TOLERANCE = 1e-9
# Rows whose item was released fewer than this many days earlier: the longest rolling
# windows (182 days) are still empty for them.
NEW_ITEM_DAYS = 182


def feature_group(name: str) -> str:
    """Families used to sum importance, so near-duplicate features are read together."""
    if name in ("item_id", "dept_id", "cat_id"):
        return "product identity"
    if name.startswith("lag_"):
        return "sales lags"
    if name.startswith("roll_mean_"):
        return "rolling means"
    if name.startswith("roll_std_"):
        return "rolling std devs"
    if name.startswith("zero_frac_"):
        return "zero-sales share"
    if name == "same_dow_mean_4":
        return "same-weekday mean"
    if name == "days_since_release":
        return "days since release"
    if name in PRICE_FEATURES:
        return "price"
    if name in ("wday", "day_of_week", "is_weekend"):
        return "weekday"
    if name in (*EVENT_COLS, "has_event"):
        return "events"
    if name == "snap":
        return "snap"
    return "other calendar"


def ranked(values: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    """Mean |SHAP| per name, largest first, with its share of the total."""
    mean_abs = np.abs(values).mean(axis=0)
    total = float(mean_abs.sum())
    order = np.argsort(-mean_abs, kind="stable")
    return [
        {
            "rank": i + 1,
            "feature": names[j],
            "mean_abs_shap": float(mean_abs[j]),
            "share": float(mean_abs[j] / total) if total > 0 else 0.0,
        }
        for i, j in enumerate(order)
    ]


def grouped(values: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    """Mean over rows of |sum of a group's SHAP values|, largest first."""
    groups = sorted({feature_group(n) for n in names})
    summed = np.column_stack(
        [values[:, [feature_group(n) == g for n in names]].sum(axis=1) for g in groups]
    )
    return [{"group": r["feature"], **{k: v for k, v in r.items() if k != "feature"}}
            for r in ranked(summed, groups)]  # fmt: skip


def rebuild(
    name: str, panel: pd.DataFrame, fold: Fold, fold_result: dict[str, Any]
) -> tuple[BoostedModel, pd.DataFrame]:
    model_class = MODELS[name]
    if not (isinstance(model_class, type) and issubclass(model_class, BoostedModel)):
        raise TypeError(f"{name} is not a boosted model")
    train, test = backtest.split(panel, fold)
    tuning = fold_result["tuning"]
    return model_class().refit(train, tuning["chosen"], tuning["chosen_rounds"]), test


def series_reading(
    model: BoostedModel, train: pd.DataFrame, test: pd.DataFrame, truth: pd.Series,
    forecast: np.ndarray, values: np.ndarray, base: np.ndarray, series_id: str, why: str,
) -> dict[str, Any]:  # fmt: skip
    """One series over the 28 test days: totals and its features' mean signed SHAP."""
    recent = train[(train["d"] > train["d"].max() - config.HORIZON) & (train["id"] == series_id)]
    rows = (test["id"] == series_id).to_numpy()
    mean_shap = values[rows].mean(axis=0)
    order = np.argsort(-np.abs(mean_shap), kind="stable")[:5]
    first = test[rows].iloc[0]
    return {
        "id": series_id,
        "why_chosen": why,
        "n_days": int(rows.sum()),
        "units_last_28_training_days": float(recent[TARGET].sum()),
        "actual_units": float(truth[rows].sum()),
        "forecast_units": float(forecast[rows].sum()),
        "days_since_release_at_start": int(first["days_since_release"]),
        "zero_frac_112_at_start": None
        if pd.isna(first["zero_frac_112"])
        else float(first["zero_frac_112"]),
        "mean_base_value": float(base[rows].mean()),
        "top_features": [
            {"feature": model.features[j], "mean_shap": float(mean_shap[j])} for j in order
        ],
    }


def pick_series(train: pd.DataFrame, test: pd.DataFrame, truth: pd.Series,
                forecast: np.ndarray) -> list[tuple[str, str]]:  # fmt: skip
    """Three series chosen by rule, not by eye: the top seller, the largest miss, the newest."""
    recent = train[train["d"] > train["d"].max() - config.HORIZON]
    top = str(recent.groupby("id", observed=True)[TARGET].sum().idxmax())
    miss = pd.Series(np.abs(forecast - truth.to_numpy()), index=test["id"].astype(str))
    worst = str(miss.groupby(level=0).sum().idxmax())
    first_day = test[test["d"] == test["d"].min()]
    newest = str(first_day.sort_values(["days_since_release", "id"])["id"].iloc[0])
    return [
        (top, "most units sold in the last 28 training days"),
        (worst, "largest total absolute error over the 28 test days"),
        (newest, "most recently released item"),
    ]


def segments(test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row masks for the parts of the data where a forecaster is most likely to struggle."""
    zero = test["zero_frac_112"].to_numpy(dtype="float64", na_value=np.nan)
    released = test["days_since_release"].to_numpy()
    price_moved = test["price_change_1w"].to_numpy(dtype="float64", na_value=np.nan)
    return {
        "all rows": np.ones(len(test), dtype=bool),
        f"new item (< {NEW_ITEM_DAYS} days on sale)": released < NEW_ITEM_DAYS,
        "regular seller (< 25% zero days)": zero < 0.25,
        "intermittent (25-75% zero days)": (zero >= 0.25) & (zero < 0.75),
        "mostly zero (>= 75% zero days)": zero >= 0.75,
        "price changed vs last week": np.nan_to_num(price_moved) != 0,
    }


def segment_errors(
    test: pd.DataFrame, truth: pd.Series, forecast: np.ndarray, values: np.ndarray,
    names: list[str],
) -> list[dict[str, Any]]:  # fmt: skip
    actual = truth.to_numpy(dtype="float64")
    out = []
    for label, mask in segments(test).items():
        if not mask.any():
            continue
        a, f = actual[mask], forecast[mask]
        out.append(
            {
                "segment": label,
                "n_rows": int(mask.sum()),
                "share_of_rows": float(mask.mean()),
                "actual_units_per_row": float(a.mean()),
                "forecast_units_per_row": float(f.mean()),
                "bias": float(f.sum() / a.sum() - 1) if a.sum() > 0 else None,
                "mae": float(np.abs(f - a).mean()),
                "mae_over_mean_actual": float(np.abs(f - a).mean() / a.mean())
                if a.mean() > 0
                else None,
                "top_features": [r["feature"] for r in ranked(values[mask], names)[:3]],
            }
        )
    return out


def snap_feature_effect(test: pd.DataFrame, values: np.ndarray, names: list[str]) -> dict[str, Any]:
    """What the `snap` feature alone adds, in the model, on SNAP days in the test window."""
    j = names.index("snap")
    on = test["snap"].to_numpy() == 1
    food = (test["cat_id"].astype(str) == "FOODS").to_numpy()
    out: dict[str, Any] = {"n_snap_rows": int(on.sum())}
    for label, mask in (("FOODS", on & food), ("non-food", on & ~food)):
        mean = float(values[mask, j].mean()) if mask.any() else None
        out[label] = {
            "mean_snap_shap": mean,
            "implied_multiplier": None if mean is None else float(np.exp(mean)),
        }
    return out


def display_frame(test: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Feature values for colouring the beeswarm; categoricals have no order, so NaN (grey)."""
    return pd.DataFrame(
        {
            n: np.full(len(test), np.nan)
            if n in CATEGORICAL_FEATURES
            else test[n].to_numpy(dtype="float64", na_value=np.nan)
            for n in names
        }
    )


def beeswarm(values: np.ndarray, frame: pd.DataFrame, title: str, out: Path) -> Path:
    rng = np.random.default_rng(PLOT_SEED)
    rows = np.sort(rng.choice(len(frame), size=min(PLOT_SAMPLE_SIZE, len(frame)), replace=False))
    plt.figure()
    with warnings.catch_warnings():
        # Categorical columns are all NaN on purpose (drawn grey); shap still takes
        # their colour-scale quantiles and numpy warns about it.
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        shap.summary_plot(
            values[rows],
            frame.iloc[rows],
            max_display=PLOT_TOP_FEATURES,
            show=False,
            plot_size=(10, 9),
            rng=np.random.default_rng(PLOT_SEED),
        )
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (effect on log of expected daily units)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def importance_chart(summary: dict[str, Any], out: Path) -> Path:
    models = list(summary["models"])
    first = summary["models"][models[0]]["mean_importance"]
    top = [r["feature"] for r in first[:15]]
    y = np.arange(len(top))
    height = 0.8 / len(models)
    colors = {"lightgbm": "#2f6fb0", "xgboost": "#3d9a5b"}
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    for i, name in enumerate(models):
        by_name = {
            r["feature"]: r["mean_abs_shap"] for r in summary["models"][name]["mean_importance"]
        }
        ax.barh(
            y + (i - (len(models) - 1) / 2) * height,
            [by_name[f] for f in top],
            height=height,
            color=colors.get(name),
            label=name,
        )
    ax.set_yticks(y, top)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| over every test-window row of the 3 folds (log units)")
    ax.set_title(
        f"Store {summary['store_id']}: top 15 features by mean |SHAP| (ranked by {models[0]})"
    )
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


Beeswarm = tuple[np.ndarray, pd.DataFrame, str, str]


def explain_models(
    panel: pd.DataFrame, report: dict[str, Any]
) -> tuple[dict[str, Any], list[Beeswarm]]:
    """SHAP for every boosted model and fold; returns the summary and the plots to draw.

    Raises ValueError, before anything is written, if a rebuilt model does not
    reproduce its recorded WRMSSE.
    """
    boosted = [n for n in report["models"] if n in MODELS and issubclass(MODELS[n], BoostedModel)]
    if not boosted:
        raise ValueError("results/metrics.json has no boosted model; run m5-backtest first")
    last_day = int(panel.loc[panel[TARGET].notna(), "d"].max())
    horizon = int(report["horizon_days"])
    folds = make_folds(last_day, len(report["folds"]), horizon)
    recorded_folds = [(f["train_end_d"], f["test_end_d"]) for f in report["folds"]]
    if [(f.train_end, f.test_end) for f in folds] != recorded_folds:
        raise ValueError("folds differ from results/metrics.json")
    hierarchy = hierarchy_of(panel)
    ids = pd.Index(hierarchy["id"])
    names = feature_columns(horizon)
    store = report["store_id"]
    latest = folds[-1]

    summary: dict[str, Any] = {"models": {}}
    plots: list[Beeswarm] = []
    for name in boosted:
        fold_entries = []
        all_values = []
        for fold, fold_result in zip(folds, report["models"][name]["folds"], strict=True):
            model, test = rebuild(name, panel, fold, fold_result)
            truth = panel.loc[test.index, TARGET]
            forecast = model.predict(test)
            actual = to_matrix(test, truth.to_numpy(dtype="float64"), ids, fold)
            score = evaluator_for(panel, hierarchy, fold).score(
                actual, to_matrix(test, forecast, ids, fold)
            )
            if abs(score.wrmsse - fold_result["wrmsse"]) > REPRODUCE_TOLERANCE:
                raise ValueError(
                    f"{name} fold {fold.index}: rebuilt WRMSSE {score.wrmsse} differs from "
                    f"the recorded {fold_result['wrmsse']}"
                )
            values, base = model.shap_values(test)
            fold_entries.append(
                {
                    "fold": fold.index,
                    "test_days": [fold.test_start, fold.test_end],
                    "params": fold_result["tuning"]["chosen"],
                    "rounds": fold_result["tuning"]["chosen_rounds"],
                    "wrmsse_recorded": fold_result["wrmsse"],
                    "wrmsse_rebuilt": score.wrmsse,
                    "n_rows": len(test),
                    "base_value": float(base[0]),
                    "max_additivity_error": float(
                        np.abs(values.sum(axis=1) + base - np.log(forecast)).max()
                    ),
                    "importance": ranked(values, names),
                    "group_importance": grouped(values, names),
                }
            )
            all_values.append(values)
            print(f"{name} fold {fold.index}: SHAP on {len(test):,} rows", flush=True)
            if fold == latest:
                train = panel[(panel["d"] <= fold.train_end) & panel[TARGET].notna()]
                readings = [
                    series_reading(model, train, test, truth, forecast, values, base, sid, why)
                    for sid, why in pick_series(train, test, truth, forecast)
                ]
                latest_detail = {
                    "series": readings,
                    "segments": segment_errors(test, truth, forecast, values, names),
                    "snap_feature": snap_feature_effect(test, values, names),
                }
                title = (
                    f"{name}, store {store}, fold {fold.index} test window: SHAP for "
                    f"{min(PLOT_SAMPLE_SIZE, len(test)):,} of {len(test):,} rows "
                    f"(random, seed {PLOT_SEED})\n"
                    "Grey: categorical features, whose values have no order"
                )
                plots.append(
                    (values, display_frame(test, names), title, f"shap_summary_{name}_{store}.png")
                )
        pooled = np.concatenate(all_values)
        summary["models"][name] = {
            "mean_importance": ranked(pooled, names),
            "mean_group_importance": grouped(pooled, names),
            "folds": fold_entries,
            "latest_fold": {"fold": latest.index, **latest_detail},
        }
    return summary, plots


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:+.1f}%"


def render_markdown(shap_summary: dict[str, Any], snap: dict[str, Any]) -> str:
    s = shap_summary
    lines = [
        "# Explanations",
        "",
        f"Generated by `m5-explain` on {s['generated_at']} - do not edit by hand.",
        f"Store {s['store_id']}. SHAP: {s['method']['algorithm']}, on every row of each "
        "fold's 28-day test window (no sampling). Values are in log(expected units): a "
        "SHAP value of +0.1 multiplies the forecast by about 1.105.",
        "",
        "## Global importance (mean |SHAP| over all 3 folds)",
        "",
    ]
    models = list(s["models"])
    head = " | ".join(f"{m} feature | {m} mean abs SHAP | share" for m in models)
    lines += [f"| Rank | {head} |", "| --- |" + " --- | --- | --- |" * len(models)]
    for i in range(TOP_FEATURES):
        cells = " | ".join(
            f"{r['feature']} | {r['mean_abs_shap']:.4f} | {100 * r['share']:.1f}%"
            for r in (s["models"][m]["mean_importance"][i] for m in models)
        )
        lines.append(f"| {i + 1} | {cells} |")
    lines += ["", "By feature group (mean |sum of the group's SHAP values|):", ""]
    lines += [f"| Rank | {' | '.join(f'{m} group | share' for m in models)} |",
              "| --- |" + " --- | --- |" * len(models)]  # fmt: skip
    n_groups = len(s["models"][models[0]]["mean_group_importance"])
    for i in range(n_groups):
        cells = " | ".join(
            f"{r['group']} | {100 * r['share']:.1f}%"
            for r in (s["models"][m]["mean_group_importance"][i] for m in models)
        )
        lines.append(f"| {i + 1} | {cells} |")
    lines += ["", "Top 5 features per fold:", "", "| Model | Fold | Rebuilt WRMSSE | Top 5 |",
              "| --- | --- | --- | --- |"]  # fmt: skip
    for m in models:
        for f in s["models"][m]["folds"]:
            top = ", ".join(r["feature"] for r in f["importance"][:5])
            lines.append(f"| {m} | {f['fold']} | {f['wrmsse_rebuilt']:.4f} | {top} |")

    for m in models:
        latest = s["models"][m]["latest_fold"]
        lines += [
            "",
            f"## {m}, fold {latest['fold']}: three series",
            "",
            "| Series | Why | Units, last 28 training days | Actual units | Forecast units "
            "| Top features (mean SHAP) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in latest["series"]:
            top = ", ".join(f"{t['feature']} {t['mean_shap']:+.2f}" for t in r["top_features"])
            lines.append(
                f"| {r['id']} | {r['why_chosen']} | {r['units_last_28_training_days']:.0f} "
                f"| {r['actual_units']:.0f} "
                f"| {r['forecast_units']:.0f} | {top} |"
            )
        lines += [
            "",
            f"Error by segment ({m}, fold {latest['fold']}):",
            "",
            "| Segment | Rows | Actual/row | Forecast/row | Bias | MAE | MAE / mean | Top 3 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for g in latest["segments"]:
            rel = "-" if g["mae_over_mean_actual"] is None else f"{g['mae_over_mean_actual']:.2f}"
            lines.append(
                f"| {g['segment']} | {g['n_rows']:,} | {g['actual_units_per_row']:.2f} "
                f"| {g['forecast_units_per_row']:.2f} | {fmt_pct(g['bias'])} | {g['mae']:.2f} "
                f"| {rel} | {', '.join(g['top_features'])} |"
            )
        sf = latest["snap_feature"]
        food = sf["FOODS"]["implied_multiplier"]
        if food is not None:
            lines += [
                "",
                f"The `snap` feature alone multiplies FOODS forecasts on the "
                f"{sf['n_snap_rows']:,} SNAP-day rows of this window by {food:.3f} on "
                "average (other features, such as day of month, can carry SNAP too).",
            ]

    lines += [
        "",
        "## SNAP-day lift, from the raw sales (no model)",
        "",
        snap["method"],
        "",
        "| Scope | Category | Matched lift | 95% CI | Cells | Cells higher on SNAP days "
        "| Unmatched lift |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scope_name, scope in snap["scopes"].items():
        rows = [*scope["categories"].items(), ("FOODS vs NON_FOOD", scope["foods_vs_non_food"])]
        for cat, r in rows:
            lo, hi = r["lift_ci95"]
            lines.append(
                f"| {scope_name} | {cat} | {fmt_pct(r['lift'])} "
                f"| {fmt_pct(lo)} to {fmt_pct(hi)} | {r['n_cells']} "
                f"| {100 * r['share_of_cells_higher_on_snap_days']:.0f}% "
                f"| {fmt_pct(r.get('unmatched_lift'))} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args(argv)

    report = json.loads((args.results_dir / "metrics.json").read_text())
    store = report["store_id"]
    panel, manifest = backtest.load_panel(args.features_dir / f"features_{store}.parquet")
    try:
        backtest.check_provenance(manifest)
        hashes = check_hashes(args.raw_dir)
    except (ValueError, DataVerificationError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    if report["data"]["raw_sha256"] != manifest["raw_sha256"] or manifest["raw_sha256"] != hashes:
        print("STOP: metrics.json, the feature table and data/raw differ", file=sys.stderr)
        return 1

    now = datetime.now(UTC).isoformat(timespec="seconds")
    env = backtest.environment()
    env["packages"]["shap"] = metadata.version("shap")
    try:
        explained, plots = explain_models(panel, report)
    except ValueError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    shap_summary = {
        "generated_by": "m5-explain",
        "generated_at": now,
        "store_id": store,
        "source": {
            "metrics_json_generated_at": report["generated_at"],
            "metrics_json_git": report["environment"]["git"],
            "raw_sha256": hashes,
        },
        "method": {
            "algorithm": "exact TreeSHAP from the boosting library (LightGBM pred_contrib, "
            "XGBoost pred_contribs)",
            "output_space": "raw model output = log(expected daily units) (Tweedie, log link)",
            "rows": "every row of each fold's 28-day test window; no sampling",
            "models": "rebuilt per fold from the parameters and rounds in "
            "results/metrics.json, trained on the fold's training days, and required to "
            f"reproduce the recorded WRMSSE within {REPRODUCE_TOLERANCE}",
            "importance": "mean |SHAP| per feature; share = its fraction of the sum over "
            "features. Group importance sums a group's SHAP values per row first.",
            "plot_sample": {"size": PLOT_SAMPLE_SIZE, "seed": PLOT_SEED, "fold": "latest"},
        },
        "environment": env,
        **explained,
    }

    calendar = load_calendar(args.raw_dir)
    store_sales = load_store_sales(args.raw_dir, store)
    state = str(store_sales["state_id"].astype(str).iloc[0])
    snap = {
        "generated_by": "m5-explain",
        "generated_at": now,
        "method": snap_method(),
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "resampled_unit": "month"},
        "raw_sha256": hashes,
        "scopes": {
            f"store {store}": snap_lift(daily_category_units(store_sales, calendar), state),
            f"state {state} (all stores)": snap_lift(
                daily_category_units(load_state_sales(args.raw_dir, state), calendar), state
            ),
        },
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "shap_summary.json").write_text(json.dumps(shap_summary, indent=2) + "\n")
    (args.results_dir / "snap_lift.json").write_text(json.dumps(snap, indent=2) + "\n")
    (args.results_dir / "explanations.md").write_text(render_markdown(shap_summary, snap))
    for values, frame, title, filename in plots:
        beeswarm(values, frame, title, args.results_dir / filename)
    importance_chart(shap_summary, args.results_dir / f"shap_importance_{store}.png")
    print(f"Wrote shap_summary.json, snap_lift.json, explanations.md and plots to "
          f"{args.results_dir}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
