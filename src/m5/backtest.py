"""Walk-forward backtest of every model on the one-store feature table.

Usage:
    m5-backtest                         # all models, default store, 3 folds
    m5-backtest --models seasonal_naive ridge

Reads data/processed/features_<store>.parquet (built by m5-features from verified data),
runs each model on each fold, logs every run to MLflow (local store in mlruns/,
gitignored) and writes results/metrics.json plus a readable results/metrics.md. Both
files are generated here and never edited by hand.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from m5 import config
from m5.download import expected_hashes, load_provenance
from m5.evaluation import (
    N_FOLDS,
    WEIGHT_DAYS,
    Fold,
    WRMSSEEvaluator,
    evaluator_for,
    hierarchy_of,
    make_folds,
    to_matrix,
)
from m5.features import TARGET
from m5.models import MODELS, SEED, BoostedModel, HurdleLogisticRidge, Model

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "mlflow", "pyarrow", "lightgbm", "xgboost"]


def load_panel(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = pq.read_table(path)
    manifest = json.loads(table.schema.metadata[b"m5_manifest"])
    return table.to_pandas(), manifest


def check_provenance(manifest: dict[str, Any]) -> None:
    """The feature table must come from the raw files recorded in provenance."""
    expected = expected_hashes(load_provenance())
    if not expected or manifest.get("raw_sha256") != expected:
        raise ValueError("feature table was not built from the verified raw files")


def split(panel: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows up to the cutoff; test rows for the horizon without their target."""
    train = panel[(panel["d"] <= fold.train_end) & panel[TARGET].notna()]
    in_window = (panel["d"] >= fold.test_start) & (panel["d"] <= fold.test_end)
    test = panel[in_window].drop(columns=[TARGET])
    return train, test


def classification_summary(
    y_sold: np.ndarray, p_sold: np.ndarray, base_rate: float
) -> dict[str, float]:
    """How well P(sales > 0) is predicted, against always predicting the training rate."""
    constant = np.full_like(p_sold, base_rate)
    brier = float(brier_score_loss(y_sold, p_sold))
    brier_ref = float(brier_score_loss(y_sold, constant))
    return {
        "auc": float(roc_auc_score(y_sold, p_sold)),
        "log_loss": float(log_loss(y_sold, p_sold, labels=[False, True])),
        "log_loss_base_rate": float(log_loss(y_sold, constant, labels=[False, True])),
        "brier": brier,
        "brier_base_rate": brier_ref,
        "brier_skill": 1.0 - brier / brier_ref,
        "share_sold": float(y_sold.mean()),
    }


def run_fold(
    model: Model, panel: pd.DataFrame, hierarchy: pd.DataFrame, fold: Fold,
    evaluator: WRMSSEEvaluator,
) -> dict[str, Any]:  # fmt: skip
    ids = pd.Index(hierarchy["id"])
    train, test = split(panel, fold)
    start = time.perf_counter()
    model.fit(train)
    prediction = model.predict(test)
    runtime = time.perf_counter() - start

    truth_rows = panel.loc[test.index]
    actual = to_matrix(truth_rows, truth_rows[TARGET].to_numpy(dtype="float64"), ids, fold)
    forecast = to_matrix(test, prediction, ids, fold)
    result: dict[str, Any] = {
        "fold": fold.index,
        **evaluator.score(actual, forecast).summary(),
        "runtime_seconds": round(runtime, 3),
        "n_train_rows": len(train),
        "n_test_rows": len(test),
    }
    if isinstance(model, HurdleLogisticRidge):
        sold = truth_rows[TARGET].to_numpy() > 0
        base_rate = float((train[TARGET] > 0).mean())
        result["classifier"] = classification_summary(
            sold, model.predict_proba_sold(test), base_rate
        )
        result["classifier"]["n_iter"] = int(model.classifier_.n_iter_[0])
    if isinstance(model, BoostedModel):
        result["tuning"] = model.tuning_
    return result


def _git_commit() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=config.REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "git": _git_commit(),
    }


def fold_record(fold: Fold, dates: pd.Series) -> dict[str, Any]:
    return {
        "fold": fold.index,
        "train_start_d": 1,
        "train_end_d": fold.train_end,
        "train_end_date": str(dates[fold.train_end].date()),
        "test_start_d": fold.test_start,
        "test_start_date": str(dates[fold.test_start].date()),
        "test_end_d": fold.test_end,
        "test_end_date": str(dates[fold.test_end].date()),
    }


def _mlflow_metrics(result: dict[str, Any]) -> dict[str, float]:
    metrics = {k: float(v) for k, v in result.items() if isinstance(v, int | float)}
    metrics.pop("fold", None)
    metrics |= {f"wrmsse_{k}": v for k, v in result["wrmsse_by_level"].items()}
    metrics |= {f"clf_{k}": float(v) for k, v in result.get("classifier", {}).items()}
    return metrics


def log_tuning(name: str, fold: Fold, tuning: dict[str, Any]) -> None:
    """Chosen parameters on the fold's run, and one nested run per search candidate."""
    mlflow.log_params({f"chosen_{k}": v for k, v in tuning["chosen"].items()})
    mlflow.log_params({"chosen_rounds": tuning["chosen_rounds"]})
    mlflow.log_dict(tuning, "tuning.json")
    for i, candidate in enumerate(tuning["candidates"]):
        with mlflow.start_run(run_name=f"{name}-fold{fold.index}-search{i}", nested=True):
            mlflow.log_params({"model": name, "fold": fold.index, **candidate["params"]})
            mlflow.log_params({"rounds": candidate["rounds"]})
            mlflow.log_metric("inner_wrmsse", candidate["inner_wrmsse"])


def _params_text(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def render_markdown(report: dict[str, Any]) -> str:
    folds = report["folds"]
    lines = [
        "# Backtest results",
        "",
        f"Generated by `m5-backtest` on {report['generated_at']} - do not edit by hand.",
        f"Store {report['store_id']}, {report['horizon_days']}-day horizon, "
        f"{len(folds)} walk-forward folds. Metric: WRMSSE over the 12 M5 levels "
        "(lower is better).",
        "",
        "| Fold | Train through | Forecast |",
        "| --- | --- | --- |",
    ]
    for f in folds:
        lines.append(
            f"| {f['fold']} | {f['train_end_date']} (d_{f['train_end_d']}) "
            f"| {f['test_start_date']} to {f['test_end_date']} |"
        )
    header = " | ".join(f"Fold {f['fold']}" for f in folds)
    lines += [
        "",
        f"| Model | {header} | Mean WRMSSE | Mean MAE | Fit+predict (s) |",
        "| --- |" + " --- |" * (len(folds) + 3),
    ]
    for name, m in report["models"].items():
        cells = " | ".join(f"{r['wrmsse']:.4f}" for r in m["folds"])
        runtime = sum(r["runtime_seconds"] for r in m["folds"])
        lines.append(
            f"| {name} | {cells} | {m['mean_wrmsse']:.4f} | {m['mean_mae']:.4f} | {runtime:.1f} |"
        )
    clf = {n: m for n, m in report["models"].items() if "classifier" in m["folds"][0]}
    if clf:
        lines += [
            "",
            "Zero vs non-zero sales classifier (LogisticRegression inside the hurdle model):",
            "",
            "| Model | Fold | AUC | Brier | Brier (base rate) | Brier skill | Share sold |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, m in clf.items():
            for r in m["folds"]:
                c = r["classifier"]
                lines.append(
                    f"| {name} | {r['fold']} | {c['auc']:.4f} | {c['brier']:.4f} "
                    f"| {c['brier_base_rate']:.4f} | {c['brier_skill']:.4f} "
                    f"| {c['share_sold']:.3f} |"
                )
    tuned = {n: m for n, m in report["models"].items() if "search_space" in m}
    if tuned:
        lines += [
            "",
            "Hyperparameter search (gradient boosting). Each fold searches the grid on its own "
            "training days only: train through the cutoff minus 28 days, score WRMSSE on the "
            "last 28 training days, boosting rounds by early stopping there, then refit the "
            "best candidate on every training day. The fold's test window is never seen.",
            "",
            "| Model | Fixed | Search grid |",
            "| --- | --- | --- |",
        ]
        for name, m in tuned.items():
            grid = "; ".join(f"{k} in {v}" for k, v in m["search_space"].items())
            lines.append(f"| {name} | {_params_text(m['fixed_params'])} | {grid} |")
        lines += [
            "",
            "| Model | Fold | Chosen | Rounds | Inner WRMSSE (chosen) | Inner WRMSSE (worst) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, m in tuned.items():
            for r in m["folds"]:
                t = r["tuning"]
                worst = max(c["inner_wrmsse"] for c in t["candidates"])
                lines.append(
                    f"| {name} | {r['fold']} | {_params_text(t['chosen'])} "
                    f"| {t['chosen_rounds']} | {t['chosen_inner_wrmsse']:.4f} | {worst:.4f} |"
                )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=config.DEFAULT_STORE, choices=config.STORES)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--features-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument(
        "--tracking-uri", default=f"sqlite:///{config.REPO_ROOT / 'mlruns' / 'mlflow.db'}"
    )
    args = parser.parse_args(argv)

    panel, manifest = load_panel(args.features_dir / f"features_{args.store}.parquet")
    try:
        check_provenance(manifest)
    except ValueError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    horizon = int(manifest["horizon_days"])
    last_day = int(panel.loc[panel[TARGET].notna(), "d"].max())
    folds = make_folds(last_day, args.n_folds, horizon)
    dates = panel.drop_duplicates("d").set_index("d")["date"]
    hierarchy = hierarchy_of(panel)
    evaluators = {f.index: evaluator_for(panel, hierarchy, f) for f in folds}

    if args.tracking_uri.startswith("sqlite:///"):
        Path(args.tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(f"m5-backtest-{args.store}")
    started = time.perf_counter()
    report: dict[str, Any] = {
        "generated_by": "m5-backtest",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "store_id": args.store,
        "horizon_days": horizon,
        "metric": "WRMSSE over the 12 M5 levels; weights = dollar sales over the last "
        f"{WEIGHT_DAYS} training days; scale = one-step naive MSE on the training period",
        "seed": SEED,
        "environment": environment(),
        "data": {"raw_sha256": manifest["raw_sha256"], "n_series": len(hierarchy)},
        "folds": [fold_record(f, dates) for f in folds],
        "models": {},
    }
    with mlflow.start_run(run_name=f"backtest-{args.store}"):
        mlflow.log_params({"store": args.store, "horizon": horizon, "seed": SEED})
        for name in args.models:
            results = []
            for fold in folds:
                model = MODELS[name]()
                with mlflow.start_run(run_name=f"{name}-fold{fold.index}", nested=True):
                    mlflow.log_params(
                        {
                            "model": name,
                            **fold_record(fold, dates),
                            "store": args.store,
                            "seed": SEED,
                        }
                    )
                    result = run_fold(model, panel, hierarchy, fold, evaluators[fold.index])
                    mlflow.log_metrics(_mlflow_metrics(result))
                    if "tuning" in result:
                        log_tuning(name, fold, result["tuning"])
                results.append(result)
                print(f"{name} fold {fold.index}: WRMSSE {result['wrmsse']:.4f} "
                      f"({result['runtime_seconds']:.1f}s)", flush=True)  # fmt: skip
            entry: dict[str, Any] = {"description": MODELS[name].description}
            model_class = MODELS[name]
            if isinstance(model_class, type) and issubclass(model_class, BoostedModel):
                entry["fixed_params"] = model_class.fixed_params
                entry["search_space"] = model_class.search_space
            report["models"][name] = {
                **entry,
                "folds": results,
                "mean_wrmsse": float(np.mean([r["wrmsse"] for r in results])),
                "mean_mae": float(np.mean([r["mae"] for r in results])),
            }
        report["total_runtime_seconds"] = round(time.perf_counter() - started, 1)
        mlflow.log_metric("total_runtime_seconds", report["total_runtime_seconds"])

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.results_dir / "metrics.md").write_text(render_markdown(report))
    print(f"Wrote {args.results_dir / 'metrics.json'} and {args.results_dir / 'metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
