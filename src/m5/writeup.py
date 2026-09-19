"""The results write-up's numbers, regenerated from the committed artifacts.

Usage:
    m5-writeup            # after m5-backtest and m5-explain: rewrite the generated regions
    m5-writeup --check    # exit 1 if any generated region differs from its artifacts

README.md, docs/RESULTS.md and docs/RESUME_CLAIMS.md carry regions marked

    <!-- GENERATED:<name>:BEGIN -->
    ...
    <!-- GENERATED:<name>:END -->

Everything between the markers is written here from results/metrics.json,
results/shap_summary.json, results/snap_lift.json and the feature manifest, never by
hand. The prose around the regions carries no measured number, so a rerun of the
backtest cannot leave a stale figure behind; tests/test_writeup.py enforces both.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from m5 import config

DOCS = ("README.md", "docs/RESULTS.md", "docs/RESUME_CLAIMS.md")
BASELINE = "seasonal_naive"
BOOSTED = ("lightgbm", "xgboost")
FAMILIES = {
    "seasonal_naive": "naive baseline",
    "linear_regression": "linear",
    "ridge": "linear",
    "hurdle_logistic_ridge": "logistic hurdle (classifier x regressor)",
    "lightgbm": "gradient boosting",
    "xgboost": "gradient boosting",
}
# Feature groups quoted in the SHAP findings, in the order they are listed.
QUOTED_GROUPS = ("rolling means", "product identity", "price", "sales lags", "snap")
TOP_FEATURES = 5
PEAK_DAYS = 3
# Mean-WRMSSE gap below which two models are called a tie (3 folds cannot separate them).
TIE_GAP = 0.005

REGION = re.compile(
    r"<!-- GENERATED:(?P<name>[a-z_]+):BEGIN -->\n(?P<body>.*?)<!-- GENERATED:(?P=name):END -->",
    re.DOTALL,
)
MARKER = re.compile(r"<!-- GENERATED:([a-z_]+):(BEGIN|END) -->")


@dataclass(frozen=True)
class Artifacts:
    metrics: dict[str, Any]
    shap: dict[str, Any]
    snap: dict[str, Any]
    manifest: dict[str, Any]

    @classmethod
    def load(cls, results_dir: Path) -> Artifacts:
        def read(name: str) -> dict[str, Any]:
            data: dict[str, Any] = json.loads((results_dir / name).read_text())
            return data

        metrics = read("metrics.json")
        return cls(
            metrics=metrics,
            shap=read("shap_summary.json"),
            snap=read("snap_lift.json"),
            manifest=read(f"features_{metrics['store_id']}_manifest.json"),
        )


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def signed_pct(value: float) -> str:
    return f"{100 * value:+.1f}%"


def change(value: float, reference: float) -> float:
    """Relative change of `value` against `reference`: negative means lower."""
    return value / reference - 1


def versus(value: float, reference: float) -> str:
    c = change(value, reference)
    return f"{pct(-c)} lower" if c < 0 else f"{pct(c)} worse"


def ranked_models(a: Artifacts) -> list[tuple[str, dict[str, Any]]]:
    return sorted(a.metrics["models"].items(), key=lambda kv: kv[1]["mean_wrmsse"])


def best(a: Artifacts) -> tuple[str, dict[str, Any]]:
    return ranked_models(a)[0]


def fold_wins(a: Artifacts, model: str, other: str) -> int:
    ms = a.metrics["models"]
    return sum(
        x["wrmsse"] < y["wrmsse"]
        for x, y in zip(ms[model]["folds"], ms[other]["folds"], strict=True)
    )


def count_words(n: int, total: int) -> str:
    return "every" if n == total else f"{n} of the {total}"


# ---- regions ------------------------------------------------------------------------


def headline(a: Artifacts) -> str:
    ms = a.metrics["models"]
    name, top = best(a)
    base = ms[BASELINE]["mean_wrmsse"]
    runner, second = ranked_models(a)[1]
    n_folds = len(a.metrics["folds"])
    gap = second["mean_wrmsse"] - top["mean_wrmsse"]
    call = "too close to call either model the winner" if gap < TIE_GAP else f"so {name} is ahead"
    lr = ms["linear_regression"]["mean_wrmsse"]
    return "\n".join(
        [
            f"- **{name}** has the lowest mean WRMSSE, **{top['mean_wrmsse']:.3f}**, over "
            f"{n_folds} walk-forward folds on the {a.metrics['data']['n_series']:,} item "
            f"series of store {a.metrics['store_id']}: **{versus(top['mean_wrmsse'], base)} "
            f"than the seasonal-naive baseline** ({base:.3f}).",
            f"- **{runner}** is {gap:.4f} behind ({second['mean_wrmsse']:.3f}), {call}. "
            f"Ridge and the hurdle model score {ms['ridge']['mean_wrmsse']:.3f} and "
            f"{ms['hurdle_logistic_ridge']['mean_wrmsse']:.3f}; the plain linear regression "
            f"({lr:.3f}) is {'worse' if lr > base else 'better'} than the baseline.",
        ]
    )


def scope(a: Artifacts) -> str:
    m, man = a.metrics, a.manifest
    lines = [
        f"- **Store {m['store_id']}** (state {man['state_id']}), {m['data']['n_series']:,} "
        f"item series, daily sales from {man['first_date']} to {man['last_history_date']}.",
        f"- **{m['horizon_days']}-day horizon**, forecast directly from "
        f"{man['n_features']} features whose sales inputs are at least "
        f"{m['horizon_days']} days old.",
        f"- **{len(m['folds'])} walk-forward folds**, each training on every day up to its "
        "cutoff and forecasting the next "
        f"{m['horizon_days']} days; the last fold ends on the last day of sales:",
        "",
        "| Fold | Train through | Forecast | Series x days scored |",
        "| --- | --- | --- | --- |",
    ]
    rows = a.metrics["models"][BASELINE]["folds"]
    for f, r in zip(m["folds"], rows, strict=True):
        lines.append(
            f"| {f['fold']} | {f['train_end_date']} | {f['test_start_date']} to "
            f"{f['test_end_date']} | {r['n_test_rows']:,} |"
        )
    return "\n".join(lines)


def model_table(a: Artifacts) -> str:
    ms = a.metrics["models"]
    folds = a.metrics["folds"]
    base = ms[BASELINE]["mean_wrmsse"]
    header = " | ".join(f"Fold {f['fold']}" for f in folds)
    lines = [
        f"| Model | Family | {header} | Mean WRMSSE | vs seasonal-naive | Mean MAE (units) |",
        "| --- | --- |" + " --- |" * (len(folds) + 3),
    ]
    for name, m in ms.items():
        cells = " | ".join(f"{r['wrmsse']:.4f}" for r in m["folds"])
        vs = "-" if name == BASELINE else versus(m["mean_wrmsse"], base)
        mean = f"**{m['mean_wrmsse']:.4f}**" if name == best(a)[0] else f"{m['mean_wrmsse']:.4f}"
        lines.append(
            f"| {name} | {FAMILIES.get(name, '-')} | {cells} | {mean} | {vs} "
            f"| {m['mean_mae']:.4f} |"
        )
    return "\n".join(lines)


def model_findings(a: Artifacts) -> str:
    ms = a.metrics["models"]
    n = len(a.metrics["folds"])
    base = ms[BASELINE]["mean_wrmsse"]
    lr = ms["linear_regression"]["mean_wrmsse"]
    ridge = ms["ridge"]["mean_wrmsse"]
    hurdle = ms["hurdle_logistic_ridge"]["mean_wrmsse"]
    boosted_worst = max(BOOSTED, key=lambda b: ms[b]["mean_wrmsse"])
    lines = [
        f"- The simple linear regression on a few sales-history features is "
        f"{versus(lr, base)} than seasonal-naive ({lr:.4f} vs {base:.4f}) and beats it in "
        f"{fold_wins(a, 'linear_regression', BASELINE)} of {n} folds. Ridge on the full "
        f"feature set is {versus(ridge, base)}, so the gain comes from the full feature set, "
        "not from regression as such."
        if lr > base
        else f"- The simple linear regression on a few sales-history features is "
        f"{versus(lr, base)} than seasonal-naive ({lr:.4f} vs {base:.4f}).",
        f'- Adding a LogisticRegression "does it sell today" stage (the hurdle model) takes '
        f"Ridge from {ridge:.4f} to {hurdle:.4f}, better than Ridge in "
        f"{count_words(fold_wins(a, 'hurdle_logistic_ridge', 'ridge'), n)} fold.",
        f"- Both boosted models beat Ridge and the hurdle model in "
        f"{count_words(min(fold_wins(a, b, o) for b in BOOSTED for o in ('ridge', 'hurdle_logistic_ridge')), n)}"  # noqa: E501
        f" fold. Even the weaker one, {boosted_worst}, is "
        f"{versus(ms[boosted_worst]['mean_wrmsse'], ridge)} than Ridge and "
        f"{versus(ms[boosted_worst]['mean_wrmsse'], hurdle)} than the hurdle model.",
    ]
    winners = [min(BOOSTED, key=lambda b: ms[b]["folds"][i]["wrmsse"]) for i in range(n)]
    per_fold = ", ".join(
        f"{w} in fold {f['fold']}" for w, f in zip(winners, a.metrics["folds"], strict=True)
    )
    verdict = (
        f"each model wins at least one fold, so {n} folds cannot separate them"
        if len(set(winners)) > 1
        else f"{winners[0]} wins every fold"
    )
    lines.append(f"- Between the two boosted models the fold winners are {per_fold}: {verdict}.")
    return "\n".join(lines)


def shap_top(a: Artifacts) -> str:
    models = list(a.shap["models"])
    head = " | ".join(f"{m} | share" for m in models)
    lines = [f"| Rank | {head} |", "| --- |" + " --- | --- |" * len(models)]
    for i in range(TOP_FEATURES):
        cells = " | ".join(
            f"{r['feature']} | {pct(r['share'])}"
            for r in (a.shap["models"][m]["mean_importance"][i] for m in models)
        )
        lines.append(f"| {i + 1} | {cells} |")
    lines += ["", "Share of total mean |SHAP| by feature group:", ""]
    lines += [f"| Group | {' | '.join(models)} |", "| --- |" + " --- |" * len(models)]
    for g in QUOTED_GROUPS:
        cells = " | ".join(
            pct(next(r["share"] for r in a.shap["models"][m]["mean_group_importance"]
                     if r["group"] == g))
            for m in models
        )  # fmt: skip
        lines.append(f"| {g} | {cells} |")
    return "\n".join(lines)


def shap_findings(a: Artifacts) -> str:
    models = list(a.shap["models"])

    def rank_of(m: str, feature: str) -> str:
        r = next(r for r in a.shap["models"][m]["mean_importance"] if r["feature"] == feature)
        return f"{ordinal(r['rank'])} in {m} ({pct(r['share'])})"

    stable = set.intersection(
        *(
            {r["feature"] for r in f["importance"][:3]}
            for m in models
            for f in a.shap["models"][m]["folds"]
        )
    )
    n_folds = len(a.shap["models"][models[0]]["folds"])
    snap = " and ".join(
        f"x{a.shap['models'][m]['latest_fold']['snap_feature']['FOODS']['implied_multiplier']:.3f}"
        for m in models
    )
    return "\n".join(
        [
            f"- The same three features are the top 3 in all {n_folds} folds of both models: "
            f"{', '.join(sorted(stable))}. Only their order changes.",
            f"- The 28-day lag ranks {' and '.join(rank_of(m, 'lag_28') for m in models)}; "
            f"sell price ranks {' and '.join(rank_of(m, 'sell_price') for m in models)}.",
            f"- On food rows of SNAP days the `snap` flag itself multiplies the forecast by "
            f"only {snap} ({models[0]} and {models[1]}, fold "
            f"{a.shap['models'][models[0]]['latest_fold']['fold']}). Day of month and recent "
            "sales can carry the same start-of-month signal, so this is not the size of the "
            "SNAP effect; the data-only measurement below is.",
        ]
    )


def weak_segments(a: Artifacts) -> str:
    models = list(a.shap["models"])
    fold = a.shap["models"][models[0]]["latest_fold"]["fold"]
    head = " | ".join(f"{m} bias | {m} MAE / mean" for m in models)
    lines = [
        f"Fold {fold}, every forecast row. Bias is total forecast over total actual minus 1; "
        "MAE / mean is the mean absolute error relative to mean actual sales.",
        "",
        f"| Segment | Rows | Actual units/row | {head} |",
        "| --- | --- | --- |" + " --- | --- |" * len(models),
    ]
    segs = [a.shap["models"][m]["latest_fold"]["segments"] for m in models]
    for rows in zip(*segs, strict=True):
        g = rows[0]
        cells = " | ".join(
            ("-" if r["bias"] is None else signed_pct(r["bias"]))
            + " | "
            + ("-" if r["mae_over_mean_actual"] is None else f"{r['mae_over_mean_actual']:.2f}")
            for r in rows
        )
        lines.append(
            f"| {g['segment']} | {g['n_rows']:,} | {g['actual_units_per_row']:.2f} | {cells} |"
        )
    return "\n".join(lines)


def segment(a: Artifacts, model: str, prefix: str) -> dict[str, Any]:
    segs = a.shap["models"][model]["latest_fold"]["segments"]
    found: dict[str, Any] = next(g for g in segs if g["segment"].startswith(prefix))
    return found


def weak_findings(a: Artifacts) -> str:
    models = list(a.shap["models"])

    def both(prefix: str, key: str, fmt: Callable[[float], str]) -> str:
        return " and ".join(f"{fmt(segment(a, m, prefix)[key])} ({m})" for m in models)

    zero = segment(a, models[0], "mostly zero")
    regular = segment(a, models[0], "regular seller")
    new = segment(a, models[0], "new item")
    ratio = min(segment(a, m, "mostly zero")["mae_over_mean_actual"] for m in models)
    size = "larger than" if ratio > 1 else "a large fraction of"
    identity_rank = max(
        next(r["rank"] for r in a.shap["models"][m]["mean_importance"] if r["feature"] == "item_id")
        for m in models
    )
    return "\n".join(
        [
            f"- **Mostly-zero items are forecast too low.** They are "
            f"{pct(zero['share_of_rows'])} of rows; their bias is "
            f"{both('mostly zero', 'bias', signed_pct)}, and their mean absolute error is "
            f"{size} their mean sales. Regular sellers are close to unbiased: "
            f"{both('regular seller', 'bias', signed_pct)} on "
            f"{pct(regular['share_of_rows'])} of rows.",
            f"- **New items are a blind spot.** Item identity ranks no lower than "
            f"{ordinal(identity_rank)} in either model and carries nothing for an item the "
            f"model has never seen. Only {new['n_rows']:,} rows in fold "
            f"{a.shap['models'][models[0]]['latest_fold']['fold']} are new items, too few to "
            f"measure this well; the bias there is {both('new item', 'bias', signed_pct)}.",
        ]
    )


def jump_example(a: Artifacts) -> str:
    models = list(a.shap["models"])
    series = [
        next(s for s in a.shap["models"][m]["latest_fold"]["series"]
             if s["why_chosen"].startswith("largest"))
        for m in models
    ]  # fmt: skip
    s = series[0]
    forecasts = " and ".join(
        f"{r['forecast_units']:.0f} ({m})" for m, r in zip(models, series, strict=True)
    )
    return (
        f"- **Sudden demand jumps are missed.** The largest miss, {s['id'].split('_CA')[0]}, "
        f"sold {s['units_last_28_training_days']:.0f} units in the last "
        f"{a.metrics['horizon_days']} training days and {s['actual_units']:.0f} in the test "
        f"window; the forecasts were {forecasts}, because every sales input is at least "
        f"{a.metrics['horizon_days']} days old."
    )


def peak_misses(a: Artifacts) -> str:
    name, m = best(a)
    days: list[tuple[date, float, float]] = []
    for f, r in zip(a.metrics["folds"], m["folds"], strict=True):
        start = date.fromisoformat(f["test_start_date"])
        units = r["store_daily_units"]
        for i, (act, fc) in enumerate(zip(units["actual"], units["forecast"], strict=True)):
            days.append((start + timedelta(days=i), act, fc))
    worst = sorted(days, key=lambda d: d[2] - d[1])[:PEAK_DAYS]
    under = sum(fc < act for _, act, fc in days)
    listed = "; ".join(
        f"{d:%a} {d.day} {d:%b %Y}: {fc:,.0f} forecast vs {act:,.0f} sold"
        for d, act, fc in sorted(worst)
    )
    return (
        f"- **Peaks are under-forecast.** Summed over the whole store, {name} forecasts "
        f"below actual sales on {under} of {len(days)} test days. Its largest shortfalls: "
        f"{listed}."
    )


def snap_table(a: Artifacts) -> str:
    lines = [
        "| Scope | Food lift | 95% CI | Non-food lift | Food beyond non-food | 95% CI |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for scope_name, s in a.snap["scopes"].items():
        food, non, diff = (
            s["categories"]["FOODS"],
            s["categories"]["NON_FOOD"],
            s["foods_vs_non_food"],
        )
        lines.append(
            f"| {scope_name} | {signed_pct(food['lift'])} "
            f"| {signed_pct(food['lift_ci95'][0])} to {signed_pct(food['lift_ci95'][1])} "
            f"| {signed_pct(non['lift'])} | {signed_pct(diff['lift'])} "
            f"| {signed_pct(diff['lift_ci95'][0])} to {signed_pct(diff['lift_ci95'][1])} |"
        )
    first = next(iter(a.snap["scopes"].values()))
    days = first["snap_days_of_month"]
    lines += [
        "",
        f"SNAP days in California are the {ordinal(days[0])} to the {ordinal(days[-1])} of "
        "each month. "
        f"Compared within {first['categories']['FOODS']['n_cells']} same-month, same-weekday "
        f"cells from {first['first_date']} to {first['last_date']}; the intervals resample "
        f"whole months ({a.snap['bootstrap']['reps']:,} draws).",
    ]
    return "\n".join(lines)


REGIONS: dict[str, Callable[[Artifacts], str]] = {
    "headline": headline,
    "scope": scope,
    "model_table": model_table,
    "model_findings": model_findings,
    "shap_top": shap_top,
    "shap_findings": shap_findings,
    "weak_segments": weak_segments,
    "weak_findings": weak_findings,
    "jump_example": jump_example,
    "peak_misses": peak_misses,
    "snap_table": snap_table,
}


# ---- region plumbing ----------------------------------------------------------------


def check_markers(text: str) -> None:
    """Refuse unbalanced, nested or unknown markers instead of silently skipping them."""
    open_name: str | None = None
    for match in MARKER.finditer(text):
        name, kind = match.groups()
        if name not in REGIONS:
            raise ValueError(f"unknown generated region {name!r}")
        if kind == "BEGIN":
            if open_name is not None:
                raise ValueError(f"region {name!r} begins inside region {open_name!r}")
            open_name = name
        else:
            if open_name != name:
                raise ValueError(f"region {name!r} ends without a matching begin")
            open_name = None
    if open_name is not None:
        raise ValueError(f"region {open_name!r} has no end marker")


def render(text: str, artifacts: Artifacts) -> str:
    """`text` with every generated region rewritten from `artifacts`."""
    check_markers(text)

    def fill(match: re.Match[str]) -> str:
        name = match["name"]
        body = REGIONS[name](artifacts)
        return f"<!-- GENERATED:{name}:BEGIN -->\n\n{body}\n\n<!-- GENERATED:{name}:END -->"

    return REGION.sub(fill, text)


def strip_regions(text: str) -> str:
    """`text` with the generated regions removed: the hand-written prose only."""
    return REGION.sub("", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--repo-root", type=Path, default=config.REPO_ROOT)
    parser.add_argument(
        "--check", action="store_true", help="only report drift; exit 1 if any region differs"
    )
    args = parser.parse_args(argv)
    artifacts = Artifacts.load(args.results_dir)
    stale = []
    for doc in DOCS:
        path = args.repo_root / doc
        text = path.read_text()
        new = render(text, artifacts)
        if new == text:
            continue
        stale.append(doc)
        if not args.check:
            path.write_text(new)
    if args.check:
        for doc in stale:
            print(f"{doc}: generated regions differ from the artifacts; run m5-writeup")
        return 1 if stale else 0
    print(f"Rewrote {', '.join(stale)}" if stale else "Every generated region is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
