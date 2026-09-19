# Product brief for the loop

**Read this first.** Every agent in this repo's improvement loop - Scout, Builder,
Auditor, Retro, Redraft - reads this file before it does anything. It is the only place
that says what this project *is*, what the owner is currently trying to achieve, and what
must be left alone. Without it, agents fall back to generic engineering hygiene.

## Keeping this current - instructions for agents

- **Read before you propose.** Ideas that contradict "Off-limits areas" or ignore
  "Current goals" should not be filed.
- **Keep it true.** If you learn something here is stale or wrong - a goal that has clearly
  been met, a description that no longer matches the code - propose an update to this file
  in the same pull request as the work that revealed it. Say plainly what changed and why.
- **Keep it short.** Aim for about 100 lines. It is loaded into every agent's context on
  every run.
- **Do not turn it into a changelog.** Metrics go in `metrics/loop-metrics.json`. This file
  describes the present, not the history.
- **Never delete a section.** If a section does not apply yet, write "Not decided yet"
  under it so the gap is visible instead of silent.

**This file vs the `scout` block in `.github/loop-config.json`:** both hold the owner's
intent. The `scout` block is the structured knob set the Scout's gate injects straight
into its prompt, edited from the loop dashboard; this file is the long-form context every
agent reads. If they conflict, the `scout` block wins for the Scout and this file governs
every other agent. A conflict is a bug: propose the fix to this file in your next PR.

---

## What this project is

A 28-day demand forecast for every product in one Walmart store, built on the real
Kaggle M5 data, with an evaluation harness designed to be honest: verified inputs, no
look-ahead, walk-forward validation and the competition's own metric (WRMSSE). It is a
portfolio piece, not a leaderboard chase. `docs/PLAN.md` is the plan, `docs/RESULTS.md`
the write-up.

**Already exists:**
- **Data** (`src/m5/download.py`, `verify.py`, `data.py`): `uv run m5-download` fetches the
  four M5 files from a pinned HuggingFace mirror and refuses any file whose sha256 differs
  from `provenance/m5_data.json`, then checks the documented M5 facts. Raw data lives in the
  gitignored `data/`. One store at a time, default `CA_1` (`--store`).
- **Features** (`src/m5/features.py`, `uv run m5-features`): 52 inputs. Sales features see
  only days at least 28 days before the forecast day; prices and calendar only up to it.
  `tests/test_leakage.py` proves it by scrambling everything after a cutoff.
- **Models and harness** (`src/m5/models.py`, `evaluation.py`, `backtest.py`):
  seasonal-naive, linear regression, Ridge, a logistic hurdle model, LightGBM and XGBoost
  (Tweedie, tuned inside each fold's training days). `uv run m5-backtest` scores all of
  them on 3 walk-forward folds and writes `results/metrics.json` and `metrics.md`.
  New models plug into `MODELS` and are covered by the leakage test in `tests/test_backtest.py`.
- **Explanations** (`src/m5/explain.py`, `snap.py`): `uv run m5-plots` and `uv run m5-explain`
  write the forecast plot, exact TreeSHAP for both boosted models and a model-free SNAP-day lift.
- **Write-up** (`src/m5/writeup.py`): `uv run m5-writeup` rewrites every
  `<!-- GENERATED:... -->` region in `README.md`, `docs/RESULTS.md` and
  `docs/RESUME_CLAIMS.md` from `results/`; `tests/test_writeup.py` fails on drift or on a
  measured number typed into the prose outside a region.
- **Gates** (`.github/workflows/ci.yml`): `uv run ruff check . && uv run ruff format --check .
  && uv run mypy && uv run pytest`, on the synthetic fixture in `tests/fixtures/` only.

**Not built, by design:** a web app, a live data feed, any deployment.

## Current goals

The weaknesses below are the ones `docs/RESULTS.md` already measures ("Where the models
are weak", "What is not shown", "What would come next"). Work that fixes one of them wins.

1. **Mostly-zero items are forecast too low.** Items that sell on fewer than a quarter of
   days are about 31% of forecast rows, and both boosted models under-forecast them by about
   24%. Try a model built for them (Croston-type, or a boosted hurdle model) and compare
   on that segment and on overall WRMSSE.
2. **Sudden demand jumps are missed**, because every sales input is at least 28 days old.
   Try faster-reacting forecasts (for example one model per forecast day with lags as
   recent as one day), extending the leakage tests to prove the new, tighter rule.
3. **One store only.** Everything is measured on `CA_1`; extend the evaluation to more
   stores in a way the owner can run and review.
4. **The other named gaps:** prediction intervals, new items forecast from similar items,
   more folds so scores get an error bar.
5. **Checks that cannot fail:** any test or gate that would stay green if the thing it
   guards broke.

**Results that move need a local rerun.** The loop's GitHub runners have 16 GB of memory
and `m5-backtest` peaks at about 18 GB, so the loop cannot regenerate `results/`. A PR
that would change a published number ships the code and tests, leaves every artifact and
generated region untouched, starts its title with `[needs-regen]` and says what it expects
to change. The owner's crew then runs `m5-backtest`, `m5-plots`, `m5-explain` and
`m5-writeup` locally before merge.

## Off-limits areas

- **Loosening a gate to go green:** never skip, deselect, xfail or loosen a test or gate
  (ruff, mypy, pytest, the leakage and write-up tests). Fix the cause.
- **Hand-edited numbers:** never edit a `GENERATED` region or anything in `results/` by
  hand. Rerun the command that writes it.
- **Fabricated data or reference values:** never invent, substitute or "fix up" data,
  provenance hashes, the M5 facts in `src/m5/config.py`, or a published number. The
  synthetic fixture never produces a published number. If data is missing, say so.
- **Tuning toward a target:** the numbers in `docs/PLAN.md` "Honesty rules" and in
  `docs/RESUME_CLAIMS.md` are targets or claims, not goals to optimise toward. Report what
  is measured, even when it misses.
- **Leaky features:** nothing may use sales newer than 28 days before the forecast day, or
  prices and calendar fields after it, unless the leakage tests are extended to prove it.
- **Scope:** no web app, live feed or deployment (`docs/PLAN.md` non-goals).
- **Loop machinery:** the loop cannot edit `.github/workflows/` (its token has no
  `workflow` scope, and a push touching it is rejected). A proposal that needs a workflow
  change is written up and left for the owner's crew. Also leave `.github/loop-config.json`,
  this file's path, `scripts/loop-metrics.mjs` and `metrics/` alone.
- **Credentials, money, history:** no secrets in issues (public repo), no key rotation, no
  force-push or history rewrite. Leave `AGENTS.md`, `CLAUDE.md` and `.claude/` alone.

## How the loop works

- **The loop drafts ideas as proposals.** The Scout files ideas as proposals; nothing is
  built from them on its own.
- **The Builder only builds proposals the owner has labelled approved.** Autonomous
  building is off (`autonomousBuildEnabled: false` in `.github/loop-config.json`).
- **Only the owner merges loop PRs.** No agent merges or auto-merges a loop PR.

## How the owner works

- An Industrial Engineering student directing (not coding) a portfolio piece aimed at
  data-science, forecasting and supply-chain roles.
- **Proposals:** one outcome each, plain English, a title that states the consequence,
  judgeable in one read on a phone.
- **Evidence:** `path:line`, and re-derive each number from its artifact in `results/`,
  never from another doc. Show a check going red before claiming it guards something.
- **Owner's call:** credentials, money, anything that changes a published result. Give 2-3
  options, a pick, and what the owner must do (ideally nothing).
