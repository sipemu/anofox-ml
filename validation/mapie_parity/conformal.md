# Conformal prediction — MAPIE parity and coverage

Issue: [#93](https://github.com/sipemu/anofox-ml/issues/93) ·
PRs: [#95](https://github.com/sipemu/anofox-ml/pull/95) (split, CV+, jackknife+),
[#96](https://github.com/sipemu/anofox-ml/pull/96) (CQR, normalised scores)

## What

`anofox-ml-conformal`: prediction intervals with finite-sample marginal
coverage for any `Fit` + `Predict` regressor.

| Rust | MAPIE 1.6.0 counterpart |
|---|---|
| `SplitConformalRegressor`, `conformalize` | `SplitConformalRegressor` (absolute score), `prefit=True` |
| `CrossConformalRegressor` (`Plus`, `MinMax`, `Base`) | `CrossConformalRegressor(method="plus" / "minmax" / "base")` |
| `CrossConformalRegressor::fit_with_splits` + `leave_one_out` | jackknife+ (`cv=LeaveOneOut()`) |
| `ConformalizedQuantileRegressor`, `conformalize_quantiles` | `ConformalizedQuantileRegressor`, `prefit=True` |
| `NormalizedConformalRegressor`, `conformalize_normalized` | `SplitConformalRegressor(conformity_score=ResidualNormalisedScore)` |

Metrics: `coverage_score`, `mean_interval_width`, `interval_score`,
`mean_pinball_loss` (anofox-ml-metrics).

## Reference

MAPIE **1.6.0** with scikit-learn **1.8.0** (pinned in
`test_harness/requirements.txt`).

## Golden test (exact parity)

- Generator: `test_harness/generators/gen_conformal.py`
- Fixture:   `crates/anofox-ml-conformal/tests/golden_data/mapie.json`
- Rust test: `crates/anofox-ml-conformal/tests/golden_mapie.rs`

Regenerate:

```bash
cd test_harness
uv run --no-project --with-requirements requirements.txt python3 generators/gen_conformal.py
cd .. && cargo test -p anofox-ml-conformal --test golden_mapie
```

Data: 80 training rows × 3 features (heteroscedastic linear model, seed 0),
15 test rows (seed 1). To compare only the conformal logic, the base model
is ordinary least squares on both sides (`LinearRegression` /
`OlsRegressor`, which agree well inside the tolerance below) and folds are
passed explicitly.

| Case | Setup | Levels |
|---|---|---|
| `split_ncal30` | prefit OLS on rows 0..50, calibration rows 50..80 | 0.9, 0.8, 0.5 |
| `split_ncal8` | calibration rows 50..58 (rank = n at 0.8) | 0.8, 0.5; **0.9 must error** |
| `cross_{plus,minmax,base}_{mean,full}` | 7 unequal folds (`i % 7`) | 0.9, 0.8, 0.5 |
| `jackknife_plus` | leave-one-out on rows 0..25 | 0.9, 0.8, 0.5 |
| `cqr_{0.9,0.8}_{asym,sym}` | OLS "quantile models" on `y − 1.5`, `y + 1.5`, `y` | the case's level |
| `normalized` | model rows 0..50, residual model rows 50..62, scores 62..80 | 0.9, 0.8, 0.5 |

Tolerance: `|got − MAPIE| ≤ 1e-9 · (1 + |MAPIE|)` on point predictions and
both bounds; infinite bounds must match exactly. The quantile rank uses
MAPIE's float formulas (`ceil(level · (n + 1)) / n` with numpy's `"lower"`
index; `(1 − 2α) + α` for the CV+ lower bound) so order statistics agree
exactly rather than within a rank.

## Coverage tests (statistical)

`crates/anofox-ml-conformal/tests/coverage.rs`, seeded, on heteroscedastic
data (noise sd `0.3 + |x0|`). The thresholds were set empirically when the
tests were written; they are the first thing to revisit (see below).

| Test | Reps × test rows | Check |
|---|---|---|
| split conformal, α = 0.1 and 0.3 | 200 × 500 (n = 100, 40 calibration) | mean coverage − (1 − α) ∈ [−0.01, 0.035) (theory: [0, 1/(n_cal + 1)]) |
| CV+, minmax, base, α = 0.1 | 100 × 500 | excess > −0.02 / −0.01 / −0.03 and < 0.05 |
| CQR (quantile GBM, 60 trees, depth 3), α = 0.2, both corrections | 20 × 400 (n = 300) | \|coverage − 0.8\| < 0.04; width–\|x0\| correlation > 0.3 |
| normalised (OLS + depth-3 tree residual model), α = 0.1 | 30 × 400 | \|coverage − 0.9\| < 0.03; correlation > 0.3 |
| plain split (control) | 5 × 400 | width–\|x0\| correlation < 0.05 (constant width) |

Behaviour tests in the same file: nesting (lower α ⇒ wider; minmax ⊇ CV+),
median point prediction by hand, too-few-rows errors / infinite bounds,
invalid splits and parameters.

## Differences from MAPIE (deliberate)

- **Median point prediction** (`PointPrediction::Median`): MAPIE 1.6.0
  returns its mean for `aggregate_point_predictions="median"`. Not in the
  golden data; checked by hand against the definition (median over training
  rows of their out-of-fold model's prediction).
- **`allow_infinite_bounds` on symmetric split**: MAPIE's split path does
  not pass it on and raises; ours returns ±∞ when asked.
- **CQR sample-size check**: MAPIE also requires
  `n ≥ max(1/α′, 1/(1 − α′))`; we only require the conformal rank to exist.
- **`Base` method**: point predictions are always the full-data model's
  (as MAPIE's no-aggregation methods), whatever `point_prediction` says.
- **Normalised score, non-prefit split**: our train / residual /
  calibration split uses our seeded RNG, not sklearn's `train_test_split`,
  so only the prefit path is compared exactly.
- Errors instead of MAPIE's warnings: none; MAPIE's quantile-ordering
  warnings (`_check_lower_upper_bounds`) are not reproduced.

## To re-run or refine later

- **When MAPIE or sklearn is bumped**: regenerate (command above) and
  re-run; update this file's versions. The *Golden Data Sync* CI job
  regenerates this fixture with the pinned versions and diffs it
  (advisory: it reports drift without failing the workflow).
- **Coverage thresholds**: replace the empirical bands with binomial /
  Clopper–Pearson bounds derived from reps × test rows, and add α = 0.05.
- **Real learners in the golden data**: compare CQR with an actual quantile
  learner and the normalised score with a non-linear residual model once a
  bit-identical pair exists on both sides (today OLS proxies stand in).
- **Not yet validated**: classification sets (LAC / APS / RAPS), Mondrian
  coverage, jackknife+-after-bootstrap, weighted (covariate-shift)
  conformal — the next milestones of #93.
