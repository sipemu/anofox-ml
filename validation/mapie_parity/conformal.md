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
| `SplitConformalClassifier`, `conformalize_classifier` (`Lac`, `Aps(Include / Exclude / Randomized)`) | `SplitConformalClassifier(conformity_score="lac" / "aps")`, `include_last_label=True / False / "randomized"` |
| `ClassificationScore::Raps(RapsParams)`, `conformalize_raps` | `SplitConformalClassifier(conformity_score=RAPSConformityScore())` |
| `MondrianConformalRegressor`, `conformalize_mondrian`, `conformalize_classifier_mondrian` | none in MAPIE 1.6.0 (validated by definition, below) |

Metrics: `coverage_score`, `mean_interval_width`, `interval_score`,
`mean_pinball_loss`, `set_coverage_score`, `mean_set_size`
(anofox-ml-metrics).

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

### Classification sets

- Generator: `test_harness/generators/gen_conformal_classification.py`
- Fixture:   `crates/anofox-ml-conformal/tests/golden_data/mapie_classification.json`
- Rust test: `crates/anofox-ml-conformal/tests/golden_mapie_classification.rs`

Both sides use a fixed softmax model `P(y = k | x) = softmax(x W)_k`
(3 features, 4 classes); 120 calibration rows with labels drawn from the
model, 25 test rows. Cases: `lac`, `aps` with `include_last_label=True`
and `False`, and RAPS (include / exclude at λ = 0.1, include at λ = 0.5),
each at levels 0.9 / 0.8 / 0.5. For RAPS the generator pins MAPIE's tuning
rows (rows 0..30; MAPIE draws them with a `StratifiedShuffleSplit`) and λ
(see the λ-search difference below); k* is computed by MAPIE from the
tuning rows, as by us. Every set (25 rows × 4
classes × 3 levels) and the point predictions must match **exactly**.
APS is compared without randomisation: MAPIE's `random_state` is a
`RandomState` whose `uniform` returns zeros.

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

Classification (`tests/classification.rs`, labels drawn from a known
softmax model, 100 calibration × 400 test rows):

| Test | Reps | Check |
|---|---|---|
| LAC, APS-include, α = 0.1 and 0.3 | 150 | mean coverage − (1 − α) > −0.01 (theory: [0, 1/(n + 1)]) |
| APS-randomised, α = 0.1 and 0.3 | 150 | \|coverage − (1 − α)\| < 0.015 (exact on average) |
| LAC vs APS-include set size, α = 0.1 | 30 | LAC sets are smaller |
| APS adaptivity | 1 × 500 | set size falls as the top probability rises; exclude ⊆ include |
| `SplitConformalClassifier` + random forest, randomised APS, α = 0.2 | 1 × 1000 | \|coverage − 0.8\| < 0.06; same sets on repeated calls |
| RAPS (tuned k*, λ), include / randomised, α = 0.1 | 150 | include: excess > −0.01; randomised: \|coverage − 0.9\| < 0.02 |
| RAPS tuning | 1 × 300 | k* in 1..=K, λ on the grid; fixed k / λ skip the tuning split |
| RAPS vs APS set size (10 classes, long tail), α = 0.1 | 20 × 300 | RAPS mean size ≤ APS |

Behaviour tests in the regression file: nesting (lower α ⇒ wider; minmax ⊇ CV+),
median point prediction by hand, too-few-rows errors / infinite bounds,
invalid splits and parameters.

Mondrian (`tests/mondrian.rs`, no reference implementation):

| Test | Check |
|---|---|
| one group = plain split conformal | regression intervals at α = 0.05 / 0.1 / 0.5 and classification sets (LAC, APS include / randomised, RAPS) **identical** to the non-Mondrian path |
| two groups, noise sd 0.2 vs 2.0, α = 0.1, 100 reps × 400 rows | plain split: > 0.97 / < 0.86 per group; Mondrian: each group excess > −0.01 and < 0.93 |
| classification, 40 % label noise in one group, LAC, 60 reps | plain under-covers the noisy group (< 0.87); Mondrian covers both (excess > −0.01) |
| errors | unseen group, wrong number of group ids, too few rows in a group (or ±∞ when allowed), Mondrian sets from a classifier calibrated without groups |

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
- **APS calibration scores**: MAPIE always randomises them (subtracting
  `u · p̂_y`), whatever `include_last_label` is; we randomise them only for
  `ApsMode::Randomized`, so `Include` / `Exclude` are deterministic.
  Randomised APS uses our seeded RNG, so it is checked for coverage, not
  against MAPIE's draws.
- **RAPS λ search**: MAPIE 1.6.0's `_find_lambda_star` regularizes the
  tuning scores with λ in place of k* (`_add_regularization` sets
  `k_star = lambda_` outside the prediction phase). We use k*, as in the
  paper. The tuned λ can therefore differ from MAPIE's; the golden data
  pins λ so the rest is compared exactly.
- **RAPS tuning split**: MAPIE stratifies it (`StratifiedShuffleSplit`,
  `size_raps = 0.2`); ours is a plain seeded split of the same size, or
  any split via `conformalize_raps`.
- **APS ties**: classes with equal probability are ordered higher index
  first (numpy's `flip(argsort)` for small arrays); MAPIE's order for large
  class counts depends on numpy's sort algorithm.
- **Too few calibration rows (classification)**: MAPIE raises (numpy
  quantile above 1); we raise too, or return the set of all classes with
  `allow_full_sets`.
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
- **Real classifiers in the golden data**: the softmax stand-in matches
  bit for bit; a comparison with an actual sklearn classifier needs a
  model pair with identical probabilities.
- **Tuned RAPS against a reference**: only possible once MAPIE's λ search
  is fixed upstream (or against the paper authors' code).
- **Mondrian against a reference**: crepes (Boström) or MAPIE < 1.0's
  `MondrianCP` could serve; today only the definition and coverage are
  tested.
- **Not yet validated**: top-k, cross-conformal classification,
  Mondrian for normalised / CQR / cross-conformal regressors,
  jackknife+-after-bootstrap, weighted (covariate-shift)
  conformal — the next milestones of #93.
