---
name: anofox-ml-boosting
description: Use the gradient-boosting estimators of the anofox-ml Rust crate (and its anofox_ml_python binding) well. Covers GradientBoosting{Classifier,Regressor} (sklearn), HistGradientBoosting{Classifier,Regressor} (sklearn 1.8 parity), LgbmClassifier/LgbmRegressor (LightGBM 4.7 parity, DART, EFB, GOSS) and CatBoostClassifier/CatBoostRegressor (catboost 1.2.10 parity incl. ordered boosting, CTRs, MVS bootstrap and its random generator). Use it whenever the user wants to train, tune, compare or reproduce a boosted-tree model with anofox-ml / anofox_ml / anofox_ml_python, asks "which boosting model should I use", wants results identical to sklearn, LightGBM or CatBoost, or has categorical features to boost on.
---

# Gradient boosting with anofox-ml

anofox-ml reimplements four boosting engines in Rust. Each is validated
against its reference library with golden tests
(`crates/anofox-ml/tests/golden_*`). They reproduce the reference library's
models, so you can choose the engine by the behaviour you want and still get
the reference's results.

## Choosing an engine

| Need | Estimator | Reference |
|---|---|---|
| Small data, sklearn's classic algorithm | `GradientBoosting{Classifier,Regressor}` | sklearn `GradientBoosting*` |
| sklearn-compatible histogram boosting, early stopping, Poisson/Gamma/quantile, native categoricals, monotonic / interaction constraints | `HistGradientBoosting{Classifier,Regressor}` | sklearn 1.8 `HistGradientBoosting*` |
| Speed on large numeric data, leaf-wise trees, DART, GOSS, sparse / one-hot data (EFB); Tweedie, cross-entropy and one-vs-all objectives | `LgbmClassifier`, `LgbmRegressor` | LightGBM 4.7 |
| Many categorical features (target statistics, combinations), robust defaults, symmetric trees, ordered boosting against target leakage | `CatBoostClassifier`, `CatBoostRegressor` | catboost 1.2.10 |

## Rust

All estimators use the type-state pattern: configure, `fit`, then
predict on the fitted value. Import `anofox_ml::prelude::*` for the
`Fit` / `Predict` traits. Estimators live in `anofox_ml::ensemble`.

```rust
use anofox_ml::prelude::*;
use anofox_ml::ensemble::{CatBoostClassifier, HistGradientBoostingRegressor, LgbmRegressor};

let hgb = HistGradientBoostingRegressor::new()
    .with_learning_rate(0.1)
    .with_max_leaf_nodes(Some(31))
    .fit(&x, &y)?;

let lgbm = LgbmRegressor::new()
    .with_n_estimators(500)
    .with_num_leaves(63)
    .with_learning_rate(0.05)
    .fit(&x, &y)?;

// Categorical columns hold category codes (any f64 values; NaN is a category).
let cat = CatBoostClassifier::new()
    .with_iterations(1000)
    .with_cat_features(vec![2, 5])
    .fit(&x, &y)?;
let proba = cat.predict_proba(&x)?;
```

Builder names follow each reference library's parameter names (with a
`with_` prefix).
- **LGBM:** `with_num_leaves`, `with_min_child_samples`, `with_reg_lambda`,
  `with_subsample`, `with_subsample_freq`, `with_colsample_bytree`,
  `with_colsample_bynode`, `with_boosting_type`, `with_goss_rates`,
  `with_dart`, `with_enable_bundle`, `with_monotone_constraints`,
  `with_early_stopping`, `with_seed` (classifier: `with_bagging_fractions`).
- **HistGB:** `with_loss`, `with_max_leaf_nodes`, `with_min_samples_leaf`,
  `with_categorical_features`, `with_monotonic_cst`, `with_interaction_cst`,
  `with_early_stopping`, `with_validation_fraction`, `with_n_iter_no_change`,
  `with_scoring`, `with_random_state`. It also has `fit_with_validation`,
  `fit_warm_start`, `staged_predict` and `feature_importances`.
- **CatBoost:** `with_depth`, `with_l2_leaf_reg`, `with_border_count`,
  `with_cat_features`, `with_one_hot_max_size`, `with_max_ctr_complexity`,
  `with_boosting_type`, `with_leaf_estimation_method`,
  `with_leaf_estimation_iterations`, `with_bootstrap_type`, `with_subsample`,
  `with_random_strength`, `with_random_seed`, `with_has_time`,
  `with_monotone_constraints`, `with_model_shrink_rate`,
  `with_early_stopping_rounds`, `with_use_best_model`, `fit_with_eval`,
  `with_grow_policy` (Depthwise / Lossguide: plain boosting; Lossguide
  takes CTR features, exact for regression, close but not always exact
  for classifiers, #154), `with_min_data_in_leaf`, `with_max_leaves`; classifier:
  `with_class_weights`, `with_auto_class_weights`.

Check the docs (`cargo doc -p anofox-ml-ensemble --open`) before relying on
any other name.

## Python

`anofox_ml_python` mirrors the sklearn API (`fit`, `predict`,
`predict_proba`, `get_params` / `set_params`, pipelines, CV):

```python
import anofox_ml_python as am
m = am.CatBoostClassifier(iterations=500, cat_features=[2]).fit(X, y)   # X may hold strings
m = am.LgbmRegressor(n_estimators=300, num_leaves=31).fit(X, y)
m = am.HistGradientBoostingClassifier(early_stopping=True).fit(X, y)
```

CatBoost text features (catboost's default processing, matching catboost):
pass `text_features=[...]` in Python (a DataFrame or object array), or call
`fit_with_text(x, &text, y, w)` / `predict_with_text` in Rust. Regression
gets BoW over word bigrams and words. Classification also gets the online
NaiveBayes. Custom processing goes through catboost's `text_processing` dict
in Python, or `with_text_options(CatBoostTextOptions { .. })` in Rust: one
`ByDelimiter` tokenizer (delimiter, lowercasing), Word dictionaries
(`gram_order`, `occurrence_lower_bound`, `max_dictionary_size`) and the
calcers `BoW:top_tokens_count=N`, `NaiveBayes` and `BM25`. NaiveBayes and
BM25 are classification only. No eval sets, rankers or MultiRMSE with text
yet.

## LightGBM: objectives

`LgbmRegressor::with_objective`: `Regression` (L2), `RegressionL1`,
`Huber`, `Quantile { alpha }`, `Poisson`, `Gamma`,
`Tweedie { variance_power }` (log link; power in `[1, 2)`, LightGBM's
default 1.5; zero-inflated counts / intermittent demand) and
`CrossEntropy` (targets in `[0, 1]`, sigmoid predictions).
`LgbmClassifier` uses binary log-loss or softmax by the number of classes;
`with_multiclass_ova(true)` is LightGBM's `multiclassova`, whose
`predict_proba` rows are per-class sigmoids that do not sum to 1. Python
takes LightGBM's names: `objective="tweedie"`, `"cross_entropy"`,
`"multiclassova"`. All match LightGBM 4.7 to ~1e-16.
Ranking: `LgbmRegressor::ranker()` (LambdaRank) or
`with_objective(LgbmObjective::RankXendcg)`, fit with
`LgbmFitOptions { group: Some(&query_sizes), .. }` and integer relevance
labels; Python `LGBMRanker().fit(X, y, group=sizes)`; registry kind
`LgbmRanker` with `fit_with_group`.
`with_linear_tree(true, lambda)` (Python `linear_tree=True`) fits a ridge
model per leaf on its path's numeric features: use it when a global model
must extrapolate trends.

## LightGBM: large data and custom losses

- Large or `f32` data: `let data = model.dataset(x_f32.view())?; drop(x);
  model.fit_dataset(&data, y, &opts)`. Peak memory is the raw matrix plus
  one byte per value. `data.subset(rows)` reuses the binning across folds.
  `bin_sample_rows` + `dataset_builder` + `push_rows` stream rows without
  the full matrix. Above 200 000 rows the bins come from LightGBM's
  200 000-row sample, so the model is still LightGBM's.
- Custom losses: implement `LgbmCustomObjective` (`n_outputs`,
  `gradients`), then `fit_custom(x, &loss)` → `FittedLgbmBooster::predict_raw`
  (`n × K`). Example: `examples/fforma_custom_objective.rs`.

## LightGBM: sampling and reproducibility

LightGBM's random generator is reproduced draw for draw: bagging,
GOSS, `colsample_bytree` / `colsample_bynode` and DART's dropping. The same
data, parameters and `seed` (LightGBM's `random_state`) give **the model
LightGBM trains** (with `n_jobs=1`, `deterministic=True,
force_row_wise=True`), to about 1e-15.
- Bagging is LightGBM's: `subsample` acts only with `subsample_freq > 0`
  (LightGBM's default 0 = no bagging). For balanced bagging of binary
  labels, use `with_bagging_fractions(pos, neg)`.
- `seed: None` (default) uses LightGBM's default seeds. `with_seed(s)`
  derives the bagging, feature-fraction and DART seeds from `s`, as
  LightGBM does.
- GOSS samples from iteration `1 / learning_rate` on and cannot be combined
  with bagging. `top_rate + other_rate` must be at most 1.

## CatBoost: defaults and reproducibility

`CatBoost*` defaults are CatBoost's own:
- `has_time=false` (the rows are permuted), `random_seed=0`;
- `random_strength=1` (score noise);
- MVS bootstrap with `subsample` 0.8 (Bayesian for multiclass);
- `max_ctr_complexity` 4 (1 below 200 iterations);
- `leaf_estimation_iterations` 10 for Logloss (1 below 200 iterations with
  fewer than 20 features);
- per `loss_function` (regressor: RMSE, MAE, Quantile, Expectile, Poisson,
  Tweedie, LogLinQuantile): Exact leaves for MAE / Quantile,
  `boost_from_average` for RMSE / MAE / Quantile; Poisson and Tweedie
  predict `exp(raw)`;
- with `monotone_constraints`, model shrinkage `model_shrink_rate=0.01`
  (CatBoost switches it on with them; pass 0 to turn it off).

CatBoost's random generator is reproduced draw for draw. The same data,
parameters and `random_seed` give **the model catboost trains** (with
catboost's `thread_count=1`), to about 1e-15.

- To compare with catboost, fit catboost with `thread_count=1` and the same
  parameters, then compare `predict_proba` / `predict`. Don't loosen the
  tolerance: a difference above ~1e-12 is a bug or a configuration mismatch.
  Exceptions are rare near-ties on large Logloss data and catboost runs
  with several threads.
- For deterministic, order-dependent fits (time series), set
  `has_time=true`. Set `random_strength=0` and `bootstrap_type=No` to remove
  the remaining randomness.
- More than two classes switch the classifier to MultiClass (softmax).
  `raw_predictions` gives the class scores; `decision_function` gives only
  the binary log-odds.
- Speed: about 2.5× slower than single-threaded catboost. For large numeric
  data without categoricals, prefer `LgbmRegressor` / `LgbmClassifier`.

## Validation habits

- Compare engines with the same CV split: `cross_val_score` in
  `anofox_ml::core::utils`, or sklearn's `cross_val_score` on the Python
  estimators.
- For probabilities, look at log loss and calibration. Boosted trees are
  often over-confident: wrap them in `CalibratedClassifierCV`.
- Report `n_estimators()` / `n_iter()` after early stopping, and
  `feature_importances` (HistGB, LGBM, GradientBoosting) with caution: they
  are split-gain importances, not causal effects.
