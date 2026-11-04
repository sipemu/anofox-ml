# Hyperparameter search (anofox-ml-tune) — validation

Issue: [#92](https://github.com/sipemu/anofox-ml/issues/92) ·
PR: [#94](https://github.com/sipemu/anofox-ml/pull/94)

## What

`anofox-ml-tune`: Optuna-style search over any registry estimator on
[optuna/rustuna](https://github.com/optuna/rustuna) (git rev `d0ec7db`,
`rustuna_core` + `rustuna_sampler`, in-memory storage). Python:
`anofox_ml_python.TuneSearchCV`.

## Reference

None for exact parity. The search loop, CV objective and result are our
code; the sampling (TPE, random, NSGA-II) is rustuna's, which the Optuna
team validates against Optuna. We do not compare trial sequences with
Optuna: rustuna's TPE uses its own RNG, so sequences differ by design.

## Tests

Rust (`crates/anofox-ml-tune`): `cargo test -p anofox-ml-tune`

| Test | Checks |
|---|---|
| `space::tests` | JSON search-space parsing; every malformed spec rejected with a message |
| `metric::tests` | metric names round-trip; default metric by estimator type |
| `finds_the_depth_a_step_function_needs` | step function with 8 levels → best `max_depth ≥ 3`, R² > 0.99; best = max of trial scores |
| `same_seed_same_search_and_refit_matches_best_params` | identical trials for the same seed, different for another seed / sampler; suggested values inside their ranges (incl. log scale); refit model = `make(best_params).fit` |
| `loss_metrics_are_minimised` | `rmse` minimised, best = min of trials |
| `probability_metrics_score_classifiers` | `log_loss` / `roc_auc` on a random forest; default `accuracy` for classifiers |
| `failing_trials_are_recorded_not_fatal` | invalid categorical value → failed trials with the error; all failed → error |
| `invalid_requests_fail_up_front` | unknown parameter, fixed + searched, unknown metric, `n_trials = 0`, unknown kind |
| `custom_splits_and_sample_weights` | group k-fold splits; weights reach the fit (and an unweighted estimator fails) |
| `result_serializes_without_the_model` | JSON result; options from JSON with defaults, unknown keys rejected |
| `multi_objective_reports_the_pareto_front` | TPE and NSGA-II; front equals a brute-force non-dominated set; best trial on the front |
| `nsgaii_is_seeded_and_fit_time_is_measured` | NSGA-II reproducible per seed; `fit_time` positive |
| `invalid_objectives_fail_up_front` | duplicate / unknown objective, `population_size < 2` |
| `mapped_search_spaces_translate_each_trial` | per-trial translation; a failing translation fails only that trial |

Python (`crates/anofox-ml-python/tests/test_tune.py`, after `maturin develop`):
sklearn parameter names and refit, label-encoded classifiers with
`log_loss`, failed trials, seeded reproducibility with log-scale floats,
sklearn `GroupKFold` with `groups`, NSGA-II Pareto front, invalid requests.

## Findings while validating

- Multi-objective: with ties on the primary metric, the first tied trial
  could be dominated on another objective; the best trial is now chosen
  from the Pareto front (caught by the Python test).
- `RandomForestClassifier` / `ExtraTreesClassifier` returned
  `predict_proba = 1.0000000000000002`; fixed in 0.2.0, regression test
  `proba_range_tests` in `anofox-ml-ensemble`.

## To re-run or refine later

- **rustuna bump**: re-run both suites; seeded trial sequences may change
  (tests compare runs with each other, not stored sequences).
- **Search quality**: no test checks that TPE beats random search on a
  benchmark; add one (median over seeds on a few estimators) once the API
  settles.
- **Pruning**: untested — rustuna has no pruners yet.
- **SQL** (`ml_tune_agg`, DataZooDE/anofox-ml#5): validated in the
  extension repo.
