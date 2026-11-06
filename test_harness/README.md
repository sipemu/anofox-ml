# Golden data (reference fixtures)

Python generators produce reference outputs (scikit-learn, MAPIE) as JSON
fixtures; Rust tests load them and compare.

| Fixtures | Generator(s) | Rust tests |
|---|---|---|
| `crates/anofox-ml/tests/golden_data/*.json` | `generators/gen_*.py`; the generators without a `__main__` block run through `generate_golden_data.py` | `crates/anofox-ml/tests/golden_*.rs` |
| `crates/anofox-ml-conformal/tests/golden_data/mapie.json` | `generators/gen_conformal.py` | `crates/anofox-ml-conformal/tests/golden_mapie.rs` |

Versions are pinned in `requirements.txt` (scikit-learn 1.8.0, MAPIE 1.6.0,
LightGBM 4.7.0 for `gen_lgbm_parity.py` / `lgbm_parity.json`, CatBoost
1.2.10 for `gen_catboost_parity.py` / `catboost_parity.json`);
per-estimator notes are in `validation/sklearn_parity/` and
`validation/mapie_parity/`.

## Regenerate and check for drift

```bash
./regenerate_all.sh          # every generator, then check_fixture_drift.py
```

`check_fixture_drift.py DIR...` compares the regenerated fixtures in the
working tree with the committed ones. Numbers are compared with a tolerance
(`--rtol 1e-9 --atol 1e-12`), because BLAS / CPU differences between
machines change the last bits; anything beyond that, and any structural
change, is reported as drift (exit status 1). Revert noise-only changes
(`git checkout -- crates/*/tests/golden_data`) instead of committing them.

Fields that legitimately differ between machines are listed in
`drift_ignore.txt` (`file:key`, with the reason), e.g. sklearn HDBSCAN's
`labels_public`, whose tie breaking depends on the CPU.

CI (*Golden Data Sync*) runs the same regeneration and check on every PR
(advisory for now).

## Fixtures without a generator

These were added in 8921972 (sklearn 1.8.0 / LightGBM 4.6) without their
generator scripts. `generators/gen_legacy_fixtures.py` re-validates the
deterministic ones: it recomputes every output from the committed inputs
with the pinned sklearn (parameters as in the Rust tests that read them), so
the drift check covers them. All matched at the time of writing (#113):
`metrics_extra`, `metrics_round2`, `model_selection_extra`,
`preprocessing_extra`, `preprocessing_round2` (except `power_transformer`,
whose Yeo-Johnson lambda moves by ~1e-8 between scipy versions and is read
by no test), `naive_bayes_extra`, `feature_selection_batch` and
`regression_batch` (except `sgd_regressor`).

Still without a generator, because they depend on seeded randomness or are
only compared loosely (accuracy / R² thresholds): `accuracy_parity`,
`ensemble_batch`, `hist_gradient_boosting`, `lgbm_edge_cases`,
`lgbm_golden`, `lgbm_round2` (LightGBM 4.6), `svm_calibration_batch`,
`svm_extra`. `ensemble_extra` and `regression_extra` were read by no test
and have been removed.
