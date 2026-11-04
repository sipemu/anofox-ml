# Golden data (reference fixtures)

Python generators produce reference outputs (scikit-learn, MAPIE) as JSON
fixtures; Rust tests load them and compare.

| Fixtures | Generator(s) | Rust tests |
|---|---|---|
| `crates/anofox-ml/tests/golden_data/*.json` | `generators/gen_*.py`; the generators without a `__main__` block run through `generate_golden_data.py` | `crates/anofox-ml/tests/golden_*.rs` |
| `crates/anofox-ml-conformal/tests/golden_data/mapie.json` | `generators/gen_conformal.py` | `crates/anofox-ml-conformal/tests/golden_mapie.rs` |

Versions are pinned in `requirements.txt` (scikit-learn 1.8.0, MAPIE 1.6.0);
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
generator scripts, so they cannot be regenerated or drift-checked:

`accuracy_parity`, `ensemble_batch`, `ensemble_extra`,
`feature_selection_batch`, `hist_gradient_boosting`, `lgbm_edge_cases`,
`lgbm_golden`, `lgbm_round2`, `metrics_extra`, `metrics_round2`,
`model_selection_extra`, `naive_bayes_extra`, `preprocessing_extra`,
`preprocessing_round2`, `regression_batch`, `regression_extra`,
`svm_calibration_batch`, `svm_extra` (`.json`).

Recreating their generators (LightGBM 4.6 for the `lgbm_*` files) would
bring them under the drift check.
