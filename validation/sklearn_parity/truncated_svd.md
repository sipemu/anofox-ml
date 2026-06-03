# TruncatedSVD — sklearn parity

Issue: [#18](https://github.com/sipemu/rustml/issues/18) (partial)

## What

Truncated singular value decomposition. Computes `X ≈ U Σ V'`, keeps top-`k`
singular triplets, and `transform(X) = X V_k`. Unlike PCA, does **not** centre
the input.

Implemented with `faer`'s thin SVD.

## Reference

`sklearn.decomposition.TruncatedSVD` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_truncated_svd.py`
- Fixture:   `crates/rustml/tests/golden_data/truncated_svd.json`
- Rust test: `crates/rustml/tests/golden_truncated_svd.rs`

40×6 random matrix multiplied by a diagonal of decaying singular values. SVD
is sign-ambiguous (any column of U/V can flip sign), so we compare:
- **Singular values** match to `1e-6`.
- **Transformed coordinates in absolute value** match to `1e-6` element-wise.

## Differences from sklearn / not implemented

- `KernelPCA`, `NMF`, `FastICA` are still pending (also in issue #18).
- No `algorithm='randomized'` — we use the dense SVD.
- No `inverse_transform` yet (sklearn provides one via `X̂ = T V'`).
