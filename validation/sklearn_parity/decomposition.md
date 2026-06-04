# KernelPCA / NMF — sklearn parity

Issue: [#18](https://github.com/sipemu/rustml/issues/18) (continued from TruncatedSVD)

## What

- **KernelPCA** (`rustml-preprocessing::kernel_pca`): eigendecomposition of the
  centered kernel matrix. Linear / RBF / Polynomial kernels. Coordinates
  returned as `α / √λ * K_new α` (sklearn convention).
- **NMF** (`rustml-preprocessing::nmf`): multiplicative-update solver (Lee &
  Seung). `X ≈ W H` with `W, H ≥ 0`.

## Reference

`sklearn.decomposition.{KernelPCA, NMF}` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_decomposition.py`
- Fixture:   `crates/rustml/tests/golden_data/decomposition.json`
- Rust test: `crates/rustml/tests/golden_decomposition.rs`

KernelPCA: eigenvalues match sklearn to `1e-6` (the centered kernel matrix
spectrum is deterministic; only the eigenvectors are sign-flippable).

NMF: reaches a reconstruction error within 2× of sklearn's. Both solvers
use multiplicative updates, but initial conditions differ, so exact match
isn't pursued.

## Not yet implemented

- **FastICA** — still pending in #18.
- KernelPCA `fit_inverse_transform=True`.
- NMF `init='nndsvd'` / `'nndsvda'` smarter inits; `solver='cd'` coordinate
  descent.
