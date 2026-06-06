# KernelPCA / NMF — sklearn parity

Issue: [#18](https://github.com/sipemu/anofox-ml/issues/18) (continued from TruncatedSVD)

## What

- **KernelPCA** (`anofox-ml-preprocessing::kernel_pca`): eigendecomposition of the
  centered kernel matrix. Linear / RBF / Polynomial kernels. Coordinates
  returned as `α / √λ * K_new α` (sklearn convention).
- **NMF** (`anofox-ml-preprocessing::nmf`): multiplicative-update solver (Lee &
  Seung). `X ≈ W H` with `W, H ≥ 0`.

## Reference

`sklearn.decomposition.{KernelPCA, NMF}` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_decomposition.py`
- Fixture:   `crates/anofox-ml/tests/golden_data/decomposition.json`
- Rust test: `crates/anofox-ml/tests/golden_decomposition.rs`

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

## Complexity

- PCA (truncated): **O(n · p · k)** via randomised SVD with `k = n_components`.
- KernelPCA: **O(n²·p)** kernel construction + **O(n³)** eigendecomposition of the centred Gram matrix.
- TruncatedSVD: **O(n · p · k)** via Lanczos / randomised SVD on possibly-sparse X.
- FastICA: **O(p² · n + p³)** for whitening + symmetric decorrelation; deflation variant **O(k · p · n)** per component.
- NMF (multiplicative update): **O(n · p · k · iter)**; coordinate descent shaves a constant factor.
- Memory dominated by **O(n·p)** for the dense data, plus **O(p²)** scratch for whitening / Gram.
