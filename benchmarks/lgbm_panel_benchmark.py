"""Panel-scale LightGBM reference for `examples/lgbm_panel_bench.rs` (#215).

Same shapes and parameters as the Rust example: `ROWS × 60` float32
features, 31 leaves, 200 iterations, learning rate 0.1, bagging 0.9 every 5
iterations, feature fraction 0.9; optionally three categorical features
(~10, ~100, ~1000 levels) and the Poisson objective.

    python benchmarks/lgbm_panel_benchmark.py ROWS [cat] [poisson] [threads=N]
    cargo run --release --example lgbm_panel_bench -- ROWS [cat] [poisson] [threads=N]

The machines' load matters at these sizes: take the minimum of several
runs, alternating the two.
"""

import resource
import sys
import time

import lightgbm as lgb
import numpy as np


def main():
    args = sys.argv[1:]
    n = int(args[0]) if args else 1_000_000
    cat = "cat" in args
    poisson = "poisson" in args
    threads = next((int(a.split("=")[1]) for a in args if a.startswith("threads=")), 14)
    p = 60
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p)).astype(np.float32)
    cats = []
    if cat:
        for j, k in zip((57, 58, 59), (10, 100, 1000)):
            X[:, j] = rng.integers(0, k, size=n)
            cats.append(j)
    eta = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.5 * X[:, 3:8].sum(axis=1)
    if cat:
        eta += np.sin(X[:, 58]) + (X[:, 59] % 7) * 0.1
    noise = rng.normal(scale=0.5, size=n)
    y = np.round(np.exp(0.3 * eta + 0.1 * noise)) if poisson else eta + noise
    model = lgb.LGBMRegressor(
        objective="poisson" if poisson else "regression", n_estimators=200, num_leaves=31,
        learning_rate=0.1, subsample=0.9, subsample_freq=5, colsample_bytree=0.9,
        n_jobs=threads, verbose=-1)
    t = time.perf_counter()
    model.fit(X, y, categorical_feature=cats or "auto")
    fit = time.perf_counter() - t
    t = time.perf_counter()
    pred = model.predict(X)
    predict = time.perf_counter() - t
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    print(f"lightgbm n={n} cat={cat} poisson={poisson} threads={threads} fit {fit:.2f}s "
          f"predict {predict:.2f}s peak {peak:.2f}GB mse {np.mean((pred - y) ** 2):.5f}")


if __name__ == "__main__":
    main()
