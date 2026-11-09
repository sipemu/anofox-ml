"""Golden data for LightGBM above `bin_construct_sample_cnt` (#214): with
more than 200 000 rows LightGBM bins from a sample of 200 000 rows drawn
with its own generator (`data_random_seed`, derived from `seed`).

The features come from an integer hash that the Rust test recomputes
(`golden_lgbm_large.rs`), so only predictions are stored. All values are
float32-exact, and each case is fitted from float64 and from float32
input, which LightGBM bins identically.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np

N = 260_000
COMMON = dict(deterministic=True, force_row_wise=True, verbose=-1, n_jobs=1)


def hashes(n, j):
    i = np.arange(n, dtype=np.uint64)
    return (i * np.uint64(2654435761) + np.uint64(j * 40503 + 17)) % np.uint64(2 ** 32)


def data(n=N):
    h = [hashes(n, j) for j in range(5)]
    x0 = ((h[0] >> np.uint64(8)) % np.uint64(4096)).astype(float) / 256 - 8
    x1 = ((h[1] >> np.uint64(4)) % np.uint64(1024)).astype(float) / 128 - 4
    cat = ((h[2] >> np.uint64(12)) % np.uint64(30)).astype(float)
    x3 = ((h[3] >> np.uint64(6)) % np.uint64(512)).astype(float) / 64 - 4
    x3[(h[3] >> np.uint64(20)) % np.uint64(97) == 0] = np.nan
    noise = ((h[4] >> np.uint64(9)) % np.uint64(1000)).astype(float) / 1000 - 0.5
    X = np.column_stack([x0, x1, cat, x3])
    y = 0.3 * x0 * x0 - 0.5 * x1 + (cat % 5 - 2) * 0.4 + 0.2 * np.nan_to_num(x3) + noise
    return X, y


def generate():
    X, y = data()
    test = np.arange(0, N, 1300)
    out = []
    base = dict(n_estimators=10, learning_rate=0.2, num_leaves=15, **COMMON)
    cases = [
        ("regression", dict(base), y),
        ("regression_seed", dict(base, random_state=11, subsample=0.9, subsample_freq=5,
                                 colsample_bytree=0.75), y),
        ("poisson_categorical", dict(base, objective="poisson"), np.exp(0.3 * y)),
    ]
    for name, params, target in cases:
        cat = [2] if "categorical" in name else "auto"
        preds = []
        for dtype in (np.float64, np.float32):
            m = lgb.LGBMRegressor(**params).fit(X.astype(dtype), target, categorical_feature=cat)
            preds.append(m.predict(X[test]))
        assert np.array_equal(preds[0], preds[1]), name
        out.append({"name": name, "params": {k: v for k, v in params.items() if k not in COMMON},
                    "test_rows": test.tolist(), "predictions": preds[0].tolist()})
    return out


if __name__ == "__main__":
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "lgbm_large.json")
    with open(path, "w") as f:
        json.dump(generate(), f)
    print(f"wrote {path}")
