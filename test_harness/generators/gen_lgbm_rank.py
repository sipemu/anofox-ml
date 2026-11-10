"""Golden data for LightGBM's ranking objectives (#207): `lambdarank`
(truncation, normalization, sigmoid, label gains) and `rank_xendcg`, with
query groups, against LightGBM 4.7.0's `LGBMRanker`.

`deterministic=True, force_row_wise=True`, one thread.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np

COMMON = dict(deterministic=True, force_row_wise=True, verbose=-1, n_jobs=1)


def generate():
    rng = np.random.default_rng(207)
    group = rng.integers(3, 25, size=50)
    n = int(group.sum())
    X = np.column_stack([rng.normal(size=n), rng.uniform(-2, 2, size=n), rng.normal(size=n),
                         rng.integers(0, 5, size=n).astype(float)])
    rel = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + rng.normal(scale=0.5, size=n)
    y = np.digitize(rel, [-0.5, 0.5, 1.5]).astype(float)
    X_test = X[::4]
    base = dict(n_estimators=15, learning_rate=0.1, num_leaves=8, min_child_samples=10,
                **COMMON)
    out = []

    def add(name, params, sw=None):
        m = lgb.LGBMRanker(**params).fit(X, y, group=group, sample_weight=sw)
        out.append({"name": name, "params": {k: v for k, v in params.items() if k not in COMMON},
                    "X": X.tolist(), "y": y.tolist(), "group": group.tolist(),
                    "sample_weight": None if sw is None else sw.tolist(),
                    "X_test": X_test.tolist(), "predictions": m.predict(X_test).tolist()})

    add("lambdarank", base)
    add("lambdarank_truncation_no_norm",
        dict(base, lambdarank_truncation_level=3, lambdarank_norm=False))
    add("lambdarank_sigmoid_gain", dict(base, sigmoid=2.0, label_gain=[0, 1, 3, 10]))
    add("lambdarank_weighted", base, sw=rng.uniform(0.5, 2.0, size=n))
    add("lambdarank_bagging", dict(base, subsample=0.8, subsample_freq=1, random_state=4))
    add("xendcg", dict(base, objective="rank_xendcg"))
    add("xendcg_seed", dict(base, objective="rank_xendcg", random_state=9))
    return out


if __name__ == "__main__":
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "lgbm_rank.json")
    with open(path, "w") as f:
        json.dump(generate(), f)
    print(f"wrote {path}")
