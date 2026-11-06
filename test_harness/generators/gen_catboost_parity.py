"""Golden data for the CatBoost-style estimators against catboost 1.2.10 (#127).

CatBoost's randomness is switched off so that its models are reproducible:
`random_strength=0` (no score noise), `bootstrap_type='No'`,
`boosting_type='Plain'` and `has_time=True` (no permutations), one thread.
Quantization borders come from `Pool.quantize()` (all borders, not only the
ones a model uses).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import tempfile

import catboost as cb
import numpy as np

COMMON = dict(random_strength=0, bootstrap_type="No", boosting_type="Plain", has_time=True,
              thread_count=1, verbose=0, allow_writing_files=False)


def nulls(a):
    return np.where(np.isnan(a), None, a).tolist()


def borders_of(X, border_count):
    pool = cb.Pool(X)
    pool.quantize(border_count=border_count)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "b.tsv")
        pool.save_quantization_borders(path)
        rows = [line.split("\t") for line in open(path).read().splitlines() if line]
    out = [[] for _ in range(X.shape[1])]
    for row in rows:
        out[int(row[0])].append(float(row[1]))
    return out


def generate():
    rng = np.random.default_rng(127)
    n = 400
    X = np.column_stack([
        rng.normal(size=n),
        rng.uniform(-2, 2, size=n),
        np.round(rng.normal(size=n) * 3),       # few distinct values
        rng.exponential(size=n),
    ])
    X_nan = X.copy()
    X_nan[rng.choice(n, 30, replace=False), 1] = np.nan
    eta = np.sin(2 * X[:, 0]) + 0.6 * X[:, 1] - 0.2 * X[:, 2] + 0.3 * X[:, 3]
    y_reg = eta + rng.normal(scale=0.4, size=n)
    y_bin = (eta + rng.logistic(size=n) > 0).astype(float)
    weights = rng.uniform(0.2, 3.0, size=n)

    out = {"quantization": [], "models": []}
    for name, data, count in (("plain", X, 254), ("nan", X_nan, 254), ("few", X, 16)):
        out["quantization"].append({"name": name, "border_count": count, "X": nulls(data),
                                    "borders": borders_of(data, count)})

    base = dict(iterations=30, learning_rate=0.2, depth=4, l2_leaf_reg=3.0, **COMMON)

    def add(name, kind, params, Xd, y, sw=None):
        cls = cb.CatBoostRegressor if kind == "regressor" else cb.CatBoostClassifier
        m = cls(**params).fit(Xd, y, sample_weight=sw)
        X_test = Xd[::4]
        pred = m.predict(X_test) if kind == "regressor" else m.predict_proba(X_test)
        out["models"].append({
            "name": name, "kind": kind,
            "params": {k: v for k, v in params.items() if k not in COMMON},
            "X": nulls(Xd), "y": y.tolist(),
            "sample_weight": None if sw is None else sw.tolist(),
            "X_test": nulls(X_test), "predictions": pred.tolist(),
        })

    add("rmse_cosine", "regressor", base, X, y_reg)
    add("rmse_l2", "regressor", dict(base, score_function="L2"), X, y_reg)
    add("rmse_depth1", "regressor", dict(base, depth=1), X, y_reg)
    add("rmse_depth6", "regressor", dict(base, depth=6, iterations=15), X, y_reg)
    add("rmse_gradient", "regressor", dict(base, leaf_estimation_method="Gradient"), X, y_reg)
    add("rmse_l2_reg", "regressor", dict(base, l2_leaf_reg=0.5, learning_rate=0.5), X, y_reg)
    add("rmse_weighted", "regressor", base, X, y_reg, sw=weights)
    add("rmse_nan", "regressor", base, X_nan, y_reg)
    add("rmse_border_16", "regressor", dict(base, border_count=16), X, y_reg)
    add("rmse_no_average", "regressor", dict(base, boost_from_average=False), X, y_reg)
    add("logloss_cosine", "classifier", base, X, y_bin)
    add("logloss_l2", "classifier", dict(base, score_function="L2"), X, y_bin)
    add("logloss_weighted", "classifier", base, X, y_bin, sw=weights)
    add("logloss_average", "classifier", dict(base, boost_from_average=True), X, y_bin)
    return out


if __name__ == "__main__":
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "catboost_parity.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
