"""Golden data for LgbmRegressor with linear trees (#208) against LightGBM
4.7.0 (`linear_tree=True`): ridge leaves on the numeric features of each
leaf's path, missing values, a categorical feature, bagging, Poisson.

`deterministic=True, force_row_wise=True`, one thread.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np

COMMON = dict(deterministic=True, force_row_wise=True, verbose=-1, n_jobs=1)


def nulls(a):
    return np.where(np.isnan(a), None, a).tolist()


def generate():
    rng = np.random.default_rng(208)
    n = 600
    X = np.column_stack([
        rng.normal(size=n),
        rng.uniform(-2, 2, size=n),
        rng.integers(0, 6, size=n).astype(float),
        rng.normal(size=n),
    ])
    # A trend the constant leaves cannot follow.
    y = 2.0 * X[:, 0] + np.sin(2 * X[:, 1]) + 0.3 * X[:, 2] + rng.normal(scale=0.3, size=n)
    Xn = X.copy()
    Xn[rng.choice(n, 40, replace=False), 0] = np.nan
    Xn[rng.choice(n, 30, replace=False), 3] = np.nan
    X_test = np.vstack([X[::6], np.column_stack([np.linspace(-4, 4, 9), np.zeros(9),
                                                 np.full(9, 2.0), np.ones(9)])])
    Xn_test = Xn[::6]
    base = dict(n_estimators=20, learning_rate=0.2, num_leaves=8, min_child_samples=20,
                linear_tree=True, **COMMON)
    out = []

    def add(name, params, X, y, X_test, sw=None, cat=None):
        m = lgb.LGBMRegressor(**params).fit(X, y, sample_weight=sw,
                                            categorical_feature=cat or "auto")
        out.append({"name": name, "params": {k: v for k, v in params.items() if k not in COMMON},
                    "categorical": cat or [], "X": nulls(X), "y": y.tolist(),
                    "sample_weight": None if sw is None else sw.tolist(),
                    "X_test": nulls(X_test), "predictions": m.predict(X_test).tolist()})

    add("linear", base, X, y, X_test)
    add("linear_lambda", dict(base, linear_lambda=2.0), X, y, X_test)
    add("linear_missing", base, Xn, y, Xn_test)
    add("linear_categorical", base, X, y, X_test, cat=[2])
    add("linear_bagging", dict(base, subsample=0.8, subsample_freq=1, random_state=3), X, y,
        X_test)
    add("linear_weighted", base, X, y, X_test, sw=rng.uniform(0.3, 2.0, size=n))
    add("linear_poisson", dict(base, objective="poisson"), X, np.round(np.exp(0.4 * y / 3)),
        X_test)
    return out


if __name__ == "__main__":
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "lgbm_linear.json")
    with open(path, "w") as f:
        json.dump(generate(), f)
    print(f"wrote {path}")
