"""Golden data for LgbmRegressor / LgbmClassifier against LightGBM 4.7.0 (#127).

Deterministic settings only (no bagging, GOSS or feature fraction, whose
draws come from LightGBM's own generators): `deterministic=True,
force_row_wise=True`, one thread.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np


COMMON = dict(deterministic=True, force_row_wise=True, verbose=-1, n_jobs=1)


def nulls(a):
    return np.where(np.isnan(a), None, a).tolist()


def generate():
    rng = np.random.default_rng(4600)
    n = 400
    X = np.column_stack([
        rng.normal(size=n),
        rng.uniform(-2, 2, size=n),
        rng.integers(0, 8, size=n).astype(float),
        rng.normal(size=n),
    ])
    X[rng.choice(n, 25, replace=False), 3] = np.nan
    eta = np.sin(2 * X[:, 0]) + 0.6 * X[:, 1] - 0.2 * X[:, 2] + 0.3 * np.nan_to_num(X[:, 3])
    weights = rng.uniform(0.2, 3.0, size=n)
    mu = np.exp(0.5 * eta)
    y_reg = eta + rng.normal(scale=0.4, size=n)
    targets = {
        "regression": y_reg,
        "regression_l1": eta + rng.standard_t(3, size=n) * 0.3,
        "huber": y_reg + (rng.uniform(size=n) < 0.05) * 8.0,
        "quantile": y_reg,
        "poisson": rng.poisson(mu).astype(float),
        "gamma": rng.gamma(2.0, mu / 2.0),
    }
    base = dict(n_estimators=25, learning_rate=0.2, num_leaves=8, **COMMON)
    X_test = X[::5]
    out = []

    def add(name, kind, params, y, sw=None, fit_kw=None, extra=None):
        cls = lgb.LGBMRegressor if kind == "regressor" else lgb.LGBMClassifier
        m = cls(**params).fit(X, y, sample_weight=sw, **(fit_kw or {}))
        pred = m.predict(X_test) if kind == "regressor" else m.predict_proba(X_test)
        case = {"name": name, "kind": kind,
                "params": {k: v for k, v in params.items() if k not in COMMON},
                "X": nulls(X), "y": y.tolist(),
                "sample_weight": None if sw is None else sw.tolist(),
                "X_test": nulls(X_test), "predictions": pred.tolist()}
        case.update(extra or {})
        out.append(case)

    for obj, y in targets.items():
        params = dict(base, objective=obj)
        if obj == "quantile":
            params["alpha"] = 0.7
        add(obj, "regressor", params, y)
        add(f"{obj}_weighted", "regressor", params, y, sw=weights)
    # LightGBM's default min_child_samples=20 with uneven hessians: its
    # count limit uses counts estimated from the hessians.
    add("gamma_min_child_20", "regressor", dict(base, objective="gamma", min_child_samples=20),
        targets["gamma"])
    add("regression_l2_l1_reg", "regressor",
        dict(base, reg_lambda=2.0, reg_alpha=0.5, min_split_gain=0.01, min_child_samples=5),
        y_reg)
    add("regression_monotone", "regressor",
        dict(base, monotone_constraints=[1, 1, -1, 0]), y_reg)
    add("regression_categorical", "regressor", dict(base, min_child_samples=5), y_reg,
        fit_kw=dict(categorical_feature=[2]))
    y_bin = (eta + rng.logistic(size=n) > 0).astype(float)
    y_multi = np.digitize(eta + rng.logistic(size=n) * 0.5, [-0.5, 0.7]).astype(float)
    add("binary", "classifier", dict(base), y_bin)
    add("binary_weighted", "classifier", dict(base), y_bin, sw=weights)
    add("binary_balanced", "classifier", dict(base, class_weight="balanced"), y_bin)
    add("multiclass", "classifier", dict(base), y_multi)
    add("multiclass_weighted", "classifier", dict(base, min_child_samples=5), y_multi,
        sw=weights)
    return out


if __name__ == "__main__":
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "lgbm_parity.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {path}")
