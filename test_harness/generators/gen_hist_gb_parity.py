"""Golden data for HistGradientBoostingRegressor losses (#127) (scikit-learn 1.8.0).

Every case is deterministic: early stopping is off and no subsampling is
used, so predictions should match to float32-gradient precision.
"""

import os

# sklearn's OpenMP threads thrash on these tiny fits (minutes instead of
# seconds on many-core machines); one thread gives the same results.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def generate():
    rng = np.random.default_rng(127)
    n = 300
    X = np.column_stack([
        rng.normal(size=n),
        rng.integers(0, 6, size=n).astype(float),
        rng.uniform(-2, 2, size=n),
    ])
    X[rng.choice(n, 15, replace=False), 2] = np.nan
    eta = 0.6 * X[:, 0] - 0.2 * X[:, 1] + 0.4 * np.nan_to_num(X[:, 2])
    mu = np.exp(eta + 0.5)
    targets = {
        "squared_error": eta + rng.normal(scale=0.3, size=n),
        "absolute_error": eta + rng.standard_t(3, size=n) * 0.3,
        "quantile": eta + rng.normal(scale=0.5, size=n),
        "poisson": rng.poisson(mu).astype(float),
        "gamma": rng.gamma(2.0, mu / 2.0),
    }
    weights = rng.uniform(0.2, 3.0, size=n)
    X_test = X[::7]

    out = []
    for loss, y in targets.items():
        for weighted in (False, True):
            params = dict(loss=loss, max_iter=25, learning_rate=0.2, max_leaf_nodes=8,
                          min_samples_leaf=10, l2_regularization=0.5, early_stopping=False)
            if loss == "quantile":
                params["quantile"] = 0.8
            sw = weights if weighted else None
            m = HistGradientBoostingRegressor(**params).fit(X, y, sample_weight=sw)
            out.append({
                "name": f"{loss}{'_weighted' if weighted else ''}",
                "params": params,
                "X": np.where(np.isnan(X), None, X).tolist(),
                "y": y.tolist(),
                "sample_weight": None if sw is None else sw.tolist(),
                "X_test": np.where(np.isnan(X_test), None, X_test).tolist(),
                "predictions": m.predict(X_test).tolist(),
            })
    out += categorical_cases()
    out += constraint_cases()
    return out


def nulls(a):
    return np.where(np.isnan(a), None, a).tolist()


def categorical_cases():
    """Categorical feature 1 (non-integer and negative codes, NaNs, a rare
    category, and an unseen category in X_test) next to numeric ones."""
    rng = np.random.default_rng(1270)
    n = 400
    codes = np.array([-1.0, 0.0, 2.5, 3.0, 7.0, 10.0, 11.0, 100.0])
    effect = np.array([1.2, -0.8, 0.3, 2.0, -1.5, 0.0, 0.9, -0.4])
    k = rng.choice(len(codes), size=n, p=[.15, .15, .15, .15, .15, .12, .1, .03])
    cat = codes[k].copy()
    cat[rng.choice(n, 20, replace=False)] = np.nan
    X = np.column_stack([rng.normal(size=n), cat, rng.uniform(-2, 2, size=n)])
    eta = effect[k] * np.where(np.isnan(cat), 0.0, 1.0) + 0.5 * X[:, 0] - 0.3 * X[:, 2]
    X_test = X[::9].copy()
    X_test[::4, 1] = 55.0  # unseen category
    base = dict(max_iter=20, learning_rate=0.2, max_leaf_nodes=8, min_samples_leaf=10,
                l2_regularization=0.5, early_stopping=False, categorical_features=[1])
    out = []
    regression = {
        "squared_error": eta + rng.normal(scale=0.3, size=n),
        "absolute_error": eta + rng.standard_t(3, size=n) * 0.3,
        "poisson": rng.poisson(np.exp(0.5 * eta)).astype(float),
    }
    for loss, y in regression.items():
        params = dict(base, loss=loss)
        m = HistGradientBoostingRegressor(**params).fit(X, y)
        out.append({"name": f"categorical_{loss}", "kind": "regressor", "params": params,
                    "X": nulls(X), "y": y.tolist(), "sample_weight": None,
                    "X_test": nulls(X_test), "predictions": m.predict(X_test).tolist()})
    classes = {
        "binary": (eta + rng.logistic(size=n) > 0).astype(float),
        "multiclass": np.digitize(eta + rng.logistic(size=n) * 0.5, [-0.5, 0.7]).astype(float),
    }
    for name, y in classes.items():
        m = HistGradientBoostingClassifier(**base).fit(X, y)
        out.append({"name": f"categorical_{name}", "kind": "classifier", "params": base,
                    "X": nulls(X), "y": y.tolist(), "sample_weight": None,
                    "X_test": nulls(X_test), "predictions": m.predict_proba(X_test).tolist()})
    return out


def constraint_cases():
    """monotonic_cst and interaction_cst on 4 numeric features (NaNs in a
    constrained one)."""
    rng = np.random.default_rng(1271)
    n = 400
    X = rng.normal(size=(n, 4))
    eta = np.sin(2 * X[:, 0]) + 0.8 * X[:, 1] - 0.5 * X[:, 2] * X[:, 3] + 0.3 * X[:, 0] * X[:, 1]
    X[rng.choice(n, 20, replace=False), 1] = np.nan
    X_test = X[::9]
    weights = rng.uniform(0.2, 3.0, size=n)
    base = dict(max_iter=20, learning_rate=0.2, max_leaf_nodes=8, min_samples_leaf=10,
                l2_regularization=0.5, early_stopping=False)
    y_reg = eta + rng.normal(scale=0.3, size=n)
    y_count = rng.poisson(np.exp(0.4 * eta)).astype(float)
    y_bin = (eta + rng.logistic(size=n) > 0).astype(float)
    y_multi = np.digitize(eta + rng.logistic(size=n) * 0.5, [-0.5, 0.7]).astype(float)
    mono = [1, 1, -1, 0]
    cases = [
        ("monotonic_squared_error", "regressor", y_reg, None, dict(monotonic_cst=mono)),
        ("monotonic_squared_error_weighted", "regressor", y_reg, weights,
         dict(monotonic_cst=mono)),
        ("monotonic_poisson", "regressor", y_count, None,
         dict(loss="poisson", monotonic_cst=mono)),
        ("monotonic_binary", "classifier", y_bin, None, dict(monotonic_cst=mono)),
        ("interaction_groups", "regressor", y_reg, None, dict(interaction_cst=[[0, 1], [2]])),
        ("interaction_pairwise", "regressor", y_reg, None, dict(interaction_cst="pairwise")),
        ("interaction_none", "regressor", y_reg, None, dict(interaction_cst="no_interactions")),
        ("interaction_multiclass", "classifier", y_multi, None,
         dict(interaction_cst=[[0, 1], [1, 2], [3]])),
        ("monotonic_and_interaction", "regressor", y_reg, None,
         dict(monotonic_cst=mono, interaction_cst=[[0, 1], [2, 3]])),
    ]
    out = []
    for name, kind, y, sw, extra in cases:
        params = dict(base, **extra)
        if kind == "regressor":
            m = HistGradientBoostingRegressor(**params).fit(X, y, sample_weight=sw)
            pred = m.predict(X_test)
        else:
            m = HistGradientBoostingClassifier(**params).fit(X, y, sample_weight=sw)
            pred = m.predict_proba(X_test)
        out.append({"name": name, "kind": kind, "params": params, "X": nulls(X),
                    "y": y.tolist(), "sample_weight": None if sw is None else sw.tolist(),
                    "X_test": nulls(X_test), "predictions": pred.tolist()})
    return out


if __name__ == "__main__":
    import json
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "hist_gb_parity.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
