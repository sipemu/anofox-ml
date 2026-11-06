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
    out += early_stopping_cases()
    out += max_features_and_class_weight_cases()
    out += scoring_and_validation_cases()
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


def early_stopping_cases():
    """early_stopping=True with sklearn's validation split (float / int /
    None validation_fraction, stratified for classifiers); n_iter_ and the
    score curves are compared too."""
    rng = np.random.default_rng(1272)
    n = 500
    X = rng.normal(size=(n, 3))
    X[rng.choice(n, 25, replace=False), 2] = np.nan
    eta = np.sin(2 * X[:, 0]) + 0.6 * X[:, 1]
    weights = rng.uniform(0.2, 3.0, size=n)
    base = dict(max_iter=200, learning_rate=0.3, max_leaf_nodes=8, min_samples_leaf=10,
                early_stopping=True, n_iter_no_change=5, random_state=7)
    y_reg = eta + rng.normal(scale=0.5, size=n)
    cases = [
        ("es_squared_error", "regressor", y_reg, None, dict()),
        ("es_squared_error_int_fraction", "regressor", y_reg, None,
         dict(validation_fraction=60, random_state=123)),
        ("es_absolute_error_train_loss", "regressor", y_reg, None,
         dict(loss="absolute_error", validation_fraction=None, tol=1e-3)),
        ("es_poisson_weighted", "regressor", rng.poisson(np.exp(0.5 * eta)).astype(float),
         weights, dict(loss="poisson", validation_fraction=0.2)),
        ("es_quantile", "regressor", y_reg, None, dict(loss="quantile", quantile=0.3)),
        ("es_binary", "classifier", (eta + rng.logistic(size=n) > 0).astype(float), None,
         dict()),
        ("es_multiclass_weighted", "classifier",
         np.digitize(eta + rng.logistic(size=n) * 0.5, [-0.5, 0.7]).astype(float), weights,
         dict(validation_fraction=0.25, random_state=99)),
    ]
    out = []
    for name, kind, y, sw, extra in cases:
        params = dict(base, **extra)
        cls = HistGradientBoostingRegressor if kind == "regressor" else HistGradientBoostingClassifier
        m = cls(**params).fit(X, y, sample_weight=sw)
        pred = m.predict(X[::7]) if kind == "regressor" else m.predict_proba(X[::7])
        staged = m.staged_predict(X[::7]) if kind == "regressor" else m.staged_predict_proba(X[::7])
        out.append({"name": name, "kind": kind, "params": params, "X": nulls(X),
                    "y": y.tolist(), "sample_weight": None if sw is None else sw.tolist(),
                    "X_test": nulls(X[::7]), "predictions": pred.tolist(),
                    "n_iter": int(m.n_iter_), "train_score": m.train_score_.tolist(),
                    "validation_score": m.validation_score_.tolist(),
                    "staged": [st.tolist() for st in staged]})
    return out


def max_features_and_class_weight_cases():
    """max_features (sklearn's per-split feature subsampling, drawn from its
    default_rng) and class_weight."""
    rng = np.random.default_rng(1273)
    n = 400
    X = rng.normal(size=(n, 6))
    X[rng.choice(n, 20, replace=False), 3] = np.nan
    eta = np.sin(2 * X[:, 0]) + 0.6 * X[:, 1] - 0.4 * X[:, 2] + 0.3 * np.nan_to_num(X[:, 3])
    weights = rng.uniform(0.2, 3.0, size=n)
    base = dict(max_iter=20, learning_rate=0.2, max_leaf_nodes=8, min_samples_leaf=10,
                early_stopping=False)
    y_reg = eta + rng.normal(scale=0.3, size=n)
    y_bin = (eta + rng.logistic(size=n) > 0.8).astype(float)  # imbalanced
    y_multi = np.digitize(eta + rng.logistic(size=n) * 0.5, [-0.5, 1.2]).astype(float)
    cases = [
        ("max_features_regressor", "regressor", y_reg, None,
         dict(max_features=0.5, random_state=11)),
        ("max_features_early_stopping", "regressor", y_reg, weights,
         dict(max_features=0.3, random_state=4, early_stopping=True, max_iter=200,
              n_iter_no_change=5)),
        ("max_features_interactions", "regressor", y_reg, None,
         dict(max_features=0.6, random_state=2, interaction_cst=[[0, 1, 2], [3, 4, 5]])),
        ("max_features_binary", "classifier", y_bin, None,
         dict(max_features=0.5, random_state=21)),
        ("max_features_multiclass", "classifier", y_multi, None,
         dict(max_features=0.4, random_state=5)),
        ("class_weight_balanced", "classifier", y_bin, None, dict(class_weight="balanced")),
        ("class_weight_dict_weighted", "classifier", y_multi, weights,
         dict(class_weight={0: 2.0, 2: 0.5})),
        ("class_weight_early_stopping", "classifier", y_bin, None,
         dict(class_weight="balanced", early_stopping=True, max_iter=200, n_iter_no_change=5,
              random_state=8)),
    ]
    out = []
    for name, kind, y, sw, extra in cases:
        params = dict(base, **extra)
        cls = HistGradientBoostingRegressor if kind == "regressor" else HistGradientBoostingClassifier
        m = cls(**params).fit(X, y, sample_weight=sw)
        pred = m.predict(X[::7]) if kind == "regressor" else m.predict_proba(X[::7])
        if isinstance(params.get("class_weight"), dict):
            params["class_weight"] = [[k, v] for k, v in params["class_weight"].items()]
        case = {"name": name, "kind": kind, "params": params, "X": nulls(X), "y": y.tolist(),
                "sample_weight": None if sw is None else sw.tolist(),
                "X_test": nulls(X[::7]), "predictions": pred.tolist()}
        if params.get("early_stopping") is True:
            case.update(n_iter=int(m.n_iter_), train_score=m.train_score_.tolist(),
                        validation_score=m.validation_score_.tolist())
        out.append(case)
    return out


def formula_data(n):
    """Data both sides rebuild from integers (no transcendental functions,
    so the floats agree bit for bit)."""
    i = np.arange(n)
    x0 = ((i * 37) % 1000) / 100.0
    x1 = ((i * 91) % 713) / 71.3
    x2 = ((i * 13) % 7).astype(float)
    noise = ((i * 7919) % 200) / 100.0 - 1.0
    y = 0.5 * x0 - 0.05 * x0 * x0 + 0.3 * x1 - 0.2 * x2 + noise
    return np.column_stack([x0, x1, x2]), y


def scoring_and_validation_cases():
    """scoring other than "loss" (on sklearn's small trainset above 10 000
    rows) and explicit X_val / y_val."""
    rng = np.random.default_rng(1274)
    n = 400
    X = rng.normal(size=(n, 3))
    eta = np.sin(2 * X[:, 0]) + 0.6 * X[:, 1]
    y_reg = eta + rng.normal(scale=0.5, size=n)
    y_bin = (eta + rng.logistic(size=n) > 0).astype(float)
    y_multi = np.digitize(eta + rng.logistic(size=n) * 0.5, [-0.5, 0.7]).astype(float)
    weights = rng.uniform(0.2, 3.0, size=n)
    Xv = rng.normal(size=(120, 3))
    eta_v = np.sin(2 * Xv[:, 0]) + 0.6 * Xv[:, 1]
    yv_reg = eta_v + rng.normal(scale=0.5, size=120)
    yv_multi = np.digitize(eta_v + rng.logistic(size=120) * 0.5, [-0.5, 0.7]).astype(float)
    wv = rng.uniform(0.2, 3.0, size=120)
    base = dict(max_iter=200, learning_rate=0.3, max_leaf_nodes=8, min_samples_leaf=10,
                early_stopping=True, n_iter_no_change=5, random_state=13)
    cases = [
        ("score_default_regressor", "regressor", X, y_reg, None, dict(scoring=None), None),
        ("score_neg_mse_weighted", "regressor", X, y_reg, weights,
         dict(scoring="neg_mean_squared_error"), None),
        ("score_neg_mae_train", "regressor", X, y_reg, None,
         dict(scoring="neg_mean_absolute_error", validation_fraction=None, tol=1e-4), None),
        ("score_r2_poisson", "regressor", X, np.exp(0.3 * y_reg), None,
         dict(scoring="r2", loss="gamma"), None),
        ("score_accuracy_binary", "classifier", X, y_bin, None, dict(scoring="accuracy"), None),
        ("score_log_loss_multiclass", "classifier", X, y_multi, weights,
         dict(scoring="neg_log_loss"), None),
        ("score_default_multiclass", "classifier", X, y_multi, None, dict(scoring=None), None),
        ("x_val_regressor", "regressor", X, y_reg, weights, dict(), (Xv, yv_reg, wv)),
        ("x_val_classifier_scored", "classifier", X, y_multi, None,
         dict(scoring="neg_log_loss"), (Xv, yv_multi, None)),
    ]
    Xb, yb = formula_data(11500)
    cases += [
        ("score_small_trainset_regressor", "regressor", Xb, yb, None,
         dict(scoring="r2", max_iter=60, early_stopping="auto"), "formula"),
        ("score_small_trainset_classifier", "classifier", Xb, (yb > 1.0).astype(float), None,
         dict(scoring="accuracy", max_iter=60, early_stopping="auto"), "formula"),
    ]
    out = []
    for name, kind, Xc, y, sw, extra, val in cases:
        params = dict(base, **extra)
        cls = HistGradientBoostingRegressor if kind == "regressor" else HistGradientBoostingClassifier
        m = cls(**params)
        fit_kw = {}
        if isinstance(val, tuple):
            fit_kw = dict(X_val=val[0], y_val=val[1], sample_weight_val=val[2])
        m.fit(Xc, y, sample_weight=sw, **fit_kw)
        Xt = Xc[::7][:80]
        pred = m.predict(Xt) if kind == "regressor" else m.predict_proba(Xt)
        case = {"name": name, "kind": kind, "params": params,
                "sample_weight": None if sw is None else sw.tolist(),
                "X_test": nulls(Xt), "predictions": pred.tolist(),
                "n_iter": int(m.n_iter_), "train_score": m.train_score_.tolist(),
                "validation_score": m.validation_score_.tolist()}
        if val == "formula":
            case["formula_rows"] = len(y)
            case["formula_classify"] = kind == "classifier"
        else:
            case["X"] = nulls(Xc)
            case["y"] = y.tolist()
        if isinstance(val, tuple):
            case["X_val"] = nulls(val[0])
            case["y_val"] = val[1].tolist()
            case["sample_weight_val"] = None if val[2] is None else val[2].tolist()
        out.append(case)
    return out


if __name__ == "__main__":
    import json
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "hist_gb_parity.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
