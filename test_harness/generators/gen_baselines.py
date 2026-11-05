"""Golden data for the gap batch 1 (#113): DummyClassifier / DummyRegressor,
NearestCentroid, RidgeClassifier and Perceptron (scikit-learn 1.8.0)."""

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Perceptron, RidgeClassifier
from sklearn.neighbors import NearestCentroid


def _data(seed, n=90, d=4, k=3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (k, d))
    y = np.arange(n) % k
    X = centers[y] + rng.normal(0, 1.5, (n, d))
    return X, y.astype(float)


def _dummy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    y_cls = np.array([2, 0, 1, 2, 2, 1, 0, 2] * 5)
    y_reg = rng.normal(5.0, 2.0, 40)
    w = rng.uniform(0.2, 3.0, 40)
    out = []
    for strategy, kw in [("most_frequent", {}), ("prior", {}), ("constant", {"constant": 1})]:
        for weighted in (False, True):
            m = DummyClassifier(strategy=strategy, **kw).fit(X, y_cls, w if weighted else None)
            out.append({"name": f"dummy_clf_{strategy}_{'w' if weighted else 'u'}",
                        "kind": "classifier", "strategy": strategy, "constant": kw.get("constant"),
                        "X": X.tolist(), "y": y_cls.astype(float).tolist(),
                        "sample_weight": w.tolist() if weighted else None,
                        "predict": m.predict(X).astype(float).tolist(),
                        "predict_proba": m.predict_proba(X).tolist(),
                        "class_prior": m.class_prior_.tolist()})
    for strategy, kw in [("mean", {}), ("median", {}), ("quantile", {"quantile": 0.3}),
                         ("quantile", {"quantile": 0.0}), ("constant", {"constant": -2.5})]:
        for weighted in (False, True):
            m = DummyRegressor(strategy=strategy, **kw).fit(X, y_reg, w if weighted else None)
            out.append({"name": f"dummy_reg_{strategy}_{kw}_{'w' if weighted else 'u'}",
                        "kind": "regressor", "strategy": strategy,
                        "quantile": kw.get("quantile"), "constant": kw.get("constant"),
                        "X": X.tolist(), "y": y_reg.tolist(),
                        "sample_weight": w.tolist() if weighted else None,
                        "constant_": float(np.ravel(m.constant_)[0])})
    return out


def _nearest_centroid():
    X, y = _data(2)
    Xt, _ = _data(3, n=30)
    out = []
    cases = [
        ("euclid", dict()),
        ("manhattan", dict(metric="manhattan")),
        ("shrink", dict(shrink_threshold=0.5)),
        ("empirical", dict(priors="empirical")),
        ("given", dict(priors=[0.6, 0.3, 0.1])),
    ]
    # Unbalanced classes for the empirical priors to matter.
    Xu, yu = X[:75], y[:75].copy()
    yu[:20] = 0.0
    for name, kw in cases:
        Xf, yf = (Xu, yu) if name in ("empirical", "given") else (X, y)
        m = NearestCentroid(**kw).fit(Xf, yf)
        case = {"name": name, "params": kw, "X": Xf.tolist(), "y": yf.tolist(),
                "X_test": Xt.tolist(), "centroids": m.centroids_.tolist(),
                "predict": m.predict(Xt).tolist()}
        if kw.get("metric", "euclidean") == "euclidean":
            case["predict_proba"] = m.predict_proba(Xt).tolist()
            case["decision_function"] = m.decision_function(Xt).tolist()
        out.append(case)
    return out


def _ridge_classifier():
    X, y = _data(4)
    yb = (y == 1).astype(float)
    w = np.random.default_rng(5).uniform(0.5, 2.0, len(y))
    out = []
    for name, yy, kw, sw in [
        ("multi", y, dict(alpha=1.0), None),
        ("multi_alpha", y, dict(alpha=25.0), None),
        ("binary", yb, dict(alpha=0.5), None),
        ("multi_balanced", y[:70], dict(alpha=1.0, class_weight="balanced"), None),
        ("multi_weighted", y, dict(alpha=2.0), w),
        ("no_intercept", y, dict(alpha=1.0, fit_intercept=False), None),
    ]:
        Xf = X[: len(yy)]
        m = RidgeClassifier(**kw).fit(Xf, yy, sample_weight=sw)
        df = m.decision_function(X)
        out.append({"name": name, "params": kw, "X": Xf.tolist(), "y": yy.tolist(),
                    "sample_weight": None if sw is None else sw.tolist(),
                    "X_test": X.tolist(), "coef": np.atleast_2d(m.coef_).tolist(),
                    "intercept": np.broadcast_to(m.intercept_, (np.atleast_2d(m.coef_).shape[0],)).tolist(),
                    "decision_function": np.atleast_2d(df.T).T.reshape(len(X), -1).tolist(),
                    "predict": m.predict(X).tolist()})
    return out


def _perceptron():
    X, y = _data(6)
    yb = (y == 2).astype(float)
    out = []
    for name, yy, kw in [
        ("binary", yb, dict()),
        ("multi", y, dict()),
        ("l2", y, dict(penalty="l2", alpha=0.01)),
        ("eta", yb, dict(eta0=0.1, max_iter=50, tol=None)),
    ]:
        m = Perceptron(shuffle=False, **kw).fit(X, yy)
        out.append({"name": name, "params": kw, "X": X.tolist(), "y": yy.tolist(),
                    "coef": m.coef_.tolist(), "intercept": m.intercept_.tolist(),
                    "n_iter": int(m.n_iter_), "predict": m.predict(X).tolist()})
    return out


def generate():
    return {"dummy": _dummy(), "nearest_centroid": _nearest_centroid(),
            "ridge_classifier": _ridge_classifier(), "perceptron": _perceptron()}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "baselines.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
