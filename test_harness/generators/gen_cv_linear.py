"""Golden data for the gap batch 7 (#113): LogisticRegressionCV and
OrthogonalMatchingPursuitCV (scikit-learn 1.8.0). LogisticRegressionCV uses
a tight tol so that each fold's fit is converged, as ours is."""

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, OrthogonalMatchingPursuitCV

warnings.filterwarnings("ignore", category=FutureWarning)


def generate():
    rng = np.random.default_rng(51)
    out = {"logistic": [], "omp": []}
    X = rng.normal(size=(120, 4))
    yb = (X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=1.2, size=120) > 0).astype(float)
    ym = np.digitize(X[:, 0] + 0.5 * X[:, 2] + rng.normal(scale=0.8, size=120), [-0.6, 0.6]).astype(float)
    for name, y, kw in [("binary", yb, {}), ("multi", ym, {}), ("binary_logloss", yb, {"scoring": "neg_log_loss"}),
                        ("multi_cs", ym, {"Cs": [0.01, 0.1, 1.0, 10.0]}), ("binary_cv3", yb, {"cv": 3})]:
        m = LogisticRegressionCV(tol=1e-10, max_iter=100000, l1_ratios=(0.0,),
                                 use_legacy_attributes=False, **kw).fit(X, y)
        scores = np.asarray(m.scores_).reshape(-1, len(np.atleast_1d(m.Cs_)))
        out["logistic"].append({"name": name, "params": {k: v for k, v in kw.items()}, "X": X.tolist(),
                                "y": y.tolist(), "C": float(np.ravel(m.C_)[0]),
                                "scores": scores.tolist(), "predict": m.predict(X).tolist(),
                                "predict_proba": m.predict_proba(X).tolist()})
    Xo = rng.normal(size=(90, 25))
    true = np.zeros(25)
    true[[2, 7, 11, 19]] = [3.0, -2.0, 1.5, 1.0]
    yo = Xo @ true + rng.normal(scale=0.3, size=90) + 4.0
    for name, kw in [("default", {}), ("max_iter12", {"max_iter": 12}), ("no_intercept", {"fit_intercept": False}),
                     ("cv3", {"cv": 3})]:
        m = OrthogonalMatchingPursuitCV(**kw).fit(Xo, yo)
        out["omp"].append({"name": name, "params": kw, "X": Xo.tolist(), "y": yo.tolist(),
                           "n_nonzero_coefs": int(m.n_nonzero_coefs_), "coef": m.coef_.tolist(),
                           "intercept": float(m.intercept_), "predict": m.predict(Xo).tolist()})
    return out


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "cv_linear.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
