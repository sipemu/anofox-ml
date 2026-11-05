"""Golden data for the gap batch 2 (#113): GenericUnivariateSelect and its
modes (SelectPercentile / SelectFpr / SelectFdr / SelectFwe / SelectKBest
ties) and TfidfTransformer (scikit-learn 1.8.0)."""

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import (
    GenericUnivariateSelect, SelectKBest, chi2, f_classif, f_regression)


def _selection():
    rng = np.random.default_rng(7)
    n, d = 120, 12
    X = rng.normal(size=(n, d))
    y_cls = (X[:, 0] + 0.5 * X[:, 3] + 0.3 * rng.normal(size=n) > 0).astype(float)
    y_reg = 2 * X[:, 1] - X[:, 5] + 0.2 * X[:, 7] + rng.normal(size=n)
    X_pos = np.abs(X) * 3
    out = []
    for score_name, func, XX, y in [("FClassif", f_classif, X, y_cls),
                                    ("FRegression", f_regression, X, y_reg),
                                    ("Chi2", chi2, X_pos, y_cls)]:
        for mode, param in [("percentile", 10), ("percentile", 33), ("percentile", 100),
                            ("k_best", 3), ("fpr", 0.05), ("fdr", 0.05), ("fdr", 1e-6),
                            ("fwe", 0.05)]:
            sel = GenericUnivariateSelect(func, mode=mode, param=param).fit(XX, y)
            out.append({"name": f"{score_name}_{mode}_{param}", "scoring": score_name,
                        "mode": mode, "param": param, "X": XX.tolist(), "y": y.tolist(),
                        "support": np.flatnonzero(sel.get_support()).tolist()})
    # Ties: identical columns; sklearn's SelectKBest keeps the higher indices.
    Xt = np.column_stack([X[:, 0], X[:, 0], X[:, 0], X[:, 4]])
    sel = SelectKBest(f_classif, k=2).fit(Xt, y_cls)
    out.append({"name": "kbest_ties", "scoring": "FClassif", "mode": "k_best", "param": 2,
                "X": Xt.tolist(), "y": y_cls.tolist(),
                "support": np.flatnonzero(sel.get_support()).tolist()})
    return out


def _tfidf():
    rng = np.random.default_rng(8)
    X = rng.poisson(0.8, size=(15, 7)).astype(float)
    X[:, 6] = 0.0  # a term in no document
    X[3] = 0.0     # an empty document
    out = []
    for norm in ("l2", "l1", None):
        for use_idf in (True, False):
            for smooth in (True, False):
                for sub in (False, True):
                    if not smooth and use_idf:
                        # df = 0 divides by zero without smoothing.
                        XX = X[:, :6]
                    else:
                        XX = X
                    t = TfidfTransformer(norm=norm, use_idf=use_idf, smooth_idf=smooth,
                                         sublinear_tf=sub).fit(XX)
                    out.append({"name": f"{norm}_{use_idf}_{smooth}_{sub}",
                                "norm": norm, "use_idf": use_idf, "smooth_idf": smooth,
                                "sublinear_tf": sub, "X": XX.tolist(),
                                "idf": t.idf_.tolist() if use_idf else None,
                                "transform": t.transform(XX).toarray().tolist()})
    return out


def generate():
    return {"selection": _selection(), "tfidf": _tfidf()}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "univariate.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
