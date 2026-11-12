"""Golden data for GraphicalLasso and GraphicalLassoCV (#259,
scikit-learn 1.8.0, mode="cd"). The coordinate descent is ported with its
screening and stopping rules, so the fits match to rounding, including a
fit that stops at max_iter."""

import warnings

import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.datasets import make_sparse_spd_matrix


def _data(seed, p, n):
    prec = make_sparse_spd_matrix(p, alpha=0.9, random_state=seed)
    rng = np.random.RandomState(seed)
    return rng.multivariate_normal(np.zeros(p), np.linalg.inv(prec), size=n) + 1.5


def generate():
    X = _data(0, 6, 40)
    single = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 0.3, "max_iter": 20},
                   {"alpha": 0.05, "assume_centered": True}, {"alpha": 0.0}]:
            est = GraphicalLasso(**kw).fit(X)
            single.append({"params": kw, "X": X.tolist(),
                           "location": est.location_.tolist(),
                           "covariance": est.covariance_.tolist(),
                           "precision": est.precision_.tolist(),
                           "costs": np.asarray(est.costs_, dtype=float).reshape(-1, 2).tolist(),
                           "n_iter": int(est.n_iter_)})
        Xc = _data(1, 5, 50)
        cv = []
        for kw in [{}, {"alphas": [0.5, 0.1, 0.02], "cv": 3},
                   {"alphas": 3, "n_refinements": 2, "cv": 4}]:
            est = GraphicalLassoCV(**kw).fit(Xc)
            res = est.cv_results_
            k = sum(1 for key in res if key.startswith("split"))
            cv.append({"params": kw, "X": Xc.tolist(), "alpha": float(est.alpha_),
                       "precision": est.precision_.tolist(),
                       "alphas": res["alphas"].tolist(),
                       "scores": np.column_stack([res[f"split{i}_test_score"]
                                                  for i in range(k)]).tolist()})
    return {"single": single, "cv": cv}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "graphical_lasso.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
