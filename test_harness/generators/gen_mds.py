"""Golden data for MDS (SMACOF, #260, scikit-learn 1.8.0): metric and
non-metric, random starts (numpy RandomState draws), the classical start
and a precomputed dissimilarity."""

import warnings

import numpy as np
from sklearn.manifold import MDS


def generate():
    X = np.random.RandomState(2600).randn(15, 3)
    cases = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in [{"random_state": 0, "n_init": 4, "init": "random"},
                   {"random_state": 2, "n_init": 2, "init": "random", "metric_mds": False},
                   {"init": "classical_mds", "n_init": 1, "n_components": 3},
                   {"random_state": 1, "n_init": 1, "init": "random", "max_iter": 5}]:
            est = MDS(**kw).fit(X)
            cases.append({"params": kw, "X": X.tolist(), "embedding": est.embedding_.tolist(),
                          "stress": float(est.stress_), "n_iter": int(est.n_iter_)})
        D = np.abs(X[:, :1] - X[:, :1].T) + np.abs(X[:, 1:2] - X[:, 1:2].T)
        est = MDS(metric="precomputed", random_state=0, n_init=4, init="random").fit(D)
        cases.append({"params": {"metric": "precomputed", "random_state": 0}, "X": D.tolist(),
                      "embedding": est.embedding_.tolist(), "stress": float(est.stress_),
                      "n_iter": int(est.n_iter_)})
    return {"cases": cases}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "mds.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
