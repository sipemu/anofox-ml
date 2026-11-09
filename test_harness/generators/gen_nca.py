"""Golden data for NeighborhoodComponentsAnalysis (#113) against
scikit-learn 1.8.0. Each init (LDA, PCA, identity) is run with sklearn's
defaults and to convergence; the L-BFGS iterates differ from SciPy's, so the
Rust test compares the objective (expected correctly classified rows) the
two reach, since NCA's map is only identified up to an orthogonal factor and
grows without bound once the classes separate."""

import json
import os

import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis


def generate():
    r = np.random.RandomState(3)
    centers = np.array([[0, 0, 0, 0], [2.0, 0.5, 0, 0], [0, 2.0, 0.5, 0]])
    X = np.round(np.vstack([c + r.randn(15, 4) * [1.0, 1.0, 3.0, 3.0] for c in centers]), 3)
    y = np.repeat([0, 1, 2], 15)
    mask = y[:, None] == y[None, :]
    cases = []
    for params in [{"n_components": 2}, {"n_components": 3}, {},
                   {"n_components": 2, "max_iter": 2000, "tol": 1e-12},
                   {"n_components": 3, "max_iter": 2000, "tol": 1e-12}]:
        m = NeighborhoodComponentsAnalysis(random_state=0, **params).fit(X, y)
        loss = -m._loss_grad_lbfgs(m.components_.ravel(), X, mask, -1.0)[0]
        cases.append({"params": params, "components": m.components_.tolist(), "loss": loss,
                      "n_iter": int(m.n_iter_), "transform": m.transform(X[:5]).tolist()})
    return {"X": X.tolist(), "y": y.tolist(), "cases": cases}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "nca.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
