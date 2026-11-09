"""Golden data for MiniBatchNMF (#113) against scikit-learn 1.8.0 (NNDSVDa
init, so no random draws)."""

import json
import os

import numpy as np
from sklearn.decomposition import MiniBatchNMF


def generate():
    r = np.random.RandomState(2)
    X = np.round(np.abs(r.randn(40, 2) @ np.abs(r.randn(2, 7)) + 0.1 * r.rand(40, 7)), 4)
    cases = []
    for params in [
        {"n_components": 2, "batch_size": 10, "random_state": 0},
        {"n_components": 3, "batch_size": 16, "max_iter": 50, "tol": 0.0,
         "max_no_improvement": None, "random_state": 0},
        {"n_components": 2, "batch_size": 8, "beta_loss": "kullback-leibler",
         "alpha_W": 0.01, "l1_ratio": 0.5, "random_state": 0},
        {"n_components": 2, "batch_size": 64, "forget_factor": 0.5, "random_state": 0},
    ]:
        m = MiniBatchNMF(**params)
        W = m.fit_transform(X)
        cases.append({"params": params, "components": m.components_.tolist(), "W": W.tolist(),
                      "n_iter": int(m.n_iter_), "n_steps": int(m.n_steps_),
                      "reconstruction_err": float(m.reconstruction_err_),
                      "transform": m.transform(X[:5]).tolist()})
    return {"X": X.tolist(), "cases": cases}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "minibatch_nmf.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
