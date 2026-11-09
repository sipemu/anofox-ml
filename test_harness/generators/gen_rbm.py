"""Golden data for BernoulliRBM (#113) against scikit-learn 1.8.0: seeded
fits draw from numpy's RandomState in sklearn's order, so they are
reproduced exactly."""

import json
import os

import numpy as np
from sklearn.neural_network import BernoulliRBM


def generate():
    r = np.random.RandomState(0)
    X = (r.rand(23, 6) > 0.6).astype(float)
    X[:, 0] = r.rand(23)  # one continuous column in [0, 1]
    cases = []
    for params in [{"n_components": 4, "random_state": 0},
                   {"n_components": 3, "batch_size": 7, "n_iter": 5, "learning_rate": 0.05,
                    "random_state": 11}]:
        m = BernoulliRBM(**params).fit(X)
        g = m.gibbs(X[:4])
        cases.append({"params": params, "components": m.components_.tolist(),
                      "intercept_hidden": m.intercept_hidden_.tolist(),
                      "intercept_visible": m.intercept_visible_.tolist(),
                      "transform": m.transform(X[:5]).tolist(),
                      "score_samples": m.score_samples(X[:5]).tolist(),
                      "gibbs": g.astype(float).tolist()})
    pf = BernoulliRBM(n_components=3, batch_size=5, random_state=4)
    for rows in (slice(0, 5), slice(5, 12), slice(12, 23)):
        pf.partial_fit(X[rows])
    partial = {"components": pf.components_.tolist(),
               "intercept_visible": pf.intercept_visible_.tolist()}
    return {"X": X.tolist(), "cases": cases, "partial_fit": partial}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "rbm.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
