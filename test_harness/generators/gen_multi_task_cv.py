"""Golden data for MultiTaskLassoCV / MultiTaskElasticNetCV (#113) against
scikit-learn 1.8.0, solved to tol=1e-12 so that the converged paths are
compared (sklearn's block coordinate descent stops on the duality gap)."""

import json
import os

import numpy as np
from sklearn.linear_model import MultiTaskElasticNetCV, MultiTaskLassoCV

from gen_multi_task import generate as base_data


def generate():
    data = base_data()
    X, Y = np.array(data["X"]), np.array(data["Y"])
    tight = {"tol": 1e-12, "max_iter": 100000}
    cases = []
    for name, cls, params in [
        ("MultiTaskLassoCV", MultiTaskLassoCV, {"alphas": 30, "cv": 5}),
        ("MultiTaskLassoCV", MultiTaskLassoCV,
         {"alphas": [0.5, 0.1, 0.02, 0.004], "cv": 4, "fit_intercept": False}),
        ("MultiTaskElasticNetCV", MultiTaskElasticNetCV,
         {"alphas": 25, "l1_ratio": [0.2, 0.7], "cv": 4}),
        ("MultiTaskElasticNetCV", MultiTaskElasticNetCV,
         {"alphas": 20, "l1_ratio": 0.5, "cv": 3, "eps": 1e-2}),
    ]:
        m = cls(**tight, **params).fit(X, Y)
        mse = np.asarray(m.mse_path_)
        n_l1 = len(np.atleast_1d(params.get("l1_ratio", 1.0)))
        alphas = np.asarray(m.alphas_).reshape(n_l1, -1)
        cases.append({"kind": name, "params": params, "alpha": m.alpha_,
                      "l1_ratio": getattr(m, "l1_ratio_", 1.0),
                      "alphas": alphas.tolist(),
                      "mse_path": mse.reshape(n_l1, alphas.shape[1], -1).tolist(),
                      "coef": m.coef_.tolist(), "intercept": np.atleast_1d(m.intercept_).tolist(),
                      "predict": m.predict(X[:3]).tolist()})
    return {"X": X.tolist(), "Y": Y.tolist(), "cases": cases}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "multi_task_cv.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
