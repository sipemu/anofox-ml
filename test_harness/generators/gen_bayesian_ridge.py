"""Golden data for BayesianRidge / ARDRegression (sklearn.linear_model)."""

import numpy as np
from sklearn.linear_model import ARDRegression, BayesianRidge


def generate():
    rng = np.random.default_rng(42)
    cases = []

    # BayesianRidge
    n, d = 80, 4
    X = rng.standard_normal((n, d))
    y = X @ np.array([1.5, -0.7, 0.0, 2.0]) + 0.1 * rng.standard_normal(n)
    m = BayesianRidge(max_iter=300, tol=1e-3, fit_intercept=True)
    m.fit(X, y)
    pred, std = m.predict(X, return_std=True)
    cases.append({
        "name": "bayesian_ridge",
        "X": X.tolist(),
        "y": y.tolist(),
        "sklearn_coef": m.coef_.tolist(),
        "sklearn_intercept": float(m.intercept_),
        "sklearn_predictions": pred.tolist(),
        "sklearn_std": std.tolist(),
    })

    # ARD: feature 1 and 2 are noise.
    n, d = 100, 5
    X = rng.standard_normal((n, d))
    # True coefs: most are zero.
    true = np.array([3.0, 0.0, 0.0, -1.5, 0.0])
    y = X @ true + 0.1 * rng.standard_normal(n)
    ar = ARDRegression(max_iter=300, tol=1e-3, threshold_lambda=1e4)
    ar.fit(X, y)
    cases.append({
        "name": "ard_sparse",
        "X": X.tolist(),
        "y": y.tolist(),
        "true_coef": true.tolist(),
        "sklearn_coef": ar.coef_.tolist(),
        "sklearn_intercept": float(ar.intercept_),
        "sklearn_predictions": ar.predict(X).tolist(),
    })

    return cases


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "bayesian_ridge.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
