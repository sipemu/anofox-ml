"""Golden data for RANSACRegressor / TheilSenRegressor.

We don't try for exact agreement (RANSAC's random sampling differs from ours,
TheilSen depends on which subset enumeration is used). Instead we verify
both implementations recover the inlier model on a contaminated dataset.
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor


def generate():
    rng = np.random.default_rng(0)
    n_in = 100
    x = np.linspace(-5, 5, n_in).reshape(-1, 1)
    y_in = 2.0 * x.ravel() + 1.0 + 0.1 * rng.standard_normal(n_in)

    n_out = 20
    x_out = rng.uniform(-5, 5, n_out).reshape(-1, 1)
    y_out = rng.uniform(15, 30, n_out)

    X = np.vstack([x, x_out])
    y = np.concatenate([y_in, y_out])
    perm = rng.permutation(len(y))
    X = X[perm]; y = y[perm]

    cases = []
    rr = RANSACRegressor(min_samples=2, residual_threshold=0.5, max_trials=200, random_state=0)
    rr.fit(X, y)
    cases.append({
        "name": "ransac_line",
        "X": X.tolist(),
        "y": y.tolist(),
        "sklearn_slope": float(rr.estimator_.coef_[0]),
        "sklearn_intercept": float(rr.estimator_.intercept_),
    })

    ts = TheilSenRegressor(random_state=0, max_subpopulation=2000)
    ts.fit(X, y)
    cases.append({
        "name": "theil_sen_line",
        "X": X.tolist(),
        "y": y.tolist(),
        "sklearn_slope": float(ts.coef_[0]),
        "sklearn_intercept": float(ts.intercept_),
    })

    return cases


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "robust.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
