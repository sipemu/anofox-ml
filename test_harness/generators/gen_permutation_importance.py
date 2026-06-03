"""Golden data for permutation_importance (sklearn.inspection.permutation_importance).

We compare the ranking (argsort descending) of feature importances. Permutation
importance is inherently stochastic so we use many repeats (50) to make the
ranking stable, and assert exact match on rank order, plus that the relative
magnitudes of the top feature are within a generous tolerance.
"""

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def generate():
    rng = np.random.default_rng(2024)
    n, d = 200, 5
    X = rng.standard_normal((n, d))
    # True coefficients: x0 is dominant, x2 is half, x4 is small noise.
    true_coef = np.array([5.0, 0.0, 2.0, 0.0, 0.3])
    y = X @ true_coef + 0.1 * rng.standard_normal(n)

    model = Ridge(alpha=1e-3, solver="cholesky").fit(X, y)

    res = permutation_importance(
        model, X, y, n_repeats=50, random_state=0, scoring="r2"
    )

    return [{
        "name": "ridge_permutation_5feat",
        "X": X.tolist(),
        "y": y.tolist(),
        "alpha": 1e-3,
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "sklearn_importances_mean": res.importances_mean.tolist(),
        "sklearn_rank_desc": np.argsort(-res.importances_mean).tolist(),
        "baseline_r2": float(r2_score(y, model.predict(X))),
    }]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "permutation_importance.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
