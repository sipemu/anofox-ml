"""Golden data for LARS / LassoLars (sklearn.linear_model)."""

import numpy as np
from sklearn.linear_model import Lars, LassoLars


def generate():
    rng = np.random.default_rng(3)
    n, d = 60, 6
    X = rng.standard_normal((n, d))
    true = np.zeros(d)
    true[1] = 4.0
    true[3] = -2.5
    true[5] = 1.5
    y = X @ true + 0.05 * rng.standard_normal(n)

    lars = Lars(n_nonzero_coefs=3, fit_intercept=True)
    lars.fit(X, y)
    pred = lars.predict(X)
    sklearn_r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    lasso = LassoLars(alpha=0.05, fit_intercept=True, max_iter=200)
    lasso.fit(X, y)
    return [
        {
            "name": "lars_3",
            "X": X.tolist(),
            "y": y.tolist(),
            "n_nonzero": 3,
            "sklearn_coef": lars.coef_.tolist(),
            "sklearn_r2": sklearn_r2,
            "expected_active": [1, 3, 5],
        },
        {
            "name": "lasso_lars",
            "X": X.tolist(),
            "y": y.tolist(),
            "alpha": 0.05,
            "sklearn_coef": lasso.coef_.tolist(),
        },
    ]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "lars.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
