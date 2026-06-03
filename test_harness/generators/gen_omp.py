"""Golden data for OrthogonalMatchingPursuit (sklearn.linear_model)."""

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit


def generate():
    rng = np.random.default_rng(3)
    n, d = 80, 8
    X = rng.standard_normal((n, d))
    # True coefs: sparse.
    true = np.zeros(d)
    true[1] = 4.0
    true[3] = -2.5
    true[6] = 1.5
    y = X @ true + 0.05 * rng.standard_normal(n)

    m = OrthogonalMatchingPursuit(n_nonzero_coefs=3, fit_intercept=True)
    m.fit(X, y)
    return [{
        "name": "omp_3",
        "X": X.tolist(),
        "y": y.tolist(),
        "n_nonzero": 3,
        "sklearn_coef": m.coef_.tolist(),
        "sklearn_intercept": float(m.intercept_),
        "sklearn_predictions": m.predict(X).tolist(),
        "expected_active": [1, 3, 6],
    }]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "omp.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
