"""Golden data for PLSRegression (sklearn.cross_decomposition)."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression


def generate():
    rng = np.random.default_rng(7)
    n, d = 80, 6
    X = rng.standard_normal((n, d))
    # Highly collinear true coefficients.
    true = np.array([2.0, -1.5, 0.0, 0.5, 0.0, 1.0])
    y = X @ true + 0.05 * rng.standard_normal(n)

    pls = PLSRegression(n_components=3, scale=True, max_iter=500, tol=1e-6)
    pls.fit(X, y)
    return [{
        "name": "pls1_3comp",
        "X": X.tolist(),
        "y": y.tolist(),
        "n_components": 3,
        "sklearn_predictions": pls.predict(X).ravel().tolist(),
    }]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "pls.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
