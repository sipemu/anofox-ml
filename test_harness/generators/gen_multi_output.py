"""Golden data for MultiOutputRegressor (sklearn.multioutput.MultiOutputRegressor).

Wraps Ridge per output column and validates predictions match sklearn
element-wise to ~1e-8 (Ridge is closed-form).
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


def generate():
    rng = np.random.default_rng(13)
    n, d = 50, 4
    X = rng.standard_normal((n, d))
    # Three correlated outputs from different linear combinations.
    coefs = np.array([
        [1.0, -0.5,  0.3, 0.0],
        [0.0,  2.0, -1.0, 0.5],
        [0.5,  0.5,  0.5, 0.5],
    ])
    Y = X @ coefs.T + 0.05 * rng.standard_normal((n, 3))

    mor = MultiOutputRegressor(Ridge(alpha=0.5, solver="cholesky"))
    mor.fit(X, Y)
    pred = mor.predict(X)

    return [{
        "name": "ridge_3_outputs",
        "X": X.tolist(),
        "Y": Y.tolist(),
        "alpha": 0.5,
        "predictions": pred.tolist(),
    }]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "multi_output.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
