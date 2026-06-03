"""Golden data for TransformedTargetRegressor (sklearn.compose.TransformedTargetRegressor).

We use log/exp on a positive-target dataset to validate that wrapping a Ridge
regressor produces the same predictions as sklearn's TransformedTargetRegressor
under the same setup.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge


def generate():
    cases = []

    rng = np.random.default_rng(0)
    n, d = 50, 4
    X = rng.standard_normal((n, d))
    # Positive target with a multiplicative structure.
    log_y = X @ np.array([1.0, -0.5, 0.3, 0.8]) + 2.0
    y = np.exp(log_y) + rng.uniform(0.1, 0.5, size=n)

    inner = Ridge(alpha=0.01, solver="cholesky")
    tt = TransformedTargetRegressor(regressor=inner, func=np.log, inverse_func=np.exp)
    tt.fit(X, y)
    predictions = tt.predict(X)

    cases.append({
        "name": "ridge_log_exp",
        "X": X.tolist(),
        "y": y.tolist(),
        "alpha": 0.01,
        "predictions": predictions.tolist(),
    })

    return cases


if __name__ == "__main__":
    import json, os, sys
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "transformed_target.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
