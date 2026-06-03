"""Golden data for TweedieRegressor / GammaRegressor (sklearn.linear_model).

We compare predictions (and coefficients) for a couple of power settings.
"""

import numpy as np
from sklearn.linear_model import TweedieRegressor, GammaRegressor


def generate():
    cases = []

    rng = np.random.default_rng(19)
    n, d = 80, 3
    X = rng.standard_normal((n, d))

    # --- Tweedie power=1.5 (compound Poisson-Gamma) -------------------------
    eta = X @ np.array([0.5, -0.2, 0.3]) + 0.5
    mu = np.exp(eta)
    # Tweedie sampling is hard; use mu + noise truncated at 0 as proxy data.
    y_tw = np.maximum(mu + 0.3 * rng.standard_normal(n), 0.01)

    m1 = TweedieRegressor(power=1.5, alpha=0.5, link="log", max_iter=200, tol=1e-6)
    m1.fit(X, y_tw)
    cases.append({
        "name": "tweedie_p1p5",
        "X": X.tolist(),
        "y": y_tw.tolist(),
        "power": 1.5,
        # sklearn alpha is per-sample-normalized; for the anofox-backed solver
        # we scale by n to match (lambda = n * alpha).
        "sklearn_alpha": 0.5,
        "anofox_lambda": 0.5 * n,
        "coef": m1.coef_.tolist(),
        "intercept": float(m1.intercept_),
        "predictions": m1.predict(X).tolist(),
    })

    # --- Gamma (power=2) ----------------------------------------------------
    eta2 = X @ np.array([0.3, 0.8, -0.4]) + 0.0
    mu2 = np.exp(eta2)
    y_g = np.maximum(mu2 + 0.1 * rng.standard_normal(n), 0.05)

    m2 = GammaRegressor(alpha=0.1, max_iter=200, tol=1e-6)
    m2.fit(X, y_g)
    cases.append({
        "name": "gamma",
        "X": X.tolist(),
        "y": y_g.tolist(),
        "power": 2.0,
        "sklearn_alpha": 0.1,
        "anofox_lambda": 0.1 * n,
        "coef": m2.coef_.tolist(),
        "intercept": float(m2.intercept_),
        "predictions": m2.predict(X).tolist(),
    })

    return cases


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "tweedie.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
