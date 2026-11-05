"""Golden data for HistGradientBoostingRegressor losses (#127) (scikit-learn 1.8.0).

Every case is deterministic: early stopping is off and no subsampling is
used, so predictions should match to float32-gradient precision.
"""

import os

# sklearn's OpenMP threads thrash on these tiny fits (minutes instead of
# seconds on many-core machines); one thread gives the same results.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def generate():
    rng = np.random.default_rng(127)
    n = 300
    X = np.column_stack([
        rng.normal(size=n),
        rng.integers(0, 6, size=n).astype(float),
        rng.uniform(-2, 2, size=n),
    ])
    X[rng.choice(n, 15, replace=False), 2] = np.nan
    eta = 0.6 * X[:, 0] - 0.2 * X[:, 1] + 0.4 * np.nan_to_num(X[:, 2])
    mu = np.exp(eta + 0.5)
    targets = {
        "squared_error": eta + rng.normal(scale=0.3, size=n),
        "absolute_error": eta + rng.standard_t(3, size=n) * 0.3,
        "quantile": eta + rng.normal(scale=0.5, size=n),
        "poisson": rng.poisson(mu).astype(float),
        "gamma": rng.gamma(2.0, mu / 2.0),
    }
    weights = rng.uniform(0.2, 3.0, size=n)
    X_test = X[::7]

    out = []
    for loss, y in targets.items():
        for weighted in (False, True):
            params = dict(loss=loss, max_iter=25, learning_rate=0.2, max_leaf_nodes=8,
                          min_samples_leaf=10, l2_regularization=0.5, early_stopping=False)
            if loss == "quantile":
                params["quantile"] = 0.8
            sw = weights if weighted else None
            m = HistGradientBoostingRegressor(**params).fit(X, y, sample_weight=sw)
            out.append({
                "name": f"{loss}{'_weighted' if weighted else ''}",
                "params": params,
                "X": np.where(np.isnan(X), None, X).tolist(),
                "y": y.tolist(),
                "sample_weight": None if sw is None else sw.tolist(),
                "X_test": np.where(np.isnan(X_test), None, X_test).tolist(),
                "predictions": m.predict(X_test).tolist(),
            })
    return out


if __name__ == "__main__":
    import json
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "hist_gb_parity.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
