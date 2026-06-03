"""Golden data for RFE / SequentialFeatureSelector (sklearn.feature_selection)."""

import numpy as np
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import Ridge


def generate():
    rng = np.random.default_rng(7)
    n, d = 100, 8
    X = rng.standard_normal((n, d))
    true = np.zeros(d)
    true[0] = 3.0
    true[2] = 2.0
    true[5] = 1.5
    y = X @ true + 0.05 * rng.standard_normal(n)

    rfe = RFE(Ridge(alpha=0.01), n_features_to_select=3, step=1)
    rfe.fit(X, y)
    sfs = SequentialFeatureSelector(
        Ridge(alpha=0.01), n_features_to_select=3, direction="forward", cv=3,
    )
    sfs.fit(X, y)

    return [{
        "name": "rfe_ridge_3",
        "X": X.tolist(),
        "y": y.tolist(),
        "n_features_to_select": 3,
        "sklearn_rfe_support": rfe.support_.astype(bool).tolist(),
        "sklearn_sfs_support": sfs.get_support().astype(bool).tolist(),
        "expected_features": [0, 2, 5],
    }]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "rfe.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
