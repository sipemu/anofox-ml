"""Golden data for IsolationForest / LocalOutlierFactor."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def generate():
    rng = np.random.default_rng(0)
    n_in = 200
    inliers = rng.standard_normal((n_in, 2)) * 0.5
    outliers = np.array([[5, 5], [-5, -5], [5, -5], [-5, 5], [6, 0]])
    X = np.vstack([inliers, outliers])
    y_true = np.concatenate([np.ones(n_in), -np.ones(len(outliers))])

    cases = []

    iso = IsolationForest(n_estimators=100, max_samples=128, contamination=5/len(X), random_state=0)
    iso.fit(X)
    cases.append({
        "name": "iso_forest",
        "X": X.tolist(),
        "y_true": y_true.tolist(),
        "sklearn_predictions": iso.predict(X).astype(float).tolist(),
    })

    lof = LocalOutlierFactor(n_neighbors=20, contamination=5/len(X))
    lof_preds = lof.fit_predict(X)
    cases.append({
        "name": "lof",
        "X": X.tolist(),
        "y_true": y_true.tolist(),
        "sklearn_predictions": lof_preds.astype(float).tolist(),
    })

    return cases


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "outlier.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
