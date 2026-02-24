"""Generate golden data for feature selection tests (VarianceThreshold, mutual_info)."""

import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif


def generate():
    cases = []

    # VarianceThreshold tests
    np.random.seed(42)
    # Column 0: constant, Col 1: low variance, Col 2-3: high variance
    X = np.column_stack([
        np.ones(30),                     # constant
        np.random.randn(30) * 0.01,      # near-zero variance
        np.random.randn(30) * 5.0,       # high variance
        np.random.randn(30) * 10.0,      # highest variance
    ])

    for threshold in [0.0, 0.001, 1.0]:
        vt = VarianceThreshold(threshold=threshold)
        X_transformed = vt.fit_transform(X)
        variances = vt.variances_
        selected = vt.get_support(indices=True).tolist()

        cases.append({
            "name": f"variance_threshold_{threshold}",
            "algorithm": "VarianceThreshold",
            "X": X.tolist(),
            "threshold": threshold,
            "variances": variances.tolist(),
            "selected_indices": selected,
            "X_transformed_shape": list(X_transformed.shape),
        })

    # Mutual information tests
    np.random.seed(42)
    # Feature 0 is highly informative, feature 1 is moderately, feature 2 is noise
    X_mi = np.random.randn(100, 3)
    y_mi = (X_mi[:, 0] > 0).astype(float)  # label depends on feature 0

    mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)

    cases.append({
        "name": "mutual_info_binary",
        "algorithm": "MutualInformation",
        "X": X_mi.tolist(),
        "y": y_mi.tolist(),
        "mi_scores": mi_scores.tolist(),
        "n_features": 3,
    })

    return cases
