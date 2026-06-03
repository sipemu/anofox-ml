"""Golden data for LDA / QDA (sklearn.discriminant_analysis)."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


def generate():
    cases = []

    # LDA: 3-class problem with shared covariance assumption.
    rng = np.random.default_rng(0)
    n_per = 60
    means = [np.array([0.0, 0.0]), np.array([3.0, 0.0]), np.array([0.0, 3.0])]
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    X_list, y_list = [], []
    for k, m in enumerate(means):
        X_list.append(rng.standard_normal((n_per, 2)) @ L.T + m)
        y_list.append(np.full(n_per, k, dtype=float))
    Xc = np.concatenate(X_list); yc = np.concatenate(y_list)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None)
    lda.fit(Xc, yc)
    cases.append({
        "name": "lda_3class",
        "X": Xc.tolist(),
        "y": yc.tolist(),
        "sklearn_predictions": lda.predict(Xc).astype(float).tolist(),
    })

    # QDA: clusters with different covariances.
    Xq, yq = make_classification(
        n_samples=200, n_features=4, n_informative=4, n_redundant=0,
        n_clusters_per_class=1, class_sep=1.5, random_state=0,
    )
    yq = yq.astype(float)
    qda = QuadraticDiscriminantAnalysis(reg_param=0.0)
    qda.fit(Xq, yq)
    cases.append({
        "name": "qda_binary",
        "X": Xq.tolist(),
        "y": yq.tolist(),
        "sklearn_predictions": qda.predict(Xq).astype(float).tolist(),
    })

    return cases


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "discriminant.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
