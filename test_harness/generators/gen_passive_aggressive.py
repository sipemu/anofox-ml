"""Golden data for PassiveAggressive Classifier/Regressor.

sklearn's PA uses its own RNG order, so exact prediction match is unlikely.
We aim for accuracy parity within a small band on synthetic problems.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.metrics import accuracy_score, r2_score


def generate():
    cases = []

    # Classifier
    Xc, yc = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=0, random_state=0, class_sep=2.0,
    )
    yc = yc.astype(float)
    clf = PassiveAggressiveClassifier(C=1.0, max_iter=200, tol=1e-3, random_state=0)
    clf.fit(Xc, yc)
    cases.append({
        "name": "pa_classifier",
        "type": "classifier",
        "X": Xc.tolist(),
        "y": yc.tolist(),
        "C": 1.0,
        "sklearn_accuracy": float(accuracy_score(yc, clf.predict(Xc))),
    })

    # Regressor
    Xr, yr = make_regression(n_samples=200, n_features=10, noise=0.5, random_state=0)
    # Scale y so that epsilon=0.1 is meaningful.
    yr = yr / yr.std()
    reg = PassiveAggressiveRegressor(C=1.0, epsilon=0.1, max_iter=500, tol=1e-3, random_state=0)
    reg.fit(Xr, yr)
    cases.append({
        "name": "pa_regressor",
        "type": "regressor",
        "X": Xr.tolist(),
        "y": yr.tolist(),
        "C": 1.0,
        "epsilon": 0.1,
        "sklearn_r2": float(r2_score(yr, reg.predict(Xr))),
    })

    return cases


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "passive_aggressive.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
