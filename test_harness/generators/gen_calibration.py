"""Golden data for the registry's CalibratedClassifierCV (#125) against
scikit-learn 1.8.0: predict_proba of `CalibratedClassifierCV(base,
method=..., ensemble=..., cv=5)` over GaussianNB (predict_proba scores) and
LinearDiscriminantAnalysis (decision_function scores), binary and 3-class.

The data is deterministic and rebuilt by the Rust test
(`crates/anofox-ml-registry/tests/calibration.rs`).
"""

import json
import os

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


def data():
    n = 60
    X = np.array([[((i * 7 + j * 3 + i * j) % 13) / 6 - 1 + 0.01 * ((i * (j + 5)) % 7)
                   for j in range(3)] for i in range(n)])
    y2 = (X[:, 0] + 0.5 * X[:, 2] + 0.3 * np.sin(7 * np.arange(n)) > 0).astype(float)
    y3 = np.where(X[:, 0] + X[:, 1] < -0.4, 0, np.where(X[:, 0] - X[:, 2] < 0.3, 1, 2)).astype(float)
    Q = np.array([[-0.9, -0.6, -0.4], [-0.2, 0.3, 0.7], [0.4, -0.6, 0.7], [0.95, 0.3, -0.4]])
    return X, y2, y3, Q


def generate():
    X, y2, y3, Q = data()
    cases = []
    for base_name, base in [("GaussianNB", GaussianNB()),
                            ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis())]:
        for classes, y in [(2, y2), (3, y3)]:
            for method in ["sigmoid", "isotonic"]:
                for ensemble in [True, False]:
                    m = CalibratedClassifierCV(base, method=method, ensemble=ensemble, cv=5).fit(X, y)
                    cases.append({"estimator": base_name, "classes": classes, "method": method,
                                  "ensemble": ensemble, "X": X.tolist(), "y": y.tolist(),
                                  "Q": Q.tolist(), "proba": m.predict_proba(Q).tolist()})
    return {"cases": cases}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "calibration.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
