"""Golden data for StackingClassifier (sklearn.ensemble.StackingClassifier).

Behavioral parity: both implementations should achieve high accuracy on a
well-separated binary problem. We don't pursue exact agreement because
sklearn defaults to predict_proba + StratifiedKFold while our implementation
uses hard predictions + sequential KFold.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def generate():
    rng = np.random.default_rng(11)
    X, y = make_classification(
        n_samples=120, n_features=6, n_informative=4,
        n_redundant=0, random_state=11, class_sep=2.5,
    )
    # Interleave so non-stratified KFold sees both classes per fold.
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    m = min(len(idx0), len(idx1))
    order = np.empty(2 * m, dtype=int)
    order[0::2] = idx0[:m]
    order[1::2] = idx1[:m]
    X = X[order]; y = y[order]

    sc = StackingClassifier(
        estimators=[
            ("t1", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ("t2", DecisionTreeClassifier(max_depth=5, random_state=0)),
        ],
        final_estimator=DecisionTreeClassifier(max_depth=3, random_state=0),
        cv=2,
        stack_method="predict",
    )
    sc.fit(X, y)
    pred = sc.predict(X)

    return [{
        "name": "two_trees_meta_tree",
        "X": X.tolist(),
        "y": y.astype(float).tolist(),
        "sklearn_accuracy": float(accuracy_score(y, pred)),
        "sklearn_predictions": pred.astype(float).tolist(),
    }]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "stacking_classifier.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
