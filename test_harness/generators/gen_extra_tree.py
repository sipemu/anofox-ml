"""Golden data for ExtraTreeRegressor / ExtraTreeClassifier and
RandomTreesEmbedding (#113) against scikit-learn 1.8.0: seeded trees drawn
by sklearn's `our_rand_r`, compared node for node through predictions and
leaf indices."""

import json
import os

import numpy as np
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor


def generate():
    r = np.random.RandomState(12)
    X = np.round(r.randn(70, 5), 3)
    X[:, 4] = np.round(X[:, 4])  # a coarse feature (ties, constants in small nodes)
    y = np.round(X[:, 0] * 2 - X[:, 1] + 0.3 * r.randn(70), 3)
    labels = np.digitize(y + r.randn(70) * 0.5, [-1.0, 1.0]).astype(float)
    w = np.round(r.uniform(0.2, 2.0, 70), 2)
    probe = np.round(r.randn(25, 5) * 1.3, 3)
    out = {"X": X.tolist(), "y": y.tolist(), "labels": labels.tolist(), "w": w.tolist(),
           "probe": probe.tolist(), "regressor": [], "classifier": [], "embedding": []}
    for params, weighted in [({"random_state": 0}, False),
                             ({"max_depth": 4, "min_samples_leaf": 3, "random_state": 5}, True),
                             ({"max_features": 2, "min_samples_split": 6, "random_state": 9},
                              False)]:
        m = ExtraTreeRegressor(**params).fit(X, y, sample_weight=w if weighted else None)
        out["regressor"].append({"params": params, "weighted": weighted,
                                 "predict": m.predict(probe).tolist(),
                                 "apply": m.apply(probe).tolist(),
                                 "node_count": int(m.tree_.node_count)})
    for params, weighted in [({"random_state": 1}, False),
                             ({"criterion": "entropy", "max_depth": 5, "random_state": 2}, True),
                             ({"max_features": None, "min_samples_leaf": 2, "random_state": 3},
                              False)]:
        m = ExtraTreeClassifier(**params).fit(X, labels, sample_weight=w if weighted else None)
        out["classifier"].append({"params": params, "weighted": weighted,
                                  "proba": m.predict_proba(probe).tolist(),
                                  "apply": m.apply(probe).tolist(),
                                  "node_count": int(m.tree_.node_count)})
    for params in [{"n_estimators": 5, "random_state": 0},
                   {"n_estimators": 3, "max_depth": 3, "min_samples_leaf": 4, "random_state": 7}]:
        m = RandomTreesEmbedding(sparse_output=False, **params).fit(X)
        out["embedding"].append({"params": params, "transform": m.transform(probe).tolist(),
                                 "apply": m.apply(probe).tolist()})
    return out


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "extra_tree.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
