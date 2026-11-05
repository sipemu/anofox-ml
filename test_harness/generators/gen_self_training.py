"""Golden data for SelfTrainingClassifier (#113, batch 8) over GaussianNB
(scikit-learn 1.8.0)."""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import SelfTrainingClassifier


def generate():
    rng = np.random.default_rng(81)
    n = 120
    yt = np.arange(n) % 3
    X = rng.normal(size=(n, 2)) * 0.9 + np.array([[0, 0], [2.0, 0.4], [0.8, 2.0]])[yt]
    y = yt.astype(float)
    y[rng.random(n) < 0.85] = -1
    y[:3] = [0, 1, 2]
    out = []
    for name, kw in [("threshold", dict(threshold=0.75)), ("threshold_high", dict(threshold=0.95, max_iter=3)),
                     ("k_best", dict(criterion="k_best", k_best=7)), ("until_done", dict(threshold=0.6, max_iter=None))]:
        m = SelfTrainingClassifier(GaussianNB(), **kw).fit(X, y)
        out.append({"name": name, "params": kw, "X": X.tolist(), "y": y.tolist(),
                    "transduction": m.transduction_.astype(float).tolist(),
                    "labeled_iter": m.labeled_iter_.astype(float).tolist(), "n_iter": int(m.n_iter_),
                    "termination": m.termination_condition_, "predict": m.predict(X).astype(float).tolist(),
                    "predict_proba": m.predict_proba(X).tolist()})
    return out


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "self_training.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
