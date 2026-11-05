"""Golden data for LabelPropagation / LabelSpreading (#113, batch 8)
(scikit-learn 1.8.0)."""

import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading


def generate():
    rng = np.random.default_rng(61)
    n = 90
    y_true = np.arange(n) % 3
    X = rng.normal(size=(n, 2)) * 0.6 + np.array([[0, 0], [2.5, 0.5], [1.0, 2.5]])[y_true]
    y = y_true.astype(float)
    y[rng.random(n) < 0.8] = -1.0
    y[:3] = [0.0, 1.0, 2.0]
    Xt = rng.normal(size=(12, 2)) + 1.0
    out = []
    for cls, name in ((LabelPropagation, "propagation"), (LabelSpreading, "spreading")):
        for kw in (dict(kernel="rbf"), dict(kernel="rbf", gamma=2.0), dict(kernel="knn", n_neighbors=7),
                   dict(kernel="knn", n_neighbors=4)):
            if name == "spreading":
                kw = {**kw, "alpha": 0.3}
            m = cls(**kw).fit(X, y)
            out.append({"name": f"{name}_{kw}", "variant": name, "params": kw, "X": X.tolist(),
                        "y": y.tolist(), "X_test": Xt.tolist(),
                        "label_distributions": m.label_distributions_.tolist(),
                        "transduction": m.transduction_.astype(float).tolist(),
                        "n_iter": int(m.n_iter_), "predict_proba": m.predict_proba(Xt).tolist(),
                        "predict": m.predict(Xt).astype(float).tolist()})
    return out


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "semi_supervised.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
