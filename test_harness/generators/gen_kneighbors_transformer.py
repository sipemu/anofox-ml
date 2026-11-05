"""Golden data for KNeighborsTransformer (#113) (scikit-learn 1.8.0)."""

import numpy as np
from sklearn.neighbors import KNeighborsTransformer


def generate():
    rng = np.random.default_rng(91)
    X, Xt = rng.normal(size=(30, 3)), rng.normal(size=(8, 3))
    out = []
    for mode in ("distance", "connectivity"):
        for k in (1, 5):
            m = KNeighborsTransformer(mode=mode, n_neighbors=k).fit(X)
            out.append({"name": f"{mode}_{k}", "mode": mode, "n_neighbors": k, "X": X.tolist(),
                        "X_test": Xt.tolist(), "train": m.transform(X).toarray().tolist(),
                        "test": m.transform(Xt).toarray().tolist()})
    return out


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "kneighbors_transformer.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
