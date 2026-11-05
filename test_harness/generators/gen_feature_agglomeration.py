"""Golden data for FeatureAgglomeration (#113, batch 8) (scikit-learn 1.8.0)."""

import numpy as np
from sklearn.cluster import FeatureAgglomeration


def generate():
    rng = np.random.default_rng(71)
    base = rng.normal(size=(50, 3))
    # 9 features in 3 noisy groups, plus 2 independent ones.
    X = np.column_stack([base[:, g] + 0.15 * rng.normal(size=50) for g in (0, 0, 1, 2, 1, 0, 2, 2, 1)]
                        + [rng.normal(size=50), rng.normal(size=50)])
    out = []
    for linkage in ("ward", "complete", "average", "single"):
        for k in (3, 5):
            m = FeatureAgglomeration(n_clusters=k, linkage=linkage).fit(X)
            Xr = m.transform(X)
            out.append({"name": f"{linkage}_{k}", "linkage": linkage, "n_clusters": k, "X": X.tolist(),
                        "labels": m.labels_.tolist(), "transform": Xr.tolist(),
                        "inverse": m.inverse_transform(Xr).tolist()})
    return out


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "feature_agglomeration.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
