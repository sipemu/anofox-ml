"""Golden data for NearestNeighbors and RadiusNeighborsTransformer (#260,
scikit-learn 1.8.0): kneighbors distances / indices and the dense radius
graphs. The data has no distance ties."""

import numpy as np
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsTransformer


def generate():
    rng = np.random.RandomState(260)
    X = rng.randn(30, 3)
    Xq = rng.randn(6, 3)
    knn = []
    for kw in [{"n_neighbors": 4}, {"n_neighbors": 3, "metric": "manhattan"},
               {"n_neighbors": 5, "algorithm": "brute"}]:
        est = NearestNeighbors(**kw).fit(X)
        dist, ind = est.kneighbors(Xq)
        knn.append({"params": kw, "X": X.tolist(), "X_query": Xq.tolist(),
                    "distances": dist.tolist(), "indices": ind.tolist()})
    radius = []
    for kw in [{"radius": 1.0}, {"radius": 1.5, "mode": "connectivity"},
               {"radius": 2.0, "metric": "manhattan"}]:
        est = RadiusNeighborsTransformer(**kw).fit(X)
        radius.append({"params": kw, "X": X.tolist(), "X_query": Xq.tolist(),
                       "transform": est.transform(Xq).toarray().tolist()})
    return {"knn": knn, "radius": radius}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "nearest_neighbors.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
