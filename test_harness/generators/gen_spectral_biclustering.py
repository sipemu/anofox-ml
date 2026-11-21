"""Golden data for SpectralBiclustering (#261, scikit-learn 1.8.0): row and
column labels for int random_state (numpy RandomState draws in the
randomized SVD and in every KMeans' k-means++), for the three
normalizations."""

from sklearn.cluster import SpectralBiclustering
from sklearn.datasets import make_checkerboard


def generate():
    cases = []
    for seed, k, shape, kw in [
        (0, 3, (30, 20), {}),
        (1, (4, 3), (36, 24), {"method": "scale", "n_init": 3}),
        (2, (2, 3), (18, 15), {"method": "log", "n_components": 4, "n_best": 2}),
        (3, 2, (20, 16), {"n_svd_vecs": 5}),
    ]:
        X, _, _ = make_checkerboard(shape, k, noise=5, shuffle=True, random_state=seed)
        est = SpectralBiclustering(k, random_state=seed, **kw).fit(X)
        cases.append({"params": {"n_clusters": list(k) if isinstance(k, tuple) else k,
                                 "random_state": seed, **kw},
                      "X": X.tolist(), "row_labels": est.row_labels_.tolist(),
                      "column_labels": est.column_labels_.tolist()})
    return {"cases": cases}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "anofox-ml", "tests",
                       "golden_data", "spectral_biclustering.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
