"""Golden data for SpectralCoclustering (#261, scikit-learn 1.8.0): row and
column labels for int random_state (numpy RandomState draws in the
randomized SVD and in KMeans' k-means++)."""

from sklearn.cluster import SpectralCoclustering
from sklearn.datasets import make_biclusters


def generate():
    cases = []
    for seed, k, shape, kw in [(0, 3, (30, 20), {}), (1, 4, (36, 24), {"n_init": 3}),
                               (2, 2, (15, 18), {"n_svd_vecs": 5})]:
        X, _, _ = make_biclusters(shape, k, noise=3, random_state=seed)
        est = SpectralCoclustering(k, random_state=seed, **kw).fit(X)
        cases.append({"params": {"n_clusters": k, "random_state": seed, **kw},
                      "X": X.tolist(), "row_labels": est.row_labels_.tolist(),
                      "column_labels": est.column_labels_.tolist()})
    return {"cases": cases}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "anofox-ml", "tests",
                       "golden_data", "spectral_coclustering.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
