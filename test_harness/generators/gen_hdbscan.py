"""Golden data for HDBSCAN (sklearn.cluster.HDBSCAN, algorithm="brute").

sklearn sorts MST edges with an unstable `np.argsort`, so datasets with many
tied mutual-reachability weights can flip between equally valid hierarchies.
For every case we therefore also run sklearn's own pipeline (same Cython
routines) with a *stable* edge sort, which is what anofox-ml does. Those
labels/probabilities are the strict reference (`labels`, `probabilities`);
the public-API result is stored as `labels_public` and only compared by
adjusted Rand index.
"""

import json
import os

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan._linkage import (
    make_single_linkage,
    mst_from_mutual_reachability,
)
from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.cluster._hdbscan._tree import tree_to_labels
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances


def _stable_labels(X, min_samples, min_cluster_size, method, allow_single):
    mr = mutual_reachability_graph(pairwise_distances(X), min_samples=min_samples)
    mst = mst_from_mutual_reachability(mr)
    mst = mst[np.argsort(mst["distance"], kind="stable")]
    labels, probs = tree_to_labels(
        make_single_linkage(mst), min_cluster_size, method, allow_single
    )
    return labels, probs


def generate():
    rng = np.random.RandomState(0)
    specs = [
        # name, centers, std, n, n_noise, min_samples, min_cluster_size, method, single
        ("three_blobs_noise", 3, 0.6, 150, 15, 5, 10, "eom", False),
        ("four_blobs_varied", 4, [0.3, 0.8, 1.2, 0.5], 240, 20, 4, 8, "eom", False),
        ("two_blobs_ms1", 2, 0.7, 100, 5, 1, 5, "eom", False),
        ("three_blobs_leaf", 3, 0.6, 150, 10, 5, 10, "leaf", False),
        ("one_blob_single", 1, 1.0, 80, 0, 5, 10, "eom", True),
    ]
    cases = []
    for name, centers, std, n, n_noise, ms, mcs, method, single in specs:
        X, _ = make_blobs(
            n_samples=n, centers=centers, cluster_std=std, random_state=rng.randint(1000)
        )
        if n_noise:
            lo, hi = X.min(axis=0) - 3, X.max(axis=0) + 3
            X = np.vstack([X, rng.uniform(lo, hi, size=(n_noise, X.shape[1]))])
        model = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            algorithm="brute",
            cluster_selection_method=method,
            allow_single_cluster=single,
            copy=True,
        ).fit(X)
        stable, stable_probs = _stable_labels(X, ms, mcs, method, single)
        cases.append(
            {
                "name": name,
                "X": X.tolist(),
                "min_samples": ms,
                "min_cluster_size": mcs,
                "cluster_selection_method": method,
                "allow_single_cluster": single,
                "labels": stable.tolist(),
                "probabilities": stable_probs.tolist(),
                "n_clusters": int(stable.max() + 1),
                "labels_public": model.labels_.tolist(),
            }
        )
    return cases


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "crates", "anofox-ml", "tests", "golden_data", "hdbscan.json",
    )
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
