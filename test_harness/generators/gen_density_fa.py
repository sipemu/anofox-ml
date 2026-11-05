"""Golden data for the gap batch 6 (#113): KernelDensity (every kernel,
several dimensions, weights, scott / silverman) and FactorAnalysis
(svd_method="lapack", with and without rotation) (scikit-learn 1.8.0)."""

import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.neighbors import KernelDensity


def _kde():
    rng = np.random.default_rng(41)
    out = []
    for d in (1, 2, 3, 5):
        X = rng.normal(size=(40, d))
        Xt = rng.normal(size=(12, d)) * 1.3
        w = rng.uniform(0.2, 2.0, 40)
        for kernel in ("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"):
            for bw, weighted in ((0.8, False), (1.6, True), ("scott", False), ("silverman", True)):
                kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X, sample_weight=w if weighted else None)
                out.append({"name": f"d{d}_{kernel}_{bw}_{weighted}", "kernel": kernel,
                            "bandwidth": bw, "X": X.tolist(), "X_test": Xt.tolist(),
                            "sample_weight": w.tolist() if weighted else None,
                            "bandwidth_": float(kde.bandwidth_),
                            "score_samples": [None if np.isneginf(v) else float(v)
                                              for v in kde.score_samples(Xt)]})
    return out


def _fa():
    rng = np.random.default_rng(42)
    Z = rng.normal(size=(200, 2))
    W = rng.normal(size=(2, 6))
    X = Z @ W + rng.normal(scale=[0.3, 0.5, 0.2, 1.0, 0.4, 0.6], size=(200, 6)) + 2.0
    Xt = rng.normal(size=(10, 6)) + 2.0
    out = []
    for k, rot in ((2, None), (3, None), (None, None), (2, "varimax"), (3, "quartimax")):
        fa = FactorAnalysis(n_components=k, svd_method="lapack", rotation=rot).fit(X)
        out.append({"name": f"{k}_{rot}", "n_components": k, "rotation": rot,
                    "X": X.tolist(), "X_test": Xt.tolist(),
                    "components": fa.components_.tolist(), "noise_variance": fa.noise_variance_.tolist(),
                    "loglike": list(map(float, fa.loglike_)), "n_iter": int(fa.n_iter_),
                    "covariance": fa.get_covariance().tolist(), "precision": fa.get_precision().tolist(),
                    "score_samples": fa.score_samples(Xt).tolist(), "transform": fa.transform(Xt).tolist()})
    return out


def generate():
    return {"kde": _kde(), "fa": _fa()}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "density_fa.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
