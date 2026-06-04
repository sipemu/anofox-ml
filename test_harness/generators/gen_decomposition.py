"""Golden data for KernelPCA / NMF (sklearn.decomposition).

KernelPCA / NMF outputs are sign-flippable and / or basis-rotatable, so we
compare reconstruction quality and rank-1 information rather than direct
element-wise equality.
"""

import numpy as np
from sklearn.decomposition import NMF, KernelPCA


def generate():
    cases = []
    rng = np.random.default_rng(0)

    # KernelPCA RBF
    X = rng.standard_normal((40, 4))
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.5, fit_inverse_transform=False)
    T = kpca.fit_transform(X)
    cases.append({
        "name": "kpca_rbf",
        "X": X.tolist(),
        "gamma": 0.5,
        "n_components": 2,
        "sklearn_eigenvalues": kpca.eigenvalues_.tolist(),
        "sklearn_transformed_abs": np.abs(T).tolist(),
    })

    # NMF on non-negative data
    W_true = rng.uniform(0.1, 1.0, size=(30, 3))
    H_true = rng.uniform(0.1, 1.0, size=(3, 6))
    X_nmf = W_true @ H_true + 0.01 * rng.standard_normal((30, 6))
    X_nmf = np.clip(X_nmf, 0.01, None)
    model = NMF(n_components=3, init="random", solver="mu", max_iter=400, random_state=0)
    W = model.fit_transform(X_nmf)
    cases.append({
        "name": "nmf_3",
        "X": X_nmf.tolist(),
        "n_components": 3,
        "sklearn_reconstruction_err": float(model.reconstruction_err_),
    })

    return cases


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "decomposition.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
