"""Golden data for the gap batch 5 (#113): KernelCenterer, Nystroem (all
rows as components, compared through the feature Gram matrix, which does
not depend on the random basis order) and johnson_lindenstrauss_min_dim
(scikit-learn 1.8.0)."""

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.random_projection import johnson_lindenstrauss_min_dim


def generate():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(20, 3))
    Y = rng.normal(size=(6, 3))
    K, Ky = rbf_kernel(X, X, gamma=0.3), rbf_kernel(Y, X, gamma=0.3)
    kc = KernelCenterer().fit(K)
    centerer = {"K": K.tolist(), "K_test": Ky.tolist(), "fit": kc.transform(K).tolist(),
                "test": kc.transform(Ky).tolist()}
    nys = []
    # The polynomial and linear kernels get enough features for a full-rank
    # K_BB: near-zero eigenvalues would amplify machine-level noise.
    X6, Y6 = rng.normal(size=(20, 6)), rng.normal(size=(6, 6))
    X25, Y25 = rng.normal(size=(20, 25)), rng.normal(size=(6, 25))
    for name, kw, XX, YY in [("rbf", dict(kernel="rbf", gamma=0.4), X, Y),
                             ("rbf_default", dict(kernel="rbf"), X, Y),
                             ("laplacian", dict(kernel="laplacian", gamma=0.2), X, Y),
                             ("poly", dict(kernel="poly", degree=2, gamma=0.5, coef0=1.0), X6, Y6),
                             ("linear", dict(kernel="linear"), X25, Y25)]:
        m = Nystroem(n_components=20, random_state=0, **kw).fit(XX)
        Z, Zy = m.transform(XX), m.transform(YY)
        nys.append({"name": name, "params": kw, "X": XX.tolist(), "X_test": YY.tolist(),
                    "gram": (Z @ Z.T).tolist(), "gram_test": (Zy @ Z.T).tolist()})
    jl = [{"n": n, "eps": e, "dim": int(johnson_lindenstrauss_min_dim(n, eps=e))}
          for n in (10, 1000, 1_000_000) for e in (0.1, 0.3, 0.5, 0.9)]
    return {"centerer": centerer, "nystroem": nys, "jl": jl}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "kernel_approx.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
