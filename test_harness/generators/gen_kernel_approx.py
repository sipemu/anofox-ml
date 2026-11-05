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
    for name, kw in [("rbf", dict(kernel="rbf", gamma=0.4)), ("rbf_default", dict(kernel="rbf")),
                     ("laplacian", dict(kernel="laplacian", gamma=0.2)),
                     ("poly", dict(kernel="poly", degree=2, gamma=0.5, coef0=1.0)),
                     ("linear", dict(kernel="linear"))]:
        m = Nystroem(n_components=20, random_state=0, **kw).fit(X)
        Z, Zy = m.transform(X), m.transform(Y)
        nys.append({"name": name, "params": kw, "X": X.tolist(), "X_test": Y.tolist(),
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
