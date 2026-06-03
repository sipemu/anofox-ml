"""Golden data for KernelRidge (sklearn.kernel_ridge.KernelRidge).

Three kernels: linear, rbf, polynomial. Predictions must match sklearn to ~1e-8
since both implementations solve the same Cholesky system.
"""

import numpy as np
from sklearn.kernel_ridge import KernelRidge


def case(name, X, y, kernel, alpha, **kw):
    kr = KernelRidge(alpha=alpha, kernel=kernel, **kw)
    kr.fit(X, y)
    pred = kr.predict(X)
    return {
        "name": name,
        "X": X.tolist(),
        "y": y.tolist(),
        "alpha": alpha,
        "kernel": kernel,
        "gamma": kw.get("gamma", None),
        "degree": kw.get("degree", None),
        "coef0": kw.get("coef0", None),
        "dual_coef": kr.dual_coef_.tolist(),
        "predictions": pred.tolist(),
    }


def generate():
    rng = np.random.default_rng(7)
    n = 30
    X = rng.standard_normal((n, 3))
    y = (
        X[:, 0] * 2.0
        + np.sin(X[:, 1] * 3.0)
        + 0.5 * X[:, 2] ** 2
        + 0.05 * rng.standard_normal(n)
    )

    return [
        case("linear", X, y, "linear", alpha=0.5),
        case("rbf_gamma0p5", X, y, "rbf", alpha=0.1, gamma=0.5),
        case("poly_deg3", X, y, "polynomial", alpha=1.0, degree=3, gamma=1.0, coef0=1.0),
    ]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "kernel_ridge.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
