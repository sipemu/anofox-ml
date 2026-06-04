"""Golden data for GaussianProcessRegressor (sklearn.gaussian_process)."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def generate():
    rng = np.random.default_rng(0)
    X = np.linspace(-3, 3, 20).reshape(-1, 1)
    y = np.sin(X.ravel()) + 0.1 * rng.standard_normal(X.shape[0])

    # Match our kernel: σ² * exp(-||x-x'||² / (2 ℓ²))  with σ²=1, ℓ=1.
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * \
             RBF(length_scale=1.0, length_scale_bounds="fixed")
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, optimizer=None, normalize_y=False)
    gp.fit(X, y)

    # Query grid.
    Xq = np.linspace(-3, 3, 40).reshape(-1, 1)
    pred, std = gp.predict(Xq, return_std=True)

    return [{
        "name": "gp_rbf_sin",
        "X": X.tolist(),
        "y": y.tolist(),
        "Xq": Xq.tolist(),
        "length_scale": 1.0,
        "signal_var": 1.0,
        "alpha": 1e-2,
        "sklearn_pred": pred.tolist(),
        "sklearn_std": std.tolist(),
    }]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "gp.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
