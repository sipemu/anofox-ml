"""Golden data for the kernel approximation samplers (#257):
AdditiveChi2Sampler, SkewedChi2Sampler and PolynomialCountSketch
(scikit-learn 1.8.0). The random ones are fitted with an int random_state,
so their draws are numpy RandomState's."""

import numpy as np
from sklearn.kernel_approximation import (AdditiveChi2Sampler, PolynomialCountSketch,
                                          SkewedChi2Sampler)


def generate():
    rng = np.random.default_rng(257)
    X = np.abs(rng.normal(size=(12, 3)))
    X[2, 1] = 0.0  # zeros map to zero columns
    Xt = np.abs(rng.normal(size=(5, 3)))
    additive = []
    for kw in [{}, {"sample_steps": 1}, {"sample_steps": 3},
               {"sample_steps": 4, "sample_interval": 0.3}]:
        est = AdditiveChi2Sampler(**kw).fit(X)
        additive.append({"params": kw, "X": X.tolist(), "transform": est.transform(X).tolist()})
    skewed = []
    for kw in [{"random_state": 0}, {"skewedness": 0.5, "n_components": 7, "random_state": 42}]:
        est = SkewedChi2Sampler(**kw).fit(X)
        skewed.append({"params": kw, "X": X.tolist(), "X_test": Xt.tolist(),
                       "random_weights": est.random_weights_.tolist(),
                       "random_offset": est.random_offset_.tolist(),
                       "transform": est.transform(Xt).tolist()})
    sketch = []
    Xs = rng.normal(size=(10, 4))
    for kw in [{"random_state": 0, "n_components": 16},
               {"degree": 3, "coef0": 1.5, "gamma": 0.7, "n_components": 9, "random_state": 3},
               {"degree": 1, "n_components": 5, "random_state": 11}]:
        est = PolynomialCountSketch(**kw).fit(Xs)
        sketch.append({"params": kw, "X": Xs.tolist(),
                       "index_hash": est.indexHash_.tolist(),
                       "bit_hash": est.bitHash_.tolist(),
                       "transform": est.transform(Xs).tolist()})
    return {"additive": additive, "skewed": skewed, "sketch": sketch}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "kernel_samplers.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
