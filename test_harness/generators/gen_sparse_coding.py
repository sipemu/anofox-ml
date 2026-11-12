"""Golden data for sparse coding (#258): SparseCoder with every
transform_algorithm, and DictionaryLearning (fit_algorithm "lars" and "cd")
(scikit-learn 1.8.0). DictionaryLearning starts from LAPACK's SVD, whose
signs are arbitrary, so the atoms are compared up to each atom's sign.
The dictionaries have fewer atoms than features: with as many or more,
the LARS codings wait on sipemu/anofox-regression#334 (the LARS path
stops early when n_samples <= n_features)."""

import numpy as np
from sklearn.decomposition import DictionaryLearning, SparseCoder


def generate():
    rng = np.random.RandomState(258)
    X = rng.randn(25, 6)
    D = rng.randn(5, 6)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    coder = []
    for alg in ["omp", "lars", "lasso_lars", "lasso_cd", "threshold"]:
        for kw in [{}, {"transform_alpha": 0.4, "transform_n_nonzero_coefs": 2},
                   {"split_sign": True}]:
            est = SparseCoder(D, transform_algorithm=alg, **kw)
            coder.append({"params": {"transform_algorithm": alg, **kw}, "X": X.tolist(),
                          "dictionary": D.tolist(), "transform": est.transform(X).tolist()})
    learning = []
    for kw in [{"n_components": 4, "alpha": 1.0},
               {"n_components": 5, "alpha": 0.5, "transform_algorithm": "lasso_lars"},
               {"n_components": 3, "alpha": 2.0, "fit_algorithm": "cd",
                "transform_algorithm": "threshold"}]:
        est = DictionaryLearning(random_state=0, **kw)
        code = est.fit_transform(X)
        learning.append({"params": kw, "X": X.tolist(),
                         "components": est.components_.tolist(), "code": code.tolist(),
                         "error": [float(e) for e in est.error_], "n_iter": int(est.n_iter_),
                         "transform": est.transform(X).tolist()})
    return {"coder": coder, "learning": learning}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "sparse_coding.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
