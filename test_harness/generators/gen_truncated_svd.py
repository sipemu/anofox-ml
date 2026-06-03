"""Golden data for TruncatedSVD (sklearn.decomposition.TruncatedSVD).

SVD is sign-ambiguous, so we compare element-wise |U Σ| not U Σ. Equivalently
we compare X V_k entries by absolute value (column-wise sign flips allowed).
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD


def generate():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((40, 6)) @ np.diag([10.0, 5.0, 2.0, 0.5, 0.2, 0.1])
    tsvd = TruncatedSVD(n_components=3, algorithm="arpack")
    transformed = tsvd.fit_transform(X)
    return [{
        "name": "truncated_svd_3",
        "X": X.tolist(),
        "n_components": 3,
        "sklearn_singular_values": tsvd.singular_values_.tolist(),
        "sklearn_transformed_abs": np.abs(transformed).tolist(),
    }]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "truncated_svd.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
