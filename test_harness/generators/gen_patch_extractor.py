"""Golden data for PatchExtractor (#261, scikit-learn 1.8.0): every patch,
and max_patches random ones (numpy RandomState draws). Images flattened
row-major per image, patches likewise."""

import numpy as np
from sklearn.feature_extraction.image import PatchExtractor


def generate():
    rng = np.random.RandomState(261)
    cases = []
    for X in (rng.rand(2, 6, 5), rng.rand(2, 5, 4, 3)):
        for kw in [{"patch_size": (2, 3)},
                   {"patch_size": (3, 2), "max_patches": 4, "random_state": 0},
                   {"patch_size": (2, 2), "max_patches": 0.5, "random_state": 7}]:
            patches = PatchExtractor(**kw).fit(X).transform(X)
            c = X.shape[3] if X.ndim == 4 else 1
            cases.append({"params": {k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in kw.items()},
                          "image_shape": [X.shape[1], X.shape[2], c],
                          "X": X.reshape(len(X), -1).tolist(),
                          "patches": patches.reshape(len(patches), -1).tolist()})
    return {"cases": cases}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "patch_extractor.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
