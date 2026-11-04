"""Golden data for conformal classification sets (anofox-ml-conformal),
from MAPIE's SplitConformalClassifier.

The classifier is a fixed softmax model, P(y = k | x) = softmax(x W)_k, so
both sides start from the same probabilities and only the conformal logic
is compared. APS is compared without randomisation: MAPIE draws its
uniforms from `random_state`, which is replaced by one returning zeros.
"""

import json
import os

import numpy as np
from mapie.classification import SplitConformalClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

LEVELS = [0.9, 0.8, 0.5]


class FixedSoftmax(ClassifierMixin, BaseEstimator):
    def __init__(self, W=None):
        self.W = W

    def fit(self, X, y):
        return self

    def __sklearn_is_fitted__(self):
        return True

    @property
    def classes_(self):
        return np.arange(np.asarray(self.W).shape[1])

    def predict_proba(self, X):
        z = np.asarray(X) @ np.asarray(self.W)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ZeroUniform(np.random.RandomState):
    """A RandomState whose uniform draws are all 0 (no APS randomisation)."""

    def uniform(self, low=0.0, high=1.0, size=None):
        return np.zeros(size)


def generate():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((3, 4)) * 1.5
    model = FixedSoftmax(W)
    X_cal = rng.standard_normal((120, 3))
    # Labels drawn from the model's own probabilities.
    P = model.predict_proba(X_cal)
    y_cal = np.array([rng.choice(4, p=p) for p in P])
    X_test = rng.standard_normal((25, 3))

    cases = []
    for score, include_last in [("lac", None), ("aps", True), ("aps", False)]:
        scc = SplitConformalClassifier(model, confidence_level=LEVELS,
                                       conformity_score=score, prefit=True,
                                       random_state=ZeroUniform(0))
        scc.conformalize(X_cal, y_cal)
        params = None if include_last is None else {"include_last_label": include_last}
        y_pred, y_sets = scc.predict_set(X_test, conformity_score_params=params)
        cases.append({
            "name": score if include_last is None else f"{score}_last_{include_last}",
            "score": score,
            "include_last_label": include_last,
            "levels": LEVELS,
            "y_pred": y_pred.tolist(),
            # sets[level][row][class]
            "sets": [y_sets[:, :, j].astype(int).tolist() for j in range(len(LEVELS))],
        })

    return {"W": W.tolist(), "X_cal": X_cal.tolist(), "y_cal": y_cal.tolist(),
            "X_test": X_test.tolist(), "cases": cases}


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "..", "crates",
                       "anofox-ml-conformal", "tests", "golden_data",
                       "mapie_classification.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {out}")
