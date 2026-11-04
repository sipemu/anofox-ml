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
from mapie.conformity_scores import RAPSConformityScore
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


class PinnedRAPS(RAPSConformityScore):
    """RAPS with a fixed tuning split and a fixed lambda.

    MAPIE draws the tuning rows with a StratifiedShuffleSplit and, while
    searching lambda, regularizes with lambda in place of k* (see
    `_add_regularization` in the non-prediction phase). Pinning both leaves
    the part we compare: k* from the tuning rows, the regularized scores,
    the threshold and the sets.
    """

    def __init__(self, tune_index, fixed_lambda):
        super().__init__(size_raps=0.2)
        self.tune_index = np.asarray(tune_index)
        self.fixed_lambda = fixed_lambda

    def split_data(self, X, y, y_enc, sample_weight=None, groups=None):
        tune = np.zeros(len(y_enc), dtype=bool)
        tune[self.tune_index] = True
        self.X_raps, self.y_raps = X[tune], y_enc[tune]
        self.y_raps_no_enc = self.label_encoder_.inverse_transform(self.y_raps)
        X, y_enc = X[~tune], y_enc[~tune]
        y = self.label_encoder_.inverse_transform(y_enc)
        self.n_samples_ = len(y_enc)
        return X, y, y_enc, sample_weight, groups

    def _find_lambda_star(self, y_raps_no_enc, y_pred_proba_raps, alpha_np,
                          include_last_label, k_star):
        if len(alpha_np) == 1:
            return float(self.fixed_lambda)
        return np.full(len(alpha_np), self.fixed_lambda)


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

    # RAPS: rows 0..30 tune k* (lambda pinned), rows 30..120 calibrate.
    tune_index = list(range(30))
    for include_last, lam in [(True, 0.1), (False, 0.1), (True, 0.5)]:
        scc = SplitConformalClassifier(model, confidence_level=LEVELS,
                                       conformity_score=PinnedRAPS(tune_index, lam),
                                       prefit=True, random_state=ZeroUniform(0))
        scc.conformalize(X_cal, y_cal)
        y_pred, y_sets = scc.predict_set(
            X_test, conformity_score_params={"include_last_label": include_last})
        cases.append({
            "name": f"raps_last_{include_last}_lambda_{lam}",
            "score": "raps",
            "include_last_label": include_last,
            "lambda": lam,
            "tune_index": tune_index,
            "levels": LEVELS,
            "y_pred": y_pred.tolist(),
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
