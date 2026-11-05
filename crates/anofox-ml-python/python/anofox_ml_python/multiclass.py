"""Multiclass and decision-threshold meta-estimators over this package's
classifiers (sklearn's ``OneVsRestClassifier``, ``OneVsOneClassifier``,
``FixedThresholdClassifier`` and ``TunedThresholdClassifierCV``).

As in sklearn, a binary model's score is its ``decision_function`` when it
has one, else ``predict_proba[:, 1]``; the threshold classifiers prefer
``predict_proba`` (``response_method="auto"``). ``estimator=None`` uses
``LogisticRegression``.
"""

from __future__ import annotations

import numpy as np

from ._base import BaseEstimator, ClassifierMixin
from .conformal import _inner, _logistic_regression


class _Meta(ClassifierMixin, BaseEstimator):
    def _base_params(self, X, y):
        kind, params = _inner(self.estimator, _logistic_regression, ClassifierMixin, X, y)
        return {"estimator": kind, "estimator_params": params}


class OneVsRestClassifier(_Meta):
    """One binary classifier per class (one for two classes)."""

    _kind = "OneVsRestClassifier"

    def __init__(self, estimator=None):
        self.estimator = estimator

    def _rust_params(self, X, y):
        return self._base_params(X, y)


class OneVsOneClassifier(_Meta):
    """One binary classifier per pair of classes, combined by votes."""

    _kind = "OneVsOneClassifier"

    def __init__(self, estimator=None):
        self.estimator = estimator

    def _rust_params(self, X, y):
        return self._base_params(X, y)


_RESPONSE = {"auto": "Auto", "predict_proba": "PredictProba",
             "decision_function": "DecisionFunction"}


class _Threshold(_Meta):
    def _pos(self):
        if self.pos_label is None:
            return None
        idx = np.flatnonzero(self.classes_ == self.pos_label)
        if idx.size == 0:
            raise ValueError(f"pos_label {self.pos_label!r} is not one of {self.classes_.tolist()}")
        return float(idx[0])

    def _response(self):
        if self.response_method not in _RESPONSE:
            raise ValueError(f"response_method must be one of {sorted(_RESPONSE)}")
        return _RESPONSE[self.response_method]


class FixedThresholdClassifier(_Threshold):
    """A binary classifier that predicts the positive class when its score
    is ``>= threshold`` (``"auto"``: 0.5 for probabilities, 0 for decision
    values)."""

    _kind = "FixedThresholdClassifier"

    def __init__(self, estimator=None, *, threshold="auto", pos_label=None,
                 response_method="auto"):
        self.estimator = estimator
        self.threshold = threshold
        self.pos_label = pos_label
        self.response_method = response_method

    def _rust_params(self, X, y):
        return {**self._base_params(X, y),
                "threshold": None if self.threshold == "auto" else float(self.threshold),
                "pos_label": self._pos(), "response_method": self._response()}


_SCORING = {"balanced_accuracy": "BalancedAccuracy", "accuracy": "Accuracy", "f1": "F1",
            "precision": "Precision", "recall": "Recall"}


class TunedThresholdClassifierCV(_Threshold):
    """Tune the decision threshold of a binary classifier by
    cross-validation (unshuffled stratified ``cv`` folds, ``thresholds``
    candidates, the threshold maximising ``scoring``), then refit on all
    rows. ``scoring`` is one of ``"balanced_accuracy"``, ``"accuracy"``,
    ``"f1"``, ``"precision"``, ``"recall"`` (positive class ``pos_label``,
    default the larger label)."""

    _kind = "TunedThresholdClassifierCV"

    def __init__(self, estimator=None, *, scoring="balanced_accuracy", response_method="auto",
                 thresholds=100, cv=5, pos_label=None):
        self.estimator = estimator
        self.scoring = scoring
        self.response_method = response_method
        self.thresholds = thresholds
        self.cv = cv
        self.pos_label = pos_label

    def _rust_params(self, X, y):
        if self.scoring not in _SCORING:
            raise ValueError(f"scoring must be one of {sorted(_SCORING)}, got {self.scoring!r}")
        return {**self._base_params(X, y), "scoring": _SCORING[self.scoring],
                "response_method": self._response(), "thresholds": int(self.thresholds),
                "cv": int(self.cv), "pos_label": self._pos()}

    @property
    def best_threshold_(self):
        return float(self._fitted_state()["threshold"])

    @property
    def best_score_(self):
        return float(self._fitted_state()["best_score"])
