"""Base classes of the sklearn-compatible estimator wrappers.

Every estimator stores its constructor arguments unchanged (sklearn's
convention), so ``get_params`` / ``set_params`` / ``sklearn.base.clone`` and
the sklearn search and CV utilities work. ``fit`` translates the parameters
into the Rust configuration (``_rust_params``), fits the Rust model with the
GIL released and returns ``self``. Fitted estimators pickle (and therefore
work with ``joblib`` / ``multiprocessing``) through the model's versioned
bincode bytes.
"""

from __future__ import annotations

import inspect
import json
from functools import lru_cache

import numpy as np

from . import _native


class NotFittedError(ValueError, AttributeError):
    """Raised when a method needs a fitted estimator (as sklearn's)."""


@lru_cache(maxsize=None)
def _rust_defaults(kind: str) -> dict:
    return json.loads(_native.Model.default_params(kind)) or {}


def _seed(random_state) -> int:
    """sklearn ``random_state`` -> u64 seed (``None`` draws a fresh one)."""
    if random_state is None:
        return int(np.random.randint(0, 2**31 - 1))
    if isinstance(random_state, (int, np.integer)):
        if random_state < 0:
            raise ValueError("random_state must be non-negative")
        return int(random_state)
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(0, 2**31 - 1))
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 2**31 - 1))
    raise ValueError(f"invalid random_state {random_state!r}")


def _nd(obj) -> np.ndarray:
    """ndarray from the serde JSON of a Rust ``ndarray`` (or a plain list)."""
    if isinstance(obj, dict) and "data" in obj and "dim" in obj:
        return np.asarray(obj["data"], dtype=np.float64).reshape(obj["dim"])
    return np.asarray(obj, dtype=np.float64)


def _check_X(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        raise ValueError(
            "Expected a 2D array, got a 1D array; reshape with X.reshape(-1, 1) "
            "for a single feature or X.reshape(1, -1) for a single sample."
        )
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {X.ndim}D")
    return np.ascontiguousarray(X)


def _check_1d(y, name="y") -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    return np.ascontiguousarray(y)


class BaseEstimator:
    """sklearn-style parameter handling, repr and pickling."""

    #: Registry kind of the Rust estimator.
    _kind: str = ""

    @classmethod
    def _get_param_names(cls):
        sig = inspect.signature(cls.__init__)
        return sorted(
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD and p.kind != p.VAR_POSITIONAL
        )

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params):
        valid = set(self._get_param_names())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {type(self).__name__}. "
                    f"Valid parameters are: {sorted(valid)!r}."
                )
            setattr(self, key, value)
        return self

    def __repr__(self):
        sig = inspect.signature(type(self).__init__)
        changed = []
        for name in self._get_param_names():
            default = sig.parameters[name].default
            value = getattr(self, name)
            try:
                same = value is default or value == default
                same = bool(same)
            except Exception:  # array-valued parameters
                same = False
            if not same:
                changed.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(changed)})"

    # -- Rust model ---------------------------------------------------------

    def _rust_params(self, X, y) -> dict:
        """Rust config overrides for this fit (subclasses override)."""
        return {}

    def _new_model(self, X, y):
        params = self._rust_params(X, y)
        payload = json.dumps(params) if params else None
        return _native.Model(self._kind, payload)

    def _model(self):
        model = self.__dict__.get("_model_")
        if model is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        return model

    def _fit_model(self, X, y=None, sample_weight=None):
        X = _check_X(X)
        if sample_weight is not None:
            sample_weight = _check_1d(sample_weight, "sample_weight")
        model = self._new_model(X, y)
        model.fit(X, y, sample_weight)
        self._model_ = model
        self.n_features_in_ = X.shape[1]
        return self

    def _checked(self, X):
        """``X`` as a 2-D float array with the fitted feature count."""
        self._model()
        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is "
                f"expecting {self.n_features_in_} features as input."
            )
        return X

    def _call(self, method, X):
        X = self._checked(X)
        return getattr(self._model(), method)(X)

    def _has(self, capability):
        return capability in _native.Model(self._kind, None).capabilities()

    @property
    def feature_importances_(self):
        """Per-feature importances (trees, forests, boosting, bagging)."""
        if not self._has("feature_importances"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute 'feature_importances_'"
            )
        return self._model().feature_importances()

    # -- pickling -----------------------------------------------------------

    def __getstate__(self):
        state = dict(self.__dict__)
        model = state.pop("_model_", None)
        if model is not None:
            state["_model_bytes_"] = bytes(model.to_bytes())
        return state

    def __setstate__(self, state):
        state = dict(state)
        data = state.pop("_model_bytes_", None)
        self.__dict__.update(state)
        if data is not None:
            self._model_ = _native.Model.from_bytes(self._kind, data)

    def __sklearn_tags__(self):
        """Tags for scikit-learn >= 1.6 (``GridSearchCV``, ``clone``, ...)."""
        from sklearn.utils import (
            ClassifierTags,
            RegressorTags,
            Tags,
            TargetTags,
            TransformerTags,
        )

        kind = getattr(self, "_estimator_type", None)
        return Tags(
            estimator_type=kind,
            target_tags=TargetTags(required=kind in ("classifier", "regressor")),
            transformer_tags=TransformerTags() if hasattr(self, "transform") else None,
            classifier_tags=ClassifierTags() if kind == "classifier" else None,
            regressor_tags=RegressorTags() if kind == "regressor" else None,
        )

    def _fitted_state(self) -> dict:
        """The fitted Rust model as a dict (for fitted attributes)."""
        return json.loads(self._model().fitted_json())


class ClassifierMixin:
    """Label encoding plus sklearn's classifier API.

    Labels of any type are encoded as ``0..n_classes-1`` for the Rust model
    and decoded on output; ``classes_`` holds the original labels in sorted
    order, which is also the column order of ``predict_proba``.
    """

    _estimator_type = "classifier"

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        return self._fit_model(X, y_enc.astype(np.float64), sample_weight)

    def _encode_class_weight(self, class_weight):
        if class_weight is None:
            return None
        if class_weight == "balanced":
            return "Balanced"
        if isinstance(class_weight, dict):
            pairs = []
            for label, weight in class_weight.items():
                idx = np.searchsorted(self.classes_, label)
                if idx < len(self.classes_) and self.classes_[idx] == label:
                    pairs.append([float(idx), float(weight)])
            return {"Manual": pairs}
        raise ValueError(f"invalid class_weight {class_weight!r}")

    def predict(self, X):
        pred = self._call("predict", X)
        return self.classes_[np.asarray(pred, dtype=np.int64)]

    def predict_proba(self, X):
        return self._call("predict_proba", X)

    def predict_log_proba(self, X):
        if self._has("predict_log_proba"):
            return self._call("predict_log_proba", X)
        with np.errstate(divide="ignore"):
            return np.log(self.predict_proba(X))

    def decision_function(self, X):
        out = self._call("decision_function", X)
        if out.ndim == 2 and out.shape[1] == 1:
            out = out.ravel()
        return out

    def score(self, X, y, sample_weight=None):
        y = np.asarray(y)
        correct = (self.predict(X) == y).astype(np.float64)
        return float(np.average(correct, weights=sample_weight))


class PartialFitMixin:
    """sklearn's ``partial_fit``: fold one batch into the model."""

    def partial_fit(self, X, y=None, sample_weight=None):
        return self._partial_fit(X, y, None, sample_weight)

    def _partial_fit(self, X, y, classes, sample_weight):
        X = _check_X(X)
        if sample_weight is not None:
            sample_weight = _check_1d(sample_weight, "sample_weight")
        model = self.__dict__.get("_model_")
        if model is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is "
                f"expecting {self.n_features_in_} features as input."
            )
        codes = None
        if getattr(self, "_estimator_type", None) == "classifier":
            y = np.asarray(y)
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            if model is None:
                if classes is None:
                    raise ValueError("classes must be passed on the first call to partial_fit")
                self.classes_ = np.unique(np.asarray(classes))
                codes = [float(k) for k in range(len(self.classes_))]
            idx = np.searchsorted(self.classes_, y)
            idx = np.clip(idx, 0, len(self.classes_) - 1)
            if not np.all(self.classes_[idx] == y):
                raise ValueError("y contains labels that are not in classes")
            y = idx.astype(np.float64)
        elif y is not None:
            y = _check_1d(y)
        if model is None:
            model = self._new_model(X, y)
        model.partial_fit(X, y, sample_weight, codes)
        self._model_ = model
        self.n_features_in_ = X.shape[1]
        return self


class ClassifierPartialFitMixin(PartialFitMixin):
    """``partial_fit(X, y, classes=None, sample_weight=None)``; ``classes``
    is required on the first call."""

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return self._partial_fit(X, y, classes, sample_weight)


class WarmStartMixin:
    """sklearn's ``warm_start=True``: a refit continues the fitted model
    (more stages or iterations) instead of starting over."""

    def _fit_model(self, X, y=None, sample_weight=None):
        model = self.__dict__.get("_model_")
        if not self.warm_start or model is None:
            return super()._fit_model(X, y, sample_weight)
        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} was fitted "
                f"with {self.n_features_in_}."
            )
        if sample_weight is not None:
            sample_weight = _check_1d(sample_weight, "sample_weight")
        params = self._rust_params(X, y)
        params.pop("seed", None)
        model.set_params_json(json.dumps(params))
        model.warm_start(X, y, sample_weight)
        return self


class StagedPredictMixin:
    """sklearn's ``staged_predict`` for the boosting estimators."""

    def staged_predict(self, X):
        """Yield the predictions after each boosting stage."""
        stages = self._call("staged_predict", X)
        classifier = getattr(self, "_estimator_type", None) == "classifier"
        for k in range(stages.shape[1]):
            column = stages[:, k]
            yield self.classes_[column.astype(np.int64)] if classifier else column


class RegressorMixin:
    _estimator_type = "regressor"

    def fit(self, X, y, sample_weight=None):
        return self._fit_model(X, _check_1d(y), sample_weight)

    def predict(self, X):
        return self._call("predict", X)

    def score(self, X, y, sample_weight=None):
        y = _check_1d(y)
        pred = self.predict(X)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, float)
        ss_res = np.sum(w * (y - pred) ** 2)
        ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)


class TransformerMixin:
    def fit(self, X, y=None, sample_weight=None):
        return self._fit_model(X, None, sample_weight)

    def transform(self, X):
        return self._call("transform", X)

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        return self.fit(X, y, **fit_params).transform(X)

    def inverse_transform(self, X):
        return self._call("inverse_transform", X)


class SupervisedTransformerMixin(TransformerMixin):
    def fit(self, X, y, sample_weight=None):
        return self._fit_model(X, _check_1d(y), sample_weight)


class ClusterMixin:
    _estimator_type = "clusterer"

    def fit(self, X, y=None, sample_weight=None):
        self._fit_model(X, None, sample_weight)
        if "labels" in self._model().capabilities():
            self.labels_ = np.asarray(self._model().labels(), dtype=np.int64)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, sample_weight=sample_weight)
        if hasattr(self, "labels_"):
            return self.labels_
        return self.predict(X)

    def predict(self, X):
        return np.asarray(self._call("predict", X), dtype=np.int64)


class OutlierMixin:
    _estimator_type = "outlier_detector"

    def fit(self, X, y=None, sample_weight=None):
        return self._fit_model(X, None, sample_weight)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
