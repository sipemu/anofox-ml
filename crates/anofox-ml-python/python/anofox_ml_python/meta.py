"""Meta-estimators over this package's estimators, run entirely in Rust:
voting, stacking, probability calibration, target transformation,
multi-output regression, recursive / sequential / importance-based feature
selection.

Each takes this package's estimators (``estimator=``, ``estimators=[(name,
est), ...]``) and sends their kind and parameters to the Rust registry, so
the fitted meta-estimator is one Rust model (it pickles, and the DuckDB
extension can load the same bytes). The given estimators stay unfitted, as
in sklearn.
"""

from __future__ import annotations

import numpy as np

from ._base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    _check_1d,
    _check_X,
)

# ---------------------------------------------------------------------------
# estimator specs
# ---------------------------------------------------------------------------


def _is_classifier(est):
    return getattr(est, "_estimator_type", None) == "classifier"


def _spec(est, X, y, *, role="estimator", mixin=None):
    """``{"estimator": kind, "estimator_params": params}`` for this fit."""
    if not isinstance(est, BaseEstimator) or not getattr(est, "_kind", ""):
        raise TypeError(
            f"{role} must be an anofox_ml_python estimator, got {type(est).__name__}"
        )
    if mixin is not None and not isinstance(est, mixin):
        what = "classifier" if mixin is ClassifierMixin else "regressor"
        raise TypeError(f"{role} must be a {what}, got {type(est).__name__}")
    est = type(est)(**est.get_params())  # leave the caller's estimator as it was
    if _is_classifier(est):
        est.classes_ = np.unique(y)
    return {"estimator": est._kind, "estimator_params": est._rust_params(X, y)}


def _named_specs(estimators, X, y, mixin):
    if not estimators:
        raise ValueError("estimators must be a non-empty list of (name, estimator) pairs")
    names = [name for name, _ in estimators]
    if len(set(names)) != len(names):
        raise ValueError(f"estimator names must be unique, got {names}")
    return [_spec(est, X, y, role=f"estimator {name!r}", mixin=mixin)
            for name, est in estimators if est != "drop"]


def _weights(weights):
    return None if weights is None else [float(w) for w in weights]


def _default(name):
    from . import _estimators

    return getattr(_estimators, name)()


# ---------------------------------------------------------------------------
# voting and stacking
# ---------------------------------------------------------------------------


class VotingRegressor(RegressorMixin, BaseEstimator):
    """The (weighted) mean of the estimators' predictions."""

    _kind = "VotingRegressor"

    def __init__(self, estimators, *, weights=None):
        self.estimators = estimators
        self.weights = weights

    def _rust_params(self, X, y):
        return {"estimators": _named_specs(self.estimators, X, y, RegressorMixin),
                "weights": _weights(self.weights)}


class VotingClassifier(ClassifierMixin, BaseEstimator):
    """Hard (weighted majority) or soft (weighted mean probability) voting."""

    _kind = "VotingClassifier"

    def __init__(self, estimators, *, voting="hard", weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def _rust_params(self, X, y):
        if self.voting not in ("hard", "soft"):
            raise ValueError(f"voting must be 'hard' or 'soft', got {self.voting!r}")
        return {"estimators": _named_specs(self.estimators, X, y, ClassifierMixin),
                "voting": self.voting, "weights": _weights(self.weights)}

    def predict_proba(self, X):
        if self.voting != "soft":
            raise AttributeError("predict_proba is not available when voting='hard'")
        return super().predict_proba(X)


def _cv(cv):
    if cv is None:
        return 5
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool) and cv >= 2:
        return int(cv)
    raise ValueError(f"cv must be an int >= 2 (KFold / StratifiedKFold folds), got {cv!r}")


class StackingRegressor(RegressorMixin, BaseEstimator):
    """The final estimator fitted on the estimators' cross-validated
    predictions (``cv`` unshuffled folds); ``final_estimator=None`` is
    ``RidgeCV``."""

    _kind = "StackingRegressor"

    def __init__(self, estimators, final_estimator=None, *, cv=None, passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough

    def _rust_params(self, X, y):
        p = {"estimators": _named_specs(self.estimators, X, y, RegressorMixin),
             "cv": _cv(self.cv), "passthrough": self.passthrough}
        if self.final_estimator is not None:
            p["final_estimator"] = _spec(self.final_estimator, X, y, role="final_estimator",
                                         mixin=RegressorMixin)
        return p


_STACK_METHOD = {"auto": "auto", "predict_proba": "predict_proba",
                 "decision_function": "decision_function", "predict": "predict"}


class StackingClassifier(ClassifierMixin, BaseEstimator):
    """The final estimator fitted on the estimators' cross-validated
    outputs (``stack_method``); ``final_estimator=None`` is
    ``LogisticRegression``."""

    _kind = "StackingClassifier"

    def __init__(self, estimators, final_estimator=None, *, cv=None, stack_method="auto",
                 passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.passthrough = passthrough

    def _rust_params(self, X, y):
        if self.stack_method not in _STACK_METHOD:
            raise ValueError(f"stack_method must be one of {sorted(_STACK_METHOD)}")
        p = {"estimators": _named_specs(self.estimators, X, y, ClassifierMixin),
             "cv": _cv(self.cv), "passthrough": self.passthrough,
             "stack_method": _STACK_METHOD[self.stack_method]}
        if self.final_estimator is not None:
            p["final_estimator"] = _spec(self.final_estimator, X, y, role="final_estimator",
                                         mixin=ClassifierMixin)
        return p


# ---------------------------------------------------------------------------
# calibration and target transformation
# ---------------------------------------------------------------------------


class CalibratedClassifierCV(ClassifierMixin, BaseEstimator):
    """Probabilities calibrated by Platt scaling (``"sigmoid"``) or
    isotonic regression on ``cv`` stratified folds; ``estimator=None`` is
    ``LinearSVC``."""

    _kind = "CalibratedClassifierCV"

    def __init__(self, estimator=None, *, method="sigmoid", cv=None, ensemble=True):
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.ensemble = ensemble

    def _rust_params(self, X, y):
        if self.method not in ("sigmoid", "isotonic"):
            raise ValueError(f"method must be 'sigmoid' or 'isotonic', got {self.method!r}")
        if self.ensemble not in (True, False):
            raise ValueError("ensemble must be True or False")
        est = self.estimator if self.estimator is not None else _default("LinearSVC")
        return {**_spec(est, X, y, mixin=ClassifierMixin), "method": self.method,
                "cv": _cv(self.cv), "ensemble": bool(self.ensemble)}


_TARGET_FUNCS = {np.log: ("log", np.exp), np.log1p: ("log1p", np.expm1),
                 np.sqrt: ("sqrt", np.square)}


class TransformedTargetRegressor(RegressorMixin, BaseEstimator):
    """The regressor learns ``func(y)`` (or ``y`` transformed by
    ``transformer``) and its predictions are mapped back. ``func`` is
    ``np.log``, ``np.log1p`` or ``np.sqrt`` (or their names), with the
    matching ``inverse_func`` or ``None``; ``transformer`` is one of this
    package's transformers, fitted on ``y`` as one column."""

    _kind = "TransformedTargetRegressor"

    def __init__(self, regressor=None, *, transformer=None, func=None, inverse_func=None):
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func

    def _func(self):
        if self.func is None:
            if self.inverse_func is not None:
                raise ValueError("inverse_func needs func")
            return None
        names = {name: inv for name, inv in _TARGET_FUNCS.values()}
        if isinstance(self.func, str) and self.func in names:
            name, inverse = self.func, names[self.func]
        elif self.func in _TARGET_FUNCS:
            name, inverse = _TARGET_FUNCS[self.func]
        else:
            raise ValueError("func must be np.log, np.log1p or np.sqrt (or None)")
        if self.inverse_func is not None and self.inverse_func is not inverse:
            raise ValueError(f"inverse_func of {name} must be np.{inverse.__name__}")
        return name

    def _rust_params(self, X, y):
        func = self._func()
        if func is not None and self.transformer is not None:
            raise ValueError("pass either func or transformer, not both")
        reg = self.regressor if self.regressor is not None else _default("LinearRegression")
        p = {"regressor": _spec(reg, X, y, role="regressor", mixin=RegressorMixin),
             "func": func}
        if self.transformer is not None:
            if not isinstance(self.transformer, TransformerMixin):
                raise TypeError("transformer must be an anofox_ml_python transformer")
            col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
            p["transformer"] = _spec(self.transformer, col, None, role="transformer")
        return p


# ---------------------------------------------------------------------------
# multi-output regression
# ---------------------------------------------------------------------------


class _MultiOutput(BaseEstimator):
    """Fit on ``(X, Y)`` with ``Y`` of shape ``n × k``; ``predict`` returns
    ``n × k``."""

    _estimator_type = "regressor"

    def fit(self, X, y):
        from ._estimators import _check_2d_y

        X = _check_X(X)
        Y = _check_2d_y(y, X.shape[0])
        model = self._new_model(X, Y)
        model.fit(np.hstack([X, Y]), None, None)
        self._model_ = model
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self._call("predict", X)

    def score(self, X, y, sample_weight=None):
        from ._estimators import _MultiTask

        return _MultiTask.score(self, X, y, sample_weight)

    def _base(self, X, Y):
        return _spec(self.estimator, X, Y[:, 0], mixin=RegressorMixin)


class MultiOutputRegressor(_MultiOutput):
    """One regressor per target column."""

    _kind = "MultiOutputRegressor"

    def __init__(self, estimator):
        self.estimator = estimator

    def _rust_params(self, X, y):
        return {**self._base(X, y), "y_columns": int(y.shape[1])}


class RegressorChain(_MultiOutput):
    """Regressors fitted in ``order`` (default: the column order), each on
    ``X`` and the targets before it."""

    _kind = "RegressorChain"

    def __init__(self, estimator, *, order=None):
        self.estimator = estimator
        self.order = order

    def _rust_params(self, X, y):
        order = self.order
        if isinstance(order, str):
            raise ValueError("order='random' is not supported; pass a list of column indices")
        order = None if order is None else [int(i) for i in order]
        return {**self._base(X, y), "y_columns": int(y.shape[1]), "order": order}


# ---------------------------------------------------------------------------
# feature selection
# ---------------------------------------------------------------------------


class _Selector(BaseEstimator):
    """Fit through the wrapped estimator, a regressor or a classifier
    (labels encoded as in ``ClassifierMixin``)."""

    @property
    def _estimator_type(self):
        return getattr(self.estimator, "_estimator_type", None)

    def _encode(self, y):
        if not _is_classifier(self.estimator):
            return _check_1d(y)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        self.classes_, codes = np.unique(y, return_inverse=True)
        return codes.astype(np.float64)

    def _spec(self, X, y):
        return _spec(self.estimator, X, y)

    def fit(self, X, y, sample_weight=None):
        self._fit_model(X, self._encode(y), sample_weight)
        self.support_ = np.asarray(self._support(), dtype=bool)
        self.n_features_ = int(self.support_.sum())
        return self

    def _support(self):
        return self._fitted_state()["support"]

    def get_support(self, indices=False):
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def transform(self, X):
        return self._call("transform", X)

    def fit_transform(self, X, y, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class _Predicting(_Selector):
    def predict(self, X):
        pred = self._call("predict", X)
        if _is_classifier(self.estimator):
            return self.classes_[np.asarray(pred, dtype=np.int64)]
        return pred

    def score(self, X, y, sample_weight=None):
        mixin = ClassifierMixin if _is_classifier(self.estimator) else RegressorMixin
        return mixin.score(self, X, y, sample_weight)


def _count(value, n_features, name):
    """An int, or a fraction of the features (sklearn's float form)."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and 0 < value < 1:
        return max(1, int(value * n_features))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 1:
        return int(value)
    raise ValueError(f"{name} must be an int >= 1 or a float in (0, 1), got {value!r}")


class RFE(_Predicting):
    """Recursive feature elimination by the squared ``coef_`` (summed over
    classes), else ``feature_importances_``."""

    _kind = "Rfe"

    def __init__(self, estimator, *, n_features_to_select=None, step=1):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step

    def _rust_params(self, X, y):
        p = X.shape[1]
        return {**self._spec(X, y),
                "n_features_to_select": _count(self.n_features_to_select, p,
                                               "n_features_to_select"),
                "step": _count(self.step, p, "step")}

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.ranking_ = np.asarray(self._fitted_state()["ranking"], dtype=np.int64)
        return self


class RFECV(_Predicting):
    """RFE with the number of features chosen by the mean ``cv``-fold score
    (R² for regressors, accuracy for classifiers)."""

    _kind = "Rfecv"

    def __init__(self, estimator, *, step=1, min_features_to_select=1, cv=None):
        self.estimator = estimator
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.cv = cv

    def _rust_params(self, X, y):
        return {**self._spec(X, y), "step": _count(self.step, X.shape[1], "step"),
                "min_features_to_select": int(self.min_features_to_select),
                "cv": _cv(self.cv)}

    def _support(self):
        return self._fitted_state()["rfe"]["support"]

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        state = self._fitted_state()
        self.ranking_ = np.asarray(state["rfe"]["ranking"], dtype=np.int64)
        self.cv_results_ = {"mean_test_score": np.asarray(state["mean_test_score"]),
                            "n_features": np.asarray(state["n_features"], dtype=np.int64)}
        return self


class SequentialFeatureSelector(TransformerMixin, _Selector):
    """Greedy forward / backward selection by the mean ``cv``-fold score
    (sklearn's with ``tol=None``); ``n_features_to_select="auto"`` or
    ``None`` selects half."""

    _kind = "SequentialFeatureSelector"

    def __init__(self, estimator, *, n_features_to_select="auto", direction="forward",
                 cv=5):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.cv = cv

    _estimator_type = None

    def fit(self, X, y, sample_weight=None):
        return _Selector.fit(self, X, y, sample_weight)

    def _rust_params(self, X, y):
        if self.direction not in ("forward", "backward"):
            raise ValueError(f"direction must be 'forward' or 'backward', got "
                             f"{self.direction!r}")
        n = self.n_features_to_select
        n = None if n == "auto" else _count(n, X.shape[1], "n_features_to_select")
        return {**self._spec(X, y), "n_features_to_select": n, "direction": self.direction,
                "cv": _cv(self.cv)}


def _l1_penalized(est):
    """sklearn's estimators whose default ``SelectFromModel`` threshold is
    1e-5 (their coefficients are exactly zero when dropped)."""
    name = type(est).__name__
    l1_ratio = getattr(est, "l1_ratio", None)
    return (getattr(est, "penalty", None) == "l1" or "Lasso" in name
            or (name in ("ElasticNet", "LogisticRegression") and l1_ratio is not None
                and np.isclose(l1_ratio, 1.0)))


def _threshold(threshold, est):
    if threshold is None:
        threshold = 1e-5 if _l1_penalized(est) else "mean"
    if isinstance(threshold, str):
        scale, _, reference = threshold.rpartition("*")
        reference = reference.strip()
        if reference not in ("mean", "median"):
            raise ValueError(f"Expected threshold='mean' or threshold='median', got "
                             f"{threshold!r}")
        rule = reference.capitalize()
        if not scale:
            return rule
        return {f"Scaled{rule}": float(scale.strip())}
    value = float(threshold)
    # JSON has no -inf; below every |importance| it selects the same.
    return {"Value": max(value, np.finfo(np.float64).min)}


class SelectFromModel(TransformerMixin, _Selector):
    """The features whose importance (``|coef_|``, the ``norm_order``-norm
    over the classes of a 2-D ``coef_``, else ``feature_importances_``) is at
    least ``threshold`` (a value, ``"mean"``, ``"median"`` or ``"1.25*mean"``;
    default the mean, 1e-5 for L1-penalized estimators), among the
    ``max_features`` largest. The estimator is fitted in Rust, so
    ``prefit=True`` is not supported and ``importance_getter`` is ``"auto"``."""

    _kind = "SelectFromModel"
    _estimator_type = None

    def __init__(self, estimator, *, threshold=None, prefit=False, norm_order=1,
                 max_features=None, importance_getter="auto"):
        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features
        self.importance_getter = importance_getter

    def _rust_params(self, X, y):
        if self.prefit:
            raise ValueError("prefit=True is not supported: the estimator is fitted in Rust")
        if self.importance_getter != "auto":
            raise ValueError("only importance_getter='auto' is supported")
        max_features = self.max_features
        if callable(max_features):
            max_features = max_features(X)
        if max_features is not None:
            if not isinstance(max_features, (int, np.integer)) or isinstance(max_features, bool) \
                    or not 0 <= max_features <= X.shape[1]:
                raise ValueError(f"max_features must be an int in [0, {X.shape[1]}], got "
                                 f"{max_features!r}")
            max_features = int(max_features)
        norm_order = None if np.isinf(self.norm_order) else float(self.norm_order)
        return {**self._spec(X, y), "threshold": _threshold(self.threshold, self.estimator),
                "max_features": max_features, "norm_order": norm_order}

    def fit(self, X, y=None, sample_weight=None):
        _Selector.fit(self, X, y, sample_weight)
        state = self._fitted_state()
        self.threshold_ = state["threshold"]
        self.max_features_ = self.max_features(np.asarray(X)) if callable(self.max_features) \
            else self.max_features
        return self
