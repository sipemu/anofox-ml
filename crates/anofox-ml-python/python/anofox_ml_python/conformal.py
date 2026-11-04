"""Conformal prediction: intervals and sets with guaranteed coverage.

Each wrapper takes one of this package's regressors or classifiers as
``estimator`` (``None`` uses MAPIE's defaults, ``LinearRegression`` /
``LogisticRegression``), splits the training data itself, and runs entirely
in Rust. ``alpha`` is the miscoverage: intervals and sets contain the true
value with probability ``>= 1 - alpha``.

- ``SplitConformalRegressor``, ``CrossConformalRegressor`` (CV+ /
  jackknife+ / minmax), ``NormalizedConformalRegressor`` (residual
  normalised), ``ConformalizedQuantileRegressor`` (CQR, ``alpha`` fixed at
  fit time) and ``MondrianConformalRegressor`` (coverage per group):
  ``predict_interval(X, alpha)`` returns an ``(n, 2)`` array of lower and
  upper bounds.
- ``SplitConformalClassifier`` (LAC / APS / RAPS) and
  ``MondrianConformalClassifier``: ``predict_set(X, alpha)`` returns an
  ``(n, n_classes)`` boolean array; column ``k`` is ``classes_[k]``.
"""

from __future__ import annotations

import json

import numpy as np

from . import _native
from ._base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    _check_1d,
    _check_X,
    _seed,
)


def _inner(estimator, default_cls, mixin, X, y):
    """``(kind, rust params)`` of the wrapped estimator for this fit."""
    if estimator is None:
        estimator = default_cls()
    if not isinstance(estimator, mixin) or not getattr(estimator, "_kind", ""):
        kind = "regressor" if mixin is RegressorMixin else "classifier"
        raise TypeError(
            f"estimator must be an anofox_ml_python {kind}, got {type(estimator).__name__}"
        )
    # A copy, so that fitting the conformal model leaves `estimator` as it was.
    est = type(estimator)(**estimator.get_params())
    if mixin is ClassifierMixin:
        est.classes_ = np.unique(y)
    return est._kind, est._rust_params(X, y)


class _Conformal(BaseEstimator):
    """Shared fit / predict of the conformal wrappers."""

    def _conformal_params(self) -> dict:
        raise NotImplementedError

    def _fit_conformal(self, X, y, default_cls, mixin):
        X = _check_X(X)
        kind, params = _inner(self.estimator, default_cls, mixin, X, y)
        config = {"estimator": kind, "estimator_params": params, **self._conformal_params()}
        model = _native.Model(self._kind, json.dumps(config))
        model.fit(X, y, None)
        self._model_ = model
        self.n_features_in_ = X.shape[1]
        return self


def _linear_regression():
    from ._estimators import LinearRegression

    return LinearRegression()


def _logistic_regression():
    from ._estimators import LogisticRegression

    return LogisticRegression()


class _ConformalRegressor(RegressorMixin, _Conformal):
    def fit(self, X, y):
        return self._fit_conformal(X, _check_1d(y), _linear_regression, RegressorMixin)

    def predict_interval(self, X, alpha=0.1):
        """``(n, 2)`` array: lower and upper bounds at miscoverage ``alpha``."""
        model = self._model()
        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is "
                f"expecting {self.n_features_in_} features as input."
            )
        return model.predict_interval(X, float(alpha))


class SplitConformalRegressor(_ConformalRegressor):
    """Split conformal intervals: absolute residuals on held-out rows."""

    _kind = "SplitConformalRegressor"

    def __init__(self, estimator=None, calibration_size=0.25, random_state=None,
                 allow_infinite_bounds=False):
        self.estimator = estimator
        self.calibration_size = calibration_size
        self.random_state = random_state
        self.allow_infinite_bounds = allow_infinite_bounds

    def _conformal_params(self):
        return {"calibration_size": float(self.calibration_size),
                "seed": _seed(self.random_state),
                "allow_infinite_bounds": bool(self.allow_infinite_bounds)}


_CROSS_METHODS = {"plus": "Plus", "minmax": "MinMax", "base": "Base"}
_POINT = {"mean": "Mean", "median": "Median", "full": "Full"}


class CrossConformalRegressor(_ConformalRegressor):
    """CV+ / jackknife+ (``cv`` = number of rows) / minmax / base intervals."""

    _kind = "CrossConformalRegressor"

    def __init__(self, estimator=None, cv=5, method="plus", point_prediction="mean",
                 random_state=None, allow_infinite_bounds=False):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.point_prediction = point_prediction
        self.random_state = random_state
        self.allow_infinite_bounds = allow_infinite_bounds

    def _conformal_params(self):
        if self.method not in _CROSS_METHODS:
            raise ValueError(f"method must be one of {sorted(_CROSS_METHODS)}, got {self.method!r}")
        if self.point_prediction not in _POINT:
            raise ValueError(
                f"point_prediction must be one of {sorted(_POINT)}, got {self.point_prediction!r}"
            )
        return {"cv": int(self.cv), "method": _CROSS_METHODS[self.method],
                "point_prediction": _POINT[self.point_prediction],
                "seed": _seed(self.random_state),
                "allow_infinite_bounds": bool(self.allow_infinite_bounds)}


class NormalizedConformalRegressor(_ConformalRegressor):
    """Residual-normalised intervals: ``residual_estimator`` models
    ``|y - prediction|``, so intervals are wide where the model is unsure."""

    _kind = "NormalizedConformalRegressor"

    def __init__(self, estimator=None, residual_estimator=None, calibration_size=0.5,
                 residual_size=0.25, random_state=None, allow_infinite_bounds=False):
        self.estimator = estimator
        self.residual_estimator = residual_estimator
        self.calibration_size = calibration_size
        self.residual_size = residual_size
        self.random_state = random_state
        self.allow_infinite_bounds = allow_infinite_bounds

    def fit(self, X, y):
        X, y = _check_X(X), _check_1d(y)
        self._residual = _inner(self.residual_estimator, _linear_regression, RegressorMixin, X, y)
        return super().fit(X, y)

    def _conformal_params(self):
        kind, params = self._residual
        return {"residual_estimator": kind, "residual_estimator_params": params,
                "calibration_size": float(self.calibration_size),
                "residual_size": float(self.residual_size),
                "seed": _seed(self.random_state),
                "allow_infinite_bounds": bool(self.allow_infinite_bounds)}


class ConformalizedQuantileRegressor(_ConformalRegressor):
    """CQR: quantile models for ``alpha / 2``, ``1 - alpha / 2`` and the
    median, corrected on held-out rows. ``estimator`` must fit quantiles
    (``QuantileRegressor``, the default, or ``GradientBoostingRegressor``);
    its quantile setting is replaced. ``alpha`` is fixed at fit time."""

    _kind = "ConformalizedQuantileRegressor"

    def __init__(self, estimator=None, alpha=0.1, calibration_size=0.25,
                 symmetric_correction=False, random_state=None, allow_infinite_bounds=False):
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.symmetric_correction = symmetric_correction
        self.random_state = random_state
        self.allow_infinite_bounds = allow_infinite_bounds

    def fit(self, X, y):
        estimator = self.estimator
        if estimator is None:
            from ._estimators import QuantileRegressor

            estimator = QuantileRegressor()
        X, y = _check_X(X), _check_1d(y)
        kind, params = _inner(estimator, None, RegressorMixin, X, y)
        config = {"estimator": kind, "estimator_params": params, **self._conformal_params()}
        model = _native.Model(self._kind, json.dumps(config))
        model.fit(X, y, None)
        self._model_ = model
        self.n_features_in_ = X.shape[1]
        return self

    def predict_interval(self, X, alpha=None):
        """Intervals for the fitted ``alpha`` (another value is an error)."""
        return super().predict_interval(X, self.alpha if alpha is None else alpha)

    def _conformal_params(self):
        return {"alpha": float(self.alpha), "calibration_size": float(self.calibration_size),
                "symmetric_correction": bool(self.symmetric_correction),
                "seed": _seed(self.random_state),
                "allow_infinite_bounds": bool(self.allow_infinite_bounds)}


def _with_groups(X, groups):
    """``X`` with the group ids appended as the last column."""
    X = _check_X(X)
    groups = np.asarray(groups)
    if groups.shape != (X.shape[0],):
        raise ValueError(f"groups must have one entry per row ({X.shape[0]}), got {groups.shape}")
    if not np.issubdtype(groups.dtype, np.integer) or (groups < 0).any():
        raise ValueError("groups must be non-negative integers")
    return np.ascontiguousarray(np.column_stack([X, groups.astype(np.float64)]))


class MondrianConformalRegressor(_ConformalRegressor):
    """Split conformal intervals calibrated per group: coverage
    ``>= 1 - alpha`` within every group (``groups``: one non-negative
    integer per row, at fit and at predict time)."""

    _kind = "MondrianConformalRegressor"

    def __init__(self, estimator=None, calibration_size=0.25, random_state=None,
                 allow_infinite_bounds=False):
        self.estimator = estimator
        self.calibration_size = calibration_size
        self.random_state = random_state
        self.allow_infinite_bounds = allow_infinite_bounds

    def fit(self, X, y, groups):
        X, y = _check_X(X), _check_1d(y)
        self.n_features_in_ = X.shape[1]
        kind, params = _inner(self.estimator, _linear_regression, RegressorMixin, X, y)
        # The Rust model reads the groups from an extra last column.
        config = {"estimator": kind, "estimator_params": params,
                  "group_column": X.shape[1], **self._conformal_params()}
        model = _native.Model(self._kind, json.dumps(config))
        model.fit(_with_groups(X, groups), y, None)
        self._model_ = model
        return self

    def _X(self, X, groups):
        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is "
                f"expecting {self.n_features_in_} features as input."
            )
        return _with_groups(X, groups)

    def predict(self, X, groups=None):
        """Point predictions (the groups do not change them)."""
        g = np.zeros(np.shape(X)[0], dtype=np.int64) if groups is None else groups
        return self._model().predict(self._X(X, g))

    def predict_interval(self, X, groups, alpha=0.1):
        """``(n, 2)`` array: bounds calibrated within each row's group."""
        return self._model().predict_interval(self._X(X, groups), float(alpha))

    def _conformal_params(self):
        return {"calibration_size": float(self.calibration_size),
                "seed": _seed(self.random_state),
                "allow_infinite_bounds": bool(self.allow_infinite_bounds)}


def _score(score, k_reg, lambda_, tuning_size, mode):
    modes = {"include": "Include", "exclude": "Exclude", "randomized": "Randomized"}
    if mode not in modes:
        raise ValueError(f"mode must be one of {sorted(modes)}, got {mode!r}")
    if score == "lac":
        return "Lac"
    if score == "aps":
        return {"Aps": modes[mode]}
    if score == "raps":
        return {"Raps": {"mode": modes[mode], "k_reg": k_reg, "lambda": lambda_,
                         "tuning_size": float(tuning_size)}}
    raise ValueError(f"score must be 'lac', 'aps' or 'raps', got {score!r}")


class _ConformalClassifier(_Conformal):
    _estimator_type = "classifier"

    def _encode(self, y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        return y_enc.astype(np.float64)

    def _conformal_params(self):
        score = _score(self.conformity_score, self.k_reg, self.lambda_reg, self.tuning_size,
                       self.mode)
        return {"score": score,
                "calibration_size": float(self.calibration_size),
                "seed": _seed(self.random_state),
                "allow_full_sets": bool(self.allow_full_sets)}

    def predict(self, X, *args):
        pred = self._model().predict(self._X(X, *args))
        return self.classes_[np.asarray(pred, dtype=np.int64)]

    def predict_proba(self, X, *args):
        return self._model().predict_proba(self._X(X, *args))


class SplitConformalClassifier(_ConformalClassifier):
    """Prediction sets from held-out calibration rows. ``conformity_score``:
    ``"lac"`` (smallest sets), ``"aps"`` (adaptive) or ``"raps"`` (APS with a
    penalty after ``k_reg`` classes; ``k_reg`` / ``lambda_reg`` left at
    ``None`` are
    tuned on ``tuning_size`` of the calibration rows). ``mode`` for APS /
    RAPS: ``"include"``, ``"exclude"`` or ``"randomized"``."""

    _kind = "SplitConformalClassifier"

    def __init__(self, estimator=None, conformity_score="lac", mode="include", k_reg=None,
                 lambda_reg=None, tuning_size=0.2, calibration_size=0.25, random_state=None,
                 allow_full_sets=False):
        self.estimator = estimator
        self.conformity_score = conformity_score
        self.mode = mode
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self.tuning_size = tuning_size
        self.calibration_size = calibration_size
        self.random_state = random_state
        self.allow_full_sets = allow_full_sets

    def fit(self, X, y):
        return self._fit_conformal(X, self._encode(y), _logistic_regression, ClassifierMixin)

    def _X(self, X):
        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is "
                f"expecting {self.n_features_in_} features as input."
            )
        return X

    def predict_set(self, X, alpha=0.1):
        """``(n, n_classes)`` boolean array; column ``k`` is ``classes_[k]``."""
        return self._model().predict_set(self._X(X), float(alpha)) == 1.0


class MondrianConformalClassifier(_ConformalClassifier):
    """``SplitConformalClassifier`` calibrated per group: set coverage
    ``>= 1 - alpha`` within every group (``groups``: one non-negative integer
    per row, at fit and at predict time)."""

    _kind = "MondrianConformalClassifier"

    def __init__(self, estimator=None, conformity_score="lac", mode="include", k_reg=None,
                 lambda_reg=None, tuning_size=0.2, calibration_size=0.25, random_state=None,
                 allow_full_sets=False):
        self.estimator = estimator
        self.conformity_score = conformity_score
        self.mode = mode
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self.tuning_size = tuning_size
        self.calibration_size = calibration_size
        self.random_state = random_state
        self.allow_full_sets = allow_full_sets

    def fit(self, X, y, groups):
        X = _check_X(X)
        y_enc = self._encode(y)
        self.n_features_in_ = X.shape[1]
        kind, params = _inner(self.estimator, _logistic_regression, ClassifierMixin, X, y_enc)
        config = {"estimator": kind, "estimator_params": params,
                  "group_column": X.shape[1], **self._conformal_params()}
        model = _native.Model(self._kind, json.dumps(config))
        model.fit(_with_groups(X, groups), y_enc, None)
        self._model_ = model
        return self

    def _X(self, X, groups=None):
        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is "
                f"expecting {self.n_features_in_} features as input."
            )
        g = np.zeros(X.shape[0], dtype=np.int64) if groups is None else groups
        return _with_groups(X, g)

    def predict_set(self, X, groups, alpha=0.1):
        """``(n, n_classes)`` boolean array; column ``k`` is ``classes_[k]``."""
        return self._model().predict_set(self._X(X, groups), float(alpha)) == 1.0
