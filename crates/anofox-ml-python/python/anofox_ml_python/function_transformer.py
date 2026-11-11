"""sklearn's ``FunctionTransformer``. It applies a Python callable, so it is
plain Python and cannot be a step of this package's (Rust) ``Pipeline``;
use it with ``sklearn.pipeline.Pipeline`` instead."""

from __future__ import annotations

import warnings

import numpy as np

from ._base import BaseEstimator, NotFittedError


class FunctionTransformer(BaseEstimator):
    """``func(X, **kw_args)`` as a transformer (the identity when ``func`` is
    ``None``); ``inverse_func`` likewise for ``inverse_transform``. With
    ``validate=True`` the input is checked to be a 2-D float array."""

    def __init__(self, func=None, inverse_func=None, *, validate=False, accept_sparse=False,
                 check_inverse=True, feature_names_out=None, kw_args=None, inv_kw_args=None):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args

    def _check_input(self, X, *, reset):
        if self.validate:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim != 2:
                raise ValueError(f"Expected a 2D array, got {X.ndim}D")
        if reset:
            if hasattr(X, "columns"):
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            if getattr(X, "ndim", 0) == 2:
                self.n_features_in_ = X.shape[1]
        elif self.validate and hasattr(self, "n_features_in_") \
                and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but FunctionTransformer is "
                             f"expecting {self.n_features_in_} features as input.")
        return X

    def _check_inverse_transform(self, X):
        X = np.asarray(X)
        rows = slice(None, None, max(1, X.shape[0] // 100))
        X_round_trip = self.inverse_transform(self.transform(X[rows]))
        try:
            ok = np.allclose(np.asarray(X_round_trip, dtype=np.float64),
                             np.asarray(X[rows], dtype=np.float64), equal_nan=True)
        except (TypeError, ValueError):
            ok = True  # not numeric: nothing to compare
        if not ok:
            warnings.warn("The provided functions are not strictly inverse of each other. "
                          "If you are sure you want to proceed regardless, set "
                          "'check_inverse=False'.", UserWarning)

    def fit(self, X, y=None):
        X = self._check_input(X, reset=True)
        if self.check_inverse and not (self.func is None or self.inverse_func is None):
            self._check_inverse_transform(X)
        self._fitted = True
        return self

    def _apply(self, func, X, kw_args):
        return X if func is None else func(X, **(kw_args or {}))

    def transform(self, X):
        X = self._check_input(X, reset=False)
        return self._apply(self.func, X, self.kw_args)

    def inverse_transform(self, X):
        if self.validate:
            X = np.asarray(X, dtype=np.float64)
        return self._apply(self.inverse_func, X, self.inv_kw_args)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if not getattr(self, "_fitted", False):
            raise NotFittedError("This FunctionTransformer instance is not fitted yet.")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None and hasattr(self, "n_features_in_"):
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        if self.feature_names_out == "one-to-one":
            return np.asarray(input_features, dtype=object)
        if callable(self.feature_names_out):
            return np.asarray(self.feature_names_out(self, input_features), dtype=object)
        raise AttributeError("This FunctionTransformer has no get_feature_names_out: set "
                             "feature_names_out='one-to-one' or a callable.")
