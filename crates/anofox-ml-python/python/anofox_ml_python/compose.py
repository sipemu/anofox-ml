"""sklearn's composition tools over this package's estimators, run in Rust:
``Pipeline`` (and ``make_pipeline``), ``FeatureUnion`` and
``ColumnTransformer``. As in ``meta``, the steps' kinds and parameters go to
the Rust registry, so a fitted composite is one Rust model (it pickles). The
given steps stay unfitted.
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
from .meta import _spec


def _skip(est):
    return est is None or (isinstance(est, str) and est in ("passthrough", "drop"))


class Pipeline(BaseEstimator):
    """The steps but the last transform in order; the last one (an
    estimator or a transformer) is fitted on their output and gets the
    sample weights. ``"passthrough"`` / ``None`` steps are skipped."""

    _kind = "Pipeline"

    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        return dict(self.steps)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.named_steps[key]
        return self.steps[key][1]

    def _final(self):
        if not self.steps:
            raise ValueError("Pipeline needs at least one step")
        return self.steps[-1][1]

    @property
    def _estimator_type(self):
        return getattr(self._final(), "_estimator_type", None)

    def _is_classifier(self):
        return self._estimator_type == "classifier"

    def _rust_params(self, X, y):
        names = [name for name, _ in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"step names must be unique, got {names}")
        specs = [_spec(est, X, y, role=f"step {name!r}")
                 for name, est in self.steps if not _skip(est)]
        if not specs:
            raise ValueError("Pipeline needs at least one estimator step")
        return {"steps": specs}

    def fit(self, X, y=None, sample_weight=None):
        if self._is_classifier():
            return ClassifierMixin.fit(self, X, y, sample_weight)
        y = None if y is None else _check_1d(y)
        return self._fit_model(X, y, sample_weight)

    def predict(self, X):
        if self._is_classifier():
            return ClassifierMixin.predict(self, X)
        return self._call("predict", X)

    def predict_proba(self, X):
        return self._call("predict_proba", X)

    def decision_function(self, X):
        if self._is_classifier():
            return ClassifierMixin.decision_function(self, X)
        return self._call("decision_function", X)

    def transform(self, X):
        return self._call("transform", X)

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).predict(X)

    def score(self, X, y, sample_weight=None):
        mixin = ClassifierMixin if self._is_classifier() else RegressorMixin
        return mixin.score(self, X, y, sample_weight)

    @property
    def classes_(self):
        classes = self.__dict__.get("_classes")
        if classes is None:
            raise AttributeError("'Pipeline' object has no attribute 'classes_'")
        return classes

    @classes_.setter
    def classes_(self, value):
        self.__dict__["_classes"] = value


def make_pipeline(*steps):
    """A ``Pipeline`` with its steps named after their lowercase class names."""
    names, out = {}, []
    for est in steps:
        name = type(est).__name__.lower()
        names[name] = names.get(name, 0) + 1
        out.append((name, est))
    counts = {}
    named = []
    for name, est in out:
        if names[name] > 1:
            counts[name] = counts.get(name, 0) + 1
            name = f"{name}-{counts[name]}"
        named.append((name, est))
    return Pipeline(named)


class FeatureUnion(TransformerMixin, BaseEstimator):
    """Every transformer on all columns; the outputs side by side, each
    multiplied by its ``transformer_weights[name]`` (default 1)."""

    _kind = "FeatureUnion"

    def __init__(self, transformer_list, *, transformer_weights=None):
        self.transformer_list = transformer_list
        self.transformer_weights = transformer_weights

    def _rust_params(self, X, y):
        kept = [(name, t) for name, t in self.transformer_list if not _skip(t)]
        if not kept:
            raise ValueError("FeatureUnion needs at least one transformer")
        weights = None
        if self.transformer_weights is not None:
            weights = [float(self.transformer_weights.get(name, 1.0)) for name, _ in kept]
        return {"transformer_list": [_spec(t, X, y, role=f"transformer {name!r}")
                                     for name, t in kept],
                "transformer_weights": weights}

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("FeatureUnion does not take sample_weight")
        y = None if y is None else _check_1d(y)
        return self._fit_model(X, y)

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)


class ColumnTransformer(TransformerMixin, BaseEstimator):
    """Each ``(name, transformer, columns)`` on its columns (indices, a
    boolean mask, a slice, or DataFrame column names), the outputs side by
    side, then the unused columns if ``remainder="passthrough"``. A
    transformer may be ``"passthrough"`` or ``"drop"``."""

    _kind = "ColumnTransformer"

    def __init__(self, transformers, *, remainder="drop"):
        self.transformers = transformers
        self.remainder = remainder

    def _columns(self, cols, n, names):
        if isinstance(cols, slice):
            return list(range(n))[cols]
        cols = [cols] if np.isscalar(cols) else list(cols)
        if cols and all(isinstance(c, (bool, np.bool_)) for c in cols):
            if len(cols) != n:
                raise ValueError("a boolean column mask needs one entry per column")
            return [i for i, keep in enumerate(cols) if keep]
        out = []
        for c in cols:
            if isinstance(c, str):
                if names is None:
                    raise ValueError("column names need a DataFrame")
                out.append(names.index(c))
            else:
                c = int(c)
                out.append(c + n if c < 0 else c)
        return out

    def _resolved(self, X):
        names = list(map(str, X.columns)) if hasattr(X, "columns") else None
        Xa = _check_X(X)
        steps = []
        for name, t, cols in self.transformers:
            idx = self._columns(cols, Xa.shape[1], names)
            if isinstance(t, str) and t == "drop":
                continue
            steps.append((name, t, idx))
        return Xa, steps

    def _rust_params(self, X, y):
        if self.remainder not in ("drop", "passthrough"):
            raise ValueError("remainder must be 'drop' or 'passthrough'")
        Xa, steps = self._resolved(X)
        return {"transformers": [
            {"transformer": None if (isinstance(t, str) and t == "passthrough")
             else _spec(t, Xa[:, idx], y, role=f"transformer {name!r}"),
             "columns": idx}
            for name, t, idx in steps],
            "remainder": self.remainder}

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("ColumnTransformer does not take sample_weight")
        y = None if y is None else _check_1d(y)
        model = self._new_model(X, y)
        Xa = _check_X(X)
        model.fit(Xa, y, None)
        self._model_ = model
        self.n_features_in_ = Xa.shape[1]
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)
