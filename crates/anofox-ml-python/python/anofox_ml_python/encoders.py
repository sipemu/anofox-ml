"""Label and dict encoders. ``LabelEncoder`` and ``OrdinalEncoder`` run in
Rust on numeric categories; other labels (strings, objects) are first coded
by their sorted order here. ``LabelBinarizer``, ``MultiLabelBinarizer`` and
``DictVectorizer`` only map labels or feature names to columns, so they are
plain Python; outputs are dense."""

from __future__ import annotations

from numbers import Number

import numpy as np

from ._base import BaseEstimator, NotFittedError, TransformerMixin


def _check_fitted(est, attr):
    if not hasattr(est, attr):
        raise NotFittedError(
            f"This {type(est).__name__} instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator."
        )


def _is_nan(v):
    return isinstance(v, (float, np.floating)) and np.isnan(v)


def _numeric(values):
    """Whether a 1-D array holds numbers only (NaN allowed)."""
    if values.dtype.kind in "biuf":
        return True
    return all(isinstance(v, Number) and not isinstance(v, (str, bytes)) for v in values)


class _Coder:
    """Codes the non-numeric categories of one column by their sorted
    order (the Rust encoders see the codes): ``None`` sorts last, as in
    sklearn, NaN is missing (NaN), and an unseen category is -1 (never a
    code, so the Rust encoder treats it as unknown)."""

    def __init__(self, values):
        present = [v for v in values if v is not None and not _is_nan(v)]
        cats = list(np.unique(np.asarray(present, dtype=object))) if present else []
        if any(v is None for v in values):
            cats.append(None)
        self.categories = np.empty(len(cats), dtype=object)
        self.categories[:] = cats
        self.index = {c: i for i, c in enumerate(cats)}

    def __call__(self, values):
        return np.asarray([np.nan if _is_nan(v) else self.index.get(v, -1.0) for v in values],
                          dtype=np.float64)


class LabelEncoder(BaseEstimator):
    """Labels as ``0..n_classes-1`` in sorted order (``classes_``)."""

    _kind = "LabelEncoder"

    def __init__(self):
        pass

    @staticmethod
    def _labels(y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y should be a 1d array")
        return y

    def _codes(self, y):
        if self._coder is None:
            return np.ascontiguousarray(y, dtype=np.float64).reshape(-1, 1)
        return self._coder(y).reshape(-1, 1)

    def fit(self, y):
        y = self._labels(y)
        self._coder = None if _numeric(y) else _Coder(y)
        self._fit_model(self._codes(y))
        if self._coder is None:
            vocab = [np.nan if v is None else v for v in self._fitted_state()["vocab"]]
            self.classes_ = np.asarray(vocab, dtype=np.float64).astype(y.dtype)
        else:
            self.classes_ = self._coder.categories.astype(y.dtype)
        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def transform(self, y):
        _check_fitted(self, "classes_")
        y = self._labels(y)
        if len(y) == 0:
            return np.empty(0, dtype=np.int64)
        codes = self._codes(y)
        unseen = codes[:, 0] < 0 if self._coder is not None else ~np.isin(y, self.classes_)
        if unseen.any():
            raise ValueError("y contains previously unseen labels: "
                             f"{np.unique(y[unseen].astype(str)).tolist()}")
        return np.asarray(self._call("transform", codes), dtype=np.int64)

    def inverse_transform(self, y):
        _check_fitted(self, "classes_")
        y = np.asarray(y)
        if len(y) == 0:
            return np.empty(0, dtype=self.classes_.dtype)
        y = np.ascontiguousarray(self._labels(y), dtype=np.float64).reshape(-1, 1)
        out = np.asarray(self._call("inverse_transform", y))
        if self._coder is None:
            return out.astype(self.classes_.dtype)
        return self.classes_[out.astype(np.int64)]


class OrdinalEncoder(TransformerMixin, BaseEstimator):
    """Each column's categories as ``0..n_categories-1`` in sorted order
    (``categories_``). A missing value (NaN / ``None``) is a category of its
    own, encoded as ``encoded_missing_value``; with
    ``handle_unknown="use_encoded_value"`` unseen categories become
    ``unknown_value``. Only ``categories="auto"`` and no infrequent
    grouping (``min_frequency`` / ``max_categories``) are supported."""

    _kind = "OrdinalEncoder"

    def __init__(self, *, categories="auto", dtype=np.float64, handle_unknown="error",
                 unknown_value=None, encoded_missing_value=np.nan, min_frequency=None,
                 max_categories=None):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    @staticmethod
    def _code_value(v):
        return None if v is None or np.isnan(v) else float(v)

    def _rust_params(self, X, y):
        if not (isinstance(self.categories, str) and self.categories == "auto"):
            raise ValueError("only categories='auto' is supported")
        if self.min_frequency is not None or self.max_categories is not None:
            raise ValueError("min_frequency / max_categories are not supported")
        if self.handle_unknown == "use_encoded_value":
            u = self.unknown_value
            if not _is_nan(u) and not (
                    isinstance(u, (int, np.integer)) and not isinstance(u, bool)):
                raise TypeError("unknown_value should be an integer or np.nan when "
                                f"handle_unknown is 'use_encoded_value', got {u}.")
            unknown = {"UseEncodedValue": self._code_value(u)}
        elif self.handle_unknown == "error":
            if self.unknown_value is not None:
                raise TypeError("unknown_value should only be set when handle_unknown is "
                                f"'use_encoded_value', got {self.unknown_value}.")
            unknown = "Error"
        else:
            raise ValueError("handle_unknown must be 'error' or 'use_encoded_value', got "
                             f"{self.handle_unknown!r}")
        return {"handle_unknown": unknown,
                "encoded_missing_value": self._code_value(self.encoded_missing_value)}

    def _columns(self, X):
        X = np.asarray(X, dtype=None if hasattr(X, "dtype") else object)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {X.ndim}D")
        if X.dtype.kind not in "biuf":
            X = X.astype(object)
        return X

    def _codes(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but OrdinalEncoder is expecting "
                             f"{self.n_features_in_} features as input.")
        out = np.empty(X.shape, dtype=np.float64)
        for j, coder in enumerate(self._coders):
            out[:, j] = X[:, j].astype(np.float64) if coder is None else coder(X[:, j])
        return out

    def fit(self, X, y=None):
        X = self._columns(X)
        self.n_features_in_ = X.shape[1]
        self._coders = [None if _numeric(X[:, j]) else _Coder(X[:, j])
                        for j in range(X.shape[1])]
        self._fit_model(self._codes(X))
        self.categories_ = []
        for coder, vocab in zip(self._coders, self._fitted_state()["vocabularies"]):
            if coder is None:
                self.categories_.append(np.asarray(
                    [np.nan if v is None else v for v in vocab], dtype=np.float64))
            else:
                missing = bool(vocab) and vocab[-1] is None  # NaN in the JSON
                cats = np.empty(len(coder.categories) + missing, dtype=object)
                cats[:len(coder.categories)] = coder.categories
                if missing:
                    cats[-1] = np.nan
                self.categories_.append(cats)
        return self

    def transform(self, X):
        _check_fitted(self, "categories_")
        out = np.asarray(self._call("transform", self._codes(self._columns(X))))
        return out.astype(self.dtype, copy=False)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Codes back to categories; ``unknown_value`` codes give ``None``."""
        _check_fitted(self, "categories_")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.categories_):
            raise ValueError("Shape of the passed X data is not correct. Expected "
                             f"{len(self.categories_)} columns, got {X.shape[-1]}.")
        unknown = (float(self.unknown_value) if self.handle_unknown == "use_encoded_value"
                   else None)
        out = np.empty(X.shape, dtype=object)
        for j, cats in enumerate(self.categories_):
            missing = len(cats) > 0 and _is_nan(cats[-1])
            n = len(cats) - int(missing)
            for i, v in enumerate(X[:, j]):
                if missing and _same(v, float(self.encoded_missing_value)):
                    out[i, j] = np.nan
                elif unknown is not None and _same(v, unknown):
                    out[i, j] = None
                elif 0 <= v < n and float(v).is_integer():
                    out[i, j] = cats[int(v)]
                else:
                    raise ValueError(f"encoded value {v} is out of range for column {j}")
        if all(c is None for c in self._coders) and not any(v is None for v in out.flat):
            return out.astype(np.float64)
        return out


def _same(a, b):
    return np.isnan(a) if np.isnan(b) else a == b


class LabelBinarizer(BaseEstimator):
    """One column per class (``pos_label`` / ``neg_label``); two classes give
    one column, as in sklearn."""

    def __init__(self, *, neg_label=0, pos_label=1):
        self.neg_label = neg_label
        self.pos_label = pos_label

    def fit(self, y):
        if self.neg_label >= self.pos_label:
            raise ValueError(f"neg_label={self.neg_label} must be strictly less than "
                             f"pos_label={self.pos_label}.")
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        self.classes_ = np.unique(y)
        self.y_type_ = "binary" if len(self.classes_) <= 2 else "multiclass"
        return self

    def transform(self, y):
        _check_fitted(self, "classes_")
        y = np.asarray(y)
        out = np.full((len(y), len(self.classes_)), self.neg_label, dtype=np.int64)
        idx = np.searchsorted(self.classes_, y)
        known = (idx < len(self.classes_)) & (self.classes_[np.minimum(idx, len(self.classes_) - 1)] == y)
        out[np.flatnonzero(known), idx[known]] = self.pos_label
        if len(self.classes_) == 1:
            return np.full((len(y), 1), self.neg_label, dtype=np.int64)
        if len(self.classes_) == 2:
            return out[:, 1:]
        return out

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, Y, threshold=None):
        _check_fitted(self, "classes_")
        Y = np.asarray(Y)
        if len(self.classes_) <= 2:
            t = (self.pos_label + self.neg_label) / 2 if threshold is None else threshold
            col = Y.reshape(len(Y), -1)[:, 0]
            if len(self.classes_) == 1:
                return np.repeat(self.classes_, len(Y))
            return self.classes_[(col > t).astype(np.int64)]
        return self.classes_[np.argmax(Y, axis=1)]


class MultiLabelBinarizer(BaseEstimator):
    """One 0/1 column per label of the label sets; ``classes`` fixes the
    column order (default: the sorted labels seen in ``fit``)."""

    def __init__(self, *, classes=None):
        self.classes = classes

    def fit(self, y):
        if self.classes is not None:
            classes = list(self.classes)
            if len(set(classes)) != len(classes):
                raise ValueError("The classes argument contains duplicate classes.")
            self.classes_ = np.empty(len(classes), dtype=object)
            self.classes_[:] = classes
        else:
            labels = sorted({label for labels in y for label in labels})
            self.classes_ = np.asarray(labels) if labels else np.empty(0)
        return self

    def transform(self, y):
        _check_fitted(self, "classes_")
        index = {c: i for i, c in enumerate(self.classes_.tolist())}
        out = np.zeros((len(y), len(index)), dtype=np.int64)
        for row, labels in enumerate(y):
            for label in labels:
                if label in index:
                    out[row, index[label]] = 1
        return out

    def fit_transform(self, y):
        y = [list(labels) for labels in y]
        return self.fit(y).transform(y)

    def inverse_transform(self, yt):
        _check_fitted(self, "classes_")
        yt = np.asarray(yt)
        if yt.shape[1] != len(self.classes_):
            raise ValueError(f"Expected indicator for {len(self.classes_)} classes, but got "
                             f"{yt.shape[1]}")
        return [tuple(self.classes_[np.flatnonzero(row)].tolist()) for row in yt]


class DictVectorizer(BaseEstimator):
    """One column per feature name; a string value ``v`` of feature ``f``
    becomes the one-hot column ``f=v`` (``separator``), lists / tuples of
    strings one column each. ``transform`` is dense; unseen features are
    dropped."""

    def __init__(self, *, separator="=", sort=True):
        self.separator = separator
        self.sort = sort

    def _items(self, d):
        for f, v in d.items():
            if isinstance(v, str):
                yield f"{f}{self.separator}{v}", 1.0
            elif isinstance(v, Number) or v is None:
                if v is not None:
                    yield f, float(v)
            elif isinstance(v, (list, tuple, set, frozenset)):
                for item in v:
                    if not isinstance(item, str):
                        raise TypeError(f"Unsupported type {type(item)} in iterable value. "
                                        "Only iterables of string are supported.")
                    yield f"{f}{self.separator}{item}", 1.0
            else:
                raise TypeError(f"Unsupported value type {type(v)} for {f}: {v}.")

    def fit(self, X, y=None):
        names = []
        seen = set()
        for d in X:
            for f, _ in self._items(d):
                if f not in seen:
                    seen.add(f)
                    names.append(f)
        if self.sort:
            names.sort()
        self.feature_names_ = names
        self.vocabulary_ = {f: i for i, f in enumerate(names)}
        return self

    def transform(self, X):
        _check_fitted(self, "vocabulary_")
        out = np.zeros((len(X), len(self.vocabulary_)))
        for row, d in enumerate(X):
            for f, v in self._items(d):
                col = self.vocabulary_.get(f)
                if col is not None:
                    out[row, col] += v
        return out

    def fit_transform(self, X, y=None):
        X = list(X)
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        _check_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)

    def inverse_transform(self, X, dict_type=dict):
        _check_fitted(self, "feature_names_")
        X = np.asarray(X)
        return [dict_type((self.feature_names_[j], row[j]) for j in np.flatnonzero(row))
                for row in X]
