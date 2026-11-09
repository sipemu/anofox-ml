"""Label and dict encoders (sklearn's ``LabelBinarizer``,
``MultiLabelBinarizer`` and ``DictVectorizer``). They only map labels or
feature names to columns, so they are plain Python; outputs are dense."""

from __future__ import annotations

from numbers import Number

import numpy as np

from ._base import BaseEstimator, NotFittedError


def _check_fitted(est, attr):
    if not hasattr(est, attr):
        raise NotFittedError(
            f"This {type(est).__name__} instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator."
        )


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
