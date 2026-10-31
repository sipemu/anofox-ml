"""Minimal model-selection helpers. The estimators are sklearn-compatible,
so ``sklearn.model_selection`` (GridSearchCV, cross_validate, ...) works too.
"""

import numpy as np

from ._base import _seed

__all__ = ["KFold", "cross_val_score", "train_test_split"]


def train_test_split(*arrays, test_size=None, train_size=None, random_state=None,
                     shuffle=True, stratify=None):
    """Split arrays into random train and test subsets (sklearn semantics for
    ``test_size`` / ``train_size`` as fractions or counts; default 25 % test)."""
    if not arrays:
        raise ValueError("At least one array required as input")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("Found input variables with inconsistent numbers of samples")

    def count(size, default):
        if size is None:
            return default
        if isinstance(size, float):
            return int(np.ceil(size * n)) if size < 1 else n
        return int(size)

    n_test = count(test_size, None)
    n_train = count(train_size, None)
    if n_test is None and n_train is None:
        n_test = int(np.ceil(0.25 * n))
    if n_test is None:
        n_test = n - n_train
    if n_train is None:
        n_train = n - n_test
    if n_train + n_test > n or n_train <= 0 or n_test <= 0:
        raise ValueError(f"invalid split sizes train={n_train}, test={n_test} for n={n}")

    rng = np.random.RandomState(_seed(random_state))
    if stratify is not None:
        if not shuffle:
            raise ValueError("stratified splits require shuffle=True")
        stratify = np.asarray(stratify)
        test_idx = []
        classes, y_idx = np.unique(stratify, return_inverse=True)
        for k in range(len(classes)):
            members = rng.permutation(np.flatnonzero(y_idx == k))
            take = int(round(n_test * len(members) / n))
            test_idx.extend(members[:take])
        test_idx = np.asarray(sorted(test_idx))
        rest = np.setdiff1d(np.arange(n), test_idx)
        train_idx = rng.permutation(rest)[:n_train]
    else:
        perm = rng.permutation(n) if shuffle else np.arange(n)
        test_idx = perm[n - n_test:] if not shuffle else perm[:n_test]
        train_idx = perm[:n_train] if not shuffle else perm[n_test:n_test + n_train]

    out = []
    for a in arrays:
        a = np.asarray(a)
        out.extend([a[train_idx], a[test_idx]])
    return out


class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            idx = np.random.RandomState(_seed(self.random_state)).permutation(n)
        sizes = np.full(self.n_splits, n // self.n_splits)
        sizes[: n % self.n_splits] += 1
        start = 0
        for size in sizes:
            test = idx[start:start + size]
            yield np.setdiff1d(idx, test, assume_unique=True), np.sort(test)
            start += size


def cross_val_score(estimator, X, y, cv=5, scoring=None):
    """Per-fold scores of a fresh copy of ``estimator`` (``scoring`` is a
    callable ``scorer(estimator, X_test, y_test)`` or ``None`` for
    ``estimator.score``)."""
    X = np.asarray(X)
    y = np.asarray(y)
    splitter = KFold(cv) if isinstance(cv, int) else cv
    scores = []
    for train, test in splitter.split(X, y):
        est = type(estimator)(**estimator.get_params())
        est.fit(X[train], y[train])
        scores.append(est.score(X[test], y[test]) if scoring is None
                      else scoring(est, X[test], y[test]))
    return np.asarray(scores)
