"""Minimal model-selection helpers. The estimators are sklearn-compatible,
so ``sklearn.model_selection`` (GridSearchCV, cross_validate, ...) works too.
"""

import json

import numpy as np

from . import _native
from ._base import _check_1d, _check_X, _seed

__all__ = [
    "Categorical",
    "Float",
    "Int",
    "KFold",
    "TuneSearchCV",
    "cross_val_score",
    "train_test_split",
]


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


# -- hyperparameter search (anofox-ml-tune, built on optuna/rustuna) --------


def Int(low, high, step=1, log=False):
    """Integer range ``[low, high]`` for :class:`TuneSearchCV`."""
    spec = {"int": [int(low), int(high)], "log": bool(log)}
    if step != 1:
        spec["step"] = int(step)
    return spec


def Float(low, high, step=None, log=False):
    """Float range ``[low, high]`` (``log`` samples on a log scale)."""
    spec = {"float": [float(low), float(high)], "log": bool(log)}
    if step is not None:
        spec["step"] = float(step)
    return spec


def Categorical(choices):
    """One of ``choices`` (JSON-serialisable values: numbers, strings,
    booleans, ``None``)."""
    return {"categorical": list(choices)}


def _space(param_distributions):
    """Search space JSON; a plain list or tuple means ``Categorical``."""
    if not isinstance(param_distributions, dict) or not param_distributions:
        raise ValueError("param_distributions must be a non-empty dict")
    space = {}
    for name, spec in param_distributions.items():
        if isinstance(spec, (list, tuple)):
            spec = Categorical(spec)
        if not isinstance(spec, dict):
            raise ValueError(
                f"{name!r}: use Int(...), Float(...), Categorical(...) or a list of choices"
            )
        space[name] = spec
    return space


class TuneSearchCV:
    """Optuna-style hyperparameter search (TPE, random or NSGA-II) for an
    anofox-ml estimator, run in Rust with the GIL released.

    ``param_distributions`` maps the estimator's (scikit-learn) parameter
    names to :func:`Int`, :func:`Float`, :func:`Categorical` or a list of
    choices; the other parameters keep the estimator's values. Each trial is
    scored by cross-validation (``cv`` folds, stratified for classifiers, or
    any splitter with ``split(X, y, groups)``) with ``scoring``: a metric
    name (``"r2"``, ``"rmse"``, ``"mae"``, ``"accuracy"``, ``"f1_macro"``,
    ``"roc_auc"``, ``"log_loss"``, ...; default accuracy for classifiers, R²
    otherwise). ``objectives`` adds further objectives (metric names,
    ``"fit_time"``, ``"model_size"``) for a multi-objective search; the
    Pareto front is in ``pareto_front_``.

    After ``fit``: ``best_params_``, ``best_score_`` (in the metric's own
    sign: losses are positive and minimised), ``best_index_``, ``trials_``
    (one dict per trial), ``pareto_front_`` and, with ``refit``,
    ``best_estimator_`` (a clone of ``estimator`` with ``best_params_``,
    fitted on all data), to which ``predict`` & co. delegate.
    """

    def __init__(self, estimator, param_distributions, *, n_trials=50, scoring=None,
                 objectives=None, cv=5, sampler="tpe", n_startup_trials=10,
                 population_size=50, random_state=None, refit=True):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.scoring = scoring
        self.objectives = objectives
        self.cv = cv
        self.sampler = sampler
        self.n_startup_trials = n_startup_trials
        self.population_size = population_size
        self.random_state = random_state
        self.refit = refit

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in (
            "estimator", "param_distributions", "n_trials", "scoring", "objectives", "cv",
            "sampler", "n_startup_trials", "population_size", "random_state", "refit")}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                raise ValueError(f"Invalid parameter {k!r} for TuneSearchCV")
            setattr(self, k, v)
        return self

    def _clone(self, params):
        est = self.estimator
        return type(est)(**{**est.get_params(), **params})

    def fit(self, X, y, groups=None, sample_weight=None):
        est = self.estimator
        kind = getattr(est, "_kind", "")
        if not kind:
            raise ValueError("estimator must be an anofox_ml_python estimator")
        valid = set(est.get_params())
        space = _space(self.param_distributions)
        unknown = sorted(set(space) - valid)
        if unknown:
            raise ValueError(
                f"{type(est).__name__} has no parameters {unknown}; valid: {sorted(valid)}"
            )

        X_arr = _check_X(X)
        if getattr(est, "_estimator_type", None) == "classifier":
            y_arr = np.asarray(y)
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_arr = y_arr.ravel()
            classes, y_enc = np.unique(y_arr, return_inverse=True)
            y_fit = np.ascontiguousarray(y_enc, dtype=np.float64)
        else:
            classes = None
            y_fit = _check_1d(y)
        w = None if sample_weight is None else _check_1d(sample_weight, "sample_weight")

        splits = None
        if not isinstance(self.cv, (int, np.integer)):
            splits = [
                (np.asarray(tr, dtype=np.int64).tolist(), np.asarray(te, dtype=np.int64).tolist())
                for tr, te in self.cv.split(X_arr, y_fit if classes is None else y_arr, groups)
            ]

        def map_params(suggested_json):
            trial_est = self._clone(json.loads(suggested_json))
            if classes is not None:
                trial_est.classes_ = classes  # used by class_weight encoding
            return json.dumps(trial_est._rust_params(X_arr, y_fit))

        options = {
            "n_trials": int(self.n_trials),
            "sampler": str(self.sampler).lower(),
            "n_startup_trials": int(self.n_startup_trials),
            "population_size": int(self.population_size),
            "seed": _seed(self.random_state),
            "cv": int(self.cv) if splits is None else 5,
            "metric": self.scoring,
            "objectives": list(self.objectives or []),
            "refit": False,
        }
        result_json, _ = _native.tune(
            kind, json.dumps(space), X_arr, y_fit, json.dumps(options), w, splits, map_params
        )
        result = json.loads(result_json)

        self.trials_ = result["trials"]
        self.best_index_ = result["best_trial"]
        self.best_params_ = self.trials_[self.best_index_]["params"]
        self.best_score_ = result["best_score"]
        self.scoring_ = result["metric"]
        self.objectives_ = [tuple(o) for o in result["objectives"]]
        self.pareto_front_ = result["pareto_front"]
        self.n_features_in_ = X_arr.shape[1]
        if classes is not None:
            self.classes_ = classes
        if self.refit:
            best = self._clone(self.best_params_)
            if w is None:
                best.fit(X, y)
            else:
                best.fit(X, y, sample_weight=w)
            self.best_estimator_ = best
        return self

    def _best(self):
        if not hasattr(self, "best_estimator_"):
            raise AttributeError("TuneSearchCV has no best_estimator_ (not fitted, or refit=False)")
        return self.best_estimator_

    def predict(self, X):
        return self._best().predict(X)

    def predict_proba(self, X):
        return self._best().predict_proba(X)

    def decision_function(self, X):
        return self._best().decision_function(X)

    def transform(self, X):
        return self._best().transform(X)

    def score(self, X, y, sample_weight=None):
        return self._best().score(X, y, sample_weight=sample_weight)
