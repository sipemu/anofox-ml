"""Model selection: splitters, ``GridSearchCV`` / ``RandomizedSearchCV``
(candidates cross-validated in Rust) and the Optuna-style ``TuneSearchCV``.
The estimators are sklearn-compatible, so ``sklearn.model_selection`` works
too.
"""

import json
import warnings
from itertools import product

import numpy as np

from . import _native
from ._base import _check_1d, _check_X, _seed

__all__ = [
    "Categorical",
    "Float",
    "GridSearchCV",
    "Int",
    "KFold",
    "ParameterGrid",
    "ParameterSampler",
    "RandomizedSearchCV",
    "StratifiedKFold",
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


def _random_state(random_state):
    """sklearn's ``check_random_state``."""
    if random_state is None:
        return np.random.mtrand._rand
    if isinstance(random_state, (int, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError(f"{random_state!r} cannot be used to seed a numpy.random.RandomState")


class StratifiedKFold:
    """K folds with the class proportions of ``y`` (sklearn's allocation, so
    the same folds as ``sklearn.model_selection.StratifiedKFold``)."""

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _test_folds(self, y):
        rng = _random_state(self.random_state)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        # Classes in order of first appearance.
        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]
        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        if np.all(self.n_splits > y_counts):
            raise ValueError(f"n_splits={self.n_splits} cannot be greater than the number of "
                             "members in each class.")
        if self.n_splits > y_counts.min():
            warnings.warn(f"The least populated class in y has only {y_counts.min()} members, "
                          f"which is less than n_splits={self.n_splits}.", UserWarning)
        y_order = np.sort(y_encoded)
        allocation = np.asarray([np.bincount(y_order[i::self.n_splits], minlength=n_classes)
                                 for i in range(self.n_splits)])
        test_folds = np.empty(len(y), dtype=np.int64)
        for k in range(n_classes):
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def split(self, X, y, groups=None):
        test_folds = self._test_folds(y)
        for i in range(self.n_splits):
            yield np.flatnonzero(test_folds != i), np.flatnonzero(test_folds == i)


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


# -- GridSearchCV / RandomizedSearchCV --------------------------------------


class ParameterGrid:
    """Every combination of the values of a dict (or of each dict of a list),
    in sklearn's order: keys sorted, the last key varying fastest."""

    def __init__(self, param_grid):
        if isinstance(param_grid, dict):
            param_grid = [param_grid]
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f"Parameter grid is not a dict ({grid!r})")
            for key, value in grid.items():
                if isinstance(value, np.ndarray) and value.ndim > 1:
                    raise ValueError(f"Parameter array for {key!r} should be one-dimensional")
                if isinstance(value, str) or not hasattr(value, "__len__"):
                    raise TypeError(f"Parameter grid for parameter {key!r} needs to be a list "
                                    f"or a numpy array, but got {value!r}")
                if len(value) == 0:
                    raise ValueError(f"Parameter grid for parameter {key!r} need to be a "
                                     f"non-empty sequence, got: {value!r}")
        self.param_grid = param_grid

    def __iter__(self):
        for grid in self.param_grid:
            items = sorted(grid.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    yield dict(zip(keys, v))

    def __len__(self):
        return sum(int(np.prod([len(v) for v in grid.values()])) if grid else 1
                   for grid in self.param_grid)

    def __getitem__(self, ind):
        for grid in self.param_grid:
            if not grid:
                if ind == 0:
                    return {}
                ind -= 1
                continue
            keys, values = zip(*sorted(grid.items())[::-1])
            sizes = [len(v) for v in values]
            total = int(np.prod(sizes))
            if ind >= total:
                ind -= total
            else:
                out = {}
                for key, v, n in zip(keys, values, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v[offset]
                return out
        raise IndexError("ParameterGrid index out of range")


def _sample_without_replacement(n_population, n_samples, rng):
    """sklearn's ``sample_without_replacement(method="auto")``."""
    ratio = n_samples / n_population if n_population else 1.0
    if 0.01 < ratio < 0.99:
        return rng.permutation(n_population)[:n_samples]
    if ratio < 0.2:
        selected, out = set(), []
        for _ in range(n_samples):
            j = rng.randint(n_population)
            while j in selected:
                j = rng.randint(n_population)
            selected.add(j)
            out.append(j)
        return out
    out = list(range(n_samples))
    for i in range(n_samples, n_population):
        j = rng.randint(0, i + 1)
        if j < n_samples:
            out[j] = i
    return out


class ParameterSampler:
    """``n_iter`` parameter sets drawn from ``param_distributions`` (a dict,
    or a list of dicts picked uniformly): lists are sampled uniformly,
    distributions with ``rvs`` (scipy.stats) by ``rvs``. When every value is
    a list, the sets are drawn from the grid without replacement. The draws
    are sklearn's for the same ``random_state``."""

    def __init__(self, param_distributions, n_iter, *, random_state=None):
        if isinstance(param_distributions, dict):
            param_distributions = [param_distributions]
        for dist in param_distributions:
            if not isinstance(dist, dict):
                raise TypeError(f"Parameter distribution is not a dict ({dist!r})")
            for key, value in dist.items():
                if not isinstance(value, (list, tuple, np.ndarray)) and not hasattr(value, "rvs"):
                    raise TypeError(f"Parameter grid for parameter {key!r} is not iterable or "
                                    f"a distribution (value={value})")
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _is_all_lists(self):
        return all(not hasattr(v, "rvs") for dist in self.param_distributions
                   for v in dist.values())

    def __iter__(self):
        rng = _random_state(self.random_state)
        if self._is_all_lists():
            grid = ParameterGrid(self.param_distributions)
            n_iter = self.n_iter
            if len(grid) < n_iter:
                warnings.warn(f"The total space of parameters {len(grid)} is smaller than "
                              f"n_iter={self.n_iter}. Running {len(grid)} iterations. For "
                              "exhaustive searches, use GridSearchCV.", UserWarning)
                n_iter = len(grid)
            for i in _sample_without_replacement(len(grid), n_iter, rng):
                yield grid[i]
        else:
            for _ in range(self.n_iter):
                dist = rng.choice(self.param_distributions)
                params = {}
                for k, v in sorted(dist.items()):
                    params[k] = v.rvs(random_state=rng) if hasattr(v, "rvs") \
                        else v[rng.randint(len(v))]
                yield params

    def __len__(self):
        if self._is_all_lists():
            return min(self.n_iter, len(ParameterGrid(self.param_distributions)))
        return self.n_iter


#: sklearn scoring name -> (metric of anofox-ml-tune, sign of sklearn's score).
_SCORERS = {
    "r2": ("r2", 1), "explained_variance": ("explained_variance", 1),
    "neg_mean_squared_error": ("mse", -1), "neg_root_mean_squared_error": ("rmse", -1),
    "neg_mean_absolute_error": ("mae", -1),
    "neg_median_absolute_error": ("median_absolute_error", -1),
    "neg_mean_absolute_percentage_error": ("mape", -1),
    "accuracy": ("accuracy", 1), "balanced_accuracy": ("balanced_accuracy", 1),
    "f1_macro": ("f1_macro", 1), "f1_weighted": ("f1_weighted", 1),
    "roc_auc": ("roc_auc", 1), "roc_auc_ovr": ("roc_auc", 1), "neg_log_loss": ("log_loss", -1),
}

# The step lists of the composites, for nested ``step__param`` names.
_STEP_LISTS = ("steps", "transformer_list", "transformers", "estimators")


def _with_params(est, params):
    """A clone of ``est`` with ``params`` (``name__param`` sets a parameter of
    the sub-estimator ``name``: a parameter, or a named step)."""
    own, nested = {}, {}
    for key, value in params.items():
        head, sep, rest = key.partition("__")
        if sep:
            nested.setdefault(head, {})[rest] = value
        else:
            own[key] = value
    current = est.get_params()
    unknown = sorted(set(own) - set(current))
    if unknown:
        raise ValueError(f"Invalid parameters {unknown} for estimator {type(est).__name__}; "
                         f"valid: {sorted(current)}")
    new = {**current, **own}
    for head, sub in nested.items():
        if head in new and hasattr(new[head], "get_params"):
            new[head] = _with_params(new[head], sub)
            continue
        for list_name in _STEP_LISTS:
            steps = new.get(list_name)
            if isinstance(steps, list) and any(step[0] == head for step in steps):
                new[list_name] = [
                    (step[0], _with_params(step[1], sub), *step[2:]) if step[0] == head
                    else step for step in steps]
                break
        else:
            raise ValueError(f"Invalid parameter {head!r} for estimator {type(est).__name__}")
    return type(est)(**new)


def _rank(scores):
    """sklearn's ``rank_test_score``: rank 1 is best, ties share the lowest
    rank, NaN scores rank last."""
    scores = np.asarray(scores, dtype=np.float64)
    if np.isnan(scores).all():
        return np.ones(len(scores), dtype=np.int32)
    scores = np.nan_to_num(scores, nan=np.nanmin(scores) - 1)
    return np.asarray([1 + np.sum(scores > s) for s in scores], dtype=np.int32)


class _BaseSearchCV:
    """Cross-validates every candidate of ``_candidates()`` in Rust (one
    GIL-released call), then refits the best one on all data, as sklearn's
    ``BaseSearchCV``: ``cv_results_``, ``best_index_``, ``best_score_``,
    ``best_params_`` and, with ``refit``, ``best_estimator_``."""

    def get_params(self, deep=True):
        import inspect

        names = [p for p in inspect.signature(type(self).__init__).parameters if p != "self"]
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params):
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for {type(self).__name__}")
            setattr(self, key, value)
        return self

    @property
    def _estimator_type(self):
        return getattr(self.estimator, "_estimator_type", None)

    def _metric(self):
        if self.scoring is None:
            return ("accuracy" if self._estimator_type == "classifier" else "r2"), 1
        if not isinstance(self.scoring, str) or self.scoring not in _SCORERS:
            raise ValueError(f"scoring must be None or one of {sorted(_SCORERS)}, got "
                             f"{self.scoring!r} (use sklearn.model_selection for callables)")
        return _SCORERS[self.scoring]

    def _splits(self, X, y, groups):
        cv = 5 if self.cv is None else self.cv
        if isinstance(cv, (int, np.integer)):
            cv = (StratifiedKFold(int(cv)) if self._estimator_type == "classifier"
                  else KFold(int(cv)))
        elif not hasattr(cv, "split"):
            cv = list(cv)  # an iterable of (train, test)
            return [(np.asarray(tr, dtype=np.int64).tolist(),
                     np.asarray(te, dtype=np.int64).tolist()) for tr, te in cv]
        return [(np.asarray(tr, dtype=np.int64).tolist(), np.asarray(te, dtype=np.int64).tolist())
                for tr, te in cv.split(X, y, groups)]

    def fit(self, X, y, groups=None, sample_weight=None):
        est = self.estimator
        kind = getattr(est, "_kind", "")
        if not kind:
            raise ValueError("estimator must be an anofox_ml_python estimator (use "
                             "sklearn.model_selection for others)")
        candidates = list(self._candidates())
        if not candidates:
            raise ValueError("No candidates to evaluate")
        for params in candidates:
            _with_params(est, params)  # fails early on an invalid name
        metric, sign = self._metric()

        X_arr = _check_X(X)
        if self._estimator_type == "classifier":
            y_arr = np.asarray(y)
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_arr = y_arr.ravel()
            classes, y_enc = np.unique(y_arr, return_inverse=True)
            y_fit = np.ascontiguousarray(y_enc, dtype=np.float64)
        else:
            classes, y_arr = None, None
            y_fit = _check_1d(y)
        w = None if sample_weight is None else _check_1d(sample_weight, "sample_weight")
        splits = self._splits(X_arr, y_fit if classes is None else y_arr, groups)

        def map_params(index_json):
            cand = _with_params(est, candidates[json.loads(index_json)["i"]])
            if classes is not None:
                cand.classes_ = classes  # used by class_weight encoding
            return json.dumps(cand._rust_params(X_arr, y_fit))

        # sklearn >= 1.8 weights the scorer too.
        options = {"metric": metric, "objectives": ["fit_time"], "weighted_scoring": True}
        records = json.loads(_native.evaluate_candidates(
            kind, json.dumps([{"i": i} for i in range(len(candidates))]), X_arr, y_fit,
            splits, json.dumps(options), w, map_params))

        n_splits = len(splits)
        fold_scores = np.empty((len(candidates), n_splits))
        fit_time = np.full(len(candidates), np.nan)
        for i, rec in enumerate(records):
            if rec["error"] is not None:
                if isinstance(self.error_score, str) and self.error_score == "raise":
                    raise ValueError(f"candidate {candidates[i]} failed: {rec['error']}")
                warnings.warn(f"candidate {candidates[i]} failed and is scored "
                              f"{self.error_score}: {rec['error']}", UserWarning)
                fold_scores[i] = self.error_score
            else:
                fold_scores[i] = sign * np.asarray(rec["fold_scores"])
                fit_time[i] = rec["values"][1]

        results = {"mean_fit_time": fit_time}
        names = sorted({k for c in candidates for k in c})
        for name in names:
            col = np.ma.MaskedArray(np.empty(len(candidates), dtype=object),
                                    mask=True)
            for i, c in enumerate(candidates):
                if name in c:
                    col[i] = c[name]
            results[f"param_{name}"] = col
        results["params"] = candidates
        for k in range(n_splits):
            results[f"split{k}_test_score"] = fold_scores[:, k]
        results["mean_test_score"] = fold_scores.mean(axis=1)
        results["std_test_score"] = fold_scores.std(axis=1)
        results["rank_test_score"] = _rank(results["mean_test_score"])

        self.cv_results_ = results
        self.best_index_ = int(results["rank_test_score"].argmin())
        self.best_score_ = float(results["mean_test_score"][self.best_index_])
        self.best_params_ = candidates[self.best_index_]
        self.n_splits_ = n_splits
        self.scorer_ = self.scoring
        self.n_features_in_ = X_arr.shape[1]
        if classes is not None:
            self.classes_ = classes
        if self.refit:
            import time

            best = _with_params(est, self.best_params_)
            start = time.perf_counter()
            if w is None:
                best.fit(X, y)
            else:
                best.fit(X, y, sample_weight=w)
            self.refit_time_ = time.perf_counter() - start
            self.best_estimator_ = best
        return self

    def _best(self):
        if not hasattr(self, "best_estimator_"):
            raise AttributeError(f"{type(self).__name__} has no best_estimator_ (not fitted, "
                                 "or refit=False)")
        return self.best_estimator_

    def predict(self, X):
        return self._best().predict(X)

    def predict_proba(self, X):
        return self._best().predict_proba(X)

    def decision_function(self, X):
        return self._best().decision_function(X)

    def transform(self, X):
        return self._best().transform(X)

    def score(self, X, y):
        """The best estimator's ``score`` (R² / accuracy) for
        ``scoring=None``, else the ``scoring`` metric of its predictions."""
        best = self._best()
        if self.scoring in (None, "r2", "accuracy"):
            return best.score(X, y)
        y = np.asarray(y, dtype=np.float64)
        err = y - np.asarray(best.predict(X), dtype=np.float64)
        losses = {"neg_mean_squared_error": lambda: np.mean(err ** 2),
                  "neg_root_mean_squared_error": lambda: np.sqrt(np.mean(err ** 2)),
                  "neg_mean_absolute_error": lambda: np.mean(np.abs(err)),
                  "neg_median_absolute_error": lambda: np.median(np.abs(err))}
        if self.scoring not in losses:
            raise NotImplementedError(f"score with scoring={self.scoring!r}")
        return -float(losses[self.scoring]())


class GridSearchCV(_BaseSearchCV):
    """Every combination of ``param_grid`` (a dict of lists, or a list of
    such dicts; ``step__param`` names reach into composites) scored by
    ``cv``-fold cross-validation in Rust: KFold, or StratifiedKFold for
    classifiers, without shuffling, as in sklearn (or any splitter with
    ``split``). ``scoring`` is ``None`` (R² / accuracy) or an sklearn scoring
    name (``"neg_mean_squared_error"``, ``"roc_auc"``, ``"neg_log_loss"``,
    ...); ``error_score`` scores failed candidates (``"raise"`` raises).
    Sample weights go to every fit and weight the fold scores, as in
    sklearn 1.8."""

    def __init__(self, estimator, param_grid, *, scoring=None, refit=True, cv=None,
                 error_score=np.nan):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.refit = refit
        self.cv = cv
        self.error_score = error_score

    def _candidates(self):
        return ParameterGrid(self.param_grid)


class RandomizedSearchCV(_BaseSearchCV):
    """``n_iter`` candidates of :class:`ParameterSampler` (the same draws as
    sklearn's for the same ``random_state``), scored as in
    :class:`GridSearchCV`."""

    def __init__(self, estimator, param_distributions, *, n_iter=10, scoring=None, refit=True,
                 cv=None, random_state=None, error_score=np.nan):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.scoring = scoring
        self.refit = refit
        self.cv = cv
        self.random_state = random_state
        self.error_score = error_score

    def _candidates(self):
        return ParameterSampler(self.param_distributions, self.n_iter,
                                random_state=self.random_state)


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
