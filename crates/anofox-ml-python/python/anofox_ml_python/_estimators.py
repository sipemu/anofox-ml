"""sklearn-compatible estimators backed by the Rust registry.

Parameter names and defaults follow scikit-learn where the Rust estimator
supports them; ``random_state`` replaces the Rust ``seed``.
"""

from __future__ import annotations

import numpy as np

from ._base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    OutlierMixin,
    RegressorMixin,
    SupervisedTransformerMixin,
    TransformerMixin,
    _nd,
    _rust_defaults,
    _seed,
)

# ---------------------------------------------------------------------------
# parameter translation helpers
# ---------------------------------------------------------------------------


def _max_features(v):
    if v is None:
        return None
    if v == "sqrt":
        return "Sqrt"
    if v == "log2":
        return "Log2"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return {"Fixed": int(v)}
    if isinstance(v, float):
        return {"Fraction": v}
    raise ValueError(f"invalid max_features {v!r}")


def _gamma(g):
    if g in ("scale", "auto"):
        return g
    return float(g)


def _kernel(kernel, gamma, degree, coef0):
    if kernel == "linear":
        return "Linear"
    if kernel == "rbf":
        return {"Rbf": {"gamma": _gamma(gamma)}}
    if kernel == "poly":
        return {"Polynomial": {"degree": int(degree), "gamma": _gamma(gamma), "coef0": float(coef0)}}
    if kernel == "sigmoid":
        return {"Sigmoid": {"gamma": _gamma(gamma), "coef0": float(coef0)}}
    if kernel == "precomputed":
        return "Precomputed"
    raise ValueError(f"unknown kernel {kernel!r}")


def _choice(value, mapping, name):
    try:
        return mapping[value]
    except (KeyError, TypeError):
        raise ValueError(f"invalid {name} {value!r}; expected one of {sorted(mapping)}") from None


def _criterion(c):
    return _choice(c, {"gini": "Gini", "entropy": "Entropy", "log_loss": "Entropy"}, "criterion")


# ---------------------------------------------------------------------------
# trees and ensembles
# ---------------------------------------------------------------------------


class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    _kind = "DecisionTreeClassifier"

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, class_weight=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"criterion": _criterion(self.criterion), "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": _max_features(self.max_features),
                "class_weight": self._encode_class_weight(self.class_weight),
                "seed": _seed(self.random_state)}


class DecisionTreeRegressor(RegressorMixin, BaseEstimator):
    _kind = "DecisionTreeRegressor"

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": _max_features(self.max_features),
                "seed": _seed(self.random_state)}


class RandomForestClassifier(ClassifierMixin, BaseEstimator):
    _kind = "RandomForestClassifier"

    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                 bootstrap=True, oob_score=False, max_samples=None, class_weight=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_estimators": self.n_estimators, "criterion": _criterion(self.criterion),
                "max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": _max_features(self.max_features), "bootstrap": self.bootstrap,
                "oob_score": self.oob_score, "max_samples": self.max_samples,
                "class_weight": self._encode_class_weight(self.class_weight),
                "seed": _seed(self.random_state)}


class RandomForestRegressor(RegressorMixin, BaseEstimator):
    _kind = "RandomForestRegressor"

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0, bootstrap=True, oob_score=False,
                 max_samples=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.max_samples = max_samples
        self.random_state = random_state

    def _rust_params(self, X, y):
        mf = None if self.max_features == 1.0 else _max_features(self.max_features)
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf, "max_features": mf,
                "bootstrap": self.bootstrap, "oob_score": self.oob_score,
                "max_samples": self.max_samples, "seed": _seed(self.random_state)}


class ExtraTreesClassifier(ClassifierMixin, BaseEstimator):
    _kind = "ExtraTreesClassifier"

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": _max_features(self.max_features),
                "seed": _seed(self.random_state)}


class ExtraTreesRegressor(RegressorMixin, BaseEstimator):
    _kind = "ExtraTreesRegressor"

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _rust_params(self, X, y):
        mf = None if self.max_features == 1.0 else _max_features(self.max_features)
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf, "max_features": mf,
                "seed": _seed(self.random_state)}


class _GradientBoosting(BaseEstimator):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2,
                 min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate,
                "max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf, "subsample": self.subsample,
                "seed": _seed(self.random_state)}


class GradientBoostingClassifier(ClassifierMixin, _GradientBoosting):
    _kind = "GradientBoostingClassifier"


class GradientBoostingRegressor(RegressorMixin, _GradientBoosting):
    _kind = "GradientBoostingRegressor"


class AdaBoostClassifier(ClassifierMixin, BaseEstimator):
    """SAMME AdaBoost over decision trees of depth ``max_depth`` (stumps)."""

    _kind = "AdaBoostClassifier"

    def __init__(self, n_estimators=50, learning_rate=1.0, max_depth=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate,
                "max_depth": self.max_depth, "seed": _seed(self.random_state)}


class AdaBoostRegressor(RegressorMixin, BaseEstimator):
    """AdaBoost.R2 over decision trees of depth ``max_depth``."""

    _kind = "AdaBoostRegressor"

    def __init__(self, n_estimators=50, learning_rate=1.0, loss="linear", max_depth=3,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate,
                "loss": _choice(self.loss, {"linear": "Linear", "square": "Square",
                                            "exponential": "Exponential"}, "loss"),
                "max_depth": self.max_depth, "seed": _seed(self.random_state)}


class _Bagging(BaseEstimator):
    """Bagged decision trees (``max_samples`` is a fraction)."""

    def __init__(self, n_estimators=10, max_samples=1.0, bootstrap=True, max_depth=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.random_state = random_state

    def _rust_params(self, X, y):
        ms = None if self.max_samples == 1.0 else float(self.max_samples)
        return {"n_estimators": self.n_estimators, "max_samples": ms,
                "bootstrap": self.bootstrap, "max_depth": self.max_depth,
                "seed": _seed(self.random_state)}


class BaggingClassifier(ClassifierMixin, _Bagging):
    _kind = "BaggingClassifier"


class BaggingRegressor(RegressorMixin, _Bagging):
    _kind = "BaggingRegressor"


class _HistGradientBoosting(BaseEstimator):
    def __init__(self, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None,
                 min_samples_leaf=20, l2_regularization=0.0, max_bins=255):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins

    def _rust_params(self, X, y):
        return {"learning_rate": self.learning_rate, "n_estimators": self.max_iter,
                "max_leaf_nodes": self.max_leaf_nodes, "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "l2_regularization": self.l2_regularization, "max_bins": self.max_bins}


class HistGradientBoostingClassifier(ClassifierMixin, _HistGradientBoosting):
    _kind = "HistGradientBoostingClassifier"


class HistGradientBoostingRegressor(RegressorMixin, _HistGradientBoosting):
    _kind = "HistGradientBoostingRegressor"

    _LOSSES = {"squared_error": "SquaredError", "absolute_error": "AbsoluteError",
               "quantile": "Quantile"}

    def __init__(self, loss="squared_error", quantile=None, learning_rate=0.1, max_iter=100,
                 max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0,
                 max_bins=255):
        super().__init__(learning_rate=learning_rate, max_iter=max_iter,
                         max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf,
                         l2_regularization=l2_regularization, max_bins=max_bins)
        self.loss = loss
        self.quantile = quantile

    def _rust_params(self, X, y):
        params = super()._rust_params(X, y)
        params["loss"] = _choice(self.loss, self._LOSSES, "loss")
        params["quantile"] = None if self.quantile is None else float(self.quantile)
        return params


class _Lgbm(BaseEstimator):
    """LightGBM-style boosting; parameters follow ``lightgbm.LGBMModel``."""

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1,
                 n_estimators=100, min_split_gain=0.0, min_child_weight=1e-3,
                 min_child_samples=20, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                 max_bin=255, random_state=None):
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.max_bin = max_bin
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"boosting_type": _choice(self.boosting_type, {"gbdt": "Gbdt", "goss": "Goss"},
                                         "boosting_type"),
                "num_leaves": self.num_leaves,
                "max_depth": None if self.max_depth is None or self.max_depth < 0
                else self.max_depth,
                "learning_rate": self.learning_rate, "n_estimators": self.n_estimators,
                "min_split_gain": self.min_split_gain,
                "min_child_weight": self.min_child_weight,
                "min_child_samples": self.min_child_samples,
                "colsample_bytree": self.colsample_bytree, "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda, "max_bins": self.max_bin,
                "seed": _seed(self.random_state)}


class LGBMClassifier(ClassifierMixin, _Lgbm):
    _kind = "LgbmClassifier"

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1,
                 n_estimators=100, min_split_gain=0.0, min_child_weight=1e-3,
                 min_child_samples=20, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                 max_bin=255, class_weight=None, random_state=None):
        super().__init__(boosting_type, num_leaves, max_depth, learning_rate, n_estimators,
                         min_split_gain, min_child_weight, min_child_samples,
                         colsample_bytree, reg_alpha, reg_lambda, max_bin, random_state)
        self.class_weight = class_weight

    def _rust_params(self, X, y):
        p = super()._rust_params(X, y)
        p["class_weight"] = self._encode_class_weight(self.class_weight)
        return p


class LGBMRegressor(RegressorMixin, _Lgbm):
    _kind = "LgbmRegressor"


class IsolationForest(OutlierMixin, BaseEstimator):
    _kind = "IsolationForest"

    def __init__(self, n_estimators=100, max_samples="auto", contamination="auto",
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def _rust_params(self, X, y):
        n = X.shape[0]
        if self.max_samples == "auto":
            ms = min(256, n)
        elif isinstance(self.max_samples, float):
            ms = int(self.max_samples * n)
        else:
            ms = int(self.max_samples)
        return {"n_estimators": self.n_estimators, "max_samples": ms,
                "contamination": None if self.contamination == "auto"
                else float(self.contamination),
                "seed": _seed(self.random_state)}

    def decision_function(self, X):
        return self._call("decision_function", X)

    def predict(self, X):
        return np.asarray(self._call("predict", X), dtype=np.int64)

    def score_samples(self, X):
        return self._call("score_samples", X)


# ---------------------------------------------------------------------------
# support vector machines
# ---------------------------------------------------------------------------


class SVC(ClassifierMixin, BaseEstimator):
    _kind = "Svc"

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0,
                 shrinking=True, probability=False, tol=1e-3, cache_size=200,
                 class_weight=None, max_iter=-1, decision_function_shape="ovr",
                 break_ties=False, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"c": float(self.C), "kernel": _kernel(self.kernel, self.gamma, self.degree,
                                                      self.coef0),
                "shrinking": self.shrinking, "probability": self.probability,
                "tol": self.tol, "cache_size": float(self.cache_size),
                "class_weight": self._encode_class_weight(self.class_weight),
                "max_iter": 10_000_000 if self.max_iter < 0 else self.max_iter,
                "decision_function_shape": _choice(self.decision_function_shape,
                                                   {"ovr": "Ovr", "ovo": "Ovo"},
                                                   "decision_function_shape"),
                "break_ties": self.break_ties, "seed": _seed(self.random_state)}


class NuSVC(ClassifierMixin, BaseEstimator):
    _kind = "NuSvc"

    def __init__(self, nu=0.5, kernel="rbf", degree=3, gamma="scale", coef0=0.0,
                 probability=False, tol=1e-3, max_iter=-1, decision_function_shape="ovr",
                 break_ties=False, random_state=None):
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.probability = probability
        self.tol = tol
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"nu": self.nu, "kernel": _kernel(self.kernel, self.gamma, self.degree,
                                                 self.coef0),
                "probability": self.probability, "tol": self.tol,
                "max_iter": 10_000_000 if self.max_iter < 0 else self.max_iter,
                "decision_function_shape": _choice(self.decision_function_shape,
                                                   {"ovr": "Ovr", "ovo": "Ovo"},
                                                   "decision_function_shape"),
                "break_ties": self.break_ties, "seed": _seed(self.random_state)}


class SVR(RegressorMixin, BaseEstimator):
    _kind = "Svr"

    def __init__(self, kernel="rbf", degree=3, gamma="scale", coef0=0.0, tol=1e-3, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200, max_iter=-1):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter

    def _rust_params(self, X, y):
        return {"kernel": _kernel(self.kernel, self.gamma, self.degree, self.coef0),
                "tol": self.tol, "c": float(self.C), "epsilon": self.epsilon,
                "shrinking": self.shrinking, "cache_size": float(self.cache_size),
                "max_iter": 10_000_000 if self.max_iter < 0 else self.max_iter}


class NuSVR(RegressorMixin, BaseEstimator):
    _kind = "NuSvr"

    def __init__(self, nu=0.5, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0,
                 shrinking=True, tol=1e-3, cache_size=200, max_iter=-1):
        self.nu = nu
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter

    def _rust_params(self, X, y):
        return {"nu": self.nu, "c": float(self.C),
                "kernel": _kernel(self.kernel, self.gamma, self.degree, self.coef0),
                "shrinking": self.shrinking, "tol": self.tol,
                "cache_size": float(self.cache_size),
                "max_iter": 10_000_000 if self.max_iter < 0 else self.max_iter}


class LinearSVC(ClassifierMixin, BaseEstimator):
    _kind = "LinearSvc"

    def __init__(self, loss="squared_hinge", tol=1e-4, C=1.0, fit_intercept=True,
                 intercept_scaling=1.0, max_iter=1000, random_state=None):
        self.loss = loss
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.max_iter = max_iter
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"loss": _choice(self.loss, {"hinge": "Hinge", "squared_hinge": "SquaredHinge"},
                                "loss"),
                "tol": self.tol, "c": float(self.C), "fit_intercept": self.fit_intercept,
                "intercept_scaling": float(self.intercept_scaling), "max_iter": self.max_iter,
                "seed": _seed(self.random_state)}


class LinearSVR(RegressorMixin, BaseEstimator):
    _kind = "LinearSvr"

    def __init__(self, epsilon=0.0, tol=1e-4, C=1.0, loss="epsilon_insensitive",
                 fit_intercept=True, intercept_scaling=1.0, max_iter=1000, random_state=None):
        self.epsilon = epsilon
        self.tol = tol
        self.C = C
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.max_iter = max_iter
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"epsilon": self.epsilon, "tol": self.tol, "c": float(self.C),
                "loss": _choice(self.loss, {
                    "epsilon_insensitive": "EpsilonInsensitive",
                    "squared_epsilon_insensitive": "SquaredEpsilonInsensitive"}, "loss"),
                "fit_intercept": self.fit_intercept,
                "intercept_scaling": float(self.intercept_scaling), "max_iter": self.max_iter,
                "seed": _seed(self.random_state)}


class OneClassSVM(OutlierMixin, BaseEstimator):
    _kind = "OneClassSvm"

    def __init__(self, kernel="rbf", degree=3, gamma="scale", coef0=0.0, tol=1e-3, nu=0.5,
                 shrinking=True, cache_size=200, max_iter=-1):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter

    def _rust_params(self, X, y):
        return {"kernel": _kernel(self.kernel, self.gamma, self.degree, self.coef0),
                "tol": self.tol, "nu": self.nu, "shrinking": self.shrinking,
                "cache_size": float(self.cache_size),
                "max_iter": 10_000_000 if self.max_iter < 0 else self.max_iter}

    def predict(self, X):
        return np.asarray(self._call("predict", X), dtype=np.int64)

    def decision_function(self, X):
        return self._call("decision_function", X)

    def score_samples(self, X):
        return self._call("score_samples", X)


# ---------------------------------------------------------------------------
# linear models
# ---------------------------------------------------------------------------

_PENALTY = {None: "None", "l2": "L2", "l1": "L1", "elasticnet": "ElasticNet"}
_LEARNING_RATE = {"constant": "Constant", "optimal": "Optimal", "invscaling": "InvScaling"}


class SGDClassifier(ClassifierMixin, BaseEstimator):
    _kind = "SgdClassifier"

    def __init__(self, loss="hinge", penalty="l2", alpha=1e-4, l1_ratio=0.15, max_iter=1000,
                 tol=1e-3, shuffle=True, random_state=None, learning_rate="optimal",
                 eta0=0.01, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, class_weight=None, average=False):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.average = average

    def _rust_params(self, X, y):
        return {"loss": _choice(self.loss, {"hinge": "Hinge", "log_loss": "Log",
                                            "modified_huber": "ModifiedHuber",
                                            "perceptron": "Perceptron"}, "loss"),
                "penalty": _choice(self.penalty, _PENALTY, "penalty"), "alpha": self.alpha,
                "l1_ratio": self.l1_ratio, "max_iter": self.max_iter, "tol": self.tol,
                "shuffle": self.shuffle, "seed": _seed(self.random_state),
                "learning_rate": _choice(self.learning_rate, _LEARNING_RATE, "learning_rate"),
                "eta0": self.eta0, "power_t": self.power_t,
                "early_stopping": self.early_stopping,
                "validation_fraction": self.validation_fraction,
                "n_iter_no_change": self.n_iter_no_change,
                "class_weight": self._encode_class_weight(self.class_weight),
                "average": int(self.average) if not isinstance(self.average, bool)
                else int(bool(self.average))}


class SGDRegressor(RegressorMixin, BaseEstimator):
    _kind = "SgdRegressor"

    def __init__(self, loss="squared_error", penalty="l2", alpha=1e-4, l1_ratio=0.15,
                 max_iter=1000, tol=1e-3, shuffle=True, epsilon=0.1, random_state=None,
                 learning_rate="invscaling", eta0=0.01, power_t=0.25, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=5, average=False):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.average = average

    def _rust_params(self, X, y):
        return {"loss": _choice(self.loss, {"squared_error": "SquaredError", "huber": "Huber",
                                            "epsilon_insensitive": "EpsilonInsensitive"},
                                "loss"),
                "penalty": _choice(self.penalty, _PENALTY, "penalty"), "alpha": self.alpha,
                "l1_ratio": self.l1_ratio, "max_iter": self.max_iter, "tol": self.tol,
                "shuffle": self.shuffle, "epsilon": self.epsilon,
                "seed": _seed(self.random_state),
                "learning_rate": _choice(self.learning_rate, _LEARNING_RATE, "learning_rate"),
                "eta0": self.eta0, "power_t": self.power_t,
                "early_stopping": self.early_stopping,
                "validation_fraction": self.validation_fraction,
                "n_iter_no_change": self.n_iter_no_change,
                "average": int(self.average) if not isinstance(self.average, bool)
                else int(bool(self.average))}


class PassiveAggressiveClassifier(ClassifierMixin, BaseEstimator):
    _kind = "PassiveAggressiveClassifier"

    def __init__(self, C=1.0, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
                 loss="hinge", random_state=None):
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.loss = loss
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"c": float(self.C), "fit_intercept": self.fit_intercept,
                "max_iter": self.max_iter, "tol": self.tol, "shuffle": self.shuffle,
                "variant": _choice(self.loss, {"hinge": "PaI", "squared_hinge": "PaII"},
                                   "loss"),
                "seed": _seed(self.random_state)}


class PassiveAggressiveRegressor(RegressorMixin, BaseEstimator):
    _kind = "PassiveAggressiveRegressor"

    def __init__(self, C=1.0, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
                 loss="epsilon_insensitive", epsilon=0.1, random_state=None):
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.loss = loss
        self.epsilon = epsilon
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"c": float(self.C), "fit_intercept": self.fit_intercept,
                "max_iter": self.max_iter, "tol": self.tol, "shuffle": self.shuffle,
                "variant": _choice(self.loss, {"epsilon_insensitive": "PaI",
                                               "squared_epsilon_insensitive": "PaII"},
                                   "loss"),
                "epsilon": self.epsilon, "seed": _seed(self.random_state)}


class _LinearAttrs:
    """``coef_`` / ``intercept_`` from the fitted Rust state."""

    @property
    def coef_(self):
        return _nd(self._fitted_state()["coef"])

    @property
    def intercept_(self):
        b = self._fitted_state()["intercept"]
        return 0.0 if b is None else float(b)


class LinearRegression(_LinearAttrs, RegressorMixin, BaseEstimator):
    _kind = "OlsRegressor"

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {"with_intercept": self.fit_intercept}


class Ridge(_LinearAttrs, RegressorMixin, BaseEstimator):
    """``||y - Xw||² + alpha ||w||²`` (sklearn's objective)."""

    _kind = "RidgeRegressor"

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {"lambda": float(self.alpha), "with_intercept": self.fit_intercept}


class Lasso(_LinearAttrs, RegressorMixin, BaseEstimator):
    """L1-penalised least squares with sklearn's ``alpha`` scale."""

    _kind = "LassoRegressor"

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {"lambda": float(self.alpha), "with_intercept": self.fit_intercept,
                "sklearn_scale": True}


class ElasticNet(_LinearAttrs, RegressorMixin, BaseEstimator):
    """Elastic-net penalised least squares (sklearn's ``alpha``, ``l1_ratio``)."""

    _kind = "ElasticNetRegressor"

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {"lambda": float(self.alpha), "alpha": float(self.l1_ratio),
                "with_intercept": self.fit_intercept, "sklearn_scale": True}


class HuberRegressor(_LinearAttrs, RegressorMixin, BaseEstimator):
    """Huber regression with sklearn's joint coefficient / scale estimate."""

    _kind = "HuberRegressor"

    def __init__(self, epsilon=1.35, max_iter=100, alpha=1e-4, fit_intercept=True, tol=1e-5):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol

    def _rust_params(self, X, y):
        return {"epsilon": self.epsilon, "max_iter": self.max_iter, "alpha": self.alpha,
                "with_intercept": self.fit_intercept, "tol": self.tol}


class QuantileRegressor(_LinearAttrs, RegressorMixin, BaseEstimator):
    """Linear quantile regression with an L1 penalty ``alpha`` (sklearn's
    objective, solved exactly as a linear program)."""

    _kind = "QuantileRegressor"

    def __init__(self, quantile=0.5, alpha=1.0, fit_intercept=True):
        self.quantile = quantile
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {"quantile": self.quantile, "alpha": float(self.alpha),
                "with_intercept": self.fit_intercept}


class RidgeCV(_LinearAttrs, RegressorMixin, BaseEstimator):
    """Ridge with ``alpha`` chosen by exact leave-one-out (``cv=None``) or k-fold CV."""

    _kind = "RidgeCrossValidated"

    def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=True, cv=None, random_state=None):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"alphas": [float(a) for a in self.alphas], "with_intercept": self.fit_intercept,
                "cv_folds": self.cv, "seed": _seed(self.random_state)}


class LassoCV(_LinearAttrs, RegressorMixin, BaseEstimator):
    _kind = "LassoCrossValidated"

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True,
                 max_iter=100_000, tol=1e-12, cv=5, random_state=None):
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"eps": self.eps, "n_alphas": self.n_alphas,
                "alphas": None if self.alphas is None else [float(a) for a in self.alphas],
                "with_intercept": self.fit_intercept, "max_iter": self.max_iter,
                "tol": self.tol,
                "folds": {"n_folds": int(self.cv), "fold_ids": None, "shuffle": False,
                          "seed": _seed(self.random_state)}}


class ElasticNetCV(LassoCV):
    _kind = "ElasticNetCrossValidated"

    def __init__(self, l1_ratio=0.5, eps=1e-3, n_alphas=100,
                 alphas=None, fit_intercept=True, max_iter=100_000, tol=1e-12, cv=5,
                 random_state=None):
        super().__init__(eps, n_alphas, alphas, fit_intercept, max_iter, tol, cv, random_state)
        self.l1_ratio = l1_ratio

    def _rust_params(self, X, y):
        p = super()._rust_params(X, y)
        ratios = self.l1_ratio if np.ndim(self.l1_ratio) else [self.l1_ratio]
        p["l1_ratios"] = [float(r) for r in ratios]
        return p


class LogisticRegression(ClassifierMixin, BaseEstimator):
    """Logistic regression: multinomial loss for more than two classes, L2 /
    L1 / elastic-net penalty via ``C`` and ``l1_ratio`` (sklearn >= 1.8)."""

    _kind = "LogisticRegressor"

    def __init__(self, C=1.0, l1_ratio=0.0, fit_intercept=True, tol=1e-8, max_iter=1000,
                 class_weight=None):
        self.C = C
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight

    def _rust_params(self, X, y):
        return {"c": float(self.C), "l1_ratio": float(self.l1_ratio),
                "with_intercept": self.fit_intercept, "tol": self.tol,
                "max_iter": self.max_iter,
                "class_weight": self._encode_class_weight(self.class_weight)}


class _Glm(_LinearAttrs, RegressorMixin, BaseEstimator):
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=100, tol=1e-8):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def _rust_params(self, X, y):
        return {"alpha": float(self.alpha), "with_intercept": self.fit_intercept,
                "max_iter": self.max_iter, "tol": self.tol}


class PoissonRegressor(_Glm):
    """Poisson GLM with log link (sklearn's ``alpha = 1.0`` default)."""

    _kind = "PoissonRegressor"


class GammaRegressor(_Glm):
    """Gamma GLM with log link (sklearn's ``alpha = 1.0`` default)."""

    _kind = "GammaRegressor"

    def _rust_params(self, X, y):
        inner = dict(_rust_defaults(self._kind)["inner"])
        inner.update(super()._rust_params(X, y))
        return {"inner": inner}


class TweedieRegressor(_Glm):
    _kind = "TweedieRegressor"

    def __init__(self, power=0.0, alpha=1.0, link="auto", fit_intercept=True, max_iter=100,
                 tol=1e-8):
        super().__init__(alpha, fit_intercept, max_iter, tol)
        self.power = power
        self.link = link

    def _rust_params(self, X, y):
        p = super()._rust_params(X, y)
        link = self.link
        if link == "auto":
            link = "identity" if self.power <= 0 else "log"
        p["power"] = float(self.power)
        p["link_power"] = _choice(link, {"identity": 1.0, "log": 0.0}, "link")
        return p


class BayesianRidge(_LinearAttrs, RegressorMixin, BaseEstimator):
    _kind = "BayesianRidge"

    def __init__(self, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6,
                 lambda_2=1e-6, fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {k: getattr(self, k) for k in ("max_iter", "tol", "alpha_1", "alpha_2",
                                              "lambda_1", "lambda_2", "fit_intercept")}


class ARDRegression(BayesianRidge):
    _kind = "ARDRegression"

    def __init__(self, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6,
                 lambda_2=1e-6, threshold_lambda=1e4, fit_intercept=True):
        super().__init__(max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, fit_intercept)
        self.threshold_lambda = threshold_lambda

    def _rust_params(self, X, y):
        p = super()._rust_params(X, y)
        p["threshold_lambda"] = self.threshold_lambda
        return p


class KernelRidge(RegressorMixin, BaseEstimator):
    _kind = "KernelRidge"

    def __init__(self, alpha=1.0, kernel="linear", gamma=None, degree=3, coef0=1.0):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _rust_params(self, X, y):
        gamma = "auto" if self.gamma is None else self.gamma
        return {"alpha": float(self.alpha),
                "kernel": _kernel(self.kernel, gamma, self.degree, self.coef0)}


class Lars(_LinearAttrs, RegressorMixin, BaseEstimator):
    _kind = "Lars"

    def __init__(self, fit_intercept=True, n_nonzero_coefs=500):
        self.fit_intercept = fit_intercept
        self.n_nonzero_coefs = n_nonzero_coefs

    def _rust_params(self, X, y):
        return {"fit_intercept": self.fit_intercept, "n_nonzero_coefs": self.n_nonzero_coefs,
                "max_iter": self.n_nonzero_coefs}


class OrthogonalMatchingPursuit(_LinearAttrs, RegressorMixin, BaseEstimator):
    _kind = "OrthogonalMatchingPursuit"

    def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {"n_nonzero_coefs": self.n_nonzero_coefs, "tol": self.tol,
                "fit_intercept": self.fit_intercept}


class RANSACRegressor(_LinearAttrs, RegressorMixin, BaseEstimator):
    """RANSAC around a linear least-squares model."""

    _kind = "RansacRegressor"

    def __init__(self, min_samples=None, residual_threshold=None, max_trials=100,
                 stop_probability=0.99, random_state=None):
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.stop_probability = stop_probability
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"min_samples": self.min_samples, "residual_threshold": self.residual_threshold,
                "max_trials": self.max_trials, "stop_probability": self.stop_probability,
                "seed": _seed(self.random_state)}


class TheilSenRegressor(_LinearAttrs, RegressorMixin, BaseEstimator):
    _kind = "TheilSenRegressor"

    def __init__(self, fit_intercept=True, max_subpopulation=10_000, n_subsamples=None,
                 max_iter=300, tol=1e-3, random_state=None):
        self.fit_intercept = fit_intercept
        self.max_subpopulation = max_subpopulation
        self.n_subsamples = n_subsamples
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"fit_intercept": self.fit_intercept,
                "max_subpopulation": self.max_subpopulation,
                "n_subsamples": self.n_subsamples, "max_iter": self.max_iter, "tol": self.tol,
                "seed": _seed(self.random_state)}


# ---------------------------------------------------------------------------
# naive Bayes, neighbours, discriminant analysis, neural networks
# ---------------------------------------------------------------------------


class GaussianNB(ClassifierMixin, BaseEstimator):
    _kind = "GaussianNB"

    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def _rust_params(self, X, y):
        return {"priors": None if self.priors is None else [float(p) for p in self.priors],
                "var_smoothing": self.var_smoothing}


class _DiscreteNB(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _rust_params(self, X, y):
        return {"alpha": float(self.alpha), "force_alpha": self.force_alpha,
                "fit_prior": self.fit_prior,
                "class_prior": None if self.class_prior is None
                else [float(p) for p in self.class_prior]}


class MultinomialNB(_DiscreteNB):
    _kind = "MultinomialNB"


class CategoricalNB(_DiscreteNB):
    _kind = "CategoricalNB"


class ComplementNB(_DiscreteNB):
    _kind = "ComplementNB"

    def __init__(self, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None,
                 norm=False):
        super().__init__(alpha, force_alpha, fit_prior, class_prior)
        self.norm = norm

    def _rust_params(self, X, y):
        p = super()._rust_params(X, y)
        p["norm"] = self.norm
        return p


class BernoulliNB(_DiscreteNB):
    _kind = "BernoulliNB"

    def __init__(self, alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True,
                 class_prior=None):
        super().__init__(alpha, force_alpha, fit_prior, class_prior)
        self.binarize = binarize

    def _rust_params(self, X, y):
        p = super()._rust_params(X, y)
        if self.binarize is None:
            raise ValueError("binarize=None is not supported; pass a threshold")
        p["binarize_threshold"] = float(self.binarize)
        return p


_METRIC = {"euclidean": "Euclidean", "l2": "Euclidean", "manhattan": "Manhattan",
           "cityblock": "Manhattan", "l1": "Manhattan", "chebyshev": "Chebyshev",
           "infinity": "Chebyshev", "cosine": "Cosine", "precomputed": "Precomputed"}
_WEIGHTS = {"uniform": "Uniform", "distance": "Distance"}
_ALGORITHM = {"auto": "Auto", "brute": "BruteForce", "kd_tree": "KdTree"}


def _neighbor_params(est):
    if est.metric == "minkowski":
        metric = {"Minkowski": float(est.p)}
    else:
        metric = _choice(est.metric, _METRIC, "metric")
    return {"weights": _choice(est.weights, _WEIGHTS, "weights"),
            "metric": metric,
            "algorithm": _choice(est.algorithm, _ALGORITHM, "algorithm")}


class KNeighborsClassifier(ClassifierMixin, BaseEstimator):
    _kind = "KnnClassifier"

    def __init__(self, n_neighbors=5, weights="uniform", algorithm="auto", p=2,
                 metric="minkowski"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.p = p
        self.metric = metric

    def _rust_params(self, X, y):
        return {"n_neighbors": self.n_neighbors, **_neighbor_params(self)}


class KNeighborsRegressor(RegressorMixin, BaseEstimator):
    _kind = "KnnRegressor"

    def __init__(self, n_neighbors=5, weights="uniform", algorithm="auto", p=2,
                 metric="minkowski"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.p = p
        self.metric = metric

    def _rust_params(self, X, y):
        return {"n_neighbors": self.n_neighbors, **_neighbor_params(self)}


class RadiusNeighborsClassifier(ClassifierMixin, BaseEstimator):
    _kind = "RadiusNeighborsClassifier"

    def __init__(self, radius=1.0, weights="uniform", algorithm="auto", p=2,
                 metric="minkowski", outlier_label=None):
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.p = p
        self.metric = metric
        self.outlier_label = outlier_label

    def _rust_params(self, X, y):
        # Labels are encoded as class indices: a training class maps to its
        # index, any other outlier label to -1 (decoded in ``predict``).
        label = self.outlier_label
        if label is None:
            enc = None
        else:
            hits = np.flatnonzero(self.classes_ == label)
            enc = float(hits[0]) if len(hits) else -1.0
        return {"radius": float(self.radius), **_neighbor_params(self),
                "outlier_label": enc}

    def predict(self, X):
        pred = np.asarray(self._call("predict", X), dtype=np.int64)
        if not (pred < 0).any():
            return self.classes_[pred]
        out = np.empty(len(pred), dtype=np.result_type(self.classes_, np.asarray(self.outlier_label)))
        out[pred >= 0] = self.classes_[pred[pred >= 0]]
        out[pred < 0] = self.outlier_label
        return out


class RadiusNeighborsRegressor(RegressorMixin, BaseEstimator):
    _kind = "RadiusNeighborsRegressor"

    def __init__(self, radius=1.0, weights="uniform", algorithm="auto", p=2,
                 metric="minkowski"):
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.p = p
        self.metric = metric

    def _rust_params(self, X, y):
        return {"radius": float(self.radius), **_neighbor_params(self)}


class LocalOutlierFactor(OutlierMixin, BaseEstimator):
    """Outlier detection on the training data (``fit_predict``)."""

    _kind = "LocalOutlierFactor"

    def __init__(self, n_neighbors=20, contamination="auto"):
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def _rust_params(self, X, y):
        return {"n_neighbors": self.n_neighbors,
                "contamination": None if self.contamination == "auto"
                else float(self.contamination)}

    def fit(self, X, y=None):
        self._fit_model(X)
        self.negative_outlier_factor_ = self._call("score_samples", X)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return np.asarray(self._model().labels(), dtype=np.int64)


class LinearDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    _kind = "LinearDiscriminantAnalysis"

    def __init__(self, solver="svd", shrinkage=None, priors=None, tol=1e-4):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.tol = tol

    def _rust_params(self, X, y):
        if self.shrinkage is None:
            shrinkage = "None"
        elif self.shrinkage == "auto":
            shrinkage = "Auto"
        else:
            shrinkage = {"Fixed": float(self.shrinkage)}
        return {"solver": _choice(self.solver, {"svd": "Svd", "lsqr": "Lsqr",
                                                "eigen": "Eigen"}, "solver"),
                "shrinkage": shrinkage,
                "priors": None if self.priors is None else [float(p) for p in self.priors],
                "tol": self.tol}

    def transform(self, X):
        return self._call("transform", X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class QuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    _kind = "QuadraticDiscriminantAnalysis"

    def __init__(self, priors=None, reg_param=0.0, tol=1e-4):
        self.priors = priors
        self.reg_param = reg_param
        self.tol = tol

    def _rust_params(self, X, y):
        return {"priors": None if self.priors is None else [float(p) for p in self.priors],
                "reg": float(self.reg_param), "tol": self.tol}


class _Mlp(BaseEstimator):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver="adam",
                 alpha=1e-4, batch_size="auto", learning_rate="constant",
                 learning_rate_init=1e-3, power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

    def _rust_params(self, X, y):
        sizes = self.hidden_layer_sizes
        sizes = [int(sizes)] if np.ndim(sizes) == 0 else [int(s) for s in sizes]
        batch = min(200, X.shape[0]) if self.batch_size == "auto" else int(self.batch_size)
        p = {"hidden_layer_sizes": sizes,
             "activation": _choice(self.activation, {"relu": "Relu", "tanh": "Tanh",
                                                     "logistic": "Sigmoid",
                                                     "identity": "Identity"}, "activation"),
             "solver": _choice(self.solver, {"adam": "Adam", "sgd": "Sgd", "lbfgs": "Lbfgs"},
                               "solver"),
             "alpha": self.alpha, "batch_size": batch,
             "learning_rate_schedule": _choice(self.learning_rate, {
                 "constant": "Constant", "invscaling": "InvScaling",
                 "adaptive": "Adaptive"}, "learning_rate"),
             "learning_rate": self.learning_rate_init, "power_t": self.power_t,
             "max_iter": self.max_iter, "shuffle": self.shuffle,
             "seed": _seed(self.random_state), "tol": self.tol, "momentum": self.momentum,
             "nesterovs_momentum": self.nesterovs_momentum,
             "early_stopping": self.early_stopping, "beta_1": self.beta_1,
             "beta_2": self.beta_2, "epsilon": self.epsilon,
             "n_iter_no_change": self.n_iter_no_change, "max_fun": self.max_fun}
        if "validation_fraction" in _rust_defaults(self._kind):
            p["validation_fraction"] = self.validation_fraction
        return p


class MLPClassifier(ClassifierMixin, _Mlp):
    _kind = "MlpClassifier"


class MLPRegressor(RegressorMixin, _Mlp):
    _kind = "MlpRegressor"


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


class StandardScaler(TransformerMixin, BaseEstimator):
    _kind = "StandardScaler"

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def _rust_params(self, X, y):
        return {"with_mean": self.with_mean, "with_std": self.with_std}


class MinMaxScaler(TransformerMixin, BaseEstimator):
    _kind = "MinMaxScaler"

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def _rust_params(self, X, y):
        lo, hi = self.feature_range
        return {"feature_min": float(lo), "feature_max": float(hi)}


class MaxAbsScaler(TransformerMixin, BaseEstimator):
    _kind = "MaxAbsScaler"

    def __init__(self):
        pass


class RobustScaler(TransformerMixin, BaseEstimator):
    _kind = "RobustScaler"

    def __init__(self, with_centering=True, with_scaling=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling

    def _rust_params(self, X, y):
        return {"with_centering": self.with_centering, "with_scaling": self.with_scaling}


class Normalizer(TransformerMixin, BaseEstimator):
    _kind = "Normalizer"

    def __init__(self, norm="l2"):
        self.norm = norm

    def _rust_params(self, X, y):
        return {"norm": _choice(self.norm, {"l1": "L1", "l2": "L2", "max": "Max"}, "norm")}


class Binarizer(TransformerMixin, BaseEstimator):
    _kind = "Binarizer"

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def _rust_params(self, X, y):
        return {"threshold": float(self.threshold)}


_IMPUTE_STRATEGY = {"mean": "Mean", "median": "Median", "most_frequent": "MostFrequent",
                    "constant": "Constant"}


class SimpleImputer(TransformerMixin, BaseEstimator):
    _kind = "SimpleImputer"

    def __init__(self, strategy="mean", fill_value=None, add_indicator=False,
                 keep_empty_features=False):
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _rust_params(self, X, y):
        return {"strategy": _choice(self.strategy, _IMPUTE_STRATEGY, "strategy"),
                "fill_value": None if self.fill_value is None else float(self.fill_value),
                "add_indicator": self.add_indicator,
                "keep_empty_features": self.keep_empty_features}


class MissingIndicator(TransformerMixin, BaseEstimator):
    _kind = "MissingIndicator"

    def __init__(self, features="missing-only", error_on_new=True):
        self.features = features
        self.error_on_new = error_on_new

    def _rust_params(self, X, y):
        return {"features": _choice(self.features, {"missing-only": "MissingOnly",
                                                    "all": "All"}, "features"),
                "error_on_new": self.error_on_new}


class KNNImputer(TransformerMixin, BaseEstimator):
    _kind = "KnnImputer"

    def __init__(self, n_neighbors=5, weights="uniform", add_indicator=False,
                 keep_empty_features=False):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _rust_params(self, X, y):
        return {"n_neighbors": self.n_neighbors,
                "weights": _choice(self.weights, {"uniform": "Uniform", "distance": "Distance"},
                                   "weights"),
                "add_indicator": self.add_indicator,
                "keep_empty_features": self.keep_empty_features}


def _bound(v):
    """IterativeImputer bound -> Rust list (JSON has no infinity)."""
    if v is None or (np.isscalar(v) and np.isinf(v)):
        return []
    if np.isscalar(v):
        return [float(v)]
    big = np.finfo(np.float64).max
    return [float(np.clip(b, -big, big)) for b in v]


class IterativeImputer(TransformerMixin, BaseEstimator):
    """sklearn's IterativeImputer with its default BayesianRidge estimator."""

    _kind = "IterativeImputer"

    def __init__(self, sample_posterior=False, max_iter=10, tol=1e-3, n_nearest_features=None,
                 initial_strategy="mean", fill_value=None, imputation_order="ascending",
                 skip_complete=False, min_value=-np.inf, max_value=np.inf, random_state=None,
                 add_indicator=False, keep_empty_features=False):
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.random_state = random_state
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _rust_params(self, X, y):
        return {"sample_posterior": self.sample_posterior, "max_iter": self.max_iter,
                "tol": float(self.tol), "n_nearest_features": self.n_nearest_features,
                "initial_strategy": _choice(self.initial_strategy, _IMPUTE_STRATEGY,
                                            "initial_strategy"),
                "fill_value": None if self.fill_value is None else float(self.fill_value),
                "imputation_order": _choice(self.imputation_order,
                                            {"ascending": "Ascending",
                                             "descending": "Descending", "roman": "Roman",
                                             "arabic": "Arabic", "random": "Random"},
                                            "imputation_order"),
                "skip_complete": self.skip_complete,
                "min_value": _bound(self.min_value), "max_value": _bound(self.max_value),
                "random_state": _seed(self.random_state),
                "add_indicator": self.add_indicator,
                "keep_empty_features": self.keep_empty_features}


class TargetEncoder(SupervisedTransformerMixin, BaseEstimator):
    """sklearn's TargetEncoder; ``fit_transform`` returns the cross-fitted encoding."""

    _kind = "TargetEncoder"

    def __init__(self, target_type="auto", smooth="auto", cv=5, random_state=None):
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.random_state = random_state

    def _rust_params(self, X, y):
        smooth = "Auto" if self.smooth == "auto" else {"Value": float(self.smooth)}
        return {"target_type": _choice(self.target_type,
                                       {"auto": "Auto", "continuous": "Continuous",
                                        "binary": "Binary", "multiclass": "Multiclass"},
                                       "target_type"),
                "smooth": smooth, "cv": self.cv, "random_state": _seed(self.random_state)}

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        return np.asarray(self._model().embedding())


class PolynomialFeatures(TransformerMixin, BaseEstimator):
    _kind = "PolynomialFeatures"

    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def _rust_params(self, X, y):
        lo, hi = (0, self.degree) if np.isscalar(self.degree) else self.degree
        return {"degree": int(hi), "min_degree": int(lo),
                "interaction_only": self.interaction_only, "include_bias": self.include_bias}


class SplineTransformer(TransformerMixin, BaseEstimator):
    """B-spline expansion of each feature (sklearn's ``SplineTransformer``;
    ``sample_weight`` and sparse output are not supported)."""

    _kind = "SplineTransformer"

    def __init__(self, n_knots=5, degree=3, *, knots="uniform", extrapolation="constant",
                 include_bias=True):
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation
        self.include_bias = include_bias

    def _rust_params(self, X, y):
        params = {"n_knots": int(self.n_knots), "degree": int(self.degree),
                  "extrapolation": _choice(self.extrapolation, {
                      "constant": "Constant", "linear": "Linear", "continue": "Continue",
                      "periodic": "Periodic", "error": "Error"}, "extrapolation"),
                  "include_bias": bool(self.include_bias)}
        if isinstance(self.knots, str):
            params["knot_placement"] = _choice(
                self.knots, {"uniform": "Uniform", "quantile": "Quantile"}, "knots")
        else:
            # sklearn: shape (n_knots, n_features); Rust: one list per feature.
            params["knots"] = np.asarray(self.knots, dtype=float).T.tolist()
        return params


class PowerTransformer(TransformerMixin, BaseEstimator):
    _kind = "PowerTransformer"

    def __init__(self, method="yeo-johnson", standardize=True):
        self.method = method
        self.standardize = standardize

    def _rust_params(self, X, y):
        return {"method": _choice(self.method, {"yeo-johnson": "YeoJohnson",
                                                "box-cox": "BoxCox"}, "method"),
                "standardize": self.standardize}


class QuantileTransformer(TransformerMixin, BaseEstimator):
    _kind = "QuantileTransformer"

    def __init__(self, n_quantiles=1000, output_distribution="uniform", subsample=10_000,
                 random_state=None):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_quantiles": self.n_quantiles,
                "output_distribution": _choice(self.output_distribution,
                                               {"uniform": "Uniform", "normal": "Normal"},
                                               "output_distribution"),
                "subsample": self.subsample, "random_state": _seed(self.random_state)}


class KBinsDiscretizer(TransformerMixin, BaseEstimator):
    _kind = "KBinsDiscretizer"

    # sklearn's default encode='onehot' returns a sparse matrix; here the
    # one-hot output is dense.
    def __init__(self, n_bins=5, encode="onehot", strategy="quantile"):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

    def _rust_params(self, X, y):
        return {"n_bins": self.n_bins,
                "encode": _choice(self.encode, {"ordinal": "Ordinal", "onehot": "Onehot",
                                                "onehot-dense": "Onehot"},
                                  "encode"),
                "strategy": _choice(self.strategy, {"uniform": "Uniform",
                                                    "quantile": "Quantile",
                                                    "kmeans": "KMeans"}, "strategy")}


class VarianceThreshold(TransformerMixin, BaseEstimator):
    _kind = "VarianceThreshold"

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def _rust_params(self, X, y):
        return {"threshold": float(self.threshold)}


class PCA(TransformerMixin, BaseEstimator):
    _kind = "Pca"

    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def _rust_params(self, X, y):
        n = min(X.shape) if self.n_components is None else int(self.n_components)
        return {"n_components": n, "whiten": self.whiten}


class IncrementalPCA(TransformerMixin, BaseEstimator):
    _kind = "IncrementalPca"

    def __init__(self, n_components=None, whiten=False, batch_size=None):
        self.n_components = n_components
        self.whiten = whiten
        self.batch_size = batch_size

    def _rust_params(self, X, y):
        return {"n_components": self.n_components, "whiten": self.whiten,
                "batch_size": self.batch_size}


class KernelPCA(TransformerMixin, BaseEstimator):
    _kind = "KernelPca"

    def __init__(self, n_components=None, kernel="linear", gamma=None, degree=3, coef0=1.0,
                 remove_zero_eig=False):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.remove_zero_eig = remove_zero_eig

    def _rust_params(self, X, y):
        gamma = 1.0 / X.shape[1] if self.gamma is None else float(self.gamma)
        if self.kernel == "linear":
            kernel = "Linear"
        elif self.kernel == "rbf":
            kernel = {"Rbf": {"gamma": gamma}}
        elif self.kernel == "poly":
            kernel = {"Polynomial": {"degree": int(self.degree), "coef0": float(self.coef0),
                                     "gamma": gamma}}
        else:
            raise ValueError(f"unknown kernel {self.kernel!r}")
        n = X.shape[0] if self.n_components is None else int(self.n_components)
        # As sklearn: with n_components=None zero-eigenvalue components are
        # always removed.
        return {"n_components": n, "kernel": kernel,
                "remove_zero_eig": self.remove_zero_eig or self.n_components is None}


class TruncatedSVD(TransformerMixin, BaseEstimator):
    _kind = "TruncatedSvd"

    def __init__(self, n_components=2):
        self.n_components = n_components

    def _rust_params(self, X, y):
        return {"n_components": self.n_components}


class FastICA(TransformerMixin, BaseEstimator):
    _kind = "FastIca"

    def __init__(self, n_components=None, *, algorithm="parallel", whiten="unit-variance",
                 fun="logcosh", fun_args=None, max_iter=200, tol=1e-4, w_init=None,
                 whiten_solver="svd", random_state=None):
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.whiten_solver = whiten_solver
        self.random_state = random_state

    def _rust_params(self, X, y):
        fun = _choice(self.fun, {"logcosh": "Logcosh", "exp": "Exp", "cube": "Cube"}, "fun")
        if fun == "Logcosh":
            fun = {"Logcosh": {"alpha": float((self.fun_args or {}).get("alpha", 1.0))}}
        whiten = {"unit-variance": "UnitVariance", "arbitrary-variance": "ArbitraryVariance",
                  False: "None"}
        w_init = None if self.w_init is None else np.asarray(self.w_init, dtype=float)
        return {"n_components": None if self.n_components is None else int(self.n_components),
                "algorithm": _choice(self.algorithm, {"parallel": "Parallel",
                                                      "deflation": "Deflation"}, "algorithm"),
                "whiten": _choice(self.whiten, whiten, "whiten"), "fun": fun,
                "max_iter": self.max_iter, "tol": self.tol,
                "w_init": None if w_init is None else {"v": 1, "dim": list(w_init.shape),
                                                       "data": w_init.ravel().tolist()},
                "whiten_solver": _choice(self.whiten_solver, {"svd": "Svd", "eigh": "Eigh"},
                                         "whiten_solver"),
                "seed": _seed(self.random_state)}

    @property
    def components_(self):
        return _nd(self._fitted_state()["components"])

    @property
    def mixing_(self):
        return _nd(self._fitted_state()["mixing"])

    @property
    def mean_(self):
        return _nd(self._fitted_state()["mean"])


class NMF(TransformerMixin, BaseEstimator):
    _kind = "Nmf"

    @property
    def components_(self):
        return _nd(self._fitted_state()["components"])

    @property
    def reconstruction_err_(self):
        return self._fitted_state()["reconstruction_err"]

    def __init__(self, n_components=None, init=None, solver="cd", beta_loss="frobenius",
                 tol=1e-4, max_iter=200, random_state=None, alpha_W=0.0, alpha_H="same",
                 l1_ratio=0.0):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio

    def _rust_params(self, X, y):
        n = X.shape[1] if self.n_components is None else int(self.n_components)
        return {"n_components": n,
                "solver": _choice(self.solver, {"cd": "Cd", "mu": "Mu"}, "solver"),
                "beta_loss": _choice(self.beta_loss, {"frobenius": "Frobenius",
                                                      "kullback-leibler": "KullbackLeibler",
                                                      "itakura-saito": "ItakuraSaito"},
                                     "beta_loss"),
                "init": _choice(self.init, {None: "Auto", "random": "Random",
                                            "nndsvd": "Nndsvd", "nndsvda": "NndsvdA",
                                            "nndsvdar": "NndsvdAr"}, "init"),
                "tol": self.tol, "max_iter": self.max_iter, "seed": _seed(self.random_state),
                "alpha_w": float(self.alpha_W),
                "alpha_h": None if self.alpha_H == "same" else float(self.alpha_H),
                "l1_ratio": float(self.l1_ratio)}


_SCORE_FUNCS = {"f_classif": "FClassif", "f_regression": "FRegression", "chi2": "Chi2",
                "mutual_info_classif": "MutualInfoClassif",
                "mutual_info_regression": "MutualInfoRegression"}


class SelectKBest(SupervisedTransformerMixin, BaseEstimator):
    """``score_func`` is one of ``"f_classif"``, ``"f_regression"``, ``"chi2"``,
    ``"mutual_info_classif"``, ``"mutual_info_regression"``."""

    _kind = "SelectKBest"

    def __init__(self, score_func="f_classif", k=10):
        self.score_func = score_func
        self.k = k

    def _rust_params(self, X, y):
        k = X.shape[1] if self.k == "all" else min(int(self.k), X.shape[1])
        return {"k": k, "scoring_fn": _choice(self.score_func, _SCORE_FUNCS, "score_func")}


class MutualInformationSelector(SupervisedTransformerMixin, BaseEstimator):
    _kind = "MutualInformationSelector"

    def __init__(self, n_features_to_select=10, n_neighbors=3, random_state=None):
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_features_to_select": min(int(self.n_features_to_select), X.shape[1]),
                "n_neighbors": self.n_neighbors, "random_state": _seed(self.random_state)}


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


class KMeans(ClusterMixin, BaseEstimator):
    _kind = "KMeans"

    def __init__(self, n_clusters=8, init="k-means++", n_init="auto", max_iter=300, tol=1e-4,
                 random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _rust_params(self, X, y):
        if isinstance(self.init, str):
            init = _choice(self.init, {"k-means++": "KMeansPlusPlus", "random": "Random"},
                           "init")
        else:
            arr = np.asarray(self.init, dtype=np.float64)
            init = {"Array": {"v": 1, "dim": list(arr.shape), "data": arr.ravel().tolist()}}
        return {"n_clusters": self.n_clusters, "init": init,
                "n_init": None if self.n_init == "auto" else int(self.n_init),
                "max_iter": self.max_iter, "tol": self.tol, "seed": _seed(self.random_state)}

    @property
    def cluster_centers_(self):
        return _nd(self._fitted_state()["centroids"])


class MiniBatchKMeans(ClusterMixin, BaseEstimator):
    _kind = "MiniBatchKMeans"

    def __init__(self, n_clusters=8, *, init="k-means++", max_iter=100, batch_size=1024,
                 random_state=None, tol=0.0, max_no_improvement=10, init_size=None,
                 n_init="auto", reassignment_ratio=0.01):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.init_size = init_size
        self.n_init = n_init
        self.reassignment_ratio = reassignment_ratio

    def _rust_params(self, X, y):
        return {"n_clusters": self.n_clusters,
                "init": _choice(self.init, {"k-means++": "KMeansPlusPlus", "random": "Random"},
                                "init"),
                "max_iter": self.max_iter, "batch_size": self.batch_size, "tol": float(self.tol),
                "max_no_improvement": self.max_no_improvement, "init_size": self.init_size,
                "n_init": None if self.n_init == "auto" else int(self.n_init),
                "reassignment_ratio": float(self.reassignment_ratio),
                "seed": _seed(self.random_state)}


class DBSCAN(ClusterMixin, BaseEstimator):
    _kind = "Dbscan"

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def _rust_params(self, X, y):
        return {"eps": float(self.eps), "min_samples": self.min_samples}


class HDBSCAN(ClusterMixin, BaseEstimator):
    _kind = "Hdbscan"

    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_method="eom",
                 allow_single_cluster=False):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster

    def _rust_params(self, X, y):
        return {"min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_cluster_size if self.min_samples is None
                else self.min_samples,
                "cluster_selection_method": _choice(self.cluster_selection_method,
                                                    {"eom": "Eom", "leaf": "Leaf"},
                                                    "cluster_selection_method"),
                "allow_single_cluster": self.allow_single_cluster}


class OPTICS(ClusterMixin, BaseEstimator):
    _kind = "Optics"

    def __init__(self, *, min_samples=5, max_eps=np.inf, cluster_method="xi", eps=None,
                 xi=0.05, predecessor_correction=True, min_cluster_size=None):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size = min_cluster_size

    def _rust_params(self, X, y):
        # f64::INFINITY does not survive JSON; use the largest float.
        big = np.finfo(np.float64).max
        max_eps = big if np.isinf(self.max_eps) else float(self.max_eps)
        if self.cluster_method == "xi":
            method = {"Xi": {"xi": float(self.xi),
                             "min_cluster_size": None if self.min_cluster_size is None
                             else int(self.min_cluster_size),
                             "predecessor_correction": bool(self.predecessor_correction)}}
            eps = big
        elif self.cluster_method == "dbscan":
            method = "Dbscan"
            eps = max_eps if self.eps is None else float(self.eps)
        else:
            raise ValueError(f"unsupported cluster_method {self.cluster_method!r}")
        return {"min_samples": int(self.min_samples), "eps_max": max_eps, "eps": eps,
                "cluster_method": method}


class MeanShift(ClusterMixin, BaseEstimator):
    _kind = "MeanShift"

    def __init__(self, *, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1,
                 cluster_all=True, max_iter=300):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.max_iter = max_iter

    def _rust_params(self, X, y):
        seeds = None
        if self.seeds is not None:
            arr = np.ascontiguousarray(self.seeds, dtype=np.float64)
            seeds = {"v": 1, "dim": list(arr.shape), "data": arr.ravel().tolist()}
        return {"bandwidth": None if self.bandwidth is None else float(self.bandwidth),
                "seeds": seeds, "bin_seeding": bool(self.bin_seeding),
                "min_bin_freq": int(self.min_bin_freq), "cluster_all": bool(self.cluster_all),
                "max_iter": int(self.max_iter)}

    @property
    def cluster_centers_(self):
        return _nd(self._fitted_state()["cluster_centers"])


class Birch(ClusterMixin, BaseEstimator):
    _kind = "Birch"

    def __init__(self, *, threshold=0.5, branching_factor=50, n_clusters=3):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters

    def _rust_params(self, X, y):
        return {"threshold": float(self.threshold),
                "branching_factor": int(self.branching_factor),
                "n_clusters": self.n_clusters}


class AgglomerativeClustering(ClusterMixin, BaseEstimator):
    _kind = "AgglomerativeClustering"

    def __init__(self, n_clusters=2, *, linkage="ward", distance_threshold=None):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold

    def _rust_params(self, X, y):
        if (self.n_clusters is None) == (self.distance_threshold is None):
            raise ValueError("Exactly one of n_clusters and distance_threshold has to be set")
        return {"n_clusters": 1 if self.n_clusters is None else int(self.n_clusters),
                "linkage": _choice(self.linkage, {"ward": "Ward", "complete": "Complete",
                                                  "average": "Average", "single": "Single"},
                                   "linkage"),
                "distance_threshold": None if self.distance_threshold is None
                else float(self.distance_threshold)}


class SpectralClustering(ClusterMixin, BaseEstimator):
    _kind = "SpectralClustering"

    def __init__(self, n_clusters=8, *, n_init=10, gamma=1.0, affinity="rbf",
                 n_neighbors=10, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def _rust_params(self, X, y):
        if self.affinity == "rbf":
            affinity = {"Rbf": float(self.gamma)}
        elif self.affinity == "nearest_neighbors":
            affinity = {"KNearest": int(self.n_neighbors)}
        else:
            raise ValueError(f"unsupported affinity {self.affinity!r}")
        return {"n_clusters": self.n_clusters, "affinity": affinity,
                "n_init": int(self.n_init), "seed": _seed(self.random_state)}


class AffinityPropagation(ClusterMixin, BaseEstimator):
    _kind = "AffinityPropagation"

    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, preference=None):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference

    def _rust_params(self, X, y):
        return {"damping": self.damping, "max_iter": self.max_iter,
                "convergence_iter": self.convergence_iter,
                "preference": None if self.preference is None else float(self.preference)}


_COV = {"full": "Full", "diag": "Diag", "tied": "Tied", "spherical": "Spherical"}
_GMM_INIT = {"kmeans": "KMeans", "k-means++": "KMeansPlusPlus", "random": "Random",
             "random_from_data": "RandomFromData"}


class GaussianMixture(ClusterMixin, BaseEstimator):
    _kind = "GaussianMixture"

    def __init__(self, n_components=1, covariance_type="full", tol=1e-3, reg_covar=1e-6,
                 max_iter=100, n_init=1, init_params="kmeans", random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_components": self.n_components,
                "covariance_type": _choice(self.covariance_type, _COV, "covariance_type"),
                "tol": self.tol, "reg_covar": self.reg_covar, "max_iter": self.max_iter,
                "n_init": self.n_init,
                "init_params": _choice(self.init_params, _GMM_INIT, "init_params"),
                "seed": _seed(self.random_state)}

    def predict_proba(self, X):
        return self._call("predict_proba", X)

    def score_samples(self, X):
        return self._call("score_samples", X)


class BayesianGaussianMixture(ClusterMixin, BaseEstimator):
    _kind = "BayesianGaussianMixture"

    def __init__(self, n_components=1, covariance_type="full", tol=1e-3, reg_covar=1e-6,
                 max_iter=100, n_init=1, init_params="kmeans",
                 weight_concentration_prior=None, mean_precision_prior=None,
                 random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.random_state = random_state

    def _rust_params(self, X, y):
        p = {"n_components": self.n_components,
             "covariance_type": _choice(self.covariance_type, _COV, "covariance_type"),
             "tol": self.tol, "reg_covar": self.reg_covar, "max_iter": self.max_iter,
             "n_init": self.n_init,
             "init_params": _choice(self.init_params, _GMM_INIT, "init_params"),
             "seed": _seed(self.random_state)}
        if self.weight_concentration_prior is not None:
            p["weight_concentration_prior"] = float(self.weight_concentration_prior)
        else:
            p["weight_concentration_prior"] = 1.0 / self.n_components
        if self.mean_precision_prior is not None:
            p["mean_precision_prior"] = float(self.mean_precision_prior)
        return p

    def predict_proba(self, X):
        return self._call("predict_proba", X)


# ---------------------------------------------------------------------------
# manifold learning
# ---------------------------------------------------------------------------


class _Embedding(BaseEstimator):
    def fit(self, X, y=None):
        self._fit_model(X)
        self.embedding_ = np.asarray(self._model().embedding())
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_


class TSNE(_Embedding):
    _kind = "TSne"

    def __init__(self, n_components=2, *, perplexity=30.0, early_exaggeration=12.0,
                 learning_rate="auto", max_iter=1000, n_iter_without_progress=300,
                 min_grad_norm=1e-7, init="pca", method="barnes_hut", angle=0.5,
                 random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.init = init
        self.method = method
        self.angle = angle
        self.random_state = random_state

    def _rust_params(self, X, y):
        if isinstance(self.init, str):
            init = _choice(self.init, {"pca": "Pca", "random": "Random"}, "init")
        else:
            arr = np.ascontiguousarray(self.init, dtype=np.float64)
            init = {"Array": {"v": 1, "dim": list(arr.shape), "data": arr.ravel().tolist()}}
        lr = None if self.learning_rate == "auto" else float(self.learning_rate)
        return {"n_components": self.n_components, "perplexity": float(self.perplexity),
                "early_exaggeration": float(self.early_exaggeration),
                "learning_rate": lr, "max_iter": int(self.max_iter),
                "n_iter_without_progress": int(self.n_iter_without_progress),
                "min_grad_norm": float(self.min_grad_norm), "init": init,
                "method": _choice(self.method, {"exact": "Exact", "barnes_hut": "BarnesHut"},
                                  "method"),
                "theta": float(self.angle), "seed": _seed(self.random_state)}


class Isomap(_Embedding):
    _kind = "Isomap"

    def __init__(self, n_neighbors=5, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def _rust_params(self, X, y):
        return {"n_neighbors": self.n_neighbors, "n_components": self.n_components}


class LocallyLinearEmbedding(_Embedding):
    _kind = "LocallyLinearEmbedding"

    def __init__(self, n_neighbors=5, n_components=2, reg=1e-3):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg

    def _rust_params(self, X, y):
        return {"n_neighbors": self.n_neighbors, "n_components": self.n_components,
                "reg": self.reg}


class ClassicalMDS(_Embedding):
    """Classical (Torgerson) multidimensional scaling, sklearn's
    ``ClassicalMDS``. (sklearn's ``MDS`` is the iterative SMACOF algorithm,
    which this package does not provide.)"""

    _kind = "ClassicalMds"

    def __init__(self, n_components=2, *, metric="euclidean"):
        self.n_components = n_components
        self.metric = metric

    def _rust_params(self, X, y):
        return {"n_components": self.n_components,
                "metric": _choice(self.metric, _METRIC, "metric")}


# ---------------------------------------------------------------------------
# baselines and simple classifiers (#113)
# ---------------------------------------------------------------------------


class DummyClassifier(ClassifierMixin, BaseEstimator):
    """Baseline that ignores the features (sklearn's ``DummyClassifier``)."""

    _kind = "DummyClassifier"

    def __init__(self, strategy="prior", random_state=None, constant=None):
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant

    def _rust_params(self, X, y):
        if self.strategy == "constant":
            if self.constant is None:
                raise ValueError("strategy='constant' needs `constant`")
            idx = np.flatnonzero(self.classes_ == self.constant)
            if idx.size == 0:
                raise ValueError(
                    f"constant {self.constant!r} is not one of the training labels "
                    f"{self.classes_.tolist()}"
                )
            strategy = {"Constant": float(idx[0])}
        else:
            strategy = _choice(self.strategy, {"most_frequent": "MostFrequent", "prior": "Prior",
                                               "stratified": "Stratified",
                                               "uniform": "Uniform"}, "strategy")
        return {"strategy": strategy, "seed": _seed(self.random_state)}


class DummyRegressor(RegressorMixin, BaseEstimator):
    """Baseline that predicts a constant (sklearn's ``DummyRegressor``)."""

    _kind = "DummyRegressor"

    def __init__(self, strategy="mean", constant=None, quantile=None):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

    def _rust_params(self, X, y):
        if self.strategy == "quantile":
            if self.quantile is None:
                raise ValueError("strategy='quantile' needs `quantile`")
            strategy = {"Quantile": float(self.quantile)}
        elif self.strategy == "constant":
            if self.constant is None:
                raise TypeError("strategy='constant' needs `constant`")
            strategy = {"Constant": float(self.constant)}
        else:
            strategy = _choice(self.strategy, {"mean": "Mean", "median": "Median"}, "strategy")
        return {"strategy": strategy}


class NearestCentroid(ClassifierMixin, BaseEstimator):
    """Nearest centroid classifier, with shrunken centroids and priors."""

    _kind = "NearestCentroid"

    def __init__(self, metric="euclidean", *, shrink_threshold=None, priors="uniform"):
        self.metric = metric
        self.shrink_threshold = shrink_threshold
        self.priors = priors

    def _rust_params(self, X, y):
        if isinstance(self.priors, str):
            priors = _choice(self.priors, {"uniform": "Uniform", "empirical": "Empirical"}, "priors")
        else:
            priors = {"Given": [float(p) for p in self.priors]}
        return {"metric": _choice(self.metric, {"euclidean": "Euclidean",
                                                "manhattan": "Manhattan"}, "metric"),
                "shrink_threshold": None if self.shrink_threshold is None
                else float(self.shrink_threshold),
                "priors": priors}


class RidgeClassifier(ClassifierMixin, BaseEstimator):
    """Ridge regression on ±1-encoded labels (sklearn's ``RidgeClassifier``)."""

    _kind = "RidgeClassifier"

    def __init__(self, alpha=1.0, *, fit_intercept=True, class_weight=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight

    def _rust_params(self, X, y):
        return {"alpha": float(self.alpha), "fit_intercept": bool(self.fit_intercept),
                "class_weight": self._encode_class_weight(self.class_weight)}

    @property
    def coef_(self):
        return _nd(self._fitted_state()["coef"])

    @property
    def intercept_(self):
        return _nd(self._fitted_state()["intercept"])


class Perceptron(ClassifierMixin, BaseEstimator):
    """The perceptron: SGD with the perceptron loss and a constant
    learning rate (sklearn's ``Perceptron``)."""

    _kind = "Perceptron"

    def __init__(self, *, penalty=None, alpha=0.0001, l1_ratio=0.15, max_iter=1000, tol=1e-3,
                 shuffle=True, eta0=1.0, random_state=0, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=5, class_weight=None):
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.eta0 = eta0
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight

    def _rust_params(self, X, y):
        return {"penalty": _choice(self.penalty, _PENALTY, "penalty"), "alpha": self.alpha,
                "l1_ratio": self.l1_ratio, "max_iter": self.max_iter, "tol": self.tol,
                "shuffle": self.shuffle, "eta0": self.eta0, "seed": _seed(self.random_state),
                "early_stopping": self.early_stopping,
                "validation_fraction": self.validation_fraction,
                "n_iter_no_change": self.n_iter_no_change,
                "class_weight": self._encode_class_weight(self.class_weight)}


# ---------------------------------------------------------------------------
# univariate selection modes and TF-IDF (#113)
# ---------------------------------------------------------------------------


class GenericUnivariateSelect(SupervisedTransformerMixin, BaseEstimator):
    """Univariate selection with ``mode`` in ``"percentile"``, ``"k_best"``,
    ``"fpr"``, ``"fdr"``, ``"fwe"`` (``score_func`` as in ``SelectKBest``;
    the p-value modes need ``f_classif``, ``f_regression`` or ``chi2``)."""

    _kind = "GenericUnivariateSelect"

    def __init__(self, score_func="f_classif", *, mode="percentile", param=1e-5):
        self.score_func = score_func
        self.mode = mode
        self.param = param

    def _mode(self, X):
        if self.mode == "k_best":
            k = X.shape[1] if self.param == "all" else int(self.param)
            return {"KBest": k}
        tag = _choice(self.mode, {"percentile": "Percentile", "fpr": "Fpr", "fdr": "Fdr",
                                  "fwe": "Fwe"}, "mode")
        return {tag: float(self.param)}

    def _rust_params(self, X, y):
        return {"scoring_fn": _choice(self.score_func, _SCORE_FUNCS, "score_func"),
                "mode": self._mode(X)}

    def get_support(self, indices=False):
        idx = np.asarray(self._fitted_state()["selected_indices"], dtype=np.int64)
        if indices:
            return idx
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[idx] = True
        return mask


class SelectPercentile(GenericUnivariateSelect):
    """Keep the best ``percentile`` percent of the features."""

    def __init__(self, score_func="f_classif", *, percentile=10):
        self.score_func = score_func
        self.percentile = percentile

    def _rust_params(self, X, y):
        return {"scoring_fn": _choice(self.score_func, _SCORE_FUNCS, "score_func"),
                "mode": {"Percentile": float(self.percentile)}}


class _PValueSelect(GenericUnivariateSelect):
    _tag = ""

    def __init__(self, score_func="f_classif", *, alpha=5e-2):
        self.score_func = score_func
        self.alpha = alpha

    def _rust_params(self, X, y):
        return {"scoring_fn": _choice(self.score_func, _SCORE_FUNCS, "score_func"),
                "mode": {self._tag: float(self.alpha)}}


class SelectFpr(_PValueSelect):
    """Keep features with p-value below ``alpha``."""

    _tag = "Fpr"


class SelectFdr(_PValueSelect):
    """Benjamini-Hochberg selection at false discovery rate ``alpha``."""

    _tag = "Fdr"


class SelectFwe(_PValueSelect):
    """Keep features with p-value below ``alpha / n_features``."""

    _tag = "Fwe"


class TfidfTransformer(TransformerMixin, BaseEstimator):
    """TF-IDF reweighting of a (dense) count matrix."""

    _kind = "TfidfTransformer"

    def __init__(self, *, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def _rust_params(self, X, y):
        return {"norm": _choice(self.norm, {"l2": "L2", "l1": "L1", None: "None"}, "norm"),
                "use_idf": bool(self.use_idf), "smooth_idf": bool(self.smooth_idf),
                "sublinear_tf": bool(self.sublinear_tf)}

    @property
    def idf_(self):
        idf = self._fitted_state()["idf"]
        if idf is None:
            raise AttributeError("idf_ is only defined when use_idf=True")
        return _nd(idf)


# ---------------------------------------------------------------------------
# covariance estimation (#113)
# ---------------------------------------------------------------------------


class _Covariance(BaseEstimator):
    """Shared API: ``location_``, ``covariance_``, ``precision_``,
    ``mahalanobis(X)`` and the Gaussian log-likelihood ``score(X)``."""

    def fit(self, X, y=None):
        return self._fit_model(X, None, None)

    def _state(self, key):
        return _nd(self._fitted_state()[key])

    @property
    def location_(self):
        return self._state("location")

    @property
    def covariance_(self):
        return self._state("covariance")

    @property
    def precision_(self):
        return self._state("precision")

    def mahalanobis(self, X):
        return self._call("transform", X)[:, 0]

    def score(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        xc = X - self.location_
        s = xc.T @ xc / X.shape[0]
        p = self.precision_
        sign, logdet = np.linalg.slogdet(p)
        logdet = logdet if sign > 0 else -np.inf
        return float((-np.sum(s * p) + logdet - p.shape[0] * np.log(2 * np.pi)) / 2)


class EmpiricalCovariance(_Covariance):
    _kind = "EmpiricalCovariance"

    def __init__(self, *, assume_centered=False):
        self.assume_centered = assume_centered

    def _rust_params(self, X, y):
        return {"assume_centered": bool(self.assume_centered)}


class ShrunkCovariance(_Covariance):
    _kind = "ShrunkCovariance"

    def __init__(self, *, assume_centered=False, shrinkage=0.1):
        self.assume_centered = assume_centered
        self.shrinkage = shrinkage

    def _rust_params(self, X, y):
        return {"assume_centered": bool(self.assume_centered), "shrinkage": float(self.shrinkage)}


class LedoitWolf(_Covariance):
    _kind = "LedoitWolf"

    def __init__(self, *, assume_centered=False):
        self.assume_centered = assume_centered

    def _rust_params(self, X, y):
        return {"assume_centered": bool(self.assume_centered)}

    @property
    def shrinkage_(self):
        return float(self._fitted_state()["shrinkage"])


class OAS(LedoitWolf):
    _kind = "Oas"


class MinCovDet(_Covariance):
    """Robust covariance (FastMCD). Its random starts come from its own
    generator, so on hard data the support can differ from sklearn's."""

    _kind = "MinCovDet"

    def __init__(self, *, assume_centered=False, support_fraction=None, random_state=None):
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"assume_centered": bool(self.assume_centered),
                "support_fraction": None if self.support_fraction is None
                else float(self.support_fraction),
                "seed": _seed(self.random_state)}

    @property
    def support_(self):
        return np.asarray(self._fitted_state()["support"], dtype=bool)

    @property
    def dist_(self):
        return self._state("dist")

    @property
    def raw_location_(self):
        return self._state("raw_location")

    @property
    def raw_covariance_(self):
        return self._state("raw_covariance")

    @property
    def raw_support_(self):
        return np.asarray(self._fitted_state()["raw_support"], dtype=bool)


class EllipticEnvelope(OutlierMixin, BaseEstimator):
    """Outliers under a Gaussian assumption, from a robust (MCD) fit."""

    _kind = "EllipticEnvelope"

    def __init__(self, *, assume_centered=False, support_fraction=None, contamination=0.1,
                 random_state=None):
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"assume_centered": bool(self.assume_centered),
                "support_fraction": None if self.support_fraction is None
                else float(self.support_fraction),
                "contamination": float(self.contamination), "seed": _seed(self.random_state)}

    def predict(self, X):
        return np.asarray(self._call("predict", X), dtype=np.int64)

    def decision_function(self, X):
        return self._call("decision_function", X)

    def score_samples(self, X):
        return self._call("score_samples", X)

    @property
    def offset_(self):
        return float(self._fitted_state()["offset"])

    @property
    def covariance_(self):
        return _nd(self._fitted_state()["mcd"]["covariance"])

    @property
    def location_(self):
        return _nd(self._fitted_state()["mcd"]["location"])


# ---------------------------------------------------------------------------
# kernel approximation and random projections (#113)
# ---------------------------------------------------------------------------


def _projection_components(n_components, eps):
    if n_components == "auto":
        return {"Auto": {"eps": float(eps)}}
    return {"Fixed": int(n_components)}


class Nystroem(TransformerMixin, BaseEstimator):
    """Low-rank kernel feature map from ``n_components`` random rows."""

    _kind = "Nystroem"

    def __init__(self, kernel="rbf", *, gamma=None, coef0=None, degree=None, n_components=100,
                 random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.n_components = n_components
        self.random_state = random_state

    def _rust_params(self, X, y):
        g = None if self.gamma is None else float(self.gamma)
        if self.kernel == "rbf":
            kernel = {"Rbf": {"gamma": g}}
        elif self.kernel == "laplacian":
            kernel = {"Laplacian": {"gamma": g}}
        elif self.kernel == "poly":
            kernel = {"Polynomial": {"degree": float(3 if self.degree is None else self.degree),
                                     "gamma": g,
                                     "coef0": float(1 if self.coef0 is None else self.coef0)}}
        elif self.kernel == "sigmoid":
            kernel = {"Sigmoid": {"gamma": g,
                                  "coef0": float(1 if self.coef0 is None else self.coef0)}}
        elif self.kernel == "linear":
            kernel = "Linear"
        else:
            raise ValueError(f"unsupported kernel {self.kernel!r}")
        return {"kernel": kernel, "n_components": int(self.n_components),
                "seed": _seed(self.random_state)}

    @property
    def components_(self):
        return _nd(self._fitted_state()["components"])

    @property
    def component_indices_(self):
        return np.asarray(self._fitted_state()["component_indices"], dtype=np.int64)


class RBFSampler(TransformerMixin, BaseEstimator):
    """Random Fourier features for the RBF kernel."""

    _kind = "RbfSampler"

    def __init__(self, *, gamma=1.0, n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"gamma": None if self.gamma == "scale" else float(self.gamma),
                "n_components": int(self.n_components), "seed": _seed(self.random_state)}


class KernelCenterer(TransformerMixin, BaseEstimator):
    """Centre a kernel matrix in feature space."""

    _kind = "KernelCenterer"

    def __init__(self):
        pass


class GaussianRandomProjection(TransformerMixin, BaseEstimator):
    _kind = "GaussianRandomProjection"

    def __init__(self, n_components="auto", *, eps=0.1, random_state=None):
        self.n_components = n_components
        self.eps = eps
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_components": _projection_components(self.n_components, self.eps),
                "seed": _seed(self.random_state)}

    @property
    def components_(self):
        return _nd(self._fitted_state()["components"])


class SparseRandomProjection(GaussianRandomProjection):
    _kind = "SparseRandomProjection"

    def __init__(self, n_components="auto", *, density="auto", eps=0.1, random_state=None):
        self.n_components = n_components
        self.density = density
        self.eps = eps
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {**super()._rust_params(X, y),
                "density": None if self.density == "auto" else float(self.density)}


# ---------------------------------------------------------------------------
# density estimation and factor analysis (#113)
# ---------------------------------------------------------------------------


class KernelDensity(BaseEstimator):
    """Kernel density estimation (exact sums, euclidean distance)."""

    _kind = "KernelDensity"

    def __init__(self, *, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def _rust_params(self, X, y):
        if self.bandwidth == "scott":
            bw = "Scott"
        elif self.bandwidth == "silverman":
            bw = "Silverman"
        else:
            bw = {"Fixed": float(self.bandwidth)}
        return {"bandwidth": bw, "kernel": _choice(
            self.kernel, {"gaussian": "Gaussian", "tophat": "Tophat",
                          "epanechnikov": "Epanechnikov", "exponential": "Exponential",
                          "linear": "Linear", "cosine": "Cosine"}, "kernel")}

    def fit(self, X, y=None, sample_weight=None):
        return self._fit_model(X, None, sample_weight)

    def score_samples(self, X):
        return self._call("score_samples", X)

    def score(self, X, y=None):
        return float(np.sum(self.score_samples(X)))

    @property
    def bandwidth_(self):
        return float(self._fitted_state()["bandwidth"])

    def sample(self, n_samples=1, random_state=None):
        """Draw from the density (gaussian and tophat kernels)."""
        if self.kernel not in ("gaussian", "tophat"):
            raise NotImplementedError("sampling needs the gaussian or tophat kernel")
        state = self._fitted_state()
        data = _nd(state["data"])
        rng = np.random.default_rng(_seed(random_state))
        w = state["weights"]
        if w is None:
            idx = rng.integers(0, len(data), n_samples)
        else:
            w = _nd(w)
            idx = rng.choice(len(data), size=n_samples, p=w / w.sum())
        h = self.bandwidth_
        if self.kernel == "gaussian":
            return data[idx] + rng.normal(scale=h, size=(n_samples, data.shape[1]))
        z = rng.normal(size=(n_samples, data.shape[1]))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        r = h * rng.uniform(size=(n_samples, 1)) ** (1.0 / data.shape[1])
        return data[idx] + z * r


class FactorAnalysis(TransformerMixin, BaseEstimator):
    """Factor analysis (exact SVD; component signs follow the SVD)."""

    _kind = "FactorAnalysis"

    def __init__(self, n_components=None, *, tol=1e-2, max_iter=1000, noise_variance_init=None,
                 rotation=None):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.noise_variance_init = noise_variance_init
        self.rotation = rotation

    def _rust_params(self, X, y):
        return {"n_components": None if self.n_components is None else int(self.n_components),
                "tol": float(self.tol), "max_iter": int(self.max_iter),
                "noise_variance_init": None if self.noise_variance_init is None
                else [float(v) for v in self.noise_variance_init],
                "rotation": None if self.rotation is None else _choice(
                    self.rotation, {"varimax": "Varimax", "quartimax": "Quartimax"}, "rotation")}

    @property
    def components_(self):
        return _nd(self._fitted_state()["components"])

    @property
    def noise_variance_(self):
        return _nd(self._fitted_state()["noise_variance"])

    @property
    def mean_(self):
        return _nd(self._fitted_state()["mean"])

    @property
    def n_iter_(self):
        return int(self._fitted_state()["n_iter"])

    @property
    def loglike_(self):
        return list(self._fitted_state()["loglike"])

    def get_covariance(self):
        w = self.components_
        return w.T @ w + np.diag(self.noise_variance_)

    def get_precision(self):
        return np.linalg.inv(self.get_covariance())

    def score_samples(self, X):
        return self._call("score_samples", X)

    def score(self, X, y=None):
        return float(np.mean(self.score_samples(X)))
