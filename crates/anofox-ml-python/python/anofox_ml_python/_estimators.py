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

    def __init__(self, n_estimators=100, max_samples=256, contamination=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_estimators": self.n_estimators, "max_samples": self.max_samples,
                "contamination": self.contamination, "seed": _seed(self.random_state)}

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
    """L1-penalised least squares.

    ``alpha`` is passed to anofox-regression's ``lambda``. Its penalty
    convention does not yet match sklearn's ``(1 / 2n) ||y - Xw||² +
    alpha ||w||₁``, so coefficients differ from sklearn's for the same
    ``alpha`` (tracked upstream, sipemu/anofox-ml#72).
    """

    _kind = "LassoRegressor"

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {"lambda": float(self.alpha), "with_intercept": self.fit_intercept}


class ElasticNet(_LinearAttrs, RegressorMixin, BaseEstimator):
    """Elastic-net penalised least squares (``alpha``, ``l1_ratio``).

    As for :class:`Lasso`, ``alpha`` is anofox-regression's ``lambda`` and
    does not yet follow sklearn's scaling (sipemu/anofox-ml#72).
    """

    _kind = "ElasticNetRegressor"

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {"lambda": float(self.alpha), "alpha": float(self.l1_ratio),
                "with_intercept": self.fit_intercept}


class HuberRegressor(_LinearAttrs, RegressorMixin, BaseEstimator):
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
    """Unpenalised linear quantile regression."""

    _kind = "QuantileRegressor"

    def __init__(self, quantile=0.5, fit_intercept=True):
        self.quantile = quantile
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {"quantile": self.quantile, "with_intercept": self.fit_intercept}


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
                "folds": {"n_folds": int(self.cv), "fold_ids": None,
                          "seed": _seed(self.random_state)}}


class ElasticNetCV(LassoCV):
    _kind = "ElasticNetCrossValidated"

    def __init__(self, l1_ratio=(0.1, 0.3, 0.5, 0.7, 0.9), eps=1e-3, n_alphas=100,
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
    """L2-penalised logistic regression (one-vs-rest for more than two classes)."""

    _kind = "LogisticRegressor"

    def __init__(self, C=1.0, fit_intercept=True, tol=1e-8, max_iter=1000, class_weight=None):
        self.C = C
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight

    def _rust_params(self, X, y):
        return {"c": float(self.C), "with_intercept": self.fit_intercept, "tol": self.tol,
                "max_iter": self.max_iter,
                "class_weight": self._encode_class_weight(self.class_weight)}


class _Glm(_LinearAttrs, RegressorMixin, BaseEstimator):
    def __init__(self, alpha=0.0, fit_intercept=True, max_iter=100, tol=1e-8):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def _rust_params(self, X, y):
        return {"alpha": float(self.alpha), "with_intercept": self.fit_intercept,
                "max_iter": self.max_iter, "tol": self.tol}


class PoissonRegressor(_Glm):
    """Poisson GLM with log link. ``alpha`` defaults to 0 (see CHANGELOG, #72)."""

    _kind = "PoissonRegressor"


class GammaRegressor(_Glm):
    """Gamma GLM with log link. ``alpha`` defaults to 0 (see CHANGELOG, #72)."""

    _kind = "GammaRegressor"

    def _rust_params(self, X, y):
        inner = dict(_rust_defaults(self._kind)["inner"])
        inner.update(super()._rust_params(X, y))
        return {"inner": inner}


class TweedieRegressor(_Glm):
    _kind = "TweedieRegressor"

    def __init__(self, power=0.0, alpha=0.0, link="auto", fit_intercept=True, max_iter=100,
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

    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def _rust_params(self, X, y):
        return {"n_neighbors": self.n_neighbors, "contamination": self.contamination}

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


class SimpleImputer(TransformerMixin, BaseEstimator):
    _kind = "SimpleImputer"

    def __init__(self, strategy="mean", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value

    def _rust_params(self, X, y):
        return {"strategy": _choice(self.strategy, {"mean": "Mean", "median": "Median",
                                                    "most_frequent": "MostFrequent",
                                                    "constant": "Constant"}, "strategy"),
                "fill_value": None if self.fill_value is None else float(self.fill_value)}


class PolynomialFeatures(TransformerMixin, BaseEstimator):
    _kind = "PolynomialFeatures"

    def __init__(self, degree=2, interaction_only=False):
        self.degree = degree
        self.interaction_only = interaction_only

    def _rust_params(self, X, y):
        return {"degree": self.degree, "interaction_only": self.interaction_only}


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

    def __init__(self, n_bins=5, encode="ordinal", strategy="quantile"):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

    def _rust_params(self, X, y):
        return {"n_bins": self.n_bins,
                "encode": _choice(self.encode, {"ordinal": "Ordinal", "onehot-dense": "Onehot"},
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
        return {"n_components": n, "kernel": kernel, "remove_zero_eig": self.remove_zero_eig}


class TruncatedSVD(TransformerMixin, BaseEstimator):
    _kind = "TruncatedSvd"

    def __init__(self, n_components=2):
        self.n_components = n_components

    def _rust_params(self, X, y):
        return {"n_components": self.n_components}


class FastICA(TransformerMixin, BaseEstimator):
    _kind = "FastIca"

    def __init__(self, n_components=None, max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _rust_params(self, X, y):
        n = X.shape[1] if self.n_components is None else int(self.n_components)
        return {"n_components": n, "max_iter": self.max_iter, "tol": self.tol,
                "seed": _seed(self.random_state)}


class NMF(TransformerMixin, BaseEstimator):
    _kind = "Nmf"

    def __init__(self, n_components=None, solver="cd", beta_loss="frobenius", tol=1e-4,
                 max_iter=200, random_state=None, alpha_W=0.0, alpha_H="same", l1_ratio=0.0):
        self.n_components = n_components
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

    def __init__(self, n_clusters=8, max_iter=100, batch_size=1024, tol=1e-4,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_clusters": self.n_clusters, "max_iter": self.max_iter,
                "batch_size": self.batch_size, "tol": self.tol,
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

    def __init__(self, min_samples=5, max_eps=np.inf, eps=None):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.eps = eps

    def _rust_params(self, X, y):
        return {"min_samples": self.min_samples,
                "eps_max": np.finfo(np.float64).max if np.isinf(self.max_eps)
                else float(self.max_eps),
                # f64::INFINITY does not survive JSON; use the largest float.
                "eps": np.finfo(np.float64).max if self.eps is None else float(self.eps)}


class MeanShift(ClusterMixin, BaseEstimator):
    _kind = "MeanShift"

    def __init__(self, bandwidth=None, max_iter=300):
        self.bandwidth = bandwidth
        self.max_iter = max_iter

    def _rust_params(self, X, y):
        bw = self.bandwidth
        if bw is None:  # median nearest-neighbour-ish heuristic
            d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)) if len(X) <= 2000 \
                else None
            bw = float(np.quantile(d[np.triu_indices(len(X), 1)], 0.3)) if d is not None \
                else float(X.std())
        return {"bandwidth": float(bw), "max_iter": self.max_iter}


class Birch(ClusterMixin, BaseEstimator):
    _kind = "Birch"

    def __init__(self, threshold=0.5, n_clusters=3):
        self.threshold = threshold
        self.n_clusters = n_clusters

    def _rust_params(self, X, y):
        return {"threshold": float(self.threshold), "n_clusters": self.n_clusters}


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


class MDS(_Embedding):
    """Classical (metric, Torgerson) multidimensional scaling."""

    _kind = "ClassicalMds"

    def __init__(self, n_components=2, *, metric="euclidean"):
        self.n_components = n_components
        self.metric = metric

    def _rust_params(self, X, y):
        return {"n_components": self.n_components,
                "metric": _choice(self.metric, _METRIC, "metric")}
