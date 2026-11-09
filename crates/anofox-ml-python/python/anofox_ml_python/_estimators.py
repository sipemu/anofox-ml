"""sklearn-compatible estimators backed by the Rust registry.

Parameter names and defaults follow scikit-learn where the Rust estimator
supports them; ``random_state`` replaces the Rust ``seed``.
"""

from __future__ import annotations

import json

import numpy as np

from . import _native
from ._base import (
    BaseEstimator,
    ClassifierMixin,
    ClassifierPartialFitMixin,
    ClusterMixin,
    OutlierMixin,
    PartialFitMixin,
    RegressorMixin,
    StagedPredictMixin,
    SupervisedTransformerMixin,
    TransformerMixin,
    WarmStartMixin,
    _check_1d,
    _check_X,
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


class _GradientBoosting(WarmStartMixin, StagedPredictMixin, BaseEstimator):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2,
                 min_samples_leaf=1, subsample=1.0, random_state=None, warm_start=False):
        self.warm_start = warm_start
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
    """Bagging of decision trees (``estimator=None``, which also takes
    ``max_depth``) or of any of this package's estimators of the same type.
    ``max_samples`` / ``max_features`` are fractions or counts."""

    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, oob_score=False, max_depth=None,
                 random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.max_depth = max_depth
        self.random_state = random_state

    @property
    def _kind(self):
        return self._tree_kind if self.estimator is None else self._estimator_kind

    def _rust_params(self, X, y):
        ms = self.max_samples
        if isinstance(ms, (int, np.integer)) and not isinstance(ms, bool):
            ms = ms / X.shape[0]
        ms = None if ms == 1.0 else float(ms)
        mf = None if self.max_features == 1.0 else _max_features(self.max_features)
        p = {"n_estimators": self.n_estimators, "max_samples": ms, "max_features": mf,
             "bootstrap": self.bootstrap, "bootstrap_features": self.bootstrap_features,
             "oob_score": self.oob_score, "seed": _seed(self.random_state)}
        if self.estimator is None:
            p["max_depth"] = self.max_depth
            return p
        if self.max_depth is not None:
            raise ValueError("max_depth applies only to the default tree estimator")
        from .meta import _spec

        return {**p, **_spec(self.estimator, X, y)}


class BaggingClassifier(ClassifierMixin, _Bagging):
    _tree_kind = "BaggingClassifier"
    _estimator_kind = "BaggingEstimatorClassifier"


class BaggingRegressor(RegressorMixin, _Bagging):
    _tree_kind = "BaggingRegressor"
    _estimator_kind = "BaggingEstimatorRegressor"


def _categorical_indices(categorical_features, n_features):
    """sklearn's ``categorical_features`` (indices or a boolean mask; ``None`` /
    ``"from_dtype"`` select none for arrays) as column indices."""
    if categorical_features is None or (isinstance(categorical_features, str)
                                        and categorical_features == "from_dtype"):
        return []
    cf = np.asarray(categorical_features)
    if cf.dtype.kind == "b":
        if cf.shape != (n_features,):
            raise ValueError("categorical_features set as a boolean mask must have "
                             f"shape (n_features,), got: {cf.shape}")
        return [int(i) for i in np.flatnonzero(cf)]
    if cf.dtype.kind not in "iu":
        raise ValueError("categorical_features must be column indices or a boolean mask")
    if cf.size and (cf.min() < 0 or cf.max() >= n_features):
        raise ValueError("categorical_features set as integer indices must be in "
                         "[0, n_features - 1]")
    return [int(i) for i in cf.ravel()]


def _interaction_cst(interaction_cst):
    """sklearn's ``interaction_cst`` as the Rust ``InteractionConstraints``."""
    if interaction_cst is None:
        return None
    if isinstance(interaction_cst, str):
        return _choice(interaction_cst, {"pairwise": "Pairwise",
                                         "no_interactions": "NoInteractions"},
                       "interaction_cst")
    try:
        return {"Groups": [[int(i) for i in group] for group in interaction_cst]}
    except TypeError:
        raise ValueError("Interaction constraints must be a sequence of tuples or lists, "
                         f"got: {interaction_cst!r}.") from None


def _early_stopping_params(est):
    """sklearn's early-stopping parameters of a HistGradientBoosting model."""
    scorings = {"loss": "Loss", "r2": "R2", "neg_mean_squared_error": "NegMeanSquaredError",
                "neg_mean_absolute_error": "NegMeanAbsoluteError", "accuracy": "Accuracy",
                "neg_log_loss": "NegLogLoss"}
    scoring = "Default" if est.scoring is None else _choice(est.scoring, scorings, "scoring")
    vf = est.validation_fraction
    if vf is None:
        validation = None
    elif isinstance(vf, (int, np.integer)) and not isinstance(vf, bool):
        validation = {"Count": int(vf)}
    else:
        validation = {"Fraction": float(vf)}
    rs = est.random_state
    if rs is not None and not isinstance(rs, (int, np.integer)):
        raise ValueError("random_state must be None or an integer")
    return {"early_stopping": _choice(est.early_stopping, {"auto": "Auto", True: "Enabled",
                                                           False: "Disabled"},
                                      "early_stopping"),
            "validation_fraction": validation,
            "n_iter_no_change": int(est.n_iter_no_change), "tol": float(est.tol),
            "random_state": 0 if rs is None else int(rs), "scoring": scoring}


class _HistGradientBoosting(StagedPredictMixin, BaseEstimator):
    def __init__(self, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None,
                 min_samples_leaf=20, l2_regularization=0.0, max_bins=255,
                 categorical_features=None, monotonic_cst=None, interaction_cst=None,
                 early_stopping="auto", scoring="loss", validation_fraction=0.1,
                 n_iter_no_change=10, tol=1e-7, random_state=None, max_features=1.0,
                 warm_start=False):
        self.warm_start = warm_start
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.categorical_features = categorical_features
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state
        self.max_features = max_features

    def _fit_model(self, X, y=None, sample_weight=None):
        model = self.__dict__.get("_model_")
        if not self.warm_start or model is None:
            return super()._fit_model(X, y, sample_weight)
        # sklearn's warm_start: more iterations on the fitted bins and trees.
        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} was fitted "
                f"with {self.n_features_in_}."
            )
        if sample_weight is not None:
            sample_weight = _check_1d(sample_weight, "sample_weight")
        model.set_params_json(json.dumps(self._rust_params(X, y)))
        model.warm_start(X, y, sample_weight)
        return self

    def _fit_validation(self, X, y, sample_weight, X_val, y_val, sample_weight_val):
        """sklearn's ``fit(..., X_val=, y_val=, sample_weight_val=)``."""
        X, X_val = _check_X(X), _check_X(X_val)
        if X_val.shape[1] != X.shape[1]:
            raise ValueError("X_val must have the features of X")
        w = None if sample_weight is None else _check_1d(sample_weight, "sample_weight")
        wv = (None if sample_weight_val is None
              else _check_1d(sample_weight_val, "sample_weight_val"))
        model = self._new_model(X, y)
        model.fit_with_validation(X, _check_1d(y), w, X_val, _check_1d(y_val, "y_val"), wv)
        self._model_ = model
        self.n_features_in_ = X.shape[1]
        return self

    @property
    def n_iter_(self):
        inner = self._fitted_state()["inner"]
        return len(inner["trees"]) if "trees" in inner else len(inner["tree_sets"][0])

    @property
    def train_score_(self):
        return np.asarray(self._fitted_state()["scores"]["train"], dtype=float)

    @property
    def validation_score_(self):
        return np.asarray(self._fitted_state()["scores"]["validation"], dtype=float)

    def _rust_params(self, X, y):
        return {**_early_stopping_params(self), "max_features": float(self.max_features),"learning_rate": self.learning_rate, "n_estimators": self.max_iter,
                "max_leaf_nodes": self.max_leaf_nodes, "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "l2_regularization": self.l2_regularization, "max_bins": self.max_bins,
                "categorical_features": _categorical_indices(self.categorical_features,
                                                             np.shape(X)[1]),
                "monotonic_cst": ([] if self.monotonic_cst is None
                                  else [int(c) for c in np.ravel(self.monotonic_cst)]),
                "interaction_cst": _interaction_cst(self.interaction_cst)}


class HistGradientBoostingClassifier(ClassifierMixin, _HistGradientBoosting):
    _kind = "HistGradientBoostingClassifier"

    def __init__(self, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None,
                 min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, max_bins=255,
                 categorical_features=None, monotonic_cst=None, interaction_cst=None,
                 early_stopping="auto", scoring="loss", validation_fraction=0.1,
                 n_iter_no_change=10, tol=1e-7, random_state=None, class_weight=None, warm_start=False):
        super().__init__(learning_rate=learning_rate, max_iter=max_iter,
                         max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf,
                         l2_regularization=l2_regularization, max_bins=max_bins,
                         categorical_features=categorical_features,
                         monotonic_cst=monotonic_cst, interaction_cst=interaction_cst,
                         early_stopping=early_stopping, scoring=scoring,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change, tol=tol,
                         random_state=random_state, max_features=max_features,
                         warm_start=warm_start)
        self.class_weight = class_weight

    def _rust_params(self, X, y):
        params = super()._rust_params(X, y)
        params["class_weight"] = self._encode_class_weight(self.class_weight)
        return params

    def fit(self, X, y, sample_weight=None, *, X_val=None, y_val=None,
            sample_weight_val=None):
        if X_val is None and y_val is None:
            return super().fit(X, y, sample_weight)
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be given together")
        y, y_val = np.ravel(np.asarray(y)), np.ravel(np.asarray(y_val))
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        idx = np.searchsorted(self.classes_, y_val)
        if np.any(idx >= len(self.classes_)) or np.any(
                self.classes_[np.minimum(idx, len(self.classes_) - 1)] != y_val):
            raise ValueError("y_val has classes that are not in y")
        return self._fit_validation(X, y_enc.astype(np.float64), sample_weight, X_val,
                                    idx.astype(np.float64), sample_weight_val)


class HistGradientBoostingRegressor(RegressorMixin, _HistGradientBoosting):
    _kind = "HistGradientBoostingRegressor"

    _LOSSES = {"squared_error": "SquaredError", "absolute_error": "AbsoluteError",
               "quantile": "Quantile", "poisson": "Poisson", "gamma": "Gamma"}

    def __init__(self, loss="squared_error", quantile=None, learning_rate=0.1, max_iter=100,
                 max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0,
                 max_bins=255, categorical_features=None, monotonic_cst=None,
                 interaction_cst=None, early_stopping="auto", scoring="loss",
                 validation_fraction=0.1, n_iter_no_change=10, tol=1e-7, random_state=None,
                 max_features=1.0, warm_start=False):
        super().__init__(learning_rate=learning_rate, max_iter=max_iter,
                         max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf,
                         l2_regularization=l2_regularization, max_bins=max_bins,
                         categorical_features=categorical_features,
                         monotonic_cst=monotonic_cst, interaction_cst=interaction_cst,
                         early_stopping=early_stopping, scoring=scoring,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change, tol=tol,
                         random_state=random_state, max_features=max_features,
                         warm_start=warm_start)
        self.loss = loss
        self.quantile = quantile

    def _rust_params(self, X, y):
        params = super()._rust_params(X, y)
        params["loss"] = _choice(self.loss, self._LOSSES, "loss")
        params["quantile"] = None if self.quantile is None else float(self.quantile)
        return params

    def fit(self, X, y, sample_weight=None, *, X_val=None, y_val=None,
            sample_weight_val=None):
        if X_val is None and y_val is None:
            return super().fit(X, y, sample_weight)
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be given together")
        return self._fit_validation(X, y, sample_weight, X_val, y_val, sample_weight_val)


class _Lgbm(StagedPredictMixin, BaseEstimator):
    """LightGBM-style boosting; parameters follow ``lightgbm.LGBMModel``."""

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1,
                 n_estimators=100, min_split_gain=0.0, min_child_weight=1e-3,
                 min_child_samples=20, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                 max_bin=255, random_state=None, drop_rate=0.1, max_drop=50, skip_drop=0.5,
                 uniform_drop=False, xgboost_dart_mode=False, drop_seed=4, enable_bundle=True,
                 n_jobs=None, subsample=1.0, subsample_freq=0, feature_fraction_bynode=1.0,
                 top_rate=0.2, other_rate=0.1):
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.feature_fraction_bynode = feature_fraction_bynode
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.boosting_type = boosting_type
        self.enable_bundle = enable_bundle
        self.n_jobs = n_jobs
        self.drop_rate = drop_rate
        self.max_drop = max_drop
        self.skip_drop = skip_drop
        self.uniform_drop = uniform_drop
        self.xgboost_dart_mode = xgboost_dart_mode
        self.drop_seed = drop_seed
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
        return {"boosting_type": _choice(self.boosting_type,
                                         {"gbdt": "Gbdt", "goss": "Goss", "dart": "Dart"},
                                         "boosting_type"),
                "dart": {"drop_rate": float(self.drop_rate), "max_drop": int(self.max_drop),
                         "skip_drop": float(self.skip_drop),
                         "uniform_drop": bool(self.uniform_drop),
                         "xgboost_dart_mode": bool(self.xgboost_dart_mode),
                         "drop_seed": int(self.drop_seed)},
                "num_leaves": self.num_leaves,
                "max_depth": None if self.max_depth is None or self.max_depth < 0
                else self.max_depth,
                "learning_rate": self.learning_rate, "n_estimators": self.n_estimators,
                "min_split_gain": self.min_split_gain,
                "min_child_weight": self.min_child_weight,
                "min_child_samples": self.min_child_samples,
                "colsample_bytree": self.colsample_bytree, "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda, "max_bins": self.max_bin,
                "enable_bundle": bool(self.enable_bundle),
                # LightGBM's n_jobs: None or <= 0 is the default (physical cores).
                "n_jobs": None if self.n_jobs is None or self.n_jobs <= 0 else int(self.n_jobs),
                "subsample": float(self.subsample), "subsample_freq": int(self.subsample_freq),
                "colsample_bynode": float(self.feature_fraction_bynode),
                "goss_top_rate": float(self.top_rate), "goss_other_rate": float(self.other_rate),
                # LightGBM's own seeds without random_state (as LightGBM).
                "seed": None if self.random_state is None else _seed(self.random_state)}


class LGBMClassifier(ClassifierMixin, _Lgbm):
    _kind = "LgbmClassifier"

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1,
                 n_estimators=100, min_split_gain=0.0, min_child_weight=1e-3,
                 min_child_samples=20, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                 max_bin=255, class_weight=None, random_state=None, drop_rate=0.1, max_drop=50,
                 skip_drop=0.5, uniform_drop=False, xgboost_dart_mode=False, drop_seed=4,
                 enable_bundle=True, n_jobs=None, subsample=1.0, subsample_freq=0,
                 feature_fraction_bynode=1.0, top_rate=0.2, other_rate=0.1,
                 pos_bagging_fraction=1.0, neg_bagging_fraction=1.0):
        super().__init__(boosting_type, num_leaves, max_depth, learning_rate, n_estimators,
                         min_split_gain, min_child_weight, min_child_samples,
                         colsample_bytree, reg_alpha, reg_lambda, max_bin, random_state,
                         drop_rate, max_drop, skip_drop, uniform_drop, xgboost_dart_mode,
                         drop_seed, enable_bundle, n_jobs, subsample, subsample_freq,
                         feature_fraction_bynode, top_rate, other_rate)
        self.class_weight = class_weight
        self.pos_bagging_fraction = pos_bagging_fraction
        self.neg_bagging_fraction = neg_bagging_fraction

    def _rust_params(self, X, y):
        p = super()._rust_params(X, y)
        p["class_weight"] = self._encode_class_weight(self.class_weight)
        p["pos_bagging_fraction"] = float(self.pos_bagging_fraction)
        p["neg_bagging_fraction"] = float(self.neg_bagging_fraction)
        return p


class LGBMRegressor(RegressorMixin, _Lgbm):
    _kind = "LgbmRegressor"


class _CatBoost(BaseEstimator):
    """CatBoost-style boosting on oblivious trees; parameters follow
    ``catboost.CatBoost`` (``learning_rate=None`` is CatBoost's automatic one)."""

    def __init__(self, iterations=1000, learning_rate=None, depth=6, l2_leaf_reg=3.0,
                 border_count=254, score_function="Cosine", leaf_estimation_method=None,
                 boost_from_average=None, cat_features=None, one_hot_max_size=2,
                 model_size_reg=0.5, boosting_type="Plain", fold_len_multiplier=2.0,
                 max_ctr_complexity=None, leaf_estimation_iterations=None,
                 leaf_estimation_backtracking="AnyImprovement", has_time=False, random_seed=0,
                 random_strength=1.0, bootstrap_type=None, subsample=None,
                 bagging_temperature=1.0, mvs_reg=None, monotone_constraints=None,
                 model_shrink_rate=None, model_shrink_mode="Constant", class_weights=None,
                 auto_class_weights=None, loss_function=None, use_best_model=None,
                 early_stopping_rounds=None, od_type=None, od_wait=None, od_pval=None,
                 grow_policy="SymmetricTree", min_data_in_leaf=1, max_leaves=31,
                 thread_count=-1):
        self.thread_count = thread_count
        self.grow_policy = grow_policy
        self.min_data_in_leaf = min_data_in_leaf
        self.max_leaves = max_leaves
        self.use_best_model = use_best_model
        self.early_stopping_rounds = early_stopping_rounds
        self.od_type = od_type
        self.od_wait = od_wait
        self.od_pval = od_pval
        self.loss_function = loss_function
        self.monotone_constraints = monotone_constraints
        self.model_shrink_rate = model_shrink_rate
        self.model_shrink_mode = model_shrink_mode
        self.class_weights = class_weights
        self.auto_class_weights = auto_class_weights
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.mvs_reg = mvs_reg
        self.random_strength = random_strength
        self.has_time = has_time
        self.random_seed = random_seed
        self.max_ctr_complexity = max_ctr_complexity
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.leaf_estimation_backtracking = leaf_estimation_backtracking
        self.boosting_type = boosting_type
        self.fold_len_multiplier = fold_len_multiplier
        self.cat_features = cat_features
        self.one_hot_max_size = one_hot_max_size
        self.model_size_reg = model_size_reg
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.score_function = score_function
        self.leaf_estimation_method = leaf_estimation_method
        self.boost_from_average = boost_from_average

    def _rust_params(self, X, y):
        params = {
            "iterations": int(self.iterations),
            "depth": int(self.depth), "l2_leaf_reg": float(self.l2_leaf_reg),
            "border_count": int(self.border_count),
            "score_function": _choice(self.score_function, {"Cosine": "Cosine", "L2": "L2"},
                                      "score_function"),
        }
        if self.leaf_estimation_method is not None:
            params["leaf_estimation_method"] = _choice(
                self.leaf_estimation_method,
                {"Newton": "Newton", "Gradient": "Gradient", "Exact": "Exact"},
                "leaf_estimation_method")
        if self.boost_from_average is not None:
            params["boost_from_average"] = bool(self.boost_from_average)
        params["cat_features"] = self._cat_indices(np.shape(X)[1])
        params["one_hot_max_size"] = int(self.one_hot_max_size)
        params["model_size_reg"] = float(self.model_size_reg)
        params["boosting_type"] = _choice(self.boosting_type,
                                          {"Plain": "Plain", "Ordered": "Ordered"},
                                          "boosting_type")
        params["fold_len_multiplier"] = float(self.fold_len_multiplier)
        if self.max_ctr_complexity is not None:
            params["max_ctr_complexity"] = int(self.max_ctr_complexity)
        if self.learning_rate is not None:
            params["learning_rate"] = float(self.learning_rate)
        params["has_time"] = bool(self.has_time)
        params["random_seed"] = int(self.random_seed)
        params["random_strength"] = float(self.random_strength)
        if self.bootstrap_type is not None:
            params["bootstrap_type"] = _choice(
                self.bootstrap_type,
                {"No": "No", "Bayesian": "Bayesian", "Bernoulli": "Bernoulli", "MVS": "MVS"},
                "bootstrap_type")
        if self.subsample is not None:
            params["subsample"] = float(self.subsample)
        params["bagging_temperature"] = float(self.bagging_temperature)
        if self.mvs_reg is not None:
            params["mvs_reg"] = float(self.mvs_reg)
        if self.leaf_estimation_iterations is not None:
            params["leaf_estimation_iterations"] = int(self.leaf_estimation_iterations)
        params["leaf_estimation_backtracking"] = _choice(
            self.leaf_estimation_backtracking, {"AnyImprovement": "AnyImprovement", "No": "No"},
            "leaf_estimation_backtracking")
        n_features = np.shape(X)[1]
        params["grow_policy"] = _choice(
            self.grow_policy, {"SymmetricTree": "SymmetricTree", "Depthwise": "Depthwise",
                               "Lossguide": "Lossguide"}, "grow_policy")
        params["min_data_in_leaf"] = int(self.min_data_in_leaf)
        params["max_leaves"] = int(self.max_leaves)
        # catboost's -1 (all cores): the physical ones here; the model does not
        # depend on it.
        params["thread_count"] = (None if self.thread_count is None or self.thread_count <= 0
                                  else int(self.thread_count))
        if self.use_best_model is not None:
            params["use_best_model"] = bool(self.use_best_model)
        if self.early_stopping_rounds is not None:
            params["od_type"] = "Iter"
            params["od_wait"] = int(self.early_stopping_rounds)
        if self.od_type is not None:
            params["od_type"] = _choice(self.od_type, {"IncToDec": "IncToDec", "Iter": "Iter"},
                                        "od_type")
        if self.od_wait is not None:
            params["od_wait"] = int(self.od_wait)
        if self.od_pval is not None:
            params["od_pval"] = float(self.od_pval)
        if self.monotone_constraints is not None:
            mc = self.monotone_constraints
            if isinstance(mc, dict):
                full = [0] * n_features
                for j, v in mc.items():
                    full[int(j)] = int(v)
                mc = full
            params["monotone_constraints"] = [int(v) for v in mc]
        if self.model_shrink_rate is not None:
            params["model_shrink_rate"] = float(self.model_shrink_rate)
        params["model_shrink_mode"] = _choice(
            self.model_shrink_mode, {"Constant": "Constant", "Decreasing": "Decreasing"},
            "model_shrink_mode")
        if self.class_weights is not None or self.auto_class_weights is not None:
            if self._kind != "CatBoostClassifier":
                raise ValueError("class weights apply to CatBoostClassifier only")
            if self.class_weights is not None:
                cw = self.class_weights
                if isinstance(cw, dict):
                    cw = [cw[c] for c in self.classes_.tolist()]
                params["class_weights"] = [float(v) for v in cw]
            if self.auto_class_weights is not None:
                params["auto_class_weights"] = _choice(
                    self.auto_class_weights,
                    {"Balanced": "Balanced", "SqrtBalanced": "SqrtBalanced"},
                    "auto_class_weights")
        return params

    def _cat_indices(self, n_features):
        if self.cat_features is None:
            return []
        idx = sorted({int(i) % n_features for i in self.cat_features})
        return idx

    def _encode(self, X, fit):
        """Numeric array with the categorical columns as category codes (any
        hashable values; unseen ones get their own code)."""
        X = np.asarray(X, dtype=object if self.cat_features else np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {X.ndim}D")
        cats = self._cat_indices(X.shape[1])
        if not cats:
            return np.ascontiguousarray(X, dtype=np.float64)
        if fit:
            self._categories_ = {}
        out = np.empty(X.shape, dtype=np.float64)
        for j in range(X.shape[1]):
            if j not in cats:
                out[:, j] = X[:, j].astype(np.float64)
                continue
            codes = self._categories_.setdefault(j, {}) if fit else self._categories_[j]
            col = out[:, j]
            for i, v in enumerate(X[:, j]):
                key = str(v)
                if fit:
                    col[i] = codes.setdefault(key, float(len(codes)))
                else:
                    col[i] = codes.get(key, -1.0)
        return out

    def _fit_model(self, X, y=None, sample_weight=None, eval_set=None):
        if eval_set is None:
            return super()._fit_model(self._encode(X, fit=True), y, sample_weight)
        X = self._encode(X, fit=True)
        Xv, yv, wv = (tuple(eval_set) + (None,))[:3]
        Xv = self._encode(Xv, fit=False)
        if Xv.shape[1] != X.shape[1]:
            raise ValueError("eval_set must have the features of X")
        w = None if sample_weight is None else _check_1d(sample_weight, "sample_weight")
        wv = None if wv is None else _check_1d(wv, "eval_set weight")
        model = self._new_model(X, y)
        model.fit_with_validation(X, y, w, Xv, self._eval_target(yv), wv)
        self._model_ = model
        self.n_features_in_ = X.shape[1]
        return self

    def _eval_target(self, yv):
        return _check_1d(yv, "eval_set y")

    @property
    def best_iteration_(self):
        return self._fitted_state().get("best_iteration")

    @property
    def evals_result_(self):
        return list(self._fitted_state().get("eval_metrics", []))

    def _call(self, method, X):
        return super()._call(method, self._encode(X, fit=False))


class CatBoostRegressor(RegressorMixin, _CatBoost):
    """``loss_function`` as in catboost: ``"RMSE"``, ``"MAE"``,
    ``"Quantile:alpha=0.3"``, ``"Expectile:alpha=0.3"``, ``"Poisson"``,
    ``"Tweedie:variance_power=1.5"`` or ``"LogLinQuantile:alpha=0.5"``."""

    _kind = "CatBoostRegressor"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        """``eval_set``: ``(X_val, y_val)`` or ``(X_val, y_val, weight_val)``."""
        return self._fit_model(X, _check_1d(y), sample_weight, eval_set)

    def _rust_params(self, X, y):
        params = super()._rust_params(X, y)
        name, _, rest = str(self.loss_function or "RMSE").partition(":")
        args = dict(kv.split("=", 1) for kv in rest.split(";") if kv)
        param = {"Quantile": "alpha", "Expectile": "alpha", "LogLinQuantile": "alpha",
                 "Tweedie": "variance_power"}
        if name in ("RMSE", "MAE", "Poisson"):
            if args:
                raise ValueError(f"{name} takes no parameters")
            params["loss_function"] = name
        elif name in param:
            key = param[name]
            if key not in args and name != "Tweedie":
                args[key] = "0.5"
            if set(args) != {key}:
                raise ValueError(f"{name} takes the parameter {key}")
            params["loss_function"] = {name: {key: float(args[key])}}
        else:
            raise ValueError(f"unsupported loss_function {self.loss_function!r}")
        return params


class CatBoostClassifier(ClassifierMixin, _CatBoost):
    """Logloss for two classes, MultiClass for more (``loss_function`` may
    name them)."""

    _kind = "CatBoostClassifier"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        """``eval_set``: ``(X_val, y_val)`` or ``(X_val, y_val, weight_val)``."""
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        return self._fit_model(X, y_enc.astype(np.float64), sample_weight, eval_set)

    def _eval_target(self, yv):
        yv = np.asarray(yv).ravel()
        idx = np.searchsorted(self.classes_, yv)
        if np.any(idx >= len(self.classes_)) or np.any(self.classes_[np.minimum(idx, len(self.classes_) - 1)] != yv):
            raise ValueError("eval_set has labels that are not in y")
        return idx.astype(np.float64)

    def _rust_params(self, X, y):
        if self.loss_function not in (None, "Logloss", "MultiClass"):
            raise ValueError(f"unsupported loss_function {self.loss_function!r}")
        return super()._rust_params(X, y)


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


class SGDClassifier(ClassifierPartialFitMixin, ClassifierMixin, BaseEstimator):
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


class SGDRegressor(PartialFitMixin, RegressorMixin, BaseEstimator):
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


class PassiveAggressiveRegressor(PartialFitMixin, RegressorMixin, BaseEstimator):
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
    """``coef_`` / ``intercept_`` from the fitted Rust state (nested under
    ``"fit"`` for the GLMs)."""

    def _linear_state(self):
        state = self._fitted_state()
        return state["fit"] if "coef" not in state and "fit" in state else state

    @property
    def coef_(self):
        return _nd(self._linear_state()["coef"])

    @property
    def intercept_(self):
        b = self._linear_state()["intercept"]
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


class _OffsetMixin:
    """``offset=`` on the link scale for ``fit`` and ``predict`` (e.g.
    ``log(exposure)`` for a log link); ``predict`` defaults to offset 0."""

    def fit(self, X, y, sample_weight=None, offset=None):
        if offset is None:
            return super().fit(X, y, sample_weight)
        X = _check_X(X)
        y = _check_1d(y)
        if sample_weight is not None:
            sample_weight = _check_1d(sample_weight, "sample_weight")
        model = self._new_model(X, y)
        model.fit_with_offset(X, y, _check_1d(offset, "offset"), sample_weight)
        self._model_ = model
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X, offset=None):
        if offset is None:
            return super().predict(X)
        X = self._checked(X)
        return self._model().predict_with_offset(X, _check_1d(offset, "offset"))


class _Glm(_OffsetMixin, _LinearAttrs, RegressorMixin, BaseEstimator):
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


class GaussianNB(ClassifierPartialFitMixin, ClassifierMixin, BaseEstimator):
    _kind = "GaussianNB"

    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def _rust_params(self, X, y):
        return {"priors": None if self.priors is None else [float(p) for p in self.priors],
                "var_smoothing": self.var_smoothing}


class _DiscreteNB(ClassifierPartialFitMixin, ClassifierMixin, BaseEstimator):
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


class _KNeighborsMixin:
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Distances to and indices of the ``n_neighbors`` nearest training
        rows of each row of ``X``, nearest first."""
        X = self._checked(X)
        k = None if n_neighbors is None else int(n_neighbors)
        dist, idx = self._model().kneighbors(X, k)
        return (dist, idx) if return_distance else idx


class KNeighborsClassifier(_KNeighborsMixin, ClassifierMixin, BaseEstimator):
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


class KNeighborsRegressor(_KNeighborsMixin, RegressorMixin, BaseEstimator):
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


class _Mlp(WarmStartMixin, BaseEstimator):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver="adam",
                 alpha=1e-4, batch_size="auto", learning_rate="constant",
                 learning_rate_init=1e-3, power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000, warm_start=False):
        self.warm_start = warm_start
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


class MLPClassifier(ClassifierPartialFitMixin, ClassifierMixin, _Mlp):
    _kind = "MlpClassifier"


class MLPRegressor(PartialFitMixin, RegressorMixin, _Mlp):
    _kind = "MlpRegressor"


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


class StandardScaler(PartialFitMixin, TransformerMixin, BaseEstimator):
    _kind = "StandardScaler"

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def _rust_params(self, X, y):
        return {"with_mean": self.with_mean, "with_std": self.with_std}


class MinMaxScaler(PartialFitMixin, TransformerMixin, BaseEstimator):
    _kind = "MinMaxScaler"

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def _rust_params(self, X, y):
        lo, hi = self.feature_range
        return {"feature_min": float(lo), "feature_max": float(hi)}


class MaxAbsScaler(PartialFitMixin, TransformerMixin, BaseEstimator):
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


class IncrementalPCA(PartialFitMixin, TransformerMixin, BaseEstimator):
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


# ---------------------------------------------------------------------------
# cross-validated linear models (#113)
# ---------------------------------------------------------------------------


class LogisticRegressionCV(ClassifierMixin, BaseEstimator):
    """Logistic regression with ``C`` chosen by unshuffled stratified
    ``cv``-fold CV over ``Cs`` (an int: ``logspace(-4, 4, Cs)``), scored by
    accuracy (``scoring=None``) or ``"neg_log_loss"``, refit on all rows."""

    _kind = "LogisticRegressionCV"

    def __init__(self, *, Cs=10, cv=5, scoring=None, l1_ratio=0.0, fit_intercept=True,
                 tol=1e-8, max_iter=1000, class_weight=None):
        self.Cs = Cs
        self.cv = cv
        self.scoring = scoring
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight

    def _rust_params(self, X, y):
        if isinstance(self.Cs, (int, np.integer)):
            cs = {"Count": int(self.Cs)}
        else:
            cs = {"Values": [float(c) for c in self.Cs]}
        return {"l1_ratio": float(self.l1_ratio), "with_intercept": self.fit_intercept,
                "tol": self.tol, "max_iter": self.max_iter,
                "class_weight": self._encode_class_weight(self.class_weight),
                "cs": cs, "cv": int(self.cv),
                "scoring": _choice(self.scoring, {None: "Accuracy", "accuracy": "Accuracy",
                                                  "neg_log_loss": "NegLogLoss"}, "scoring")}

    @property
    def C_(self):
        return float(self._fitted_state()["c"])

    @property
    def Cs_(self):
        return np.asarray(self._fitted_state()["cs"], dtype=float)

    @property
    def scores_(self):
        return _nd(self._fitted_state()["scores"])


class OrthogonalMatchingPursuitCV(RegressorMixin, BaseEstimator):
    """OMP with the number of non-zero coefficients chosen by unshuffled
    ``cv``-fold CV."""

    _kind = "OrthogonalMatchingPursuitCV"

    def __init__(self, *, max_iter=None, cv=5, fit_intercept=True):
        self.max_iter = max_iter
        self.cv = cv
        self.fit_intercept = fit_intercept

    def _rust_params(self, X, y):
        return {"max_iter": None if self.max_iter is None else int(self.max_iter),
                "cv": int(self.cv), "fit_intercept": bool(self.fit_intercept)}

    @property
    def n_nonzero_coefs_(self):
        return int(self._fitted_state()["n_nonzero_coefs"])

    @property
    def coef_(self):
        return _nd(self._fitted_state()["model"]["coef"])

    @property
    def intercept_(self):
        return float(self._fitted_state()["model"]["intercept"])


# ---------------------------------------------------------------------------
# semi-supervised (#113)
# ---------------------------------------------------------------------------


class _LabelGraph(ClassifierMixin, BaseEstimator):
    """Unlabelled rows have the label ``-1`` (sklearn's convention)."""

    def fit(self, X, y):
        y = np.asarray(y)
        labelled = y != -1
        self.classes_ = np.unique(y[labelled])
        y_enc = np.full(len(y), -1.0)
        y_enc[labelled] = np.searchsorted(self.classes_, y[labelled])
        return self._fit_model(X, y_enc, None)

    def _kernel(self):
        if self.kernel == "rbf":
            return {"Rbf": {"gamma": float(self.gamma)}}
        if self.kernel == "knn":
            return {"Knn": {"n_neighbors": int(self.n_neighbors)}}
        raise ValueError(f"kernel must be 'rbf' or 'knn', got {self.kernel!r}")

    @property
    def label_distributions_(self):
        return _nd(self._fitted_state()["label_distributions"])

    @property
    def transduction_(self):
        idx = np.asarray(_nd(self._fitted_state()["transduction"]), dtype=np.int64)
        return self.classes_[idx]

    @property
    def n_iter_(self):
        return int(self._fitted_state()["n_iter"])


class LabelPropagation(_LabelGraph):
    _kind = "LabelPropagation"

    def __init__(self, kernel="rbf", *, gamma=20, n_neighbors=7, max_iter=1000, tol=1e-3):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol

    def _rust_params(self, X, y):
        return {"kernel": self._kernel(), "max_iter": int(self.max_iter), "tol": float(self.tol)}


class LabelSpreading(_LabelGraph):
    _kind = "LabelSpreading"

    def __init__(self, kernel="rbf", *, gamma=20, n_neighbors=7, alpha=0.2, max_iter=30, tol=1e-3):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _rust_params(self, X, y):
        return {"kernel": self._kernel(), "alpha": float(self.alpha),
                "max_iter": int(self.max_iter), "tol": float(self.tol)}


class FeatureAgglomeration(TransformerMixin, BaseEstimator):
    """Cluster the features (agglomerative) and pool each cluster by its mean."""

    _kind = "FeatureAgglomeration"

    def __init__(self, n_clusters=2, *, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def _rust_params(self, X, y):
        return {"n_clusters": int(self.n_clusters), "linkage": _choice(
            self.linkage, {"ward": "Ward", "complete": "Complete", "average": "Average",
                           "single": "Single"}, "linkage")}

    @property
    def labels_(self):
        return np.asarray(self._fitted_state()["labels"], dtype=np.int64)

    def inverse_transform(self, X):
        # The input has one column per cluster, not the training width.
        from ._base import _check_X

        return self._model().inverse_transform(_check_X(X))


class KNeighborsTransformer(TransformerMixin, BaseEstimator):
    """k-nearest-neighbour graph of ``X`` against the training rows, as a
    dense array (sklearn returns it sparse)."""

    _kind = "KNeighborsTransformer"

    def __init__(self, *, mode="distance", n_neighbors=5):
        self.mode = mode
        self.n_neighbors = n_neighbors

    def _rust_params(self, X, y):
        return {"mode": _choice(self.mode, {"distance": "Distance",
                                            "connectivity": "Connectivity"}, "mode"),
                "n_neighbors": int(self.n_neighbors)}


# ---------------------------------------------------------------------------
# registry kinds: LARS-IC, GLMs, isotonic, Gaussian processes, PLS / CCA,
# multi-task linear models, spectral embedding, splines
# ---------------------------------------------------------------------------


def _check_2d_y(Y, n_rows) -> np.ndarray:
    """Targets as an ``n × k`` float array (a 1-D ``y`` is one column)."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError(f"y must be 1- or 2-dimensional, got {Y.ndim}D")
    if Y.shape[0] != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {Y.shape[0]}")
    return np.ascontiguousarray(Y)


class LassoLarsIC(RegressorMixin, BaseEstimator):
    """Lasso on the LARS path, with ``alpha`` chosen by AIC or BIC."""

    _kind = "LassoLarsIC"

    def __init__(self, criterion="aic", *, fit_intercept=True, max_iter=None,
                 noise_variance=None):
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.noise_variance = noise_variance

    def _rust_params(self, X, y):
        return {"criterion": _choice(self.criterion, {"aic": "Aic", "bic": "Bic"}, "criterion"),
                "fit_intercept": self.fit_intercept, "max_features": self.max_iter,
                "noise_variance": self.noise_variance}

    @property
    def coef_(self):
        return _nd(self._fitted_state()["fitted"]["coef"])

    @property
    def intercept_(self):
        return float(self._fitted_state()["fitted"]["intercept"])

    @property
    def alpha_(self):
        return float(self._fitted_state()["alpha"])

    @property
    def criterion_(self):
        return _nd(self._fitted_state()["criterion_path"])

    @property
    def noise_variance_(self):
        return float(self._fitted_state()["noise_variance"])


class BinomialRegressor(_OffsetMixin, _LinearAttrs, RegressorMixin, BaseEstimator):
    """Binomial GLM for proportions in ``[0, 1]`` (weights: trial counts);
    ``predict`` returns the mean. ``alpha`` is an L2 penalty."""

    _kind = "BinomialRegressor"

    def __init__(self, link="logit", *, alpha=0.0, fit_intercept=True, max_iter=100,
                 tol=1e-8):
        self.link = link
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def _rust_params(self, X, y):
        return {"link": _choice(self.link, {"logit": "Logit", "probit": "Probit",
                                            "cloglog": "Cloglog"}, "link"),
                "alpha": float(self.alpha), "with_intercept": self.fit_intercept,
                "max_iter": self.max_iter, "tol": self.tol}


class IsotonicRegression(RegressorMixin, BaseEstimator):
    """Monotone piecewise-linear fit of one feature (``X`` may be 1-D)."""

    _kind = "IsotonicRegressor"

    def __init__(self, *, increasing=True, out_of_bounds="nan"):
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def _rust_params(self, X, y):
        if self.increasing == "auto":
            raise ValueError("increasing='auto' is not supported; pass True or False")
        return {"increasing": bool(self.increasing),
                "out_of_bounds": _choice(self.out_of_bounds, {"nan": "Nan", "clip": "Clip",
                                                              "raise": "Raise"},
                                         "out_of_bounds")}

    @staticmethod
    def _column(X):
        X = np.asarray(X, dtype=np.float64)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def fit(self, X, y, sample_weight=None):
        return super().fit(self._column(X), y, sample_weight)

    def predict(self, T):
        return super().predict(self._column(T))

    def transform(self, T):
        return self.predict(T)

    @property
    def X_thresholds_(self):
        return np.asarray(self._fitted_state()["x_thresholds"], dtype=np.float64)

    @property
    def y_thresholds_(self):
        return np.asarray(self._fitted_state()["y_thresholds"], dtype=np.float64)


class GaussianProcessRegressor(RegressorMixin, BaseEstimator):
    """Gaussian process regression. ``kernel=None`` is sklearn's default,
    ``ConstantKernel(1.0, "fixed") * RBF(1.0, "fixed")``; other kernels are
    given as the Rust kernel spec (a dict, as in ``default_params``)."""

    _kind = "GaussianProcessRegressor"

    def __init__(self, kernel=None, *, alpha=1e-10, optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0, normalize_y=False, random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.random_state = random_state

    def _rust_params(self, X, y):
        p = {"alpha": float(self.alpha), "optimizer": self.optimizer is not None,
             "n_restarts_optimizer": self.n_restarts_optimizer,
             "normalize_y": self.normalize_y, "seed": _seed(self.random_state)}
        if self.kernel is not None:
            p["kernel"] = self.kernel
        return p

    def predict(self, X, return_std=False):
        mean = self._call("predict", X)
        if not return_std:
            return mean
        return mean, self._call("predict_std", X)

    @property
    def log_marginal_likelihood_value_(self):
        return float(self._fitted_state()["log_marginal_likelihood_value"])


class GaussianProcessClassifier(ClassifierMixin, BaseEstimator):
    """Gaussian process classification (Laplace approximation, one-vs-rest
    for more than two classes). ``kernel`` as in
    :class:`GaussianProcessRegressor`."""

    _kind = "GaussianProcessClassifier"

    def __init__(self, kernel=None, *, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 max_iter_predict=100, random_state=None):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.random_state = random_state

    def _rust_params(self, X, y):
        p = {"optimizer": self.optimizer is not None,
             "n_restarts_optimizer": self.n_restarts_optimizer,
             "max_iter": self.max_iter_predict, "seed": _seed(self.random_state)}
        if self.kernel is not None:
            p["kernel"] = self.kernel
        return p


class PLSRegression(RegressorMixin, BaseEstimator):
    """PLS regression (NIPALS) of one target."""

    _kind = "PlsRegression"

    def __init__(self, n_components=2, *, scale=True, max_iter=500, tol=1e-6):
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol

    def _rust_params(self, X, y):
        return {"n_components": self.n_components, "scale": self.scale,
                "max_iter": self.max_iter, "tol": self.tol}

    def transform(self, X):
        return self._call("transform", X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    @property
    def coef_(self):
        return _nd(self._fitted_state()["coef"]).reshape(1, -1)

    @property
    def intercept_(self):
        """sklearn's: the target mean (``predict`` centres ``X`` first)."""
        return np.array([float(self._fitted_state()["y_mean"])])

    def _attr(self, name):
        return _nd(self._fitted_state()[name])

    x_weights_ = property(lambda self: self._attr("x_weights"))
    x_loadings_ = property(lambda self: self._attr("x_loadings"))
    x_rotations_ = property(lambda self: self._attr("x_rotations"))
    x_scores_ = property(lambda self: self._attr("x_scores"))
    y_loadings_ = property(lambda self: self._attr("y_loadings").reshape(1, -1))


class _TwoBlock(TransformerMixin, BaseEstimator):
    """Fit on ``(X, Y)``; ``transform(X, Y=None)`` returns the X scores, or
    ``(x_scores, y_scores)`` with ``Y``."""

    def _rust_params(self, X, y):
        return {"n_components": self.n_components, "scale": self.scale,
                "y_columns": int(y.shape[1])}

    def fit(self, X, y):
        X = _check_X(X)
        Y = _check_2d_y(y, X.shape[0])
        model = self._new_model(X, Y)
        model.fit(np.hstack([X, Y]), None, None)
        self._model_ = model
        self.n_features_in_ = X.shape[1]
        self._n_targets = Y.shape[1]
        return self

    def transform(self, X, y=None):
        x_scores = self._call("transform", X)
        if y is None:
            return x_scores
        Y = _check_2d_y(y, x_scores.shape[0])
        if Y.shape[1] != self._n_targets:
            raise ValueError(f"y has {Y.shape[1]} columns, expected {self._n_targets}")
        return x_scores, self._model().transform_y(Y)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)

    def _attr(self, name):
        return _nd(self._fitted_state()[name])

    x_weights_ = property(lambda self: self._attr("x_weights"))
    y_weights_ = property(lambda self: self._attr("y_weights"))


class _NipalsTwoBlock(_TwoBlock):
    def __init__(self, n_components=2, *, scale=True, max_iter=500, tol=1e-6):
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol

    def _rust_params(self, X, y):
        return {**super()._rust_params(X, y), "max_iter": self.max_iter, "tol": self.tol}

    x_loadings_ = property(lambda self: self._attr("x_loadings"))
    y_loadings_ = property(lambda self: self._attr("y_loadings"))
    x_rotations_ = property(lambda self: self._attr("x_rotations"))
    y_rotations_ = property(lambda self: self._attr("y_rotations"))


class CCA(_NipalsTwoBlock):
    """Canonical correlation analysis (NIPALS)."""

    _kind = "Cca"


class PLSCanonical(_NipalsTwoBlock):
    """Canonical PLS (NIPALS, mode A)."""

    _kind = "PlsCanonical"


class PLSSVD(_TwoBlock):
    """PLS from the SVD of the cross-covariance ``XᵀY``."""

    _kind = "PlsSvd"

    def __init__(self, n_components=2, *, scale=True):
        self.n_components = n_components
        self.scale = scale


class _MultiTask(BaseEstimator):
    """Fit on ``(X, Y)`` with ``Y`` of shape ``n × k``; the ℓ2,1 penalty
    selects the features jointly for all targets."""

    _estimator_type = "regressor"

    def _rust_params(self, X, y):
        return {"alpha": float(self.alpha), "l1_ratio": float(self._l1_ratio()),
                "fit_intercept": self.fit_intercept, "max_iter": self.max_iter,
                "y_columns": int(y.shape[1])}

    def fit(self, X, y):
        X = _check_X(X)
        Y = _check_2d_y(y, X.shape[0])
        if Y.shape[1] < 1:
            raise ValueError("y needs at least one column")
        model = self._new_model(X, Y)
        model.fit(np.hstack([X, Y]), None, None)
        self._model_ = model
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self._call("predict", X)

    def score(self, X, y, sample_weight=None):
        """R² averaged uniformly over the targets (sklearn's default)."""
        Y = _check_2d_y(y, len(X))
        pred = self.predict(X).reshape(Y.shape)
        w = np.ones(len(Y)) if sample_weight is None else np.asarray(sample_weight, float)
        ss_res = (w[:, None] * (Y - pred) ** 2).sum(0)
        ss_tot = (w[:, None] * (Y - np.average(Y, axis=0, weights=w)) ** 2).sum(0)
        r2 = np.where(ss_tot == 0, np.where(ss_res == 0, 1.0, 0.0),
                      1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot))
        return float(r2.mean())

    @property
    def coef_(self):
        return np.asarray(self._model().coef())

    @property
    def intercept_(self):
        return _nd(self._fitted_state()["intercept"])


class MultiTaskLasso(_MultiTask):
    """Multi-task Lasso, solved by block coordinate descent to convergence
    (``max_iter`` caps the sweeps)."""

    _kind = "MultiTaskLasso"

    def __init__(self, alpha=1.0, *, fit_intercept=True, max_iter=100_000):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def _l1_ratio(self):
        return 1.0


class MultiTaskElasticNet(_MultiTask):
    """Multi-task elastic net, solved as :class:`MultiTaskLasso`."""

    _kind = "MultiTaskElasticNet"

    def __init__(self, alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, max_iter=100_000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def _l1_ratio(self):
        return self.l1_ratio


class SpectralEmbedding(_Embedding):
    """Laplacian eigenmaps on a k-NN (``n_neighbors`` default
    ``max(n / 10, 1)``) or RBF (``gamma`` default ``1 / n_features``)
    affinity."""

    _kind = "SpectralEmbedding"

    def __init__(self, n_components=2, *, affinity="nearest_neighbors", gamma=None,
                 n_neighbors=None):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors

    def _rust_params(self, X, y):
        return {"n_components": self.n_components,
                "affinity": _choice(self.affinity, {"nearest_neighbors": "NearestNeighbors",
                                                    "rbf": "Rbf"}, "affinity"),
                "gamma": self.gamma, "n_neighbors": self.n_neighbors}


class RcsTransformer(TransformerMixin, BaseEstimator):
    """Restricted cubic spline basis per feature (Harrell's ``rcs``):
    ``n_knots`` quantile knots, or explicit ``knots`` (one list per
    feature). ``norm`` scales the nonlinear terms: ``"range"`` divides by
    ``(t_k − t_1)²`` (``rms::rcs``'s default), ``"last_interval"`` by
    ``(t_k − t_{k−1})³``."""

    _kind = "RcsTransformer"

    def __init__(self, n_knots=5, *, knots=None, norm="range"):
        self.n_knots = n_knots
        self.knots = knots
        self.norm = norm

    def _rust_params(self, X, y):
        knots = None if self.knots is None else [[float(t) for t in k] for k in self.knots]
        return {"n_knots": self.n_knots, "knots": knots,
                "norm": _choice(self.norm, {"none": "None", "last_interval": "LastInterval",
                                            "range": "Range"}, "norm")}

    @property
    def knots_(self):
        return [np.asarray(k, dtype=np.float64) for k in self._fitted_state()["knots"]]


class _MultiTaskCV(_MultiTask):
    """``alphas``: the grid size per ``l1_ratio`` (sklearn's grid) or the
    alphas; ``cv`` contiguous folds (sklearn's ``KFold``)."""

    def _rust_params(self, X, y):
        p = {"l1_ratios": [float(r) for r in np.atleast_1d(self._l1_ratio())],
             "eps": float(self.eps), "cv": int(5 if self.cv is None else self.cv),
             "fit_intercept": self.fit_intercept, "max_iter": self.max_iter,
             "y_columns": int(y.shape[1])}
        if isinstance(self.alphas, (int, np.integer)) and not isinstance(self.alphas, bool):
            p["n_alphas"] = int(self.alphas)
        else:
            p["alphas"] = [float(a) for a in self.alphas]
        return p

    @property
    def coef_(self):
        return np.asarray(self._model().coef())

    @property
    def intercept_(self):
        return _nd(self._fitted_state()["fitted"]["intercept"])

    @property
    def alpha_(self):
        return float(self._fitted_state()["alpha"])

    @property
    def alphas_(self):
        a = _nd(self._fitted_state()["alphas"])
        return a[0] if a.shape[0] == 1 else a

    @property
    def mse_path_(self):
        m = _nd(self._fitted_state()["mse_path"])
        return m[0] if m.shape[0] == 1 else m


class MultiTaskLassoCV(_MultiTaskCV):
    """:class:`MultiTaskLasso` with ``alpha`` chosen by cross-validation."""

    _kind = "MultiTaskLassoCV"

    def __init__(self, *, eps=1e-3, alphas=100, fit_intercept=True, max_iter=100_000, cv=None):
        self.eps = eps
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.cv = cv

    def _l1_ratio(self):
        return 1.0


class MultiTaskElasticNetCV(_MultiTaskCV):
    """:class:`MultiTaskElasticNet` with ``alpha`` and ``l1_ratio`` (a float
    or a list) chosen by cross-validation."""

    _kind = "MultiTaskElasticNetCV"

    def __init__(self, *, l1_ratio=0.5, eps=1e-3, alphas=100, fit_intercept=True,
                 max_iter=100_000, cv=None):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.cv = cv

    def _l1_ratio(self):
        return self.l1_ratio

    @property
    def l1_ratio_(self):
        return float(self._fitted_state()["l1_ratio"])


class LarsCV(RegressorMixin, BaseEstimator):
    """LARS with ``alpha`` chosen by the mean held-out squared residue over
    ``cv`` contiguous folds; the refit stops the path at ``alpha_``."""

    _kind = "LarsCV"

    def __init__(self, *, fit_intercept=True, max_iter=500, cv=None, max_n_alphas=1000):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.cv = cv
        self.max_n_alphas = max_n_alphas

    def _rust_params(self, X, y):
        return {"fit_intercept": self.fit_intercept, "max_iter": self.max_iter,
                "cv": int(5 if self.cv is None else self.cv),
                "max_n_alphas": self.max_n_alphas}

    def _state(self, name):
        return self._fitted_state()[name]

    coef_ = property(lambda self: _nd(self._state("fitted")["coef"]))
    intercept_ = property(lambda self: float(self._state("fitted")["intercept"]))
    alpha_ = property(lambda self: float(self._state("alpha")))
    cv_alphas_ = property(lambda self: _nd(self._state("cv_alphas")))
    mse_path_ = property(lambda self: _nd(self._state("mse_path")))
    alphas_ = property(lambda self: _nd(self._state("fitted")["alphas"]))
    coef_path_ = property(lambda self: _nd(self._state("fitted")["coef_path"]))
    active_ = property(lambda self: list(self._state("fitted")["active_set"]))


class LassoLarsCV(LarsCV):
    """:class:`LarsCV` on the Lasso path."""

    _kind = "LassoLarsCV"


# ---------------------------------------------------------------------------
# text vectorizers (documents in, dense arrays out)
# ---------------------------------------------------------------------------


def _documents(raw_documents):
    if isinstance(raw_documents, str):
        raise ValueError("Iterable over raw text documents expected, string object received.")
    return [str(d) for d in raw_documents]


class _TextVectorizer(TransformerMixin, BaseEstimator):
    """sklearn's default analyzer: lowercase, ``token_pattern =
    r"(?u)\\b\\w\\w+\\b"``, unigrams. ``transform`` returns a dense array
    (sklearn returns a sparse matrix)."""

    def fit(self, raw_documents, y=None):
        docs = _documents(raw_documents)
        model = self._new_model(None, None)
        model.fit_text(docs)
        self._model_ = model
        return self

    def transform(self, raw_documents):
        return self._model().transform_text(_documents(raw_documents))

    def fit_transform(self, raw_documents, y=None):
        docs = _documents(raw_documents)
        return self.fit(docs).transform(docs)


class _VocabularyVectorizer(_TextVectorizer):
    def _df_params(self):
        if not isinstance(self.min_df, (int, np.integer)) or isinstance(self.min_df, bool):
            raise ValueError("min_df must be an int (a document count)")
        if not isinstance(self.max_df, (float, np.floating)) or not 0 <= self.max_df <= 1:
            raise ValueError("max_df must be a float in [0, 1] (a document fraction)")
        return {"min_df": int(self.min_df), "max_df_frac": float(self.max_df)}

    def _vocabulary(self):
        state = self._fitted_state()
        return state.get("vocabulary", state.get("counts", {}).get("vocabulary"))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self._vocabulary(), dtype=object)

    @property
    def vocabulary_(self):
        return {t: i for i, t in enumerate(self._vocabulary())}


class CountVectorizer(_VocabularyVectorizer):
    """Term counts over the training vocabulary."""

    _kind = "CountVectorizer"

    def __init__(self, *, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df

    def _rust_params(self, X, y):
        return self._df_params()


class TfidfVectorizer(_VocabularyVectorizer):
    """Counts weighted by the smoothed IDF, rows L2-normalised (``norm="l2"``)
    or not (``norm=None``)."""

    _kind = "TfidfVectorizer"

    def __init__(self, *, min_df=1, max_df=1.0, norm="l2"):
        self.min_df = min_df
        self.max_df = max_df
        self.norm = norm

    def _rust_params(self, X, y):
        if self.norm not in ("l2", None):
            raise ValueError("norm must be 'l2' or None")
        return {**self._df_params(), "norm_l2": self.norm == "l2"}

    @property
    def idf_(self):
        return np.asarray(self._fitted_state()["idf"], dtype=np.float64)


class HashingVectorizer(_TextVectorizer):
    """Signed MurmurHash3 columns (sklearn's); stateless, ``fit`` only checks
    the parameters. Keep ``n_features`` small: the output is dense."""

    _kind = "HashingVectorizer"

    def __init__(self, *, n_features=2**20, alternate_sign=True, norm="l2"):
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        self.norm = norm

    def _rust_params(self, X, y):
        if self.norm not in ("l2", None):
            raise ValueError("norm must be 'l2' or None")
        return {"n_features": int(self.n_features), "alternate_sign": self.alternate_sign,
                "norm_l2": self.norm == "l2"}

    def transform(self, raw_documents):
        if "_model_" not in self.__dict__:
            self.fit([])
        return super().transform(raw_documents)


class FeatureHasher(TransformerMixin, BaseEstimator):
    """The hashing trick on named features (sklearn's signed MurmurHash3):
    ``input_type="dict"`` (``{name: value}``; a string value ``v`` adds
    ``name=v`` with value 1), ``"pair"`` (``(name, value)`` pairs) or
    ``"string"`` (names, value 1). Stateless; ``transform`` is dense, so keep
    ``n_features`` small."""

    def __init__(self, n_features=2**20, *, input_type="dict", alternate_sign=True):
        self.n_features = n_features
        self.input_type = input_type
        self.alternate_sign = alternate_sign

    def fit(self, X=None, y=None):
        self._validate()
        return self

    def _validate(self):
        if self.input_type not in ("dict", "pair", "string"):
            raise ValueError("input_type must be 'dict', 'pair' or 'string'")
        n = self.n_features
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or not 1 <= n < 2**31:
            raise ValueError(f"n_features must be an int in [1, 2**31 - 1], got {n!r}")

    @staticmethod
    def _pairs(items):
        out = []
        for name, value in items:
            if isinstance(value, str):
                name, value = f"{name}={value}", 1.0
            if not isinstance(name, str):
                raise TypeError(f"feature names must be strings, got {type(name).__name__}")
            out.append((name, float(value)))
        return out

    def transform(self, raw_X):
        self._validate()
        if self.input_type == "dict":
            samples = [self._pairs(d.items()) for d in raw_X]
        elif self.input_type == "pair":
            samples = [self._pairs(x) for x in raw_X]
        else:
            if isinstance(raw_X, str) or any(isinstance(x, str) for x in raw_X):
                raise ValueError("Samples can not be a single string. The input must be an "
                                 "iterable over iterables of strings.")
            samples = [self._pairs((f, 1.0) for f in x) for x in raw_X]
        return _native.feature_hash(samples, int(self.n_features), bool(self.alternate_sign))

    def fit_transform(self, raw_X, y=None):
        return self.fit(raw_X).transform(raw_X)


class NeighborhoodComponentsAnalysis(SupervisedTransformerMixin, BaseEstimator):
    """Linear map maximising the stochastic nearest-neighbour accuracy;
    ``init`` is ``"auto"``, ``"pca"``, ``"lda"``, ``"identity"`` or an
    ``(n_components, n_features)`` array. The optimiser is L-BFGS like
    sklearn's, but not SciPy's, so results agree to its tolerance."""

    def __init__(self, n_components=None, *, init="auto", max_iter=50, tol=1e-5):
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol

    _kind = "NeighborhoodComponentsAnalysis"

    def _rust_params(self, X, y):
        if isinstance(self.init, str):
            init = _choice(self.init, {"auto": "Auto", "pca": "Pca", "lda": "Lda",
                                       "identity": "Identity"}, "init")
        else:
            init = {"Matrix": np.asarray(self.init, dtype=np.float64).tolist()}
        return {"n_components": self.n_components, "init": init,
                "max_iter": int(self.max_iter), "tol": float(self.tol)}

    def fit(self, X, y):
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        return self._fit_model(X, codes.astype(np.float64))

    @property
    def components_(self):
        return _nd(self._fitted_state()["components"])

    @property
    def n_iter_(self):
        return int(self._fitted_state()["n_iter"])


class BernoulliRBM(PartialFitMixin, TransformerMixin, BaseEstimator):
    """Bernoulli restricted Boltzmann machine (persistent contrastive
    divergence). A seeded fit draws from numpy's ``RandomState`` in
    sklearn's order and equals sklearn's; ``transform`` is ``P(h=1|v)``."""

    _kind = "BernoulliRbm"

    def __init__(self, n_components=256, *, learning_rate=0.1, batch_size=10, n_iter=10,
                 verbose=0, random_state=None):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_components": int(self.n_components),
                "learning_rate": float(self.learning_rate),
                "batch_size": int(self.batch_size), "n_iter": int(self.n_iter),
                "seed": _seed(self.random_state) % 2**32}

    def score_samples(self, X):
        """Pseudo-likelihood of each row (one random bit flipped)."""
        return self._call("score_samples", X)

    def _attr(self, name):
        return _nd(self._fitted_state()[name])

    components_ = property(lambda self: self._attr("components"))
    intercept_hidden_ = property(lambda self: self._attr("intercept_hidden"))
    intercept_visible_ = property(lambda self: self._attr("intercept_visible"))
    h_samples_ = property(lambda self: self._attr("h_samples"))


class LatentDirichletAllocation(TransformerMixin, BaseEstimator):
    """Topic model by variational Bayes (``learning_method`` ``"batch"`` or
    ``"online"``). A seeded fit draws numpy ``RandomState`` gammas in
    sklearn's order and uses sklearn's digamma, so it equals sklearn's;
    ``transform`` returns normalised document-topic distributions."""

    _kind = "LatentDirichletAllocation"

    def __init__(self, n_components=10, *, doc_topic_prior=None, topic_word_prior=None,
                 learning_method="batch", learning_decay=0.7, learning_offset=10.0,
                 max_iter=10, batch_size=128, evaluate_every=-1, perp_tol=1e-1,
                 mean_change_tol=1e-3, max_doc_update_iter=100, random_state=None):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.random_state = random_state

    def _rust_params(self, X, y):
        if self.learning_method not in ("batch", "online"):
            raise ValueError("learning_method must be 'batch' or 'online'")
        return {"n_components": int(self.n_components),
                "doc_topic_prior": self.doc_topic_prior,
                "topic_word_prior": self.topic_word_prior,
                "learning_method": self.learning_method,
                "learning_decay": float(self.learning_decay),
                "learning_offset": float(self.learning_offset),
                "max_iter": int(self.max_iter), "batch_size": int(self.batch_size),
                "evaluate_every": int(self.evaluate_every), "perp_tol": float(self.perp_tol),
                "mean_change_tol": float(self.mean_change_tol),
                "max_doc_update_iter": int(self.max_doc_update_iter),
                "seed": _seed(self.random_state) % 2**32}

    components_ = property(lambda self: _nd(self._fitted_state()["components"]))
    n_iter_ = property(lambda self: int(self._fitted_state()["n_iter"]))
    bound_ = property(lambda self: float(self._fitted_state()["bound"]))
    doc_topic_prior_ = property(lambda self: float(self._fitted_state()["doc_topic_prior"]))
    topic_word_prior_ = property(lambda self: float(self._fitted_state()["topic_word_prior"]))


class BisectingKMeans(ClusterMixin, BaseEstimator):
    """Divisive k-means: split the leaf with the largest inertia (or the
    most rows) by 2-means until ``n_clusters`` leaves remain. A seeded fit
    draws its inits from numpy's ``RandomState`` in sklearn's order and
    equals sklearn's."""

    _kind = "BisectingKMeans"

    def __init__(self, n_clusters=8, *, init="random", n_init=1, random_state=None,
                 max_iter=300, tol=1e-4, bisecting_strategy="biggest_inertia"):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.bisecting_strategy = bisecting_strategy

    def _rust_params(self, X, y):
        return {"n_clusters": int(self.n_clusters),
                "init": _choice(self.init, {"random": "Random",
                                            "k-means++": "KMeansPlusPlus"}, "init"),
                "n_init": int(self.n_init), "max_iter": int(self.max_iter),
                "tol": float(self.tol),
                "bisecting_strategy": _choice(self.bisecting_strategy, {
                    "biggest_inertia": "BiggestInertia",
                    "largest_cluster": "LargestCluster"}, "bisecting_strategy"),
                "seed": _seed(self.random_state) % 2**32}

    cluster_centers_ = property(lambda self: _nd(self._fitted_state()["cluster_centers"]))
    inertia_ = property(lambda self: float(self._fitted_state()["inertia"]))


class SparsePCA(TransformerMixin, BaseEstimator):
    """Sparse components by dictionary learning (``method="lars"``);
    ``transform`` is a ridge projection (``ridge_alpha``). Equals sklearn's
    up to the LARS solver's rounding."""

    _kind = "SparsePca"

    def __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000,
                 tol=1e-8, method="lars", random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.random_state = random_state

    def _rust_params(self, X, y):
        if self.method != "lars":
            raise ValueError("only method='lars' is supported")
        return {"n_components": self.n_components, "alpha": float(self.alpha),
                "ridge_alpha": float(self.ridge_alpha), "max_iter": int(self.max_iter),
                "tol": float(self.tol), "seed": _seed(self.random_state) % 2**32}

    components_ = property(lambda self: _nd(self._fitted_state()["components"]))
    mean_ = property(lambda self: _nd(self._fitted_state()["mean"]))
    error_ = property(lambda self: np.asarray(self._fitted_state()["error"]))
    n_iter_ = property(lambda self: int(self._fitted_state()["n_iter"]))


class MiniBatchNMF(TransformerMixin, BaseEstimator):
    """NMF by online multiplicative updates over contiguous mini-batches
    (no ``fresh_restarts``). With the default NNDSVDa init it draws no
    random numbers and equals sklearn's; ``fit_transform`` returns the
    training ``W``."""

    _kind = "MiniBatchNmf"

    def __init__(self, n_components="auto", *, init=None, batch_size=1024,
                 beta_loss="frobenius", tol=1e-4, max_no_improvement=10, max_iter=200,
                 alpha_W=0.0, alpha_H="same", l1_ratio=0.0, forget_factor=0.7,
                 transform_max_iter=None, random_state=None):
        self.n_components = n_components
        self.init = init
        self.batch_size = batch_size
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.max_iter = max_iter
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.forget_factor = forget_factor
        self.transform_max_iter = transform_max_iter
        self.random_state = random_state

    def _rust_params(self, X, y):
        k = self.n_components
        k = X.shape[1] if k in ("auto", None) else int(k)
        return {"n_components": k,
                "init": _choice(self.init, {None: "Auto", "random": "Random",
                                            "nndsvd": "Nndsvd", "nndsvda": "NndsvdA",
                                            "nndsvdar": "NndsvdAr"}, "init"),
                "batch_size": int(self.batch_size),
                "beta_loss": _choice(self.beta_loss, {"frobenius": "Frobenius",
                                                      "kullback-leibler": "KullbackLeibler",
                                                      "itakura-saito": "ItakuraSaito"},
                                     "beta_loss"),
                "tol": float(self.tol),
                "max_no_improvement": None if self.max_no_improvement is None
                else int(self.max_no_improvement),
                "max_iter": int(self.max_iter), "alpha_w": float(self.alpha_W),
                "alpha_h": None if self.alpha_H == "same" else float(self.alpha_H),
                "l1_ratio": float(self.l1_ratio), "forget_factor": float(self.forget_factor),
                "transform_max_iter": self.transform_max_iter,
                "seed": _seed(self.random_state)}

    def fit_transform(self, X, y=None, **fit_params):
        return np.asarray(self.fit(X)._model().embedding())

    components_ = property(lambda self: _nd(self._fitted_state()["components"]))
    reconstruction_err_ = property(lambda self: float(self._fitted_state()["reconstruction_err"]))
    n_iter_ = property(lambda self: int(self._fitted_state()["n_iter"]))
    n_steps_ = property(lambda self: int(self._fitted_state()["n_steps"]))


def _tree_max_features(v):
    if v is None:
        return None
    if v in ("sqrt", "log2"):
        return _max_features(v)
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return {"Fixed": int(v)}
    if isinstance(v, float):
        return {"Fraction": v}
    raise ValueError(f"invalid max_features {v!r}")


class _ExtraTree(BaseEstimator):
    """sklearn's random-splitter tree, reproduced draw for draw: a seeded
    fit has sklearn's nodes. ``min_samples_split`` / ``min_samples_leaf``
    are counts."""

    def _tree_params(self, X):
        return {"max_depth": self.max_depth, "min_samples_split": int(self.min_samples_split),
                "min_samples_leaf": int(self.min_samples_leaf),
                "min_weight_fraction_leaf": float(self.min_weight_fraction_leaf),
                "max_features": _tree_max_features(self.max_features),
                "min_impurity_decrease": float(self.min_impurity_decrease),
                "seed": _seed(self.random_state) % 2**32}


class ExtraTreeRegressor(RegressorMixin, _ExtraTree):
    _kind = "ExtraTreeRegressor"

    def __init__(self, *, criterion="squared_error", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0,
                 random_state=None, min_impurity_decrease=0.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease

    def _rust_params(self, X, y):
        if self.criterion != "squared_error":
            raise ValueError("only criterion='squared_error' is supported")
        return self._tree_params(X)


class ExtraTreeClassifier(ClassifierMixin, _ExtraTree):
    _kind = "ExtraTreeClassifier"

    def __init__(self, *, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="sqrt",
                 random_state=None, min_impurity_decrease=0.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease

    def _rust_params(self, X, y):
        crit = _choice(self.criterion, {"gini": "Gini", "entropy": "Entropy",
                                        "log_loss": "Entropy"}, "criterion")
        return {"criterion": crit, **self._tree_params(X)}


class RandomTreesEmbedding(TransformerMixin, BaseEstimator):
    """One-hot leaf indicators of totally random trees (dense output). A
    seeded fit has sklearn's trees and columns."""

    _kind = "RandomTreesEmbedding"

    def __init__(self, n_estimators=100, *, max_depth=5, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    def _rust_params(self, X, y):
        return {"n_estimators": int(self.n_estimators), "max_depth": self.max_depth,
                "min_samples_split": int(self.min_samples_split),
                "min_samples_leaf": int(self.min_samples_leaf),
                "min_weight_fraction_leaf": float(self.min_weight_fraction_leaf),
                "min_impurity_decrease": float(self.min_impurity_decrease),
                "seed": _seed(self.random_state) % 2**32}


class OneHotEncoder(TransformerMixin, BaseEstimator):
    """One-hot encoding of numeric category codes (dense output; sklearn's
    default is sparse). ``drop`` is ``None``, ``"first"`` or
    ``"if_binary"``; rare categories are grouped by ``min_frequency`` /
    ``max_categories``."""

    _kind = "OneHotEncoder"

    def __init__(self, *, categories="auto", drop=None, sparse_output=False,
                 handle_unknown="error", min_frequency=None, max_categories=None):
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def _rust_params(self, X, y):
        if self.categories != "auto":
            raise ValueError("only categories='auto' is supported")
        mf = self.min_frequency
        if mf is not None:
            mf = ({"Count": int(mf)} if isinstance(mf, (int, np.integer))
                  else {"Fraction": float(mf)})
        return {"handle_unknown": _choice(self.handle_unknown, {
                    "error": "Error", "ignore": "Ignore",
                    "infrequent_if_exist": "InfrequentIfExist"}, "handle_unknown"),
                "drop": _choice(self.drop, {None: "None", "first": "First",
                                            "if_binary": "IfBinary"}, "drop"),
                "min_frequency": mf, "max_categories": self.max_categories}

    @property
    def categories_(self):
        return [np.asarray(c["categories"], dtype=np.float64)
                for c in self._fitted_state()["columns"]]
