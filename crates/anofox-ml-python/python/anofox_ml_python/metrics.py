"""Metrics computed in Rust (``y_true`` first, as in sklearn)."""

import numpy as np

from . import _native

__all__ = [
    "accuracy_score", "adjusted_rand_score", "average_precision_score",
    "balanced_accuracy_score", "brier_score_loss", "cohen_kappa_score", "confusion_matrix",
    "explained_variance_score", "f1_score", "log_loss", "matthews_corrcoef", "max_error",
    "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_error",
    "mean_squared_log_error", "median_absolute_error", "normalized_mutual_info_score",
    "precision_score", "r2_score", "recall_score", "roc_auc_score", "silhouette_score",
]


def _a(y):
    return np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())


def _pair(name):
    fn = getattr(_native, name)

    def metric(y_true, y_pred):
        return fn(_a(y_true), _a(y_pred))

    metric.__name__ = name
    metric.__doc__ = f"sklearn.metrics.{name} (computed in Rust)."
    return metric


accuracy_score = _pair("accuracy_score")
balanced_accuracy_score = _pair("balanced_accuracy_score")
cohen_kappa_score = _pair("cohen_kappa_score")
matthews_corrcoef = _pair("matthews_corrcoef")
explained_variance_score = _pair("explained_variance_score")
max_error = _pair("max_error")
mean_absolute_percentage_error = _pair("mean_absolute_percentage_error")
mean_squared_log_error = _pair("mean_squared_log_error")
median_absolute_error = _pair("median_absolute_error")
adjusted_rand_score = _pair("adjusted_rand_score")
normalized_mutual_info_score = _pair("normalized_mutual_info_score")
roc_auc_score = _pair("roc_auc_score")
average_precision_score = _pair("average_precision_score")
brier_score_loss = _pair("brier_score_loss")
r2_score = _pair("r2_score")
mean_squared_error = _pair("mse")
mean_absolute_error = _pair("mae")


def log_loss(y_true, y_prob):
    """Binary log loss; ``y_prob`` is the probability of the positive class."""
    return _native.log_loss(_a(y_true), _a(y_prob))


def precision_score(y_true, y_pred, average="macro"):
    return _native.precision_score(_a(y_true), _a(y_pred), average)


def recall_score(y_true, y_pred, average="macro"):
    return _native.recall_score(_a(y_true), _a(y_pred), average)


def f1_score(y_true, y_pred, average="macro"):
    return _native.f1_score(_a(y_true), _a(y_pred), average)


def confusion_matrix(y_true, y_pred):
    return _native.confusion_matrix(_a(y_true), _a(y_pred)).astype(np.int64)


def silhouette_score(X, labels):
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    return _native.silhouette_score(X, _a(labels))
