"""sklearn's ``FrozenEstimator``: a fitted estimator whose ``fit`` does
nothing, so meta-estimators and searches use it as it is. It holds a Python
object, so it is plain Python."""

from __future__ import annotations

from ._base import BaseEstimator, NotFittedError


class FrozenEstimator(BaseEstimator):
    """Wraps a fitted ``estimator``: ``fit`` / ``fit_transform`` /
    ``fit_predict`` leave it unchanged, everything else is delegated."""

    def __init__(self, estimator):
        self.estimator = estimator

    def __getattr__(self, name):
        # Only reached for attributes not on the wrapper (sklearn's rule).
        if name == "estimator":
            raise AttributeError(name)
        return getattr(self.estimator, name)

    def __sklearn_clone__(self):
        return self

    def _check_fitted(self):
        fitted = getattr(self.estimator, "_model_", None) is not None or any(
            k.endswith("_") and not k.startswith("_") for k in vars(self.estimator))
        if not fitted:
            raise NotFittedError("FrozenEstimator needs a fitted estimator")

    def fit(self, X=None, y=None, *args, **kwargs):
        self._check_fitted()
        return self

    def fit_transform(self, X, y=None, *args, **kwargs):
        return self.fit().transform(X)

    def fit_predict(self, X, y=None, *args, **kwargs):
        return self.fit().predict(X)

    def set_params(self, **params):
        estimator = params.pop("estimator", None)
        if params:
            raise ValueError("You cannot set parameters of the inner estimator in a "
                             "frozen estimator since calling `fit` has no effect.")
        if estimator is not None:
            self.estimator = estimator
        return self

    def get_params(self, deep=True):
        return {"estimator": self.estimator}
