import pickle
import warnings

import numpy as np
import pytest
import sklearn.covariance as skc
import sklearn.model_selection as skms
from sklearn.datasets import make_sparse_spd_matrix

import anofox_ml_python as am

prec = make_sparse_spd_matrix(7, alpha=0.9, random_state=0)
X = np.random.RandomState(0).multivariate_normal(np.zeros(7), np.linalg.inv(prec), 50) + 2.0


def close(a, b, tol=1e-9):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


def _same(ours, ref, tol=1e-9):
    close(ours.location_, ref.location_, tol)
    close(ours.covariance_, ref.covariance_, tol)
    close(ours.precision_, ref.precision_, tol)
    assert ours.n_iter_ == ref.n_iter_
    costs_o = np.asarray(ours.costs_).reshape(-1, 2)
    costs_r = np.asarray(ref.costs_).reshape(-1, 2)
    if tol <= 1e-9:  # the whole trajectory
        close(costs_o, costs_r, tol)
    else:  # mid-oscillation gaps differ more; the final cost does not
        close(costs_o[-1, 0], costs_r[-1, 0], tol)
    close(ours.score(X), ref.score(X), tol)
    close(ours.mahalanobis(X), ref.mahalanobis(X), tol)


@pytest.mark.parametrize("kw, tol", [
    ({"alpha": 0.01}, 1e-9), ({"alpha": 0.1}, 1e-9), ({"alpha": 0.4, "max_iter": 15}, 1e-9),
    ({"alpha": 0.0}, 1e-9),
    # Uncentred data with mean 2: neither solver converges (the dual gap
    # oscillates for 91 iterations, the inner solves hit their limit), so
    # the first iterations agree to 1e-14 and the rounding grows to ~5e-8.
    ({"alpha": 0.05, "assume_centered": True}, 1e-6),
])
def test_graphical_lasso_matches_sklearn(kw, tol):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _same(am.GraphicalLasso(**kw).fit(X), skc.GraphicalLasso(**kw).fit(X), tol)


def test_precomputed_and_errors():
    S = np.cov(X.T, bias=True)
    ours = am.GraphicalLasso(alpha=0.05, covariance="precomputed").fit(S)
    ref = skc.GraphicalLasso(alpha=0.05, covariance="precomputed").fit(S)
    close(ours.precision_, ref.precision_)
    close(ours.location_, ref.location_)
    close(pickle.loads(pickle.dumps(ours)).precision_, ref.precision_)
    with pytest.raises(ValueError, match="mode"):
        am.GraphicalLasso(mode="lars").fit(X)


@pytest.mark.parametrize("kw", [{}, {"alphas": [0.5, 0.1, 0.02], "cv": 3},
                                {"alphas": 3, "n_refinements": 2, "cv": 4},
                                {"cv": skms.KFold(3, shuffle=True, random_state=0)}])
def test_graphical_lasso_cv_matches_sklearn(kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours, ref = am.GraphicalLassoCV(**kw).fit(X), skc.GraphicalLassoCV(**kw).fit(X)
    assert ours.alpha_ == pytest.approx(ref.alpha_, rel=1e-12)
    for key, value in ref.cv_results_.items():
        close(ours.cv_results_[key], value)
    close(ours.precision_, ref.precision_)
    assert ours.n_iter_ == ref.n_iter_
