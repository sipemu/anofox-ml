import pickle

import numpy as np
import pytest
import sklearn.kernel_approximation as ska

import anofox_ml_python as am

rng = np.random.RandomState(0)
X = np.abs(rng.randn(30, 4))
X[3, 2] = 0.0
Xs = rng.randn(30, 4)


def close(a, b, tol=1e-10):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("kw", [{}, {"sample_steps": 1}, {"sample_steps": 3},
                                {"sample_steps": 5, "sample_interval": 0.2}])
def test_additive_chi2_sampler_matches_sklearn(kw):
    ours, ref = am.AdditiveChi2Sampler(**kw).fit(X), ska.AdditiveChi2Sampler(**kw).fit(X)
    close(ours.transform(X), ref.transform(X))
    np.testing.assert_array_equal(ours.get_feature_names_out(), ref.get_feature_names_out())
    with pytest.raises(ValueError):
        ours.transform(-X)
    with pytest.raises(ValueError, match="sample_interval"):
        am.AdditiveChi2Sampler(sample_steps=4).fit(X)


@pytest.mark.parametrize("kw", [{"random_state": 0}, {"skewedness": 0.3, "n_components": 11,
                                                      "random_state": 9}])
def test_skewed_chi2_sampler_matches_sklearn(kw):
    ours, ref = am.SkewedChi2Sampler(**kw).fit(X), ska.SkewedChi2Sampler(**kw).fit(X)
    close(ours.random_weights_, ref.random_weights_, 1e-12)
    close(ours.random_offset_, ref.random_offset_, 1e-12)
    close(ours.transform(X), ref.transform(X))
    close(pickle.loads(pickle.dumps(ours)).transform(X), ref.transform(X))
    with pytest.raises(ValueError, match="skewedness"):
        ours.transform(X - 10)


@pytest.mark.parametrize("kw", [{"random_state": 0},
                                {"degree": 3, "coef0": 2, "gamma": 0.5, "n_components": 37,
                                 "random_state": 4},
                                {"degree": 4, "n_components": 8, "random_state": 1}])
def test_polynomial_count_sketch_matches_sklearn(kw):
    ours, ref = am.PolynomialCountSketch(**kw).fit(Xs), ska.PolynomialCountSketch(**kw).fit(Xs)
    np.testing.assert_array_equal(ours.indexHash_, ref.indexHash_)
    np.testing.assert_array_equal(ours.bitHash_, ref.bitHash_)
    close(ours.transform(Xs), ref.transform(Xs))
