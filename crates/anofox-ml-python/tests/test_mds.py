import pickle
import warnings

import numpy as np
import pytest
import sklearn.manifold as skm

import anofox_ml_python as am

X = np.random.RandomState(0).randn(25, 4)


def close(a, b, tol=1e-10):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("kw", [
    {"random_state": 0}, {"random_state": 3, "n_init": 2, "n_components": 3},
    {"random_state": 0, "metric_mds": False}, {"init": "classical_mds"},
    {"init": "classical_mds", "n_components": 3},  # sklearn: still 2-D
    {"init": "classical_mds", "metric_mds": False}, {"random_state": 1, "max_iter": 7},
    {"random_state": 0, "normalized_stress": True},
])
def test_mds_matches_sklearn(kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = skm.MDS(**kw).fit(X)
    ours = am.MDS(**kw).fit(X)
    close(ours.embedding_, ref.embedding_)
    close(ours.stress_, ref.stress_)
    assert ours.n_iter_ == ref.n_iter_
    close(ours.dissimilarity_matrix_, ref.dissimilarity_matrix_)
    close(am.MDS(**kw).fit_transform(X), ref.embedding_)


def test_precomputed_and_errors():
    D = np.abs(X[:, :1] - X[:, :1].T) + np.abs(X[:, 1:2] - X[:, 1:2].T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = skm.MDS(metric="precomputed", random_state=0).fit(D)
    ours = am.MDS(metric="precomputed", random_state=0).fit(D)
    close(ours.embedding_, ref.embedding_)
    close(pickle.loads(pickle.dumps(ours)).embedding_, ref.embedding_)
    with pytest.raises(ValueError, match="metric"):
        am.MDS(metric="manhattan").fit(X)
    with pytest.raises(ValueError, match="symmetric"):
        am.MDS(metric="precomputed").fit(np.triu(D))
