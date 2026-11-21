import pickle

import numpy as np
import pytest
from sklearn.cluster import SpectralBiclustering as SkSpectralBiclustering
from sklearn.datasets import make_checkerboard

import anofox_ml_python as am


@pytest.mark.parametrize("seed, k, shape, kw", [
    (0, 3, (30, 20), {}), (1, (4, 3), (40, 25), {"method": "scale"}),
    (2, (2, 3), (18, 15), {"method": "log", "n_components": 4, "n_best": 2}),
    (3, 2, (20, 16), {"n_svd_vecs": 5, "n_init": 2}),
])
def test_spectral_biclustering_matches_sklearn(seed, k, shape, kw):
    X, _, _ = make_checkerboard(shape, k, noise=5, shuffle=True, random_state=seed)
    ours = am.SpectralBiclustering(k, random_state=seed, **kw).fit(X)
    ref = SkSpectralBiclustering(k, random_state=seed, **kw).fit(X)
    np.testing.assert_array_equal(ours.row_labels_, ref.row_labels_)
    np.testing.assert_array_equal(ours.column_labels_, ref.column_labels_)
    np.testing.assert_array_equal(ours.rows_, ref.rows_)
    np.testing.assert_array_equal(ours.columns_, ref.columns_)
    assert ours.get_shape(1) == ref.get_shape(1)
    np.testing.assert_array_equal(ours.get_submatrix(2, X), ref.get_submatrix(2, X))
    clone = pickle.loads(pickle.dumps(ours))
    np.testing.assert_array_equal(clone.column_labels_, ref.column_labels_)


def test_options():
    X, _, _ = make_checkerboard((20, 10), 2, random_state=0)
    with pytest.raises(ValueError, match="method"):
        am.SpectralBiclustering(method="nope").fit(X)
    with pytest.raises(ValueError, match="Incorrect parameter n_clusters"):
        am.SpectralBiclustering((1, 2, 3)).fit(X)
    with pytest.raises(ValueError, match="n_best"):
        am.SpectralBiclustering(n_components=2, n_best=3).fit(X)
    with pytest.raises(ValueError, match="randomized"):
        am.SpectralBiclustering(svd_method="arpack").fit(X)
