import pickle

import numpy as np
import pytest
from sklearn.cluster import SpectralCoclustering as SkSpectralCoclustering
from sklearn.datasets import make_biclusters

import anofox_ml_python as am


@pytest.mark.parametrize("seed, k, shape, kw", [
    (0, 3, (30, 20), {}), (1, 4, (40, 25), {}), (2, 2, (15, 18), {"n_svd_vecs": 6}),
    (3, 5, (50, 30), {"n_init": 2}),
])
def test_spectral_coclustering_matches_sklearn(seed, k, shape, kw):
    X, _, _ = make_biclusters(shape, k, noise=3, random_state=seed)
    ours = am.SpectralCoclustering(k, random_state=seed, **kw).fit(X)
    ref = SkSpectralCoclustering(k, random_state=seed, **kw).fit(X)
    np.testing.assert_array_equal(ours.row_labels_, ref.row_labels_)
    np.testing.assert_array_equal(ours.column_labels_, ref.column_labels_)
    np.testing.assert_array_equal(ours.rows_, ref.rows_)
    np.testing.assert_array_equal(ours.columns_, ref.columns_)
    assert ours.get_shape(0) == ref.get_shape(0)
    np.testing.assert_array_equal(ours.get_submatrix(1, X), ref.get_submatrix(1, X))
    clone = pickle.loads(pickle.dumps(ours))
    np.testing.assert_array_equal(clone.row_labels_, ref.row_labels_)


def test_options():
    X, _, _ = make_biclusters((20, 10), 2, random_state=0)
    with pytest.raises(ValueError, match="randomized"):
        am.SpectralCoclustering(svd_method="arpack").fit(X)
    with pytest.raises(ValueError, match="n_clusters"):
        am.SpectralCoclustering(30).fit(X)
