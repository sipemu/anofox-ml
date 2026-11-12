import pickle

import numpy as np
import pytest
import sklearn.decomposition as skd

import anofox_ml_python as am

rng = np.random.RandomState(0)
X = rng.randn(40, 8)
D = rng.randn(6, 8)
D /= np.linalg.norm(D, axis=1, keepdims=True)


def close(a, b, tol=1e-10):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


# Fewer atoms than features: with as many or more, the LARS codings wait
# on sipemu/anofox-regression#334.
@pytest.mark.parametrize("algorithm", ["omp", "lars", "lasso_lars", "lasso_cd", "threshold"])
@pytest.mark.parametrize("kw", [{}, {"transform_alpha": 0.3, "transform_n_nonzero_coefs": 3},
                                {"split_sign": True}])
def test_sparse_coder_matches_sklearn(algorithm, kw):
    ours = am.SparseCoder(D, transform_algorithm=algorithm, **kw)
    ref = skd.SparseCoder(D, transform_algorithm=algorithm, **kw)
    # sklearn's coordinate descent stops at tol=1e-8.
    close(ours.transform(X), ref.transform(X), 1e-6 if algorithm == "lasso_cd" else 1e-12)
    code = ref.transform(X)
    close(ours.inverse_transform(code), ref.inverse_transform(code), 1e-12)
    close(pickle.loads(pickle.dumps(ours.fit(X))).transform(X), ours.transform(X))


def _aligned(ours, ref):
    """Each atom's sign, so that ours points as sklearn's (LAPACK's SVD signs
    are arbitrary; the problem is symmetric in them)."""
    s = np.sign((ours.components_ * ref.components_).sum(axis=1))
    return np.where(s == 0, 1.0, s)


@pytest.mark.parametrize("kw, tol", [
    ({"n_components": 5, "alpha": 1.0}, 1e-8),
    ({"n_components": 7, "alpha": 0.5, "transform_algorithm": "lasso_lars"}, 1e-5),
    ({"n_components": 4, "alpha": 1.0, "fit_algorithm": "cd",
      "transform_algorithm": "threshold", "transform_alpha": 0.2}, 1e-3),
])
def test_dictionary_learning_matches_sklearn_up_to_atom_signs(kw, tol):
    ours = am.DictionaryLearning(random_state=0, **kw)
    ref = skd.DictionaryLearning(random_state=0, **kw)
    code_o, code_r = ours.fit_transform(X), ref.fit_transform(X)
    s = _aligned(ours, ref)
    close(ours.components_ * s[:, None], ref.components_, tol)
    close(code_o * s, code_r, tol)
    assert ours.n_iter_ == ref.n_iter_
    close(ours.error_[-1], ref.error_[-1], tol)
    close(ours.transform(X) * s, ref.transform(X), tol)
    close(ours.inverse_transform(code_r * s), ref.inverse_transform(code_r), tol)


def test_dictionary_learning_options():
    with pytest.raises(ValueError, match="not supported"):
        am.DictionaryLearning(dict_init=np.eye(8)).fit(X)
    with pytest.raises(ValueError, match="positive_code"):
        am.SparseCoder(D, positive_code=True).transform(X)
    split = am.DictionaryLearning(n_components=3, split_sign=True, random_state=0)
    assert split.fit_transform(X).shape == (40, 6)


# The mini-batch ones start from randomized_svd with svd_flip, so they are
# compared exactly (no sign alignment).
@pytest.mark.parametrize("kw, tol", [
    ({"n_components": 5, "batch_size": 8}, 1e-10),
    ({"n_components": 4, "batch_size": 16, "shuffle": False, "alpha": 0.5,
      "transform_algorithm": "lasso_lars"}, 1e-10),
    # sklearn stops each Lasso-CD code at tol=1e-8; over ~2500 steps that
    # compounds to ~1e-5.
    ({"n_components": 6, "batch_size": 10, "fit_algorithm": "cd"}, 1e-4),
    ({"n_components": 3, "batch_size": 7, "max_iter": 5, "tol": 0, "split_sign": True}, 1e-10),
])
def test_minibatch_dictionary_learning_matches_sklearn(kw, tol):
    Xm = np.random.RandomState(1).randn(60, 10)
    ours = am.MiniBatchDictionaryLearning(random_state=0, **kw).fit(Xm)
    ref = skd.MiniBatchDictionaryLearning(random_state=0, **kw).fit(Xm)
    close(ours.components_, ref.components_, tol)
    assert ours.n_steps_ == ref.n_steps_ and ours.n_iter_ == ref.n_iter_
    close(ours.transform(Xm), ref.transform(Xm), tol)
    code = ref.transform(Xm)
    close(ours.inverse_transform(code), ref.inverse_transform(code), tol)
    close(pickle.loads(pickle.dumps(ours)).transform(Xm), ours.transform(Xm))


@pytest.mark.parametrize("kw, tol", [
    ({"n_components": 4}, 1e-10),
    ({"n_components": 6, "alpha": 0.5, "batch_size": 5}, 1e-10),
    ({"n_components": 3, "method": "cd", "shuffle": False}, 1e-6),
])
def test_minibatch_sparse_pca_matches_sklearn(kw, tol):
    Xs = np.random.RandomState(2).randn(30, 12)
    ours = am.MiniBatchSparsePCA(random_state=0, **kw).fit(Xs)
    ref = skd.MiniBatchSparsePCA(random_state=0, **kw).fit(Xs)
    close(ours.components_, ref.components_, tol)
    close(ours.mean_, ref.mean_)
    assert ours.n_iter_ == ref.n_iter_
    close(ours.transform(Xs), ref.transform(Xs), tol)
    assert not hasattr(ours, "error_")
