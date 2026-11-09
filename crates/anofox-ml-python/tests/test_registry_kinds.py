"""Parity of the registry kinds added for the DuckDB extension (#125, #113)
and of the DynModel methods behind them (partial_fit, warm_start,
kneighbors, predict_log_proba, offsets) with scikit-learn 1.8."""

import numpy as np
import pytest

import anofox_ml_python as am

sk = pytest.importorskip("sklearn")

rng = np.random.RandomState(0)
X = rng.randn(60, 4)
y = X @ np.array([1.5, -2.0, 0.5, 0.0]) + 0.3 * rng.randn(60)
Y = np.column_stack([y, X[:, 1] - X[:, 2] + 0.3 * rng.randn(60), X[:, 3] + rng.randn(60)])
y_cls = np.where(y + rng.randn(60) > 0, "a", "b")


def _same_up_to_sign(a, b, atol):
    """Columns equal up to their sign (eigen / singular vectors)."""
    signs = np.sign(np.sum(a * b, axis=0))
    np.testing.assert_allclose(a * signs, b, atol=atol)


# ------------------------------------------------------------- new kinds


def test_lasso_lars_ic_matches_sklearn():
    from sklearn.linear_model import LassoLarsIC

    for criterion in ("aic", "bic"):
        ours = am.LassoLarsIC(criterion=criterion).fit(X, y)
        ref = LassoLarsIC(criterion=criterion).fit(X, y)
        np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-10)
        np.testing.assert_allclose(ours.alpha_, ref.alpha_, atol=1e-12)
        np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-10)


def test_isotonic_matches_sklearn():
    from sklearn.isotonic import IsotonicRegression

    x = rng.rand(50) * 10
    t = np.log1p(x) + 0.3 * rng.randn(50)
    grid = np.linspace(-1, 11, 40)
    for increasing in (True, False):
        ours = am.IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(x, t)
        ref = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(x, t)
        np.testing.assert_allclose(ours.predict(grid), ref.predict(grid), atol=1e-12)


def test_gaussian_process_regressor_matches_sklearn():
    from sklearn.gaussian_process import GaussianProcessRegressor

    xs, ts = X[:30], y[:30]
    ours = am.GaussianProcessRegressor(alpha=1e-2).fit(xs, ts)
    ref = GaussianProcessRegressor(alpha=1e-2).fit(xs, ts)
    mean, std = ours.predict(X, return_std=True)
    ref_mean, ref_std = ref.predict(X, return_std=True)
    np.testing.assert_allclose(mean, ref_mean, atol=1e-8)
    np.testing.assert_allclose(std, ref_std, atol=1e-8)


def test_gaussian_process_classifier_matches_sklearn():
    from sklearn.gaussian_process import GaussianProcessClassifier

    xs, ts = X[:40], y_cls[:40]
    ours = am.GaussianProcessClassifier().fit(xs, ts)
    ref = GaussianProcessClassifier().fit(xs, ts)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-6)


def test_pls_regression_matches_sklearn():
    from sklearn.cross_decomposition import PLSRegression

    ours = am.PLSRegression(n_components=2).fit(X, y)
    ref = PLSRegression(n_components=2).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-10)
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-10)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_, atol=1e-10)
    np.testing.assert_allclose(ours.transform(X), ref.transform(X), atol=1e-10)


@pytest.mark.parametrize("name", ["CCA", "PLSCanonical", "PLSSVD"])
def test_two_block_matches_sklearn(name):
    from sklearn import cross_decomposition

    ours = getattr(am, name)(n_components=2).fit(X, Y)
    ref = getattr(cross_decomposition, name)(n_components=2).fit(X, Y)
    xs, ys = ours.transform(X, Y)
    ref_xs, ref_ys = ref.transform(X, Y)
    np.testing.assert_allclose(xs, ref_xs, atol=1e-8)
    np.testing.assert_allclose(ys, ref_ys, atol=1e-8)
    np.testing.assert_allclose(ours.x_weights_, ref.x_weights_, atol=1e-8)
    np.testing.assert_allclose(ours.transform(X), ref_xs, atol=1e-8)
    with pytest.raises(ValueError, match="columns"):
        ours.transform(X, Y[:, :2])


@pytest.mark.parametrize("name", ["MultiTaskLasso", "MultiTaskElasticNet"])
def test_multi_task_matches_sklearn(name):
    from sklearn import linear_model

    ours = getattr(am, name)(alpha=0.2).fit(X, Y)
    ref = getattr(linear_model, name)(alpha=0.2, tol=1e-14, max_iter=100_000).fit(X, Y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-8)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_, atol=1e-8)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-8)
    np.testing.assert_allclose(ours.score(X, Y), ref.score(X, Y), atol=1e-10)


def test_spectral_embedding_matches_sklearn():
    from sklearn.manifold import SpectralEmbedding

    for affinity in ("nearest_neighbors", "rbf"):
        ours = am.SpectralEmbedding(affinity=affinity).fit_transform(X)
        ref = SpectralEmbedding(affinity=affinity, random_state=0).fit_transform(X)
        _same_up_to_sign(ours, ref, atol=1e-6)


def test_binomial_regressor_matches_unpenalised_logistic():
    from sklearn.linear_model import LogisticRegression

    t = (y + 2 * rng.randn(60) > 0).astype(float)
    ours = am.BinomialRegressor().fit(X, t)
    ref = LogisticRegression(penalty=None, tol=1e-12, max_iter=10_000).fit(X, t)
    np.testing.assert_allclose(ours.coef_, ref.coef_.ravel(), atol=1e-6)
    np.testing.assert_allclose(ours.predict(X), ref.predict_proba(X)[:, 1], atol=1e-7)


def test_rcs_transformer_shape_and_knots():
    out = am.RcsTransformer(n_knots=4).fit_transform(X[:, :2])
    assert out.shape == (60, 2 * 3)
    knots = am.RcsTransformer(knots=[[-1, 0, 1], [-1, 0, 1]]).fit(X[:, :2]).knots_
    np.testing.assert_array_equal(knots[0], [-1, 0, 1])


# ---------------------------------------------------------- new methods


def test_predict_log_proba_is_native_for_naive_bayes():
    from sklearn.naive_bayes import GaussianNB

    far = X * 40  # predict_proba underflows to 0 for some classes
    ours = am.GaussianNB().fit(X, y_cls)
    ref = GaussianNB().fit(X, y_cls)
    got = ours.predict_log_proba(far)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, ref.predict_log_proba(far), rtol=1e-9)


def test_partial_fit_matches_sklearn():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler

    ours, ref = am.GaussianNB(), GaussianNB()
    for rows in np.array_split(np.arange(60), 3):
        ours.partial_fit(X[rows], y_cls[rows], classes=["a", "b"])
        ref.partial_fit(X[rows], y_cls[rows], classes=["a", "b"])
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-12)
    assert list(ours.classes_) == ["a", "b"]

    ours, ref = am.StandardScaler(), StandardScaler()
    for rows in np.array_split(np.arange(60), 4):
        ours.partial_fit(X[rows])
        ref.partial_fit(X[rows])
    np.testing.assert_allclose(ours.transform(X), ref.transform(X), atol=1e-12)


def test_partial_fit_errors():
    with pytest.raises(ValueError, match="classes"):
        am.GaussianNB().partial_fit(X, y_cls)
    clf = am.GaussianNB().partial_fit(X, y_cls, classes=["a", "b"])
    with pytest.raises(ValueError, match="not in classes"):
        clf.partial_fit(X[:3], ["a", "c", "b"])
    with pytest.raises(ValueError, match="features"):
        clf.partial_fit(X[:3, :2], ["a", "a", "b"])


def test_kneighbors_matches_sklearn():
    from sklearn.neighbors import KNeighborsRegressor

    ours = am.KNeighborsRegressor(n_neighbors=4).fit(X, y)
    ref = KNeighborsRegressor(n_neighbors=4).fit(X, y)
    dist, idx = ours.kneighbors(X[:10])
    ref_dist, ref_idx = ref.kneighbors(X[:10])
    np.testing.assert_allclose(dist, ref_dist, atol=1e-12)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_array_equal(ours.kneighbors(X[:10], 2, return_distance=False),
                                  ref.kneighbors(X[:10], 2, return_distance=False))


def test_mlp_warm_start_continues():
    kw = dict(hidden_layer_sizes=(8,), max_iter=20, random_state=0)
    cold = am.MLPRegressor(**kw).fit(X, y)
    warm = am.MLPRegressor(warm_start=True, **kw).fit(X, y)
    np.testing.assert_array_equal(warm.predict(X), cold.predict(X))
    warm.fit(X, y)  # 20 more epochs from the fitted weights
    restart = am.MLPRegressor(**kw).fit(X, y)
    assert not np.allclose(warm.predict(X), restart.predict(X))
    assert warm.score(X, y) > cold.score(X, y)


def test_poisson_offset_equals_rate_model():
    """A log(exposure) offset equals fitting the rate y / e with weights e."""
    e = rng.uniform(0.5, 3.0, 60)
    counts = rng.poisson(e * np.exp(0.3 * X[:, 0]))
    off = am.PoissonRegressor(alpha=0.0).fit(X, counts, offset=np.log(e))
    rate = am.PoissonRegressor(alpha=0.0).fit(X, counts / e, sample_weight=e)
    np.testing.assert_allclose(off.coef_, rate.coef_, atol=1e-7)
    np.testing.assert_allclose(off.predict(X, offset=np.log(e)), e * rate.predict(X),
                               rtol=1e-7)
    np.testing.assert_allclose(off.predict(X), rate.predict(X), rtol=1e-7)
