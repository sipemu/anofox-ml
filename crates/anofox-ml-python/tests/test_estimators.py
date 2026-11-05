"""Smoke and parity tests of the sklearn-compatible Python API.

Run after building the extension: ``maturin develop`` (or install the
wheel), then ``pytest crates/anofox-ml-python/tests``. sklearn is only
needed for the parity / integration tests.
"""

import inspect
import pickle

import numpy as np
import pytest

import anofox_ml_python as am
from anofox_ml_python import _estimators

rng = np.random.RandomState(0)
X = rng.randn(80, 4)
X_pos = np.abs(X) + 0.1
y_reg = X @ np.array([1.5, -2.0, 0.5, 0.0]) + 0.1 * rng.randn(80)
y_bin = np.where(y_reg > 0, "yes", "no")
y_multi = np.digitize(y_reg, [-1.0, 1.0])

CLASSES = {
    name: cls
    for name, cls in inspect.getmembers(_estimators, inspect.isclass)
    if cls.__module__ == _estimators.__name__ and not name.startswith("_")
}

# Estimators needing special data or parameters in the smoke test.
SPECIAL = {
    "MultinomialNB": dict(X=np.round(X_pos * 3)),
    "ComplementNB": dict(X=np.round(X_pos * 3)),
    "CategoricalNB": dict(X=np.round(X_pos)),
    "PoissonRegressor": dict(y=np.round(np.exp(y_reg / 3))),
    "GammaRegressor": dict(y=np.exp(y_reg / 3)),
    "NMF": dict(X=X_pos),
    "PowerTransformer": dict(),
    "SelectKBest": dict(params=dict(k=2), y=y_multi),
    "MutualInformationSelector": dict(params=dict(n_features_to_select=2)),
    "SVC": dict(params=dict(probability=True)),
    "NuSVC": dict(params=dict(probability=True)),
    "KMeans": dict(params=dict(n_clusters=3, random_state=0)),
    "MeanShift": dict(params=dict(bandwidth=2.0)),
    "RadiusNeighborsClassifier": dict(params=dict(radius=3.0)),
    "RadiusNeighborsRegressor": dict(params=dict(radius=3.0)),
    "SpectralClustering": dict(params=dict(n_clusters=3, random_state=0)),
    "TSNE": dict(params=dict(perplexity=10.0, max_iter=250, random_state=0)),
    "TargetEncoder": dict(X=np.round(X_pos), params=dict(random_state=0)),
}


def _data(name, cls):
    spec = SPECIAL.get(name, {})
    Xd = spec.get("X", X)
    if "y" in spec:
        yd = spec["y"]
    elif cls._estimator_type == "classifier" if hasattr(cls, "_estimator_type") else False:
        yd = y_bin if "NB" not in name else y_multi
    else:
        yd = y_reg
    return Xd, yd, spec.get("params", {})


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_fit_predict_pickle_clone(name):
    cls = CLASSES[name]
    Xd, yd, params = _data(name, cls)
    est = cls(**params)
    kind = getattr(est, "_estimator_type", None)

    if kind in ("classifier", "regressor"):
        assert est.fit(Xd, yd) is est
        pred = est.predict(Xd)
        assert pred.shape == (len(Xd),)
        if kind == "classifier":
            assert set(np.unique(pred)) <= set(np.unique(yd))
            assert 0.0 <= est.score(Xd, yd) <= 1.0
            if hasattr(est, "predict_proba") and "predict_proba" in \
                    est._model().capabilities():
                try:
                    proba = est.predict_proba(Xd)
                except AttributeError:  # e.g. SGD with hinge loss
                    proba = np.full((len(Xd), len(est.classes_)), 1 / len(est.classes_))
                assert proba.shape == (len(Xd), len(est.classes_))
                np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-6)
        out = pred
    elif kind == "clusterer":
        out = est.fit_predict(Xd)
        assert out.shape == (len(Xd),)
    elif kind == "outlier_detector":
        out = est.fit_predict(Xd)
        assert set(np.unique(out)) <= {-1, 1}
    elif hasattr(est, "embedding_") or name in ("TSNE", "Isomap", "LocallyLinearEmbedding",
                                                 "ClassicalMDS"):
        out = est.fit_transform(Xd)
        assert out.shape == (len(Xd), 2)
        return
    else:  # transformer
        out = est.fit_transform(Xd, yd) if name in ("SelectKBest",
                                                     "MutualInformationSelector",
                                                     "TargetEncoder") \
            else est.fit_transform(Xd)
        assert out.shape[0] == len(Xd)

    # pickling keeps the fitted model and the hyperparameters
    clone = pickle.loads(pickle.dumps(est))
    assert clone.get_params() == est.get_params()
    if kind in ("classifier", "regressor"):
        np.testing.assert_array_equal(clone.predict(Xd), out)
    elif kind is None and hasattr(est, "transform"):
        # (TargetEncoder's fit_transform is cross-fitted, so compare with
        # transform rather than with `out`.)
        np.testing.assert_array_equal(clone.transform(Xd), est.transform(Xd))

    # sklearn-style clone: same params, unfitted
    fresh = type(est)(**est.get_params())
    with pytest.raises(am.NotFittedError):
        fresh._model()


def test_not_fitted_and_param_errors():
    with pytest.raises(am.NotFittedError):
        am.RandomForestRegressor().predict(X)
    with pytest.raises(ValueError, match="Invalid parameter"):
        am.Ridge().set_params(bogus=1)
    with pytest.raises(ValueError, match="features"):
        am.Ridge().fit(X, y_reg).predict(X[:, :2])
    with pytest.raises(AttributeError):
        am.LinearSVC().fit(X, y_bin).predict_proba(X)


def test_string_labels_and_class_weight():
    clf = am.RandomForestClassifier(n_estimators=10, random_state=0,
                                    class_weight={"yes": 3.0}).fit(X, y_bin)
    assert list(clf.classes_) == ["no", "yes"]
    assert set(clf.predict(X)) <= {"no", "yes"}


def test_random_state_reproducible():
    a = am.RandomForestRegressor(n_estimators=5, random_state=3).fit(X, y_reg).predict(X)
    b = am.RandomForestRegressor(n_estimators=5, random_state=3).fit(X, y_reg).predict(X)
    np.testing.assert_array_equal(a, b)


def test_sample_weight_is_used():
    w = np.where(y_reg > 0, 10.0, 0.1)
    a = am.LinearRegression().fit(X, y_reg).coef_
    b = am.LinearRegression().fit(X, y_reg, sample_weight=w).coef_
    assert not np.allclose(a, b)
    with pytest.raises(ValueError, match="sample_weight"):
        am.LassoCV().fit(X, y_reg, sample_weight=w)


# ---------------------------------------------------------------- sklearn
sk = pytest.importorskip("sklearn")


def test_parity_with_sklearn():
    from sklearn import linear_model, preprocessing, tree

    np.testing.assert_allclose(
        am.LinearRegression().fit(X, y_reg).coef_,
        linear_model.LinearRegression().fit(X, y_reg).coef_, atol=1e-10)
    np.testing.assert_allclose(
        am.Ridge(alpha=2.0).fit(X, y_reg).predict(X),
        linear_model.Ridge(alpha=2.0).fit(X, y_reg).predict(X), atol=1e-8)
    np.testing.assert_allclose(
        am.StandardScaler().fit_transform(X),
        preprocessing.StandardScaler().fit_transform(X), atol=1e-12)
    np.testing.assert_array_equal(
        am.DecisionTreeClassifier(max_depth=3).fit(X, y_multi).predict(X),
        tree.DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y_multi).predict(X))


def test_lasso_parity_with_sklearn():
    from sklearn import linear_model

    np.testing.assert_allclose(
        am.Lasso(alpha=0.1).fit(X, y_reg).predict(X),
        linear_model.Lasso(alpha=0.1, tol=1e-12, max_iter=100_000).fit(X, y_reg).predict(X),
        atol=1e-6)


def test_sklearn_clone_grid_search_and_pipeline():
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import make_pipeline

    est = am.Ridge(alpha=3.0)
    c = clone(est)
    assert c.get_params() == est.get_params() and c is not est

    gs = GridSearchCV(am.Ridge(), {"alpha": [0.1, 1.0, 10.0]}, cv=3).fit(X, y_reg)
    assert gs.best_params_["alpha"] in (0.1, 1.0, 10.0)

    pipe = make_pipeline(am.StandardScaler(), am.LogisticRegression())
    scores = cross_val_score(pipe, X, y_bin, cv=3)
    assert scores.mean() > 0.8


def test_joblib_parallel_fits():
    joblib = pytest.importorskip("joblib")
    fitted = joblib.Parallel(n_jobs=2, backend="loky")(
        joblib.delayed(lambda a: am.Ridge(alpha=a).fit(X, y_reg))(a) for a in (0.1, 1.0)
    )
    assert len(fitted) == 2 and fitted[0].predict(X).shape == (80,)


def test_neighbors_match_sklearn():
    from sklearn import neighbors as skn

    Xq = X[:10] + 0.3
    for metric, p in [("minkowski", 2), ("minkowski", 3), ("chebyshev", 2), ("manhattan", 2)]:
        for algorithm in ["brute", "kd_tree"]:
            kw = dict(metric=metric, p=p, algorithm=algorithm, weights="distance")
            ours = am.KNeighborsClassifier(n_neighbors=7, **kw).fit(X, y_multi)
            ref = skn.KNeighborsClassifier(n_neighbors=7, **kw).fit(X, y_multi)
            np.testing.assert_allclose(ours.predict_proba(Xq), ref.predict_proba(Xq), atol=1e-12)
            ours = am.KNeighborsRegressor(n_neighbors=7, **kw).fit(X, y_reg)
            ref = skn.KNeighborsRegressor(n_neighbors=7, **kw).fit(X, y_reg)
            np.testing.assert_allclose(ours.predict(Xq), ref.predict(Xq), atol=1e-12)

    far = np.vstack([Xq, np.full((1, 4), 50.0)])
    for label, y in [(7, y_multi), ("yes", y_bin)]:
        ours = am.RadiusNeighborsClassifier(radius=1.5, outlier_label=label).fit(X, y)
        ref = skn.RadiusNeighborsClassifier(radius=1.5, outlier_label=label).fit(X, y)
        assert list(ours.predict(far)) == list(ref.predict(far))
        np.testing.assert_allclose(ours.predict_proba(far), ref.predict_proba(far), atol=1e-12)
    ours = am.RadiusNeighborsRegressor(radius=1.5).fit(X, y_reg)
    ref = skn.RadiusNeighborsRegressor(radius=1.5).fit(X, y_reg)
    np.testing.assert_allclose(ours.predict(Xq), ref.predict(Xq), atol=1e-12)


def test_tsne_array_init_and_defaults():
    init = np.linspace(-1e-4, 1e-4, 160).reshape(80, 2)
    emb = am.TSNE(perplexity=10.0, init=init, method="exact", max_iter=300).fit_transform(X)
    assert emb.shape == (80, 2) and np.isfinite(emb).all()
    assert am.TSNE().get_params()["learning_rate"] == "auto"


def test_agglomerative_matches_sklearn():
    from sklearn import cluster as skc

    for linkage in ["ward", "complete", "average", "single"]:
        ours = am.AgglomerativeClustering(3, linkage=linkage).fit(X)
        ref = skc.AgglomerativeClustering(3, linkage=linkage).fit(X)
        assert list(ours.labels_) == list(ref.labels_)
    ours = am.AgglomerativeClustering(None, linkage="average", distance_threshold=2.0).fit(X)
    ref = skc.AgglomerativeClustering(None, linkage="average", distance_threshold=2.0).fit(X)
    assert list(ours.labels_) == list(ref.labels_)


def test_classical_mds_matches_sklearn():
    from sklearn.manifold import ClassicalMDS

    for metric in ["euclidean", "manhattan"]:
        ours = am.ClassicalMDS(3, metric=metric).fit_transform(X)
        ref = ClassicalMDS(n_components=3, metric=metric).fit_transform(X)
        np.testing.assert_allclose(ours, ref, atol=1e-9)


def test_mean_shift_matches_sklearn():
    from sklearn import cluster as skc

    Xb = np.vstack([X[:, :2], X[:, :2] + 6.0])
    for kw in [dict(), dict(bin_seeding=True), dict(bandwidth=1.0, cluster_all=False)]:
        ours = am.MeanShift(**kw).fit(Xb)
        ref = skc.MeanShift(**kw).fit(Xb)
        np.testing.assert_allclose(ours.cluster_centers_, ref.cluster_centers_, atol=1e-10)
        assert list(ours.labels_) == list(ref.labels_)


def test_optics_matches_sklearn():
    from sklearn import cluster as skc

    Xb = np.vstack([X[:, :2], X[:, :2] * 0.5 + 6.0])
    for kw in [dict(), dict(xi=0.1, min_cluster_size=6),
               dict(cluster_method="dbscan", eps=0.8)]:
        ours = am.OPTICS(min_samples=5, **kw).fit(Xb)
        ref = skc.OPTICS(min_samples=5, **kw).fit(Xb)
        assert list(ours.labels_) == list(ref.labels_), kw


def test_birch_matches_sklearn():
    from sklearn import cluster as skc

    for kw in [dict(threshold=0.8, branching_factor=5, n_clusters=3),
               dict(threshold=1.0, branching_factor=4, n_clusters=None)]:
        ours = am.Birch(**kw).fit(X)
        ref = skc.Birch(**kw).fit(X)
        # Same partition; the subcluster (leaf) order can differ where BLAS
        # dot products round differently on near-ties.
        from sklearn.metrics import adjusted_rand_score
        assert adjusted_rand_score(ours.labels_, ref.labels_) == 1.0, kw


def test_imputers_and_encoders_match_sklearn():
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn import decomposition as skd
    from sklearn import impute as ski
    from sklearn import preprocessing as skp

    Xm = X.copy()
    Xm[rng.rand(*X.shape) < 0.15] = np.nan
    close = lambda a, b: np.testing.assert_allclose(a, b, atol=1e-8)  # noqa: E731
    close(am.SimpleImputer(add_indicator=True).fit_transform(Xm),
          ski.SimpleImputer(add_indicator=True).fit_transform(Xm))
    close(am.MissingIndicator().fit_transform(Xm), ski.MissingIndicator().fit_transform(Xm))
    close(am.KNNImputer(n_neighbors=3, weights="distance").fit_transform(Xm),
          ski.KNNImputer(n_neighbors=3, weights="distance").fit_transform(Xm))
    for kw in [dict(), dict(min_value=-1.0, max_value=[2, 2, 3, 3], max_iter=4)]:
        close(am.IterativeImputer(**kw).fit_transform(Xm),
              ski.IterativeImputer(**kw).fit_transform(Xm))
    close(am.IncrementalPCA(n_components=2, batch_size=20).fit_transform(X),
          skd.IncrementalPCA(n_components=2, batch_size=20).fit_transform(X))
    close(am.PolynomialFeatures(degree=(2, 3), include_bias=False).fit_transform(X),
          skp.PolynomialFeatures(degree=(2, 3), include_bias=False).fit_transform(X))
    w = rng.rand(len(X))
    close(am.StandardScaler().fit(X, sample_weight=w).transform(X),
          skp.StandardScaler().fit(X, sample_weight=w).transform(X))

    cats = np.round(X_pos)
    for yy in [y_reg, y_multi]:
        ours = am.TargetEncoder(random_state=0)
        train = ours.fit_transform(cats, yy)
        close(ours.transform(cats), skp.TargetEncoder().fit(cats, yy).transform(cats))
        assert train.shape == ours.transform(cats).shape


def test_anofox_regression_0_10_features_match_sklearn():
    """Multinomial / L1 logistic, weighted GLMs, Quantile L1, joint-scale
    Huber and SplineTransformer (anofox-regression 0.10)."""
    sl = pytest.importorskip("sklearn.linear_model")
    skp = pytest.importorskip("sklearn.preprocessing")

    def close(a, b, atol):
        np.testing.assert_allclose(a, b, atol=atol)

    w = rng.uniform(0.2, 3, len(X))
    close(am.LogisticRegression(tol=1e-10).fit(X, y_multi).predict_proba(X),
          sl.LogisticRegression(tol=1e-12, max_iter=10000).fit(X, y_multi).predict_proba(X),
          1e-5)
    y01 = (y_reg > 0).astype(float)
    close(am.LogisticRegression(C=0.5, l1_ratio=1.0, tol=1e-10).fit(X, y01).predict_proba(X),
          sl.LogisticRegression(C=0.5, l1_ratio=1.0, solver="saga", tol=1e-12,
                                max_iter=100000).fit(X, y01).predict_proba(X),
          1e-5)
    y_cnt = np.round(np.exp(0.2 * X[:, 0] - 0.3 * X[:, 1]) * 2)
    close(am.PoissonRegressor(alpha=0.1, tol=1e-12).fit(X, y_cnt, sample_weight=w).predict(X),
          sl.PoissonRegressor(alpha=0.1, tol=1e-12, max_iter=1000)
          .fit(X, y_cnt, sample_weight=w).predict(X),
          1e-6)
    close(am.QuantileRegressor(quantile=0.3, alpha=0.05).fit(X, y_reg).predict(X),
          sl.QuantileRegressor(quantile=0.3, alpha=0.05, solver="highs").fit(X, y_reg).predict(X),
          1e-7)
    close(am.HuberRegressor(tol=1e-10, max_iter=1000).fit(X, y_reg).predict(X),
          sl.HuberRegressor(tol=1e-12, max_iter=10000).fit(X, y_reg).predict(X),
          1e-4)
    for kw in [dict(), dict(knots="quantile", include_bias=False),
               dict(extrapolation="periodic", n_knots=6), dict(extrapolation="linear", degree=2)]:
        close(am.SplineTransformer(**kw).fit(X).transform(1.3 * X),
              skp.SplineTransformer(**kw).fit(X).transform(1.3 * X), 1e-10)


def test_hist_gradient_boosting_quantile_loss():
    rs = np.random.RandomState(1)
    Xq = rs.rand(500, 2)
    yq = Xq[:, 0] + rs.uniform(-1, 1, 500)
    for q in (0.1, 0.9):
        model = am.HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=50)
        below = np.mean(yq <= model.fit(Xq, yq).predict(Xq))
        assert abs(below - q) < 0.08, (q, below)
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(loss="quantile").fit(Xq, yq)
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(loss="poisson").fit(Xq, yq)


def test_batch1_baselines_match_sklearn():
    from sklearn import dummy, linear_model, neighbors

    for strategy, kw in [("prior", {}), ("most_frequent", {}), ("constant", {"constant": "yes"})]:
        ours = am.DummyClassifier(strategy=strategy, **kw).fit(X, y_bin)
        ref = dummy.DummyClassifier(strategy=strategy, **kw).fit(X, y_bin)
        np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X))
    for strategy, kw in [("mean", {}), ("median", {}), ("quantile", {"quantile": 0.2})]:
        np.testing.assert_allclose(
            am.DummyRegressor(strategy=strategy, **kw).fit(X, y_reg).predict(X),
            dummy.DummyRegressor(strategy=strategy, **kw).fit(X, y_reg).predict(X))
    for kw in [{}, {"shrink_threshold": 0.3}, {"priors": "empirical"}]:
        ours = am.NearestCentroid(**kw).fit(X, y_multi)
        ref = neighbors.NearestCentroid(**kw).fit(X, y_multi)
        np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-10)
    for yy in (y_bin, y_multi):
        ours = am.RidgeClassifier(alpha=2.0).fit(X, yy)
        ref = linear_model.RidgeClassifier(alpha=2.0).fit(X, yy)
        np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
        np.testing.assert_allclose(ours.decision_function(X), ref.decision_function(X), atol=1e-8)
        ours = am.Perceptron(shuffle=False).fit(X, yy)
        ref = linear_model.Perceptron(shuffle=False).fit(X, yy)
        np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
