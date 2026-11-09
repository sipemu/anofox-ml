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
from anofox_ml_python._base import SupervisedTransformerMixin

rng = np.random.RandomState(0)
X = rng.randn(80, 4)
X_pos = np.abs(X) + 0.1
y_reg = X @ np.array([1.5, -2.0, 0.5, 0.0]) + 0.1 * rng.randn(80)
y_bin = np.where(y_reg > 0, "yes", "no")
y_multi = np.digitize(y_reg, [-1.0, 1.0])
Y_two = np.column_stack([y_reg, X[:, 1] - X[:, 2] + 0.1 * rng.randn(80)])

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
    "GenericUnivariateSelect": dict(y=y_multi, params=dict(mode="k_best", param=2)),
    "SelectPercentile": dict(y=y_multi, params=dict(percentile=50)),
    "SelectFpr": dict(y=y_multi),
    "SelectFdr": dict(y=y_multi),
    "SelectFwe": dict(y=y_multi),
    "TfidfTransformer": dict(X=np.round(X_pos)),
    "KernelCenterer": dict(X=X[:4] @ X[:4].T),
    "GaussianRandomProjection": dict(params=dict(n_components=2)),
    "SparseRandomProjection": dict(params=dict(n_components=2)),
    "BinomialRegressor": dict(y=1 / (1 + np.exp(-y_reg))),
    "NeighborhoodComponentsAnalysis": dict(y=y_multi),
    "BernoulliRBM": dict(X=(X > 0).astype(float), params=dict(n_components=3,
                                                                 random_state=0)),
    "LatentDirichletAllocation": dict(X=np.round(X_pos * 3),
                                      params=dict(n_components=3, random_state=0)),
    "BisectingKMeans": dict(params=dict(n_clusters=3, random_state=0)),
    "MiniBatchNMF": dict(X=X_pos, params=dict(n_components=2, batch_size=16)),
    "RandomTreesEmbedding": dict(params=dict(n_estimators=4, random_state=0)),
    "OneHotEncoder": dict(X=np.round(X_pos)),
    "FeatureHasher": dict(X=[{"a": 1.0, "b": 2.0}, {"c": "x"}, {"a": -1.0}, {}],
                          params=dict(n_features=16)),
    "CountVectorizer": dict(X=["a fox", "the dog", "the fox ran", "dogs run"]),
    "TfidfVectorizer": dict(X=["a fox", "the dog", "the fox ran", "dogs run"]),
    "HashingVectorizer": dict(X=["a fox", "the dog", "the fox ran", "dogs run"],
                              params=dict(n_features=32)),
    "IsotonicRegression": dict(X=X[:, :1]),
    "CCA": dict(y=Y_two),
    "PLSCanonical": dict(y=Y_two),
    "PLSSVD": dict(y=Y_two),
    "MultiTaskLasso": dict(y=Y_two, params=dict(alpha=0.1)),
    "MultiTaskElasticNet": dict(y=Y_two, params=dict(alpha=0.1)),
    "MultiTaskLassoCV": dict(y=Y_two, params=dict(alphas=10)),
    "MultiTaskElasticNetCV": dict(y=Y_two, params=dict(alphas=10)),
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
        assert pred.shape == (len(Xd),) + np.shape(yd)[1:]
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
    elif name == "KernelDensity":
        out = est.fit(Xd).score_samples(Xd)
        assert out.shape == (len(Xd),)
        assert est.sample(5, random_state=0).shape == (5, Xd.shape[1])
    elif hasattr(est, "mahalanobis"):  # covariance estimators
        out = est.fit(Xd).mahalanobis(Xd)
        assert out.shape == (len(Xd),)
        assert est.covariance_.shape == (Xd.shape[1], Xd.shape[1])
    else:  # transformer
        supervised = isinstance(est, SupervisedTransformerMixin) or name in (
            "SelectKBest", "MutualInformationSelector", "TargetEncoder", "CCA", "PLSCanonical",
            "PLSSVD")
        out = est.fit_transform(Xd, yd) if supervised else est.fit_transform(Xd)
        if isinstance(out, tuple):  # (x_scores, y_scores)
            out = out[0]
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
        am.HistGradientBoostingRegressor(loss="huber").fit(Xq, yq)


def test_hist_gradient_boosting_categorical_matches_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier as SkC
    from sklearn.ensemble import HistGradientBoostingRegressor as SkR

    rs = np.random.RandomState(7)
    cat = rs.choice([-2.0, 0.5, 3.0, 8.0, 40.0], size=300)
    cat[::23] = np.nan
    X = np.column_stack([rs.normal(size=300), cat])
    effect = {-2.0: 1.0, 0.5: -1.0, 3.0: 0.5, 8.0: 2.0, 40.0: -0.5}
    y = np.array([effect.get(c, 0.0) for c in cat]) + 0.3 * X[:, 0] + rs.normal(scale=0.2, size=300)
    Xt = X[::5].copy()
    Xt[::3, 1] = 99.0  # unseen category
    kw = dict(max_iter=20, max_leaf_nodes=8, min_samples_leaf=10)
    for cf in ([1], [False, True]):
        ours = am.HistGradientBoostingRegressor(categorical_features=cf, **kw).fit(X, y)
        ref = SkR(categorical_features=cf, early_stopping=False, **kw).fit(X, y)
        np.testing.assert_allclose(ours.predict(Xt), ref.predict(Xt), rtol=1e-10)
    yc = (y > 0.3).astype(int)
    ours = am.HistGradientBoostingClassifier(categorical_features=[1], **kw).fit(X, yc)
    ref = SkC(categorical_features=[1], early_stopping=False, **kw).fit(X, yc)
    np.testing.assert_allclose(ours.predict_proba(Xt), ref.predict_proba(Xt), rtol=1e-10)
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(categorical_features=[2]).fit(X, y)
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(categorical_features=[0], max_bins=10).fit(X, y)


def test_hist_gradient_boosting_constraints_match_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier as SkC
    from sklearn.ensemble import HistGradientBoostingRegressor as SkR

    rs = np.random.RandomState(8)
    X = rs.normal(size=(300, 3))
    y = np.sin(2 * X[:, 0]) + X[:, 1] - 0.5 * X[:, 1] * X[:, 2] + rs.normal(scale=0.2, size=300)
    kw = dict(max_iter=20, max_leaf_nodes=8, min_samples_leaf=10)
    for extra in (dict(monotonic_cst=[1, 1, 0]), dict(interaction_cst=[[0], [1, 2]]),
                  dict(interaction_cst="pairwise"), dict(interaction_cst="no_interactions"),
                  dict(monotonic_cst=np.array([-1, 0, 1]), interaction_cst=[(0, 1)])):
        ours = am.HistGradientBoostingRegressor(**kw, **extra).fit(X, y).predict(X)
        ref = SkR(early_stopping=False, **kw, **extra).fit(X, y).predict(X)
        np.testing.assert_allclose(ours, ref, rtol=1e-10)
    yc = (y > 0).astype(int)
    ours = am.HistGradientBoostingClassifier(monotonic_cst=[1, 1, 0], **kw).fit(X, yc)
    ref = SkC(monotonic_cst=[1, 1, 0], early_stopping=False, **kw).fit(X, yc)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-10)
    with pytest.raises(ValueError):
        am.HistGradientBoostingClassifier(monotonic_cst=[1, 0, 0]).fit(X, np.digitize(y, [0, 1]))
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(interaction_cst=[[0, 5]]).fit(X, y)
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(interaction_cst="sometimes").fit(X, y)


def test_hist_gradient_boosting_early_stopping_matches_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier as SkC
    from sklearn.ensemble import HistGradientBoostingRegressor as SkR

    rs = np.random.RandomState(9)
    X = rs.normal(size=(400, 3))
    y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] + rs.normal(scale=0.5, size=400)
    kw = dict(max_iter=300, learning_rate=0.3, max_leaf_nodes=8, min_samples_leaf=10,
              early_stopping=True, n_iter_no_change=5)
    for extra in (dict(random_state=3), dict(validation_fraction=50, random_state=8),
                  dict(validation_fraction=None, tol=1e-3)):
        ours = am.HistGradientBoostingRegressor(**kw, **extra).fit(X, y)
        ref = SkR(**kw, **extra).fit(X, y)
        assert ours.n_iter_ == ref.n_iter_ < 300
        np.testing.assert_allclose(ours.train_score_, ref.train_score_, rtol=1e-10)
        np.testing.assert_allclose(ours.validation_score_, ref.validation_score_, rtol=1e-10)
        np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-10)
    yc = np.digitize(y, [-0.5, 0.5])
    ours = am.HistGradientBoostingClassifier(**kw, random_state=5).fit(X, yc)
    ref = SkC(**kw, random_state=5).fit(X, yc)
    assert ours.n_iter_ == ref.n_iter_
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-10)
    off = am.HistGradientBoostingRegressor(max_iter=20).fit(X, y)
    assert off.n_iter_ == 20 and off.train_score_.size == 0
    for scoring in (None, "r2", "neg_mean_squared_error"):
        ours = am.HistGradientBoostingRegressor(scoring=scoring, random_state=2, **kw).fit(X, y)
        ref = SkR(scoring=scoring, random_state=2, **kw).fit(X, y)
        assert ours.n_iter_ == ref.n_iter_
        np.testing.assert_allclose(ours.validation_score_, ref.validation_score_, rtol=1e-10)
    for scoring in (None, "neg_log_loss"):
        ours = am.HistGradientBoostingClassifier(scoring=scoring, random_state=6, **kw).fit(X, yc)
        ref = SkC(scoring=scoring, random_state=6, **kw).fit(X, yc)
        assert ours.n_iter_ == ref.n_iter_
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-10)
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(scoring="accuracy", early_stopping=True).fit(X, y)
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(scoring="roc_auc", early_stopping=True).fit(X, y)


def test_hist_gradient_boosting_max_features_class_weight_match_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier as SkC
    from sklearn.ensemble import HistGradientBoostingRegressor as SkR

    rs = np.random.RandomState(10)
    X = rs.normal(size=(300, 5))
    y = np.sin(2 * X[:, 0]) + X[:, 1] - 0.5 * X[:, 2] + rs.normal(scale=0.3, size=300)
    kw = dict(max_iter=20, max_leaf_nodes=8, min_samples_leaf=10)
    ours = am.HistGradientBoostingRegressor(max_features=0.4, random_state=3, **kw).fit(X, y)
    ref = SkR(max_features=0.4, random_state=3, early_stopping=False, **kw).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-10)
    codes = np.digitize(y, [0.0, 1.5])
    names = np.array(["a", "b", "c"])[codes]
    # sklearn applies a class_weight dict to the encoded labels, so the dict
    # case uses labels 0..2 (where both agree).
    for labels, extra in ((names, dict(class_weight="balanced")),
                          (codes, dict(class_weight={0: 3.0, 2: 0.5})),
                          (names, dict(class_weight="balanced", max_features=0.6,
                                       random_state=7))):
        ours = am.HistGradientBoostingClassifier(**kw, **extra).fit(X, labels)
        ref = SkC(early_stopping=False, **kw, **extra).fit(X, labels)
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-10)
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(max_features=0.0).fit(X, y)


def test_lgbm_bundling_matches_lightgbm():
    lgb = pytest.importorskip("lightgbm")
    rs = np.random.RandomState(13)
    level = rs.randint(0, 12, size=400)
    sparse = rs.normal(size=(400, 4)) * (rs.uniform(size=(400, 4)) < 0.1)
    X = np.column_stack([rs.normal(size=400), np.eye(12)[level], sparse])
    y = X[:, 0] + 0.2 * level + 2 * sparse.sum(axis=1) + rs.normal(scale=0.3, size=400)
    kw = dict(n_estimators=25, learning_rate=0.2, num_leaves=8, min_child_samples=5)
    ref = lgb.LGBMRegressor(**kw, deterministic=True, force_row_wise=True, verbose=-1,
                            n_jobs=1).fit(X, y)
    for enable_bundle in (True, False):
        ours = am.LGBMRegressor(**kw, enable_bundle=enable_bundle).fit(X, y)
        np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-8)


def test_lgbm_dart_matches_lightgbm():
    lgb = pytest.importorskip("lightgbm")
    rs = np.random.RandomState(12)
    X = rs.normal(size=(300, 3))
    y = np.sin(2 * X[:, 0]) + X[:, 1] + rs.normal(scale=0.3, size=300)
    kw = dict(boosting_type="dart", n_estimators=30, learning_rate=0.2, num_leaves=8,
              drop_rate=0.3, skip_drop=0.2, drop_seed=9)
    ref = lgb.LGBMRegressor(**kw, deterministic=True, force_row_wise=True, verbose=-1,
                            n_jobs=1).fit(X, y)
    ours = am.LGBMRegressor(**kw).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-8)
    yc = (y > 0).astype(int)
    ref = lgb.LGBMClassifier(**kw, deterministic=True, force_row_wise=True, verbose=-1,
                             n_jobs=1).fit(X, yc)
    ours = am.LGBMClassifier(**kw).fit(X, yc)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-8)


def test_hist_gradient_boosting_log_link_losses_match_sklearn():
    from sklearn.ensemble import HistGradientBoostingRegressor as SkHGB

    rs = np.random.RandomState(4)
    X = rs.normal(size=(300, 3))
    mu = np.exp(0.5 * X[:, 0] - 0.3 * X[:, 1])
    for loss, y in [("poisson", rs.poisson(mu).astype(float)), ("gamma", rs.gamma(2.0, mu / 2.0))]:
        kw = dict(loss=loss, max_iter=30, max_leaf_nodes=8, min_samples_leaf=10)
        ours = am.HistGradientBoostingRegressor(**kw).fit(X, y).predict(X)
        ref = SkHGB(early_stopping=False, **kw).fit(X, y).predict(X)
        np.testing.assert_allclose(ours, ref, rtol=1e-10)
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(loss="gamma").fit(X, X[:, 0])


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


def test_batch2_selection_and_tfidf_match_sklearn():
    from sklearn import feature_extraction, feature_selection as fs

    for ours, ref in [
        (am.SelectPercentile(percentile=50), fs.SelectPercentile(percentile=50)),
        (am.SelectFpr(alpha=0.01), fs.SelectFpr(alpha=0.01)),
        (am.SelectFdr(alpha=0.05), fs.SelectFdr(alpha=0.05)),
        (am.SelectFwe(alpha=0.05), fs.SelectFwe(alpha=0.05)),
        (am.GenericUnivariateSelect(mode="k_best", param=2),
         fs.GenericUnivariateSelect(mode="k_best", param=2)),
    ]:
        ours.fit(X, y_multi)
        ref.fit(X, y_multi)
        np.testing.assert_array_equal(ours.get_support(), ref.get_support())
        np.testing.assert_allclose(ours.transform(X), ref.transform(X))
    counts = np.round(X_pos * 2)
    for kw in [{}, {"norm": "l1", "sublinear_tf": True}, {"use_idf": False}]:
        np.testing.assert_allclose(
            am.TfidfTransformer(**kw).fit_transform(counts),
            feature_extraction.text.TfidfTransformer(**kw).fit_transform(counts).toarray(),
            atol=1e-12)


def test_batch4_covariance_matches_sklearn():
    from sklearn import covariance as cv

    rs = np.random.RandomState(3)
    Xc = rs.randn(150, 3) @ rs.randn(3, 3)
    Xc[:15] += 10.0
    for ours, ref in [(am.EmpiricalCovariance(), cv.EmpiricalCovariance()),
                      (am.ShrunkCovariance(shrinkage=0.2), cv.ShrunkCovariance(shrinkage=0.2)),
                      (am.LedoitWolf(), cv.LedoitWolf()), (am.OAS(), cv.OAS()),
                      (am.MinCovDet(random_state=0), cv.MinCovDet(random_state=0))]:
        ours.fit(Xc)
        ref.fit(Xc)
        np.testing.assert_allclose(ours.covariance_, ref.covariance_, atol=1e-9)
        np.testing.assert_allclose(ours.mahalanobis(Xc), ref.mahalanobis(Xc), rtol=1e-7)
        np.testing.assert_allclose(ours.score(Xc), ref.score(Xc), rtol=1e-9)
    env = am.EllipticEnvelope(random_state=0).fit(Xc)
    renv = cv.EllipticEnvelope(random_state=0).fit(Xc)
    np.testing.assert_array_equal(env.predict(Xc), renv.predict(Xc))
    np.testing.assert_allclose(env.decision_function(Xc), renv.decision_function(Xc), rtol=1e-7)


def test_batch5_kernel_approximation():
    from sklearn import kernel_approximation as ka, preprocessing
    from sklearn.metrics.pairwise import rbf_kernel

    K = rbf_kernel(X, X, gamma=0.2)
    np.testing.assert_allclose(am.KernelCenterer().fit_transform(K),
                               preprocessing.KernelCenterer().fit_transform(K), atol=1e-12)
    for kw in [dict(kernel="rbf", gamma=0.3), dict(kernel="poly", degree=2)]:
        Z = am.Nystroem(n_components=len(X), random_state=0, **kw).fit_transform(X)
        R = ka.Nystroem(n_components=len(X), random_state=0, **kw).fit_transform(X)
        np.testing.assert_allclose(Z @ Z.T, R @ R.T, atol=1e-8)
    Z = am.RBFSampler(gamma=0.2, n_components=5000, random_state=0).fit_transform(X)
    assert np.abs(Z @ Z.T - K).max() < 0.1


def test_batch6_density_and_factor_analysis_match_sklearn():
    from sklearn import decomposition, neighbors

    for kw in [dict(kernel="gaussian", bandwidth=0.7), dict(kernel="epanechnikov", bandwidth=2.0),
               dict(kernel="exponential", bandwidth="scott")]:
        np.testing.assert_allclose(am.KernelDensity(**kw).fit(X).score_samples(X),
                                   neighbors.KernelDensity(**kw).fit(X).score_samples(X),
                                   rtol=1e-10)
    for kw in [dict(n_components=2), dict(n_components=2, rotation="varimax")]:
        ours = am.FactorAnalysis(**kw).fit(X)
        ref = decomposition.FactorAnalysis(svd_method="lapack", **kw).fit(X)
        np.testing.assert_allclose(ours.get_covariance(), ref.get_covariance(), atol=1e-8)
        np.testing.assert_allclose(ours.score_samples(X), ref.score_samples(X), rtol=1e-8)
        np.testing.assert_allclose(np.abs(ours.components_), np.abs(ref.components_), atol=1e-7)


def test_batch7_cv_linear_models_match_sklearn():
    import warnings
    from sklearn import linear_model

    warnings.filterwarnings("ignore", category=FutureWarning)
    for yy in (y_bin, y_multi):
        ours = am.LogisticRegressionCV(tol=1e-10, max_iter=100000).fit(X, yy)
        ref = linear_model.LogisticRegressionCV(tol=1e-10, max_iter=100000, l1_ratios=(0.0,),
                                                use_legacy_attributes=False).fit(X, yy)
        assert abs(ours.C_ - float(np.ravel(ref.C_)[0])) < 1e-12
        np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    Xo = np.hstack([X, X ** 2, np.sin(X)])
    ours = am.OrthogonalMatchingPursuitCV().fit(Xo, y_reg)
    ref = linear_model.OrthogonalMatchingPursuitCV().fit(Xo, y_reg)
    assert ours.n_nonzero_coefs_ == ref.n_nonzero_coefs_
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-9)


def test_batch8_label_propagation_matches_sklearn():
    from sklearn import semi_supervised as ss

    yl = np.array(y_multi)
    yl[np.arange(len(yl)) % 3 != 0] = -1
    for name, kw in [("LabelPropagation", dict(gamma=0.5)), ("LabelSpreading", dict(kernel="knn"))]:
        ours = getattr(am, name)(**kw).fit(X, yl)
        ref = getattr(ss, name)(**kw).fit(X, yl)
        np.testing.assert_array_equal(ours.transduction_, ref.transduction_)
        np.testing.assert_allclose(ours.label_distributions_, ref.label_distributions_, atol=1e-10)
        np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def test_feature_agglomeration_matches_sklearn():
    from sklearn import cluster

    for linkage in ("ward", "average"):
        ours = am.FeatureAgglomeration(n_clusters=2, linkage=linkage).fit(X)
        ref = cluster.FeatureAgglomeration(n_clusters=2, linkage=linkage).fit(X)
        np.testing.assert_array_equal(ours.labels_, ref.labels_)
        np.testing.assert_allclose(ours.transform(X), ref.transform(X), atol=1e-12)
        np.testing.assert_allclose(ours.inverse_transform(ours.transform(X)),
                                   ref.inverse_transform(ref.transform(X)), atol=1e-12)


def test_kneighbors_transformer_matches_sklearn():
    from sklearn import neighbors

    Xt = X[:10] + 0.05
    for mode in ("distance", "connectivity"):
        ours = am.KNeighborsTransformer(mode=mode, n_neighbors=4).fit(X)
        ref = neighbors.KNeighborsTransformer(mode=mode, n_neighbors=4).fit(X)
        np.testing.assert_allclose(ours.transform(X), ref.transform(X).toarray(), atol=1e-12)
        np.testing.assert_allclose(ours.transform(Xt), ref.transform(Xt).toarray(), atol=1e-12)


def test_feature_importances_on_tree_and_boosting_estimators():
    for est in (am.DecisionTreeRegressor(), am.RandomForestRegressor(n_estimators=5),
                am.GradientBoostingRegressor(n_estimators=10),
                am.HistGradientBoostingRegressor(max_iter=10),
                am.LGBMRegressor(n_estimators=10, min_child_samples=5)):
        imp = est.fit(X, y_reg).feature_importances_
        assert imp.shape == (4,)
        assert abs(imp.sum() - 1.0) < 1e-9
        # The informative features carry the importance.
        assert imp[[0, 1]].sum() > imp[[2, 3]].sum(), type(est).__name__
    assert not hasattr(am.LinearRegression().fit(X, y_reg), "feature_importances_")
    with pytest.raises(AttributeError):
        am.GradientBoostingRegressor().feature_importances_


def test_staged_predict_ends_at_predict_and_matches_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier as SkHgb

    for est, y in ((am.GradientBoostingRegressor(n_estimators=8), y_reg),
                   (am.GradientBoostingClassifier(n_estimators=8), y_bin),
                   (am.HistGradientBoostingRegressor(max_iter=8), y_reg),
                   (am.LGBMClassifier(n_estimators=8, min_child_samples=5), y_multi)):
        est.fit(X, y)
        stages = list(est.staged_predict(X))
        assert len(stages) == 8
        np.testing.assert_array_equal(stages[-1], est.predict(X))
    ours = am.HistGradientBoostingClassifier(max_iter=6, min_samples_leaf=5).fit(X, y_bin)
    ref = SkHgb(max_iter=6, min_samples_leaf=5, early_stopping=False).fit(X, y_bin)
    for a, b in zip(ours.staged_predict(X), ref.staged_predict(X)):
        np.testing.assert_array_equal(a, b)
    assert not hasattr(am.RandomForestRegressor(), "staged_predict")


def test_gradient_boosting_warm_start_adds_stages():
    for cls, y in ((am.GradientBoostingRegressor, y_reg),
                   (am.GradientBoostingClassifier, y_bin)):
        warm = cls(n_estimators=5, subsample=0.7, random_state=3, warm_start=True).fit(X, y)
        warm.set_params(n_estimators=12).fit(X, y)
        once = cls(n_estimators=12, subsample=0.7, random_state=3).fit(X, y)
        assert len(list(warm.staged_predict(X))) == 12
        np.testing.assert_array_equal(warm.predict(X), once.predict(X))
        with pytest.raises(ValueError):
            warm.set_params(n_estimators=4).fit(X, y)
        # warm_start=False refits from scratch.
        cold = cls(n_estimators=5, random_state=3).fit(X, y)
        cold.set_params(n_estimators=7).fit(X, y)
        assert len(list(cold.staged_predict(X))) == 7


def test_hist_gradient_boosting_explicit_validation_set_matches_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier as SkC
    from sklearn.ensemble import HistGradientBoostingRegressor as SkR

    r = np.random.RandomState(5)
    Xt, Xv = r.randn(300, 4), r.randn(100, 4)
    f = lambda A: A[:, 0] - 0.5 * A[:, 1]  # noqa: E731
    yt, yv = f(Xt) + r.randn(300), f(Xv) + r.randn(100)
    kw = dict(max_iter=200, early_stopping=True, n_iter_no_change=3, min_samples_leaf=5)
    ours = am.HistGradientBoostingRegressor(**kw).fit(Xt, yt, X_val=Xv, y_val=yv)
    ref = SkR(**kw).fit(Xt, yt, X_val=Xv, y_val=yv)
    assert ours.n_iter_ == ref.n_iter_ < 200
    np.testing.assert_allclose(ours.predict(Xv), ref.predict(Xv), rtol=1e-12, atol=1e-12)
    labels = np.where(yt > 0, "a", "b")
    labels_v = np.where(yv > 0, "a", "b")
    ours = am.HistGradientBoostingClassifier(**kw).fit(Xt, labels, X_val=Xv, y_val=labels_v)
    ref = SkC(**kw).fit(Xt, labels, X_val=Xv, y_val=labels_v)
    assert ours.n_iter_ == ref.n_iter_
    np.testing.assert_allclose(ours.predict_proba(Xv), ref.predict_proba(Xv), rtol=1e-12,
                               atol=1e-12)
    with pytest.raises(ValueError):
        am.HistGradientBoostingClassifier(**kw).fit(Xt, labels, X_val=Xv,
                                                    y_val=np.full(100, "c"))
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(**kw).fit(Xt, yt, X_val=Xv)


def test_hist_gradient_boosting_warm_start_matches_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier as SkC
    from sklearn.ensemble import HistGradientBoostingRegressor as SkR

    for ours_cls, ref_cls, y in ((am.HistGradientBoostingRegressor, SkR, y_reg),
                                 (am.HistGradientBoostingClassifier, SkC, y_multi)):
        ours = ours_cls(max_iter=4, min_samples_leaf=5, warm_start=True).fit(X, y)
        ref = ref_cls(max_iter=4, min_samples_leaf=5, warm_start=True).fit(X, y)
        ours.set_params(max_iter=11).fit(X, y)
        ref.set_params(max_iter=11).fit(X, y)
        assert ours.n_iter_ == ref.n_iter_ == 11
        out = ours.predict_proba(X) if hasattr(ref, "predict_proba") else ours.predict(X)
        exp = ref.predict_proba(X) if hasattr(ref, "predict_proba") else ref.predict(X)
        np.testing.assert_allclose(out, exp, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        am.HistGradientBoostingRegressor(max_iter=4, warm_start=True, early_stopping=True,
                                         ).fit(X, y_reg).set_params(max_iter=8).fit(X, y_reg)


def test_catboost_matches_catboost():
    cb = pytest.importorskip("catboost")
    off = dict(random_strength=0, bootstrap_type="No", boosting_type="Plain", has_time=True,
               thread_count=1, verbose=0, allow_writing_files=False)
    kw = dict(iterations=25, learning_rate=0.2, depth=4)
    r = np.random.RandomState(9)
    Xc = r.randn(300, 4)
    Xc[r.rand(300) < 0.05, 1] = np.nan
    yc = np.sin(Xc[:, 0]) + np.nan_to_num(Xc[:, 1]) + 0.2 * r.randn(300)
    ref = cb.CatBoostRegressor(**kw, **off).fit(Xc, yc)
    ours = am.CatBoostRegressor(has_time=off.get("has_time", False),
                              random_strength=off.get("random_strength", 1.0),
                              bootstrap_type=off.get("bootstrap_type"), **kw).fit(Xc, yc)
    np.testing.assert_allclose(ours.predict(Xc), ref.predict(Xc), rtol=1e-12, atol=1e-12)
    off_ordered = dict(off, boosting_type="Ordered")
    ref = cb.CatBoostRegressor(**kw, **off_ordered).fit(Xc, yc)
    ours = am.CatBoostRegressor(has_time=off.get("has_time", False),
                              random_strength=off.get("random_strength", 1.0),
                              bootstrap_type=off.get("bootstrap_type"), **kw, boosting_type="Ordered").fit(Xc, yc)
    np.testing.assert_allclose(ours.predict(Xc), ref.predict(Xc), rtol=1e-12, atol=1e-12)
    labels = np.where(yc > 0, "pos", "neg")
    ref = cb.CatBoostClassifier(**kw, score_function="L2", **off).fit(Xc, labels)
    ours = am.CatBoostClassifier(has_time=off.get("has_time", False),
                              random_strength=off.get("random_strength", 1.0),
                              bootstrap_type=off.get("bootstrap_type"), **kw, score_function="L2").fit(Xc, labels)
    np.testing.assert_allclose(ours.predict_proba(Xc), ref.predict_proba(Xc), rtol=1e-12,
                               atol=1e-12)
    np.testing.assert_array_equal(ours.predict(Xc), ref.predict(Xc).ravel())


def test_catboost_categorical_features_match_catboost():
    cb = pytest.importorskip("catboost")
    off = dict(random_strength=0, bootstrap_type="No", boosting_type="Plain", has_time=True,
               thread_count=1, verbose=0, allow_writing_files=False, max_ctr_complexity=1)
    r = np.random.RandomState(4)
    n = 400
    city = r.choice(["berlin", "paris", "rome", "oslo", "lima", "kyiv", "baku"], size=n)
    flag = r.choice(["yes", "no"], size=n)
    num = r.randn(n)
    effect = {"berlin": 1.0, "paris": 0.5, "rome": -0.5, "oslo": -1.0, "lima": 0.2,
              "kyiv": 0.8, "baku": -0.3}
    y = (num + np.array([effect[c] for c in city]) + (flag == "yes") + r.logistic(size=n) * 0.5
         > 0.5).astype(int)
    X = np.empty((n, 3), dtype=object)
    X[:, 0], X[:, 1], X[:, 2] = num, city, flag
    kw = dict(iterations=30, learning_rate=0.2, depth=4, cat_features=[1, 2])
    ref = cb.CatBoostClassifier(**kw, **off).fit(X, y)
    ours = am.CatBoostClassifier(has_time=off.get("has_time", False),
                              random_strength=off.get("random_strength", 1.0),
                              bootstrap_type=off.get("bootstrap_type"), **kw).fit(X, y)
    X_new = X[:50].copy()
    X_new[0, 1] = "madrid"  # unseen
    np.testing.assert_allclose(ours.predict_proba(X_new), ref.predict_proba(X_new),
                               rtol=1e-12, atol=1e-12)
    yr = num + np.array([effect[c] for c in city]) + 0.3 * r.randn(n)
    ref = cb.CatBoostRegressor(**kw, **off).fit(X, yr)
    ours = am.CatBoostRegressor(has_time=off.get("has_time", False),
                              random_strength=off.get("random_strength", 1.0),
                              bootstrap_type=off.get("bootstrap_type"), **kw).fit(X, yr)
    np.testing.assert_allclose(ours.predict(X_new), ref.predict(X_new), rtol=1e-12, atol=1e-12)


def test_catboost_multiclass_matches_catboost():
    cb = pytest.importorskip("catboost")
    off = dict(random_strength=0, bootstrap_type="No", has_time=True, thread_count=1,
               verbose=0, allow_writing_files=False, max_ctr_complexity=1)
    r = np.random.RandomState(12)
    n = 300
    X = np.empty((n, 3), dtype=object)
    X[:, 0], X[:, 1] = r.randn(n), r.uniform(-2, 2, n)
    X[:, 2] = r.choice(["a", "b", "c", "d", "e"], size=n)
    eta = X[:, 0].astype(float) + 0.5 * X[:, 1].astype(float) + (X[:, 2] == "b")
    y = np.array(["low", "mid", "high"])[np.digitize(eta + r.logistic(size=n) * 0.5, [-0.5, 0.7])]
    kw = dict(iterations=25, learning_rate=0.3, depth=4, cat_features=[2])
    for boosting in ("Plain", "Ordered"):
        ref = cb.CatBoostClassifier(**kw, boosting_type=boosting, **off).fit(X, y)
        ours = am.CatBoostClassifier(has_time=off.get("has_time", False),
                              random_strength=off.get("random_strength", 1.0),
                              bootstrap_type=off.get("bootstrap_type"), **kw, boosting_type=boosting).fit(X, y)
        np.testing.assert_array_equal(ours.classes_, ref.classes_)
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-12,
                                   atol=1e-12)
        np.testing.assert_array_equal(ours.predict(X), ref.predict(X).ravel())
        raw = ref.predict(X, prediction_type="RawFormulaVal")
        np.testing.assert_allclose(ours.decision_function(X), raw, rtol=1e-12, atol=1e-12)


def test_catboost_ctr_combinations_match_catboost():
    cb = pytest.importorskip("catboost")
    off = dict(random_strength=0, bootstrap_type="No", has_time=True, thread_count=1,
               verbose=0, allow_writing_files=False)
    r = np.random.RandomState(21)
    n = 500
    X = np.empty((n, 4), dtype=object)
    X[:, 0] = r.randn(n)
    X[:, 1] = r.choice(["a", "b", "c", "d", "e", "f"], size=n)
    X[:, 2] = r.choice(["x", "y", "z", "w"], size=n)
    X[:, 3] = r.choice(["on", "off"], size=n)
    eta = X[:, 0].astype(float) + ((X[:, 1] == "a") ^ (X[:, 2] == "x")) + (X[:, 3] == "on")
    y = (eta + r.logistic(size=n) * 0.5 > 1.0).astype(int)
    kw = dict(iterations=40, learning_rate=0.2, depth=5, cat_features=[1, 2, 3],
              max_ctr_complexity=3)
    ref = cb.CatBoostClassifier(**kw, **off).fit(X, y)
    ours = am.CatBoostClassifier(has_time=off.get("has_time", False),
                              random_strength=off.get("random_strength", 1.0),
                              bootstrap_type=off.get("bootstrap_type"), **kw).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-12,
                               atol=1e-12)


def test_catboost_leaf_estimation_iterations_match_catboost():
    cb = pytest.importorskip("catboost")
    off = dict(random_strength=0, bootstrap_type="No", has_time=True, thread_count=1,
               verbose=0, allow_writing_files=False)
    r = np.random.RandomState(5)
    X = r.randn(600, 4)
    y = (X[:, 0] - X[:, 1] ** 2 + r.logistic(size=600) > 0).astype(int)
    # 200 iterations: CatBoost's Logloss default of 10 Newton steps.
    for kw in (dict(iterations=200, learning_rate=0.05, depth=4),
               dict(iterations=30, learning_rate=0.2, depth=4, leaf_estimation_iterations=5,
                    boosting_type="Ordered")):
        ref = cb.CatBoostClassifier(**kw, **off).fit(X, y)
        ours = am.CatBoostClassifier(has_time=off.get("has_time", False),
                              random_strength=off.get("random_strength", 1.0),
                              bootstrap_type=off.get("bootstrap_type"), **kw).fit(X, y)
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-12,
                                   atol=1e-12)


def test_catboost_permutations_match_catboost():
    cb = pytest.importorskip("catboost")
    off = dict(random_strength=0, bootstrap_type="No", thread_count=1, verbose=0,
               allow_writing_files=False)
    r = np.random.RandomState(8)
    n = 500
    X = np.empty((n, 3), dtype=object)
    X[:, 0], X[:, 1] = r.randn(n), r.randn(n)
    X[:, 2] = r.choice(list("abcdefgh"), size=n)
    y = X[:, 0].astype(float) + (X[:, 2] == "c") + 0.3 * r.randn(n)
    for kw in (dict(boosting_type="Plain", random_seed=3), dict(boosting_type="Ordered")):
        kw.update(iterations=30, learning_rate=0.2, depth=4, cat_features=[2])
        ref = cb.CatBoostRegressor(**kw, **off).fit(X, y)
        ours = am.CatBoostRegressor(has_time=off.get("has_time", False),
                              random_strength=off.get("random_strength", 1.0),
                              bootstrap_type=off.get("bootstrap_type"), **kw).fit(X, y)
        np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-12, atol=1e-12)


def test_catboost_score_noise_matches_catboost():
    cb = pytest.importorskip("catboost")
    off = dict(thread_count=1, verbose=0, allow_writing_files=False)
    r = np.random.RandomState(13)
    n = 500
    X = np.empty((n, 4), dtype=object)
    X[:, 0], X[:, 1] = r.randn(n), r.randn(n)
    X[:, 2] = r.choice(list("abcdefg"), size=n)
    X[:, 3] = r.choice(list("uvwxyz"), size=n)
    y = (X[:, 0].astype(float) + ((X[:, 2] == "a") ^ (X[:, 3] == "u")) + r.logistic(size=n)
         > 0.5).astype(int)
    kw = dict(iterations=40, learning_rate=0.2, depth=5, cat_features=[2, 3],
              max_ctr_complexity=3, random_strength=1.0, bootstrap_type="No")
    ref = cb.CatBoostClassifier(**kw, **off).fit(X, y)
    ours = am.CatBoostClassifier(**kw).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-12,
                               atol=1e-12)


def test_catboost_defaults_match_catboost():
    """CatBoost's defaults: MVS bootstrap, score noise, permutations."""
    cb = pytest.importorskip("catboost")
    off = dict(thread_count=1, verbose=0, allow_writing_files=False)
    r = np.random.RandomState(17)
    n = 600
    X = np.empty((n, 3), dtype=object)
    X[:, 0], X[:, 1] = r.randn(n), r.randn(n)
    X[:, 2] = r.choice(list("abcdefghij"), size=n)
    y = X[:, 0].astype(float) * 2 + (X[:, 2] == "d") + 0.3 * r.randn(n)
    kw = dict(iterations=60, learning_rate=0.1, cat_features=[2])
    ref = cb.CatBoostRegressor(**kw, **off).fit(X, y)
    ours = am.CatBoostRegressor(**kw).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-12, atol=1e-12)


def test_lgbm_sampling_matches_lightgbm():
    # LightGBM's own generator: bagging, GOSS and feature fractions (#152).
    lgb = pytest.importorskip("lightgbm")
    rs = np.random.RandomState(21)
    X = rs.normal(size=(500, 5))
    y = np.sin(2 * X[:, 0]) + X[:, 1] - 0.5 * X[:, 2] + rs.normal(scale=0.3, size=500)
    yc = (y > 0).astype(int)
    base = dict(n_estimators=25, learning_rate=0.2, num_leaves=8)
    cases = [
        dict(subsample=0.7, subsample_freq=2, colsample_bytree=0.8, random_state=3),
        dict(boosting_type="goss", feature_fraction_bynode=0.6),
        dict(subsample=0.8, subsample_freq=1, feature_fraction_bynode=0.7, random_state=9),
    ]
    ref_kw = dict(deterministic=True, force_row_wise=True, verbose=-1, n_jobs=1)
    for kw in cases:
        ref = lgb.LGBMRegressor(**base, **kw, **ref_kw).fit(X, y)
        ours = am.LGBMRegressor(**base, **kw).fit(X, y)
        np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-8)
        ref = lgb.LGBMClassifier(**base, **kw, **ref_kw).fit(X, yc)
        ours = am.LGBMClassifier(**base, **kw).fit(X, yc)
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-8)
    kw = dict(subsample_freq=1, pos_bagging_fraction=0.5, neg_bagging_fraction=0.8)
    ref = lgb.LGBMClassifier(**base, **kw, **ref_kw).fit(X, yc)
    ours = am.LGBMClassifier(**base, **kw).fit(X, yc)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-8)


def test_catboost_class_weights_and_monotone_match_catboost():
    cb = pytest.importorskip("catboost")
    rs = np.random.RandomState(31)
    X = rs.normal(size=(300, 4))
    eta = np.sin(2 * X[:, 0]) + X[:, 1] - 0.5 * X[:, 2]
    y = eta + rs.normal(scale=0.3, size=300)
    yc = (eta + rs.logistic(size=300) > 1.0).astype(int)
    kw = dict(iterations=30, learning_rate=0.2, depth=4, random_strength=0, bootstrap_type="No",
              has_time=True)
    ref_kw = dict(thread_count=1, verbose=0, allow_writing_files=False)
    for extra in (dict(auto_class_weights="Balanced"), dict(class_weights={0: 1.0, 1: 3.0})):
        ref = cb.CatBoostClassifier(**kw, **extra, **ref_kw).fit(X, yc)
        ours = am.CatBoostClassifier(**kw, **extra).fit(X, yc)
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-10)
    mono = dict(monotone_constraints=[1, 1, -1, 0])
    ref = cb.CatBoostRegressor(**kw, **mono, **ref_kw).fit(X, y)
    ours = am.CatBoostRegressor(**kw, **mono).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-10)


def test_catboost_regression_losses_match_catboost():
    cb = pytest.importorskip("catboost")
    rs = np.random.RandomState(41)
    X = rs.normal(size=(300, 4))
    eta = np.sin(2 * X[:, 0]) + X[:, 1] - 0.5 * X[:, 2]
    y = eta + rs.normal(scale=0.3, size=300)
    counts = rs.poisson(np.exp(0.5 * eta)).astype(float)
    kw = dict(iterations=30, learning_rate=0.2, depth=4, random_strength=0, bootstrap_type="No",
              has_time=True)
    ref_kw = dict(thread_count=1, verbose=0, allow_writing_files=False)
    for loss, target in (("MAE", y), ("Quantile:alpha=0.3", y), ("Expectile:alpha=0.7", y),
                         ("Poisson", counts), ("Tweedie:variance_power=1.5", counts)):
        ref = cb.CatBoostRegressor(loss_function=loss, **kw, **ref_kw).fit(X, target)
        ours = am.CatBoostRegressor(loss_function=loss, **kw).fit(X, target)
        np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-10, err_msg=loss)


def test_catboost_eval_set_matches_catboost():
    cb = pytest.importorskip("catboost")
    rs = np.random.RandomState(51)
    X = rs.normal(size=(400, 4))
    eta = np.sin(2 * X[:, 0]) + X[:, 1] - 0.5 * X[:, 2]
    y = eta + rs.normal(scale=0.4, size=400)
    labels = np.where(eta + rs.logistic(size=400) > 0, "yes", "no")
    tr, va = slice(0, 300), slice(300, None)
    kw = dict(iterations=200, learning_rate=0.4, depth=4, random_strength=0, bootstrap_type="No",
              has_time=True, early_stopping_rounds=5)
    ref_kw = dict(thread_count=1, verbose=0, allow_writing_files=False)
    ref = cb.CatBoostRegressor(**kw, **ref_kw).fit(X[tr], y[tr], eval_set=(X[va], y[va]))
    ours = am.CatBoostRegressor(**kw).fit(X[tr], y[tr], eval_set=(X[va], y[va]))
    assert ours.best_iteration_ == ref.get_best_iteration()
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-10)
    np.testing.assert_allclose(ours.evals_result_, ref.get_evals_result()["validation"]["RMSE"],
                               rtol=1e-10)
    kw["auto_class_weights"] = "Balanced"
    ref = cb.CatBoostClassifier(**kw, **ref_kw).fit(X[tr], labels[tr], eval_set=(X[va], labels[va]))
    ours = am.CatBoostClassifier(**kw).fit(X[tr], labels[tr], eval_set=(X[va], labels[va]))
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-10)
    explicit = am.CatBoostClassifier(iterations=5, class_weights={"no": 1.0, "yes": 2.0})
    explicit.fit(X[tr], labels[tr])


def test_catboost_grow_policies_match_catboost():
    cb = pytest.importorskip("catboost")
    rs = np.random.RandomState(61)
    X = rs.normal(size=(300, 4))
    y = np.sin(2 * X[:, 0]) + X[:, 1] + rs.normal(scale=0.3, size=300)
    kw = dict(iterations=30, learning_rate=0.2, depth=5, random_strength=0, bootstrap_type="No",
              has_time=True)
    ref_kw = dict(thread_count=1, verbose=0, allow_writing_files=False)
    for extra in (dict(grow_policy="Depthwise", min_data_in_leaf=5),
                  dict(grow_policy="Lossguide", max_leaves=10)):
        ref = cb.CatBoostRegressor(**kw, **extra, **ref_kw).fit(X, y)
        ours = am.CatBoostRegressor(**kw, **extra).fit(X, y)
        np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-10)
