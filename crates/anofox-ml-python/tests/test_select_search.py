import pickle
import warnings

import numpy as np
import pytest
import sklearn.feature_selection as skfs
import sklearn.linear_model as skl
import sklearn.model_selection as skms
import sklearn.pipeline as skp
import sklearn.preprocessing as skpp
from scipy import stats

import anofox_ml_python as am

rng = np.random.RandomState(0)
X = rng.randn(90, 6) * [1, 10, 0.1, 3, 1, 2]
y = X @ [1.0, 0.0, 5.0, -0.5, 0.0, 0.3] + rng.randn(90) * 0.1
labels = np.where(X[:, 0] + X[:, 3] / 3 > 0, "hi", "lo")
labels3 = np.asarray(["a", "b", "c"])[(X[:, 0] > 0).astype(int) + (X[:, 3] > 1)]


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


# -- encoders -----------------------------------------------------------------


@pytest.mark.parametrize("labs", [
    np.asarray(["b", "a", "c", "a"]),
    np.asarray([10, 2, 2, 7]),
    np.asarray([2.5, -1.0, 2.5, 0.0]),
])
def test_label_encoder_matches_sklearn(labs):
    ours, ref = am.LabelEncoder().fit(labs), skpp.LabelEncoder().fit(labs)
    np.testing.assert_array_equal(ours.classes_, ref.classes_)
    assert ours.classes_.dtype == ref.classes_.dtype
    np.testing.assert_array_equal(ours.transform(labs), ref.transform(labs))
    np.testing.assert_array_equal(ours.inverse_transform([2, 0]), ref.inverse_transform([2, 0]))
    np.testing.assert_array_equal(ours.fit_transform(labs), ref.fit_transform(labs))
    with pytest.raises(ValueError, match="unseen"):
        ours.transform(np.asarray([99]).astype(labs.dtype))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(ours)).transform(labs),
                                  ref.transform(labs))


def test_ordinal_encoder_numeric_and_missing_matches_sklearn():
    nan = np.nan
    Xn = np.asarray([[10.0, 2.0], [2.0, nan], [10.0, 3.0], [nan, 2.0]])
    kw = dict(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
    ours, ref = am.OrdinalEncoder(**kw).fit(Xn), skpp.OrdinalEncoder(**kw).fit(Xn)
    for a, b in zip(ours.categories_, ref.categories_):
        np.testing.assert_array_equal(a, b)
    test = np.asarray([[2.0, 3.0], [5.0, nan], [nan, 7.0]])
    np.testing.assert_array_equal(ours.transform(test), ref.transform(test))
    codes = np.asarray([[0, 1], [-1, -2], [-2, 0]], dtype=float)
    got, want = ours.inverse_transform(codes), ref.inverse_transform(codes)
    assert got.shape == want.shape
    for g, w in zip(got.ravel(), want.ravel()):
        assert (g is None and w is None) or g == w or (np.isnan(g) and np.isnan(w))
    # Default: missing values stay NaN, unknown ones raise.
    ours, ref = am.OrdinalEncoder().fit(Xn), skpp.OrdinalEncoder().fit(Xn)
    np.testing.assert_array_equal(ours.transform(Xn), ref.transform(Xn))
    with pytest.raises(ValueError):
        ours.transform([[5.0, 2.0]])


def test_ordinal_encoder_strings_match_sklearn():
    Xs = np.asarray([["red", "s"], ["blue", "m"], ["green", "s"], ["red", None]], dtype=object)
    kw = dict(handle_unknown="use_encoded_value", unknown_value=np.nan)
    ours, ref = am.OrdinalEncoder(**kw).fit(Xs), skpp.OrdinalEncoder(**kw).fit(Xs)
    for a, b in zip(ours.categories_, ref.categories_):  # None sorts last, as a category
        np.testing.assert_array_equal(a, b)
    test = np.asarray([["green", "m"], ["pink", "s"]], dtype=object)
    np.testing.assert_array_equal(ours.transform(test), ref.transform(test))
    np.testing.assert_array_equal(ours.transform(Xs), ref.transform(Xs))
    np.testing.assert_array_equal(ours.inverse_transform([[2, 2]]), ref.inverse_transform([[2, 2]]))
    Xm = np.asarray([["a"], [np.nan], ["b"]], dtype=object)  # NaN is the missing value
    ours, ref = am.OrdinalEncoder().fit(Xm), skpp.OrdinalEncoder().fit(Xm)
    np.testing.assert_array_equal(ours.transform(Xm), ref.transform(Xm))
    np.testing.assert_array_equal(ours.categories_[0][:2], ref.categories_[0][:2])
    assert np.isnan(ours.categories_[0][2]) and np.isnan(ref.categories_[0][2])
    with pytest.raises(TypeError):
        am.OrdinalEncoder(unknown_value=-1).fit(Xs)


# -- SelectFromModel ------------------------------------------------------------


@pytest.mark.parametrize("make, kw", [
    (lambda m: m.Ridge(alpha=1.0), {}),
    (lambda m: m.Lasso(alpha=0.1), {}),
    (lambda m: m.Ridge(alpha=1.0), {"threshold": "median"}),
    (lambda m: m.Ridge(alpha=1.0), {"threshold": "0.5*mean"}),
    (lambda m: m.Ridge(alpha=1.0), {"threshold": 0.2}),
    (lambda m: m.Ridge(alpha=1.0), {"threshold": -np.inf, "max_features": 2}),
])
def test_select_from_model_regression_matches_sklearn(make, kw):
    ours = am.SelectFromModel(make(am), **kw).fit(X, y)
    ref = skfs.SelectFromModel(make(skl), **kw).fit(X, y)
    np.testing.assert_array_equal(ours.get_support(), ref.get_support())
    close(ours.transform(X), ref.transform(X))
    if np.isfinite(ref.threshold_):
        close(ours.threshold_, ref.threshold_)


def test_select_from_model_classifier_and_forest():
    ours = am.SelectFromModel(am.LogisticRegression(), max_features=3).fit(X, labels3)
    ref = skfs.SelectFromModel(skl.LogisticRegression(), max_features=3).fit(X, labels3)
    np.testing.assert_array_equal(ours.get_support(), ref.get_support())
    # Same importances as the fitted Rust forest.
    forest = am.RandomForestRegressor(n_estimators=20, random_state=0)
    sel = am.SelectFromModel(forest).fit(X, y)
    imp = forest.fit(X, y).feature_importances_
    np.testing.assert_array_equal(sel.get_support(), imp >= imp.mean())
    assert sel.transform(X).shape[1] == sel.get_support().sum()
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(sel)).transform(X), sel.transform(X))
    with pytest.raises(ValueError, match="prefit"):
        am.SelectFromModel(am.Ridge(), prefit=True).fit(X, y)


# -- FunctionTransformer ----------------------------------------------------------


def test_function_transformer_matches_sklearn():
    ours = am.FunctionTransformer(np.log1p, np.expm1, feature_names_out="one-to-one")
    ref = skpp.FunctionTransformer(np.log1p, np.expm1, feature_names_out="one-to-one")
    Xp = np.abs(X)
    close(ours.fit_transform(Xp), ref.fit_transform(Xp))
    close(ours.inverse_transform(ours.transform(Xp)), Xp)
    np.testing.assert_array_equal(ours.get_feature_names_out(), ref.get_feature_names_out())
    ident = am.FunctionTransformer().fit(X)
    assert ident.transform(X) is X
    with pytest.warns(UserWarning, match="not strictly inverse"):
        am.FunctionTransformer(np.log1p, np.exp).fit(Xp)
    kw = am.FunctionTransformer(np.round, kw_args={"decimals": 1}).fit(X)
    close(kw.transform(X), np.round(X, 1))
    # It works as a step of sklearn's Pipeline.
    pipe = skp.make_pipeline(am.FunctionTransformer(np.abs), am.Ridge()).fit(X, y)
    close(pipe.predict(X), am.Ridge().fit(np.abs(X), y).predict(np.abs(X)))


# -- splitters and parameter sets -----------------------------------------------------


def test_stratified_kfold_and_samplers_match_sklearn():
    for shuffle, rs in [(False, None), (True, 3)]:
        ours = am.StratifiedKFold(4, shuffle=shuffle, random_state=rs).split(X, labels3)
        ref = skms.StratifiedKFold(4, shuffle=shuffle, random_state=rs).split(X, labels3)
        for (a, b), (c, d) in zip(ours, ref):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)
    grid = [{"a": [1, 2, 3], "b": ["x", "y"]}, {}, {"c": [0.5]}]
    assert list(am.ParameterGrid(grid)) == list(skms.ParameterGrid(grid))
    g = am.ParameterGrid(grid)
    assert [g[i] for i in range(len(g))] == list(g)
    lists = {"a": list(range(10)), "b": list(range(30))}
    for n_iter in (2, 50, 299):  # tracking selection, permutation, reservoir
        assert list(am.ParameterSampler(lists, n_iter, random_state=1)) == \
            list(skms.ParameterSampler(lists, n_iter, random_state=1))
    dists = [{"alpha": stats.loguniform(1e-3, 10), "fit_intercept": [True, False]},
             {"alpha": stats.uniform(0, 1)}]
    assert list(am.ParameterSampler(dists, 7, random_state=5)) == \
        list(skms.ParameterSampler(dists, 7, random_state=5))


# -- GridSearchCV / RandomizedSearchCV ---------------------------------------------------


def _same_search(ours, ref):
    """Our search against sklearn's search of the same (sklearn-compatible)
    anofox estimator: the scores only differ by how the folds are scored."""
    for key in ["mean_test_score", "std_test_score", "rank_test_score"] + \
            [f"split{k}_test_score" for k in range(ref.n_splits_)]:
        close(ours.cv_results_[key], ref.cv_results_[key])
    assert ours.cv_results_["params"] == ref.cv_results_["params"]
    assert ours.best_index_ == ref.best_index_
    assert ours.best_params_ == ref.best_params_
    close(ours.best_score_, ref.best_score_)


@pytest.mark.parametrize("scoring", [None, "neg_mean_squared_error",
                                     "neg_mean_absolute_error", "neg_root_mean_squared_error"])
def test_grid_search_regressor_matches_sklearn(scoring):
    grid = {"alpha": [0.01, 1.0, 100.0], "fit_intercept": [True, False]}
    ours = am.GridSearchCV(am.Ridge(), grid, scoring=scoring, cv=4).fit(X, y)
    ref = skms.GridSearchCV(am.Ridge(), grid, scoring=scoring, cv=4).fit(X, y)
    _same_search(ours, ref)
    close(ours.predict(X), ref.predict(X))
    close(ours.score(X, y), ref.score(X, y))
    np.testing.assert_array_equal(ours.cv_results_["param_alpha"],
                                  ref.cv_results_["param_alpha"])
    assert ours.cv_results_["mean_fit_time"].shape == (6,)


@pytest.mark.parametrize("scoring", [None, "neg_log_loss", "f1_macro", "balanced_accuracy"])
def test_grid_search_classifier_matches_sklearn(scoring):
    grid = {"C": [0.01, 1.0, 10.0]}
    ours = am.GridSearchCV(am.LogisticRegression(), grid, scoring=scoring).fit(X, labels3)
    ref = skms.GridSearchCV(am.LogisticRegression(), grid, scoring=scoring).fit(X, labels3)
    _same_search(ours, ref)
    np.testing.assert_array_equal(ours.classes_, ref.classes_)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    close(ours.predict_proba(X), ref.predict_proba(X))


def test_grid_search_roc_auc_and_pipeline_and_list_grid():
    ours = am.GridSearchCV(am.LogisticRegression(), {"C": [0.1, 1.0]}, scoring="roc_auc",
                           cv=3).fit(X, labels)
    ref = skms.GridSearchCV(am.LogisticRegression(), {"C": [0.1, 1.0]}, scoring="roc_auc",
                            cv=3).fit(X, labels)
    _same_search(ours, ref)
    # Nested names reach into our Pipeline; the folds are the same as sklearn's
    # search of sklearn's pipeline.
    grid = [{"pca__n_components": [2, 4], "r__alpha": [0.1, 10.0]}, {"r__alpha": [1.0]}]
    ours = am.GridSearchCV(am.Pipeline([("s", am.StandardScaler()), ("pca", am.PCA()),
                                        ("r", am.Ridge())]), grid, cv=3).fit(X, y)
    ref = skms.GridSearchCV(skp.Pipeline([("s", skpp.StandardScaler()),
                                          ("pca", __import__("sklearn.decomposition").decomposition.PCA()),
                                          ("r", skl.Ridge())]), grid, cv=3).fit(X, y)
    _same_search(ours, ref)
    assert ours.best_estimator_.named_steps["r"].alpha == ref.best_params_["r__alpha"]


def test_randomized_search_matches_sklearn():
    dists = {"alpha": stats.loguniform(1e-3, 100)}
    ours = am.RandomizedSearchCV(am.Ridge(), dists, n_iter=6, cv=3, random_state=0).fit(X, y)
    ref = skms.RandomizedSearchCV(am.Ridge(), dists, n_iter=6, cv=3, random_state=0).fit(X, y)
    _same_search(ours, ref)
    lists = {"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = am.RandomizedSearchCV(am.Ridge(), lists, n_iter=4, random_state=2).fit(X, y)
        ref = skms.RandomizedSearchCV(am.Ridge(), lists, n_iter=4, random_state=2).fit(X, y)
    _same_search(ours, ref)


def test_search_errors_and_weights():
    with pytest.warns(UserWarning, match="failed"):
        s = am.GridSearchCV(am.Ridge(), {"alpha": [1.0, -1.0]}, cv=3).fit(X, y)
    assert np.isnan(s.cv_results_["mean_test_score"][1])
    assert s.cv_results_["rank_test_score"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="failed"):
        am.GridSearchCV(am.Ridge(), {"alpha": [-1.0]}, cv=3, error_score="raise").fit(X, y)
    with pytest.raises(ValueError, match="Invalid parameter"):
        am.GridSearchCV(am.Ridge(), {"nope": [1]}).fit(X, y)
    with pytest.raises(ValueError, match="scoring"):
        am.GridSearchCV(am.Ridge(), {"alpha": [1.0]}, scoring=len).fit(X, y)
    w = np.linspace(0.5, 2.0, len(y))
    grid = {"alpha": [0.1, 10.0]}
    ours = am.GridSearchCV(am.Ridge(), grid, cv=3).fit(X, y, sample_weight=w)
    ref = skms.GridSearchCV(am.Ridge(), grid, cv=3).fit(X, y, sample_weight=w)
    _same_search(ours, ref)
    no_refit = am.GridSearchCV(am.Ridge(), grid, cv=3, refit=False).fit(X, y)
    with pytest.raises(AttributeError):
        no_refit.predict(X)
