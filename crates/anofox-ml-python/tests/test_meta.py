"""Parity of the meta-estimators (``anofox_ml_python.meta``) and generic
bagging with scikit-learn 1.8."""

import pickle

import numpy as np
import pytest

import anofox_ml_python as am

sk = pytest.importorskip("sklearn")
from sklearn import (  # noqa: E402
    calibration,
    compose,
    ensemble,
    feature_selection,
    linear_model,
    multioutput,
    preprocessing,
    tree,
)
import sklearn.neighbors  # noqa: E402,F401

rng = np.random.RandomState(0)
X = rng.randn(80, 5)
y = X @ np.array([1.5, -2.0, 0.5, 0.0, 0.2]) + 0.5 * rng.randn(80)
Y = np.column_stack([y, X[:, 1] - X[:, 2] + 0.3 * rng.randn(80), X[:, 3] + rng.randn(80)])
y_cls = np.where(y + rng.randn(80) > 0, "pos", "neg")
y3 = np.digitize(y + rng.randn(80), [-1.0, 1.0])


def _pickled(est):
    return pickle.loads(pickle.dumps(est))


def test_voting_regressor():
    ours = am.VotingRegressor([("ols", am.LinearRegression()), ("ridge", am.Ridge(alpha=5.0))],
                              weights=[1, 3]).fit(X, y)
    ref = ensemble.VotingRegressor([("ols", linear_model.LinearRegression()),
                                    ("ridge", linear_model.Ridge(alpha=5.0))],
                                   weights=[1, 3]).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-10)
    np.testing.assert_allclose(_pickled(ours).predict(X), ref.predict(X), atol=1e-10)


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_voting_classifier(voting):
    ests = [("lr", am.LogisticRegression()), ("nb", am.GaussianNB()),
            ("knn", am.KNeighborsClassifier())]
    tight = dict(tol=1e-12, max_iter=10_000)  # sklearn's lbfgs stops at tol=1e-4
    ref_ests = [("lr", linear_model.LogisticRegression(**tight)),
                ("nb", sk.naive_bayes.GaussianNB()),
                ("knn", sk.neighbors.KNeighborsClassifier())]
    ours = am.VotingClassifier(ests, voting=voting).fit(X, y3)
    ref = ensemble.VotingClassifier(ref_ests, voting=voting).fit(X, y3)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    if voting == "soft":
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-6)
    else:
        with pytest.raises(AttributeError):
            ours.predict_proba(X)


def test_stacking_regressor():
    ests = [("ols", am.LinearRegression()), ("tree", am.DecisionTreeRegressor(max_depth=3))]
    ref_ests = [("ols", linear_model.LinearRegression()),
                ("tree", tree.DecisionTreeRegressor(max_depth=3, random_state=0))]
    ours = am.StackingRegressor(ests, final_estimator=am.Ridge(alpha=1.0)).fit(X, y)
    ref = ensemble.StackingRegressor(ref_ests,
                                     final_estimator=linear_model.Ridge(alpha=1.0)).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-9)


def test_stacking_classifier():
    ests = [("nb", am.GaussianNB()), ("tree", am.DecisionTreeClassifier(max_depth=3))]
    ref_ests = [("nb", sk.naive_bayes.GaussianNB()),
                ("tree", tree.DecisionTreeClassifier(max_depth=3, random_state=0))]
    ours = am.StackingClassifier(ests).fit(X, y_cls)
    final = linear_model.LogisticRegression(tol=1e-12, max_iter=10_000)
    ref = ensemble.StackingClassifier(ref_ests, final_estimator=final).fit(X, y_cls)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-5)
    assert list(ours.classes_) == ["neg", "pos"]


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrated_classifier(method):
    ours = am.CalibratedClassifierCV(am.GaussianNB(), method=method, cv=3).fit(X, y_cls)
    ref = calibration.CalibratedClassifierCV(sk.naive_bayes.GaussianNB(), method=method,
                                             cv=3).fit(X, y_cls)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-6)


def test_transformed_target_regressor():
    t = np.exp(y / 3)
    ours = am.TransformedTargetRegressor(am.LinearRegression(), func=np.log,
                                         inverse_func=np.exp).fit(X, t)
    ref = compose.TransformedTargetRegressor(linear_model.LinearRegression(), func=np.log,
                                             inverse_func=np.exp).fit(X, t)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-10)

    ours = am.TransformedTargetRegressor(transformer=am.StandardScaler()).fit(X, y)
    ref = compose.TransformedTargetRegressor(
        transformer=preprocessing.StandardScaler()).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-10)

    with pytest.raises(ValueError, match="inverse_func"):
        am.TransformedTargetRegressor(func=np.log, inverse_func=np.square).fit(X, t)


def test_multi_output_and_chain():
    ours = am.MultiOutputRegressor(am.Ridge(alpha=2.0)).fit(X, Y)
    ref = multioutput.MultiOutputRegressor(linear_model.Ridge(alpha=2.0)).fit(X, Y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-10)
    assert ours.score(X, Y) == pytest.approx(ref.score(X, Y), abs=1e-10)

    ours = am.RegressorChain(am.LinearRegression(), order=[2, 0, 1]).fit(X, Y)
    ref = multioutput.RegressorChain(linear_model.LinearRegression(),
                                     order=[2, 0, 1]).fit(X, Y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-10)


def test_rfe_and_rfecv():
    ours = am.RFE(am.LinearRegression(), n_features_to_select=2).fit(X, y)
    ref = feature_selection.RFE(linear_model.LinearRegression(),
                                n_features_to_select=2).fit(X, y)
    np.testing.assert_array_equal(ours.support_, ref.support_)
    np.testing.assert_array_equal(ours.ranking_, ref.ranking_)
    np.testing.assert_allclose(ours.transform(X), ref.transform(X))
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-10)

    clf = am.RFE(am.LogisticRegression(), n_features_to_select=0.4).fit(X, y_cls)
    ref = feature_selection.RFE(linear_model.LogisticRegression(),
                                n_features_to_select=0.4).fit(X, y_cls)
    np.testing.assert_array_equal(clf.support_, ref.support_)
    np.testing.assert_array_equal(clf.predict(X), ref.predict(X))

    ours = am.RFECV(am.LinearRegression(), cv=4).fit(X, y)
    ref = feature_selection.RFECV(linear_model.LinearRegression(), cv=4).fit(X, y)
    np.testing.assert_array_equal(ours.support_, ref.support_)
    np.testing.assert_allclose(ours.cv_results_["mean_test_score"],
                               ref.cv_results_["mean_test_score"], atol=1e-10)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_sequential_feature_selector(direction):
    ours = am.SequentialFeatureSelector(am.LinearRegression(), n_features_to_select=2,
                                        direction=direction).fit(X, y)
    ref = feature_selection.SequentialFeatureSelector(
        linear_model.LinearRegression(), n_features_to_select=2,
        direction=direction).fit(X, y)
    np.testing.assert_array_equal(ours.get_support(), ref.get_support())
    np.testing.assert_allclose(ours.transform(X), ref.transform(X))


def test_bagging_over_an_estimator():
    ours = am.BaggingRegressor(am.LinearRegression(), n_estimators=5,
                               random_state=0).fit(X, y)
    assert ours._kind == "BaggingEstimatorRegressor"
    np.testing.assert_allclose(_pickled(ours).predict(X), ours.predict(X))
    assert ours.score(X, y) > 0.8
    clf = am.BaggingClassifier(am.GaussianNB(), n_estimators=5, random_state=0).fit(X, y_cls)
    assert set(clf.predict(X)) <= {"neg", "pos"}
    np.testing.assert_allclose(clf.predict_proba(X).sum(1), 1.0)
    with pytest.raises(ValueError, match="max_depth"):
        am.BaggingRegressor(am.LinearRegression(), max_depth=3).fit(X, y)


def test_estimator_type_errors():
    with pytest.raises(TypeError, match="classifier"):
        am.VotingClassifier([("r", am.LinearRegression())]).fit(X, y_cls)
    with pytest.raises(TypeError, match="anofox_ml_python"):
        am.RFE(linear_model.LinearRegression()).fit(X, y)
    with pytest.raises(ValueError, match="unique"):
        am.VotingRegressor([("a", am.Ridge()), ("a", am.Ridge())]).fit(X, y)
