"""Conformal wrappers: shapes, coverage, groups, cloning and pickling."""

import pickle

import numpy as np
import pytest

import anofox_ml_python as am
from anofox_ml_python import conformal as cf

rng = np.random.RandomState(0)
n = 600
X = rng.randn(n, 3)
noise_sd = np.where(X[:, 0] > 0, 2.0, 0.2)
y = X @ np.array([1.0, -1.0, 0.5]) + noise_sd * rng.randn(n)
groups = (X[:, 0] > 0).astype(np.int64)
y_cls = np.array(["a", "b", "c"])[np.digitize(X[:, 1] + 0.3 * rng.randn(n), [-0.5, 0.5])]
train, test = slice(0, 400), slice(400, None)


def coverage(iv, y):
    return np.mean((iv[:, 0] <= y) & (y <= iv[:, 1]))


@pytest.mark.parametrize(
    "model",
    [
        cf.SplitConformalRegressor(random_state=0),
        cf.SplitConformalRegressor(am.DecisionTreeRegressor(max_depth=3), random_state=0),
        cf.CrossConformalRegressor(am.Ridge(alpha=1.0), cv=5, method="plus", random_state=0),
        cf.CrossConformalRegressor(method="minmax", random_state=0),
        cf.NormalizedConformalRegressor(residual_estimator=am.KNeighborsRegressor(), random_state=0),
        cf.ConformalizedQuantileRegressor(alpha=0.1, random_state=0),
        cf.ConformalizedQuantileRegressor(
            am.GradientBoostingRegressor(n_estimators=30), alpha=0.1, random_state=0
        ),
    ],
)
def test_regressors_cover(model):
    model.fit(X[train], y[train])
    iv = model.predict_interval(X[test], 0.1) if not isinstance(
        model, cf.ConformalizedQuantileRegressor
    ) else model.predict_interval(X[test])
    assert iv.shape == (200, 2)
    assert (iv[:, 0] <= iv[:, 1]).all()
    assert coverage(iv, y[test]) >= 0.8
    assert model.predict(X[test]).shape == (200,)
    back = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        back.predict_interval(X[test], 0.1) if not isinstance(
            model, cf.ConformalizedQuantileRegressor
        ) else back.predict_interval(X[test]),
        iv,
    )


def test_cqr_alpha_is_fixed_at_fit():
    model = cf.ConformalizedQuantileRegressor(alpha=0.2, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="refit"):
        model.predict_interval(X, alpha=0.1)


def test_estimator_is_left_unfitted_and_validated():
    est = am.DecisionTreeRegressor(max_depth=2)
    cf.SplitConformalRegressor(est, random_state=0).fit(X, y)
    assert not hasattr(est, "_model_")
    with pytest.raises(TypeError):
        cf.SplitConformalRegressor(object()).fit(X, y)
    with pytest.raises(TypeError):
        cf.SplitConformalRegressor(am.LogisticRegression()).fit(X, y)
    with pytest.raises(ValueError, match="method"):
        cf.CrossConformalRegressor(method="nope").fit(X, y)


def test_mondrian_regressor_is_wider_in_the_noisy_group():
    model = cf.MondrianConformalRegressor(random_state=0)
    model.fit(X[train], y[train], groups[train])
    iv = model.predict_interval(X[test], groups[test], 0.1)
    width = iv[:, 1] - iv[:, 0]
    g = groups[test]
    assert width[g == 1].mean() > 4 * width[g == 0].mean()
    for k in (0, 1):
        assert coverage(iv[g == k], y[test][g == k]) >= 0.8
    with pytest.raises(ValueError, match="non-negative"):
        model.predict_interval(X[test], -groups[test] - 1, 0.1)


@pytest.mark.parametrize("score", ["lac", "aps", "raps"])
def test_classifier_sets(score):
    model = cf.SplitConformalClassifier(
        am.RandomForestClassifier(n_estimators=30, random_state=0),
        conformity_score=score,
        random_state=0,
    ).fit(X[train], y_cls[train])
    sets = model.predict_set(X[test], 0.1)
    assert sets.shape == (200, 3) and sets.dtype == bool
    idx = np.searchsorted(model.classes_, y_cls[test])
    assert sets[np.arange(200), idx].mean() >= 0.8
    assert set(model.predict(X[test])) <= set(model.classes_)
    np.testing.assert_allclose(model.predict_proba(X[test]).sum(axis=1), 1.0)
    back = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(back.predict_set(X[test], 0.1), sets)


def test_mondrian_classifier():
    model = cf.MondrianConformalClassifier(random_state=0).fit(X[train], y_cls[train], groups[train])
    sets = model.predict_set(X[test], groups[test], 0.2)
    assert sets.shape == (200, 3)
    assert model.predict(X[test]).shape == (200,)


def test_sklearn_clone_and_params():
    sklearn = pytest.importorskip("sklearn")
    model = cf.CrossConformalRegressor(am.Ridge(alpha=2.0), cv=3)
    clone = sklearn.base.clone(model)
    assert clone.get_params()["cv"] == 3
    assert clone.get_params()["estimator"].alpha == 2.0


def test_meta_estimators_match_sklearn():
    from sklearn import model_selection as ms, multiclass as mc, naive_bayes

    for ours, ref in [
        (am.OneVsRestClassifier(am.GaussianNB()), mc.OneVsRestClassifier(naive_bayes.GaussianNB())),
        (am.OneVsOneClassifier(am.GaussianNB()), mc.OneVsOneClassifier(naive_bayes.GaussianNB())),
    ]:
        np.testing.assert_array_equal(ours.fit(X, y_cls).predict(X), ref.fit(X, y_cls).predict(X))
    ovr = am.OneVsRestClassifier(am.GaussianNB()).fit(X, y_cls)
    ref = mc.OneVsRestClassifier(naive_bayes.GaussianNB()).fit(X, y_cls)
    np.testing.assert_allclose(ovr.predict_proba(X), ref.predict_proba(X), atol=1e-10)
    yb = np.where(y_cls == "a", "pos", "neg")
    fixed = am.FixedThresholdClassifier(am.GaussianNB(), threshold=0.3, pos_label="pos").fit(X, yb)
    rfix = ms.FixedThresholdClassifier(naive_bayes.GaussianNB(), threshold=0.3,
                                       pos_label="pos").fit(X, yb)
    np.testing.assert_array_equal(fixed.predict(X), rfix.predict(X))
    tuned = am.TunedThresholdClassifierCV(am.GaussianNB()).fit(X, yb)
    rtun = ms.TunedThresholdClassifierCV(naive_bayes.GaussianNB()).fit(X, yb)
    assert abs(tuned.best_score_ - rtun.best_score_) < 1e-9
    assert abs(tuned.best_threshold_ - rtun.best_threshold_) < 1e-6
    np.testing.assert_array_equal(tuned.predict(X), rtun.predict(X))
