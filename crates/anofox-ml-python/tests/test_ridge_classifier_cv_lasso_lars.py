import pickle

import numpy as np
import pytest
import sklearn.linear_model as skl

import anofox_ml_python as am

rng = np.random.RandomState(0)
X = rng.randn(80, 5)
y = X @ [1.0, 0.0, 2.0, 0.0, -1.0] + rng.randn(80) * 0.5
labels4 = np.digitize(X[:, 0] + X[:, 1], [-1.0, 0.0, 1.0])
labels2 = np.where(labels4 > 1, "hi", "lo")
alphas = (0.01, 0.3, 3.0, 30.0, 300.0)


def close(a, b, tol=1e-10):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("alpha", [0.001, 0.05, 0.2, 1.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_lasso_lars_matches_sklearn(alpha, fit_intercept):
    ours = am.LassoLars(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)
    ref = skl.LassoLars(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)
    close(ours.coef_, ref.coef_)
    close(ours.intercept_, ref.intercept_)
    close(ours.predict(X), ref.predict(X))
    close(pickle.loads(pickle.dumps(ours)).predict(X), ref.predict(X))
    with pytest.raises(ValueError, match="positive"):
        am.LassoLars(positive=True).fit(X, y)


@pytest.mark.parametrize("labels", [labels2, labels4])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_classifier_cv_loo_matches_sklearn(labels, class_weight, fit_intercept):
    kw = dict(alphas=alphas, class_weight=class_weight, fit_intercept=fit_intercept)
    ours = am.RidgeClassifierCV(**kw).fit(X, labels)
    ref = skl.RidgeClassifierCV(**kw).fit(X, labels)
    assert ours.alpha_ == ref.alpha_
    close(ours.best_score_, ref.best_score_)
    close(ours.coef_, ref.coef_)
    close(ours.intercept_, ref.intercept_)
    np.testing.assert_array_equal(ours.classes_, ref.classes_)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    close(ours.decision_function(X), ref.decision_function(X))


def test_ridge_classifier_cv_weights_and_kfold():
    w = np.linspace(0.2, 3.0, len(X))
    ours = am.RidgeClassifierCV(alphas=alphas).fit(X, labels4, sample_weight=w)
    ref = skl.RidgeClassifierCV(alphas=alphas).fit(X, labels4, sample_weight=w)
    assert ours.alpha_ == ref.alpha_
    close(ours.best_score_, ref.best_score_)
    close(ours.coef_, ref.coef_)
    # cv=k: sklearn's GridSearchCV over RidgeClassifier (accuracy).
    for scoring in (None, "balanced_accuracy"):
        ours = am.RidgeClassifierCV(alphas=alphas, cv=4, scoring=scoring).fit(X, labels4)
        ref = skl.RidgeClassifierCV(alphas=alphas, cv=4, scoring=scoring).fit(X, labels4)
        assert ours.alpha_ == ref.alpha_
        close(ours.best_score_, ref.best_score_)
        close(ours.coef_, ref.coef_)
    fitted = am.RidgeClassifierCV(alphas=alphas).fit(X, labels2)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fitted)).predict(X),
                                  fitted.predict(X))
    with pytest.raises(ValueError, match="scoring"):
        am.RidgeClassifierCV(scoring="accuracy").fit(X, labels2)
