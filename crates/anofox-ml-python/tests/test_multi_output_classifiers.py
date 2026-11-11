import pickle

import numpy as np
import pytest
import sklearn.ensemble as ske
import sklearn.frozen as skfr
import sklearn.linear_model as skl
import sklearn.multiclass as skmc
import sklearn.multioutput as skmo

import anofox_ml_python as am

rng = np.random.RandomState(0)
X = rng.randn(120, 4)
Y = np.column_stack([
    (X[:, 0] + 0.3 * rng.randn(120) > 0).astype(int),
    (X[:, 1] - X[:, 2] > 0.2).astype(int),
    (X[:, 0] + X[:, 3] > 0).astype(int) * 2 + 1,  # labels {1, 3}
])
y4 = np.digitize(X[:, 0] + 0.5 * X[:, 1], [-1.0, 0.0, 1.0])  # 4 classes


# The trees inside sklearn's meta-estimators are ours (they are
# sklearn-compatible), so the comparison is of the meta logic alone; a
# single tree can split ties differently from sklearn's.


def close(a, b, tol=1e-6):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


def test_multi_output_classifier_matches_sklearn():
    ours = am.MultiOutputClassifier(am.DecisionTreeClassifier(max_depth=3, random_state=0)).fit(X, Y)
    ref = skmo.MultiOutputClassifier(am.DecisionTreeClassifier(max_depth=3, random_state=0)).fit(X, Y)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    for a, b in zip(ours.classes_, ref.classes_):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(ours.predict_proba(X), ref.predict_proba(X)):
        close(a, b, 1e-12)
    assert ours.score(X, Y) == ref.score(X, Y)
    # String labels, and the fitted model pickles.
    Ys = np.where(Y == 1, "yes", "no")
    s = am.MultiOutputClassifier(am.DecisionTreeClassifier(max_depth=3, random_state=0)).fit(X, Ys)
    np.testing.assert_array_equal(s.predict(X), np.where(ref.predict(X) == 1, "yes", "no"))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(s)).predict(X), s.predict(X))


@pytest.mark.parametrize("order", [None, [2, 0, 1]])
def test_classifier_chain_matches_sklearn(order):
    ours = am.ClassifierChain(am.LogisticRegression(), order=order).fit(X, Y)
    ref = skmo.ClassifierChain(skl.LogisticRegression(tol=1e-10), order=order).fit(X, Y)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    close(ours.predict_proba(X), ref.predict_proba(X))
    close(ours.decision_function(X), ref.decision_function(X))
    np.testing.assert_array_equal(ours.order_, ref.order_)
    tree = am.ClassifierChain(am.DecisionTreeClassifier(max_depth=4, random_state=0)).fit(X, Y)
    tref = skmo.ClassifierChain(am.DecisionTreeClassifier(max_depth=4, random_state=0)).fit(X, Y)
    np.testing.assert_array_equal(tree.predict(X), tref.predict(X))
    close(tree.predict_proba(X), tref.predict_proba(X), 1e-12)


def test_classifier_chain_random_order():
    ours = am.ClassifierChain(am.LogisticRegression(), order="random", random_state=3).fit(X, Y)
    ref = skmo.ClassifierChain(skl.LogisticRegression(tol=1e-10), order="random", random_state=3).fit(X, Y)
    np.testing.assert_array_equal(ours.order_, ref.order_)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    with pytest.raises(ValueError, match="cv"):
        am.ClassifierChain(am.LogisticRegression(), cv=3).fit(X, Y)


@pytest.mark.parametrize("code_size, random_state", [(1.5, 0), (2.0, 7), (0.6, 1)])
def test_output_code_classifier_matches_sklearn(code_size, random_state):
    kw = dict(code_size=code_size, random_state=random_state)
    # A decision function: codes 1 / -1.
    ours = am.OutputCodeClassifier(am.LogisticRegression(), **kw).fit(X, y4)
    ref = skmc.OutputCodeClassifier(skl.LogisticRegression(tol=1e-10), **kw).fit(X, y4)
    np.testing.assert_array_equal(ours.code_book_, ref.code_book_)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    # Probabilities only: codes 1 / 0 (sklearn's own tree has no
    # decision_function; our wrappers always define one, so sklearn's
    # meta-estimator over our tree would draw -1 codes). The prediction is
    # checked by hand: our trees on each code column, P(1), nearest code.
    ours = am.OutputCodeClassifier(am.DecisionTreeClassifier(max_depth=3, random_state=0),
                                   **kw).fit(X, y4)
    import sklearn.tree as skt
    ref = skmc.OutputCodeClassifier(skt.DecisionTreeClassifier(max_depth=3), **kw).fit(X, y4)
    np.testing.assert_array_equal(ours.code_book_, ref.code_book_)
    book = ours.code_book_
    scores = []
    for j in range(book.shape[1]):
        col = book[y4, j]
        if len(np.unique(col)) == 1:
            scores.append(np.full(len(X), col[0]))
        else:
            tree = am.DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, col)
            scores.append(tree.predict_proba(X)[:, 1])
    dist = ((np.column_stack(scores)[:, None, :] - book[None]) ** 2).sum(axis=2)
    np.testing.assert_array_equal(ours.predict(X), dist.argmin(axis=1))
    labels = np.asarray(["a", "b", "c", "d"])[y4]
    s = am.OutputCodeClassifier(am.LogisticRegression(), **kw).fit(X, labels)
    ref = skmc.OutputCodeClassifier(skl.LogisticRegression(tol=1e-10), **kw).fit(X, labels)
    np.testing.assert_array_equal(s.predict(X), ref.predict(X))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(s)).predict(X), s.predict(X))


def test_frozen_estimator_matches_sklearn():
    fitted = am.LogisticRegression().fit(X, y4)
    frozen = am.FrozenEstimator(fitted)
    before = frozen.predict_proba(X)
    assert frozen.fit(X[:10], y4[:10]) is frozen  # fit does nothing
    np.testing.assert_array_equal(frozen.predict_proba(X), before)
    np.testing.assert_array_equal(frozen.classes_, fitted.classes_)
    assert frozen.get_params() == {"estimator": fitted}
    with pytest.raises(ValueError):
        frozen.set_params(C=2.0)
    with pytest.raises(am.NotFittedError):
        am.FrozenEstimator(am.LogisticRegression()).fit(X, y4)
    # As in sklearn: a frozen model inside a meta-estimator is not refitted.
    ref = skfr.FrozenEstimator(skl.LogisticRegression(tol=1e-10).fit(X, y4))
    vote = ske.VotingClassifier([("a", frozen), ("b", ref)], voting="soft").fit(X[:5], y4[:5])
    close(vote.predict_proba(X), (before + ref.predict_proba(X)) / 2, 1e-12)
