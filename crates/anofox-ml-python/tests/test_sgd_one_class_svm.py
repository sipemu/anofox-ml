import pickle
import warnings

import numpy as np
import pytest
from sklearn.linear_model import SGDOneClassSVM as SkSGDOneClassSVM

import anofox_ml_python as am

rng = np.random.RandomState(1)
X = np.vstack([rng.normal(0, 1, (60, 4)) + 1.5, rng.uniform(-5, 5, (10, 4))])
Xt = rng.uniform(-5, 5, (15, 4))
W = rng.uniform(0.1, 3.0, len(X))


@pytest.mark.parametrize("seed, kw, weighted", [
    (0, {}, False),
    (7, {"nu": 0.05, "learning_rate": "invscaling", "eta0": 0.1, "power_t": 0.25}, False),
    (2, {"nu": 0.3, "learning_rate": "constant", "eta0": 0.005, "shuffle": False}, True),
    (3, {"nu": 0.6, "average": 10}, False),
    (4, {"learning_rate": "adaptive", "eta0": 0.5, "tol": 1e-5, "max_iter": 200}, True),
    (5, {"nu": 0.2, "fit_intercept": False, "tol": None, "max_iter": 15}, False),
])
def test_sgd_one_class_svm_matches_sklearn(seed, kw, weighted):
    sw = W if weighted else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = SkSGDOneClassSVM(random_state=seed, **kw).fit(X, sample_weight=sw)
    ours = am.SGDOneClassSVM(random_state=seed, **kw).fit(X, sample_weight=sw)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ours.offset_, ref.offset_, rtol=1e-9, atol=1e-12)
    assert ours.n_iter_ == ref.n_iter_ and ours.t_ == ref.t_
    np.testing.assert_allclose(ours.decision_function(Xt), ref.decision_function(Xt),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ours.score_samples(Xt), ref.score_samples(Xt),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(ours.predict(Xt), ref.predict(Xt))
    clone = pickle.loads(pickle.dumps(ours))
    np.testing.assert_array_equal(clone.predict(X), ref.predict(X))


def test_options():
    with pytest.raises(ValueError, match="nu"):
        am.SGDOneClassSVM(nu=0.0).fit(X)
    with pytest.raises(ValueError, match="learning_rate"):
        am.SGDOneClassSVM(learning_rate="pa1").fit(X)
    with pytest.raises(ValueError, match="warm_start"):
        am.SGDOneClassSVM(warm_start=True).fit(X)
    with pytest.raises(ValueError, match="coef_init"):
        am.SGDOneClassSVM().fit(X, coef_init=np.zeros(4))
