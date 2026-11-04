"""TuneSearchCV: hyperparameter search through anofox-ml-tune (rustuna)."""

import numpy as np
import pytest

import anofox_ml_python as am
from anofox_ml_python.model_selection import Categorical, Float, Int, KFold, TuneSearchCV

rng = np.random.RandomState(0)
X = np.column_stack([np.arange(128) % 32, rng.randint(0, 5, 128)]).astype(float)
y_step = np.floor(X[:, 0] / 4.0)  # needs a tree of depth >= 3
y_lab = np.where(X[:, 0] + 0.5 * X[:, 1] > 16, "high", "low")


def test_regressor_search_uses_sklearn_names_and_refits():
    search = TuneSearchCV(am.DecisionTreeRegressor(), {"max_depth": Int(1, 10)},
                          n_trials=20, cv=4, random_state=0).fit(X, y_step)
    assert search.scoring_ == "r2"
    assert search.best_params_["max_depth"] >= 3
    assert search.best_score_ > 0.99
    assert len(search.trials_) == 20
    best = search.best_estimator_
    assert isinstance(best, am.DecisionTreeRegressor)
    assert best.get_params()["max_depth"] == search.best_params_["max_depth"]
    np.testing.assert_allclose(search.predict(X), best.predict(X))
    assert search.score(X, y_step) > 0.99


def test_classifier_keeps_labels_and_translates_parameters():
    est = am.RandomForestClassifier(n_estimators=15, random_state=0)
    space = {"max_depth": Int(2, 8), "max_features": ["sqrt", None],
             "min_samples_leaf": Int(1, 8, step=1)}
    search = TuneSearchCV(est, space, n_trials=8, cv=3, scoring="log_loss",
                          random_state=1).fit(X, y_lab)
    assert all(t["error"] is None for t in search.trials_), search.trials_
    assert search.best_params_["max_features"] in ("sqrt", None)
    assert list(search.classes_) == ["high", "low"]
    assert set(search.predict(X)) <= {"high", "low"}
    assert search.predict_proba(X).shape == (128, 2)
    # log loss is minimised and reported with its own sign
    assert search.best_score_ == min(t["score"] for t in search.trials_)
    assert search.best_estimator_.n_estimators == 15


def test_failing_trials_are_recorded():
    space = {"max_features": ["sqrt", "bogus"]}
    search = TuneSearchCV(am.RandomForestRegressor(n_estimators=5), space, n_trials=8,
                          cv=3, sampler="random", random_state=2).fit(X, y_step)
    failed = [t for t in search.trials_ if t["score"] is None]
    assert failed and all("bogus" in t["error"] for t in failed)
    assert search.best_params_ == {"max_features": "sqrt"}


def test_same_seed_same_trials_and_float_log_ranges():
    space = {"max_depth": Int(1, 6), "learning_rate": Float(1e-3, 0.5, log=True)}

    def run(seed):
        return TuneSearchCV(am.HistGradientBoostingRegressor(max_iter=20), space, n_trials=12,
                            cv=3, random_state=seed, refit=False).fit(X, y_step).trials_

    a, b = run(5), run(5)
    params = lambda trials: [t["params"] for t in trials]  # noqa: E731
    assert params(a) == params(b)
    assert all(t["error"] is None for t in a), a
    assert all(1e-3 <= t["params"]["learning_rate"] <= 0.5 for t in a)


def test_custom_splitter_and_groups():
    from sklearn.model_selection import GroupKFold

    groups = np.arange(128) % 8
    search = TuneSearchCV(am.DecisionTreeRegressor(), {"max_depth": Int(1, 6)}, n_trials=5,
                          cv=GroupKFold(n_splits=4), random_state=0).fit(X, y_step, groups=groups)
    assert all(len(t["fold_scores"]) == 4 for t in search.trials_)
    search = TuneSearchCV(am.DecisionTreeRegressor(), {"max_depth": Int(1, 6)}, n_trials=3,
                          cv=KFold(3), random_state=0).fit(X, y_step)
    assert all(len(t["fold_scores"]) == 3 for t in search.trials_)


def test_multi_objective_pareto_front():
    search = TuneSearchCV(am.DecisionTreeRegressor(),
                          {"max_depth": Int(1, 10), "min_samples_leaf": Int(1, 20)},
                          objectives=["model_size"], sampler="nsgaii", population_size=6,
                          n_trials=18, cv=3, random_state=0).fit(X, y_step)
    assert search.objectives_ == [("r2", True), ("model_size", False)]
    assert len(search.pareto_front_) >= 2
    assert search.best_index_ in search.pareto_front_


def test_invalid_requests():
    with pytest.raises(ValueError, match="no parameters \\['depth'\\]"):
        TuneSearchCV(am.DecisionTreeRegressor(), {"depth": Int(1, 3)}, n_trials=2).fit(X, y_step)
    with pytest.raises(ValueError, match="unknown metric"):
        TuneSearchCV(am.DecisionTreeRegressor(), {"max_depth": Int(1, 3)}, n_trials=2,
                     scoring="auc").fit(X, y_step)
    with pytest.raises(ValueError, match="Int"):
        TuneSearchCV(am.DecisionTreeRegressor(), {"max_depth": 3}, n_trials=2).fit(X, y_step)
    with pytest.raises(AttributeError, match="best_estimator_"):
        TuneSearchCV(am.DecisionTreeRegressor(), {"max_depth": Int(1, 3)}).predict(X)
