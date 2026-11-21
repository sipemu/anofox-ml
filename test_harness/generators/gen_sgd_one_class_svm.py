"""Golden data for SGDOneClassSVM (#261, scikit-learn 1.8.0): coef_, offset_,
n_iter_, t_ and decision_function for int random_state (the epoch shuffle is
SequentialDataset's our_rand_r Fisher-Yates, seeded by
RandomState(random_state).randint(0, 2**31 - 1))."""

import numpy as np
from sklearn.linear_model import SGDOneClassSVM


def generate():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(0, 1, (40, 3)) + [2, 2, 0], rng.uniform(-4, 4, (8, 3))])
    Xt = rng.uniform(-4, 4, (10, 3))
    w = rng.uniform(0.2, 2.0, len(X))
    cases = []
    for seed, kw, weighted in [
        (0, {}, False),
        (1, {"nu": 0.1, "learning_rate": "invscaling", "eta0": 0.05}, False),
        (2, {"nu": 0.3, "learning_rate": "constant", "eta0": 0.01, "shuffle": False}, False),
        (3, {"nu": 0.2, "average": True, "max_iter": 50}, False),
        (4, {"nu": 0.4, "learning_rate": "adaptive", "eta0": 0.1, "tol": 1e-4}, False),
        (5, {"nu": 0.25, "fit_intercept": False, "tol": None, "max_iter": 20}, False),
        (6, {"nu": 0.15, "average": 30, "power_t": 0.3, "learning_rate": "invscaling",
             "eta0": 0.02}, True),
    ]:
        est = SGDOneClassSVM(random_state=seed, **kw)
        est.fit(X, sample_weight=w if weighted else None)
        cases.append({"params": {"random_state": seed, **kw}, "weighted": weighted,
                      "coef": est.coef_.tolist(), "offset": float(est.offset_[0]),
                      "n_iter": int(est.n_iter_), "t": float(est.t_),
                      "decision_function": est.decision_function(Xt).tolist()})
    return {"X": X.tolist(), "X_test": Xt.tolist(), "sample_weight": w.tolist(),
            "cases": cases}


if __name__ == "__main__":
    import json, os, warnings
    warnings.simplefilter("ignore")
    out = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "anofox-ml", "tests",
                       "golden_data", "sgd_one_class_svm.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
