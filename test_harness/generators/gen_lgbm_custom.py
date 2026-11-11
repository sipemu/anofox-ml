"""Golden data for custom LightGBM objectives (#213): `objective` given as
a Python callable (LightGBM 4.7 trains with `objective="none"` and the
callable's gradients), one or several outputs (`num_class`).

The single-output loss is pseudo-Huber against `y`; the multi-output one is
FFORMA's softmax combination loss over per-method errors `E` (anofox-
orchestration): with `p = softmax(score_i)` and `L = Σ_j p_j E_ij`,
`g_j = p_j (E_ij − L)`, `h_j = max(E_ij p_j (1 − p_j) − g_j p_j, 1e-6)`.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np

COMMON = dict(deterministic=True, force_row_wise=True, verbose=-1, num_threads=1)


def pseudo_huber(y, delta):
    def fobj(preds, _):
        r = preds - y
        s = np.sqrt(1 + (r / delta) ** 2)
        return r / s, 1 / s ** 3
    return fobj


def fforma(errors):
    def fobj(preds, _):
        z = preds - preds.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        loss = (p * errors).sum(axis=1, keepdims=True)
        g = p * (errors - loss)
        h = np.maximum(errors * p * (1 - p) - g * p, 1e-6)
        return g, h
    return fobj


def generate():
    rng = np.random.default_rng(213)
    n = 500
    X = np.column_stack([
        rng.normal(size=n),
        rng.uniform(-2, 2, size=n),
        rng.integers(0, 6, size=n).astype(float),
        rng.normal(size=n),
    ])
    X[rng.choice(n, 20, replace=False), 3] = np.nan
    y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] + rng.standard_t(3, size=n) * 0.3
    m = 4
    base_err = np.abs(rng.normal(size=(n, m))) + 0.1
    errors = base_err * np.exp(np.outer(X[:, 0], np.linspace(-0.6, 0.6, m)))
    X_test = X[::7]
    base = dict(num_leaves=15, learning_rate=0.1, min_data_in_leaf=20, **COMMON)
    out = []

    def add(name, params, fobj, k, rounds=30, side=None, cat=None):
        params = dict(params, objective=fobj)
        if k > 1:
            params["num_class"] = k
        ds = lgb.Dataset(X, label=np.zeros(n), categorical_feature=cat or "auto",
                         free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=rounds)
        raw = booster.predict(X_test, raw_score=True).reshape(len(X_test), k)
        out.append({"name": name, "outputs": k, "rounds": rounds,
                    "n_trees": booster.num_trees(),
                    "params": {a: b for a, b in params.items()
                               if a not in COMMON and a != "objective"},
                    "categorical": cat or [], "side": side,
                    "X": np.where(np.isnan(X), None, X).tolist(),
                    "X_test": np.where(np.isnan(X_test), None, X_test).tolist(),
                    "raw": raw.tolist()})

    add("pseudo_huber", base, pseudo_huber(y, 0.5), 1, side={"y": y.tolist(), "delta": 0.5})
    side = {"errors": errors.tolist()}
    add("fforma", base, fforma(errors), m, side=side)
    # anofox-orchestration's settings: bagging, feature fraction, a seed.
    add("fforma_sampling", dict(base, bagging_fraction=0.9, bagging_freq=5,
                                feature_fraction=0.9, seed=7), fforma(errors), m, side=side)
    add("fforma_categorical", dict(base, min_data_in_leaf=10), fforma(errors), m, side=side,
        cat=[2])
    add("fforma_goss", dict(base, boosting="goss"), fforma(errors), m, rounds=40, side=side)
    # A large min_data_in_leaf: few splits per tree.
    add("pseudo_huber_min_leaf", dict(base, min_data_in_leaf=200), pseudo_huber(y, 0.5), 1,
        rounds=10, side={"y": y.tolist(), "delta": 0.5})

    # anofox-orchestration's FFORMA loop on a larger matrix (#243): many
    # features (feature_fraction samples), hessians clamped at 1e-6, full
    # trees. LightGBM is driven as its C API loop does (objective "none" from
    # the start, `__boost` each round): Python's callable objective resets
    # the parameters first, which re-seeds the feature sampler.
    env = lambda k, d: type(d)(os.environ.get(k, d))
    r = np.random.default_rng(env("CAPI_SEED", 243))
    nl, dl, ml = env("CAPI_N", 3000), env("CAPI_D", 12), 3
    Xl = np.round(r.normal(size=(nl, dl)), 3)
    Xl[:, 3] = np.round(Xl[:, 3] * 2)
    El = np.abs(r.standard_t(2, size=(nl, ml))) ** env("CAPI_POW", 3.0) * np.exp(
        np.outer(Xl[:, 0] - 0.5 * Xl[:, 1], [-0.8, 0.0, 0.8]) * env("CAPI_SCALE", 2.0))
    El = np.round(El, 5) + 1e-5
    Xl_test = Xl[::11]
    rounds_l = env("CAPI_ROUNDS", 300)
    for name, extra, rounds in (("fforma_capi", {}, rounds_l),
                                ("fforma_capi_sampling", dict(feature_fraction=0.8,
                                                              bagging_fraction=0.8,
                                                              bagging_freq=5), rounds_l)):
        params = dict(num_leaves=31, learning_rate=0.05, min_data_in_leaf=20, objective="none",
                      num_class=ml, **COMMON, **extra)
        booster = lgb.Booster(params, lgb.Dataset(Xl, label=np.zeros(nl), free_raw_data=False))
        fobj = fforma(El)
        for _ in range(rounds):
            g, h = fobj(booster._Booster__inner_predict(data_idx=0), None)
            booster._Booster__boost(grad=g, hess=h)
        raw = booster.predict(Xl_test, raw_score=True).reshape(len(Xl_test), ml)
        out.append({"name": name, "outputs": ml, "rounds": rounds,
                    "n_trees": booster.num_trees(),
                    "params": {a: b for a, b in params.items()
                               if a not in COMMON and a not in ("objective", "num_class")},
                    "categorical": [], "side": {"errors": El.tolist()},
                    "X": Xl.tolist(), "X_test": Xl_test.tolist(), "raw": raw.tolist()})
    return out


if __name__ == "__main__":
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "lgbm_custom.json")
    with open(path, "w") as f:
        json.dump(generate(), f)
    print(f"wrote {path}")
