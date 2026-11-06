"""Golden data for the CatBoost-style estimators against catboost 1.2.10 (#127).

CatBoost's randomness is switched off so that its models are reproducible:
`random_strength=0` (no score noise), `bootstrap_type='No'`,
`boosting_type='Plain'` and `has_time=True` (no permutations), one thread.
Quantization borders come from `Pool.quantize()` (all borders, not only the
ones a model uses).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import tempfile

import catboost as cb
import numpy as np

COMMON = dict(random_strength=0, bootstrap_type="No", has_time=True,
              thread_count=1, verbose=0, allow_writing_files=False)


def nulls(a):
    return np.where(np.isnan(a), None, a).tolist()


def borders_of(X, border_count):
    pool = cb.Pool(X)
    pool.quantize(border_count=border_count)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "b.tsv")
        pool.save_quantization_borders(path)
        rows = [line.split("\t") for line in open(path).read().splitlines() if line]
    out = [[] for _ in range(X.shape[1])]
    for row in rows:
        out[int(row[0])].append(float(row[1]))
    return out


def generate():
    rng = np.random.default_rng(127)
    n = 400
    X = np.column_stack([
        rng.normal(size=n),
        rng.uniform(-2, 2, size=n),
        np.round(rng.normal(size=n) * 3),       # few distinct values
        rng.exponential(size=n),
    ])
    X_nan = X.copy()
    X_nan[rng.choice(n, 30, replace=False), 1] = np.nan
    eta = np.sin(2 * X[:, 0]) + 0.6 * X[:, 1] - 0.2 * X[:, 2] + 0.3 * X[:, 3]
    y_reg = eta + rng.normal(scale=0.4, size=n)
    y_bin = (eta + rng.logistic(size=n) > 0).astype(float)
    weights = rng.uniform(0.2, 3.0, size=n)

    out = {"quantization": [], "models": []}
    for name, data, count in (("plain", X, 254), ("nan", X_nan, 254), ("few", X, 16)):
        out["quantization"].append({"name": name, "border_count": count, "X": nulls(data),
                                    "borders": borders_of(data, count)})

    base = dict(iterations=30, learning_rate=0.2, depth=4, l2_leaf_reg=3.0, boosting_type="Plain",
                **COMMON)

    def add(name, kind, params, Xd, y, sw=None):
        cls = cb.CatBoostRegressor if kind == "regressor" else cb.CatBoostClassifier
        m = cls(**params).fit(Xd, y, sample_weight=sw)
        X_test = Xd[::4]
        pred = m.predict(X_test) if kind == "regressor" else m.predict_proba(X_test)
        out["models"].append({
            "name": name, "kind": kind,
            "params": {k: v for k, v in params.items() if k not in COMMON},
            "X": nulls(Xd), "y": y.tolist(),
            "sample_weight": None if sw is None else sw.tolist(),
            "X_test": nulls(X_test), "predictions": pred.tolist(),
        })

    add("rmse_cosine", "regressor", base, X, y_reg)
    add("rmse_l2", "regressor", dict(base, score_function="L2"), X, y_reg)
    add("rmse_depth1", "regressor", dict(base, depth=1), X, y_reg)
    add("rmse_depth6", "regressor", dict(base, depth=6, iterations=15), X, y_reg)
    add("rmse_gradient", "regressor", dict(base, leaf_estimation_method="Gradient"), X, y_reg)
    add("rmse_l2_reg", "regressor", dict(base, l2_leaf_reg=0.5, learning_rate=0.5), X, y_reg)
    add("rmse_weighted", "regressor", base, X, y_reg, sw=weights)
    add("rmse_nan", "regressor", base, X_nan, y_reg)
    add("rmse_border_16", "regressor", dict(base, border_count=16), X, y_reg)
    add("rmse_no_average", "regressor", dict(base, boost_from_average=False), X, y_reg)
    add("logloss_cosine", "classifier", base, X, y_bin)
    add("logloss_l2", "classifier", dict(base, score_function="L2"), X, y_bin)
    add("logloss_weighted", "classifier", base, X, y_bin, sw=weights)
    add("logloss_average", "classifier", dict(base, boost_from_average=True), X, y_bin)

    # Categorical features (integer codes): ordered target statistics
    # (CTRs) without feature combinations (`max_ctr_complexity=1`), and
    # one-hot splits for at most `one_hot_max_size` categories.
    c_many = rng.integers(0, 12, size=n)
    c_few = rng.integers(0, 4, size=n)
    c_bin = rng.integers(0, 2, size=n)
    effect = np.linspace(-1.2, 1.2, 12)[c_many] + 0.5 * (c_few == 2) - 0.4 * c_bin
    y_cat = (eta * 0.5 + effect + rng.logistic(size=n) * 0.7 > 0).astype(float)
    Xc = np.column_stack([X[:, 0], c_many, X[:, 1], c_few, c_bin]).astype(float)
    cat_cols = [1, 3, 4]
    cat = dict(base, cat_features=cat_cols, max_ctr_complexity=1)

    def add_cat(name, params, y, kind="classifier", sw=None):
        cls = cb.CatBoostRegressor if kind == "regressor" else cb.CatBoostClassifier
        Xo = Xc.astype(object)
        for c in cat_cols:
            Xo[:, c] = Xc[:, c].astype(int)
        m = cls(**params).fit(Xo, y, sample_weight=sw)
        X_test = np.vstack([Xc[::4], [[0.1, 99, 0.2, 1, 0]]])  # an unseen category
        Xt = X_test.astype(object)
        for c in cat_cols:
            Xt[:, c] = X_test[:, c].astype(int)
        pred = m.predict(Xt) if kind == "regressor" else m.predict_proba(Xt)
        out["models"].append({
            "name": name, "kind": kind,
            "params": {k: v for k, v in params.items() if k not in COMMON},
            "X": nulls(Xc), "y": y.tolist(),
            "sample_weight": None if sw is None else sw.tolist(),
            "X_test": nulls(X_test), "predictions": pred.tolist(),
        })

    # Ordered boosting (prefixes of the given order, `has_time=True`).
    ordered = dict(base, boosting_type="Ordered")
    add("ordered_rmse", "regressor", ordered, X, y_reg)
    add("ordered_rmse_l2_depth6", "regressor", dict(ordered, score_function="L2", depth=6,
                                                     iterations=15), X, y_reg)
    add("ordered_rmse_weighted", "regressor", ordered, X, y_reg, sw=weights)
    add("ordered_rmse_gradient", "regressor", dict(ordered, leaf_estimation_method="Gradient",
                                                    fold_len_multiplier=1.5), X, y_reg)
    add("ordered_logloss", "classifier", ordered, X, y_bin)
    big = rng.normal(size=(1200, 3))
    y_big = big[:, 0] - big[:, 1] ** 2 + rng.normal(scale=0.3, size=1200)
    add("ordered_rmse_1200", "regressor", dict(ordered, iterations=15), big, y_big)
    add_cat("cat_logloss", cat, y_cat)
    add_cat("cat_logloss_l2", dict(cat, score_function="L2", depth=6), y_cat)
    add_cat("cat_logloss_weighted", cat, y_cat, sw=weights)
    add_cat("cat_logloss_no_onehot", dict(cat, one_hot_max_size=1), y_cat)
    add_cat("cat_only", dict(cat, cat_features=[1, 3, 4]), y_cat)
    add_cat("ordered_cat_logloss", dict(cat, boosting_type="Ordered"), y_cat)
    y_cat_reg = eta * 0.5 + effect + rng.normal(scale=0.3, size=n)
    add_cat("cat_rmse", cat, y_cat_reg, kind="regressor")
    add_cat("cat_rmse_ordered_l2", dict(cat, boosting_type="Ordered", score_function="L2"),
            y_cat_reg, kind="regressor")
    add_cat("cat_rmse_weighted", cat, y_cat_reg, kind="regressor", sw=weights)

    # MultiClass (more than two classes): softmax over one approximation per
    # class, CatBoost's full Newton step and multiclass CTRs (one target
    # border per class but the last). `leaf_estimation_iterations=1` is
    # CatBoost's default for Newton; Gradient defaults to 10 for MultiClass.
    y_mc = np.digitize(eta + rng.logistic(size=n) * 0.5, [-0.6, 0.4])
    y_mc5 = np.digitize(eta + rng.logistic(size=n) * 0.5, [-1.2, -0.4, 0.3, 1.0])
    mc = dict(base, loss_function="MultiClass")
    add("mc_cosine", "classifier", mc, X, y_mc.astype(float))
    add("mc_l2", "classifier", dict(mc, score_function="L2"), X, y_mc.astype(float))
    add("mc_depth6", "classifier", dict(mc, depth=6, iterations=15), X, y_mc.astype(float))
    add("mc_weighted", "classifier", mc, X, y_mc.astype(float), sw=weights)
    add("mc_gradient", "classifier", dict(mc, leaf_estimation_method="Gradient",
                                          leaf_estimation_iterations=1), X, y_mc.astype(float))
    add("mc_l2_reg", "classifier", dict(mc, l2_leaf_reg=0.5, learning_rate=0.5), X,
        y_mc.astype(float))
    add("mc_5_classes", "classifier", mc, X_nan, y_mc5.astype(float))
    add("mc_labels", "classifier", mc, X, np.array([-1.0, 3.0, 7.0])[y_mc])
    mc_ordered = dict(mc, boosting_type="Ordered")
    add("mc_ordered", "classifier", mc_ordered, X, y_mc.astype(float))
    add("mc_ordered_weighted", "classifier", mc_ordered, X, y_mc.astype(float), sw=weights)
    add("mc_ordered_gradient", "classifier", dict(mc_ordered, leaf_estimation_method="Gradient",
                                                  leaf_estimation_iterations=1),
        X, y_mc.astype(float))
    y_cat_mc = np.digitize(eta * 0.5 + effect + rng.logistic(size=n) * 0.7, [-0.5, 0.5]).astype(float)
    cat_mc = dict(cat, loss_function="MultiClass")
    add_cat("cat_mc", cat_mc, y_cat_mc)
    add_cat("cat_mc_no_onehot", dict(cat_mc, one_hot_max_size=1, score_function="L2"), y_cat_mc)
    add_cat("cat_mc_weighted", cat_mc, y_cat_mc, sw=weights)
    add_cat("cat_mc_ordered", dict(cat_mc, boosting_type="Ordered"), y_cat_mc)

    # CTRs of feature combinations (`max_ctr_complexity` > 1): the tree's
    # float and one-hot split conditions and its CTR projections, each
    # extended by a categorical feature.
    combo = dict(cat, max_ctr_complexity=4)
    add_cat("combo_logloss", combo, y_cat)
    add_cat("combo_logloss_2", dict(combo, max_ctr_complexity=2), y_cat)
    add_cat("combo_logloss_l2_depth6", dict(combo, score_function="L2", depth=6), y_cat)
    add_cat("combo_rmse", combo, y_cat_reg, kind="regressor")
    add_cat("combo_mc", dict(combo, loss_function="MultiClass", depth=6, score_function="L2",
                             iterations=60), y_cat_mc)
    add_cat("combo_ordered", dict(combo, boosting_type="Ordered"), y_cat)
    add_cat("combo_weighted", dict(combo, depth=6), y_cat, sw=weights)
    # 200 iterations: `max_ctr_complexity` defaults to 4 (1 below). Logloss
    # then also defaults to 10 leaf estimation iterations, which we do not
    # have, so that one is pinned.
    default_combo = {k: v for k, v in cat.items() if k != "max_ctr_complexity"}
    add_cat("combo_default_200", dict(default_combo, iterations=200, learning_rate=0.05,
                                      leaf_estimation_iterations=1), y_cat)
    return out


if __name__ == "__main__":
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "catboost_parity.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
