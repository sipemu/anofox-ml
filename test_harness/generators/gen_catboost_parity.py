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
            "params": {k: v for k, v in params.items() if k not in COMMON or v != COMMON[k]},
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
            "params": {k: v for k, v in params.items() if k not in COMMON or v != COMMON[k]},
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

    # Several leaf estimation steps (`leaf_estimation_iterations`), with
    # CatBoost's backtracking (a step is halved until the objective
    # improves) or without. Defaults: Logloss takes 10 Newton steps from 200
    # iterations or 20 features on.
    add("leaf10_logloss", "classifier", dict(base, leaf_estimation_iterations=10), X, y_bin)
    add("leaf10_logloss_no_backtracking", "classifier",
        dict(base, leaf_estimation_iterations=10, leaf_estimation_backtracking="No"), X, y_bin)
    add("leaf10_logloss_weighted", "classifier", dict(base, leaf_estimation_iterations=10), X,
        y_bin, sw=weights)
    add("leaf4_logloss_gradient", "classifier", dict(base, leaf_estimation_method="Gradient",
                                                     leaf_estimation_iterations=4), X, y_bin)
    add("leaf_default_logloss_200", "classifier", dict(base, iterations=200, learning_rate=0.05),
        X, y_bin)
    X20 = np.column_stack([X, rng.normal(size=(n, 16))])
    add("leaf_default_logloss_20_features", "classifier", base, X20, y_bin)
    add("leaf_default_mc_gradient_20_features", "classifier",
        dict(mc, leaf_estimation_method="Gradient"), X20, y_mc.astype(float))
    add("leaf3_rmse", "regressor", dict(base, leaf_estimation_iterations=3), X, y_reg)
    add("leaf5_mc", "classifier", dict(mc, leaf_estimation_iterations=5), X, y_mc.astype(float))
    add("leaf10_ordered_logloss", "classifier", dict(ordered, leaf_estimation_iterations=10), X,
        y_bin)
    add("leaf5_ordered_mc", "classifier", dict(mc_ordered, leaf_estimation_iterations=5), X,
        y_mc.astype(float))
    add_cat("leaf10_cat_logloss", dict(cat, leaf_estimation_iterations=10), y_cat)

    # Permutations (`has_time=False`): CatBoost's generator (seeded by
    # `random_seed`) shuffles the learning folds but the first and, with
    # CTRs, the averaging fold; each iteration searches its tree on a fold
    # drawn at random. Score noise and bootstrap stay off.
    perm = dict(has_time=False)
    add_cat("perm_cat_logloss", dict(cat, **perm), y_cat)
    add_cat("perm_cat_logloss_seed7", dict(cat, random_seed=7, **perm), y_cat)
    add_cat("perm_cat_rmse", dict(cat, **perm), y_cat_reg, kind="regressor")
    add_cat("perm_cat_mc", dict(cat_mc, **perm), y_cat_mc)
    add_cat("perm_cat_weighted", dict(cat, **perm), y_cat, sw=weights)
    add_cat("perm_combo", dict(combo, **perm), y_cat)
    add_cat("perm_ordered_cat", dict(cat, boosting_type="Ordered", **perm), y_cat)
    add_cat("perm_leaf10_cat", dict(cat, leaf_estimation_iterations=10, **perm), y_cat)
    add("perm_ordered_rmse", "regressor", dict(ordered, **perm), X, y_reg)
    add("perm_ordered_logloss", "classifier", dict(ordered, **perm), X, y_bin)
    add("perm_ordered_mc", "classifier", dict(mc_ordered, **perm), X, y_mc.astype(float))
    add("perm_plain_rmse", "regressor", dict(base, **perm), X, y_reg)
    add("perm_ordered_rmse_1200", "regressor", dict(ordered, iterations=15, **perm), big, y_big)

    # Score noise (`random_strength`, CatBoost's default 1): normal noise on
    # every split's score from a generator per candidate, and on every
    # candidate's best score from the main one.
    noise = dict(perm, random_strength=1.0)
    add("noise_rmse", "regressor", dict(base, **noise), X, y_reg)
    add("noise_rmse_half", "regressor", dict(base, **dict(noise, random_strength=0.5)), X, y_reg)
    add("noise_logloss", "classifier", dict(base, **noise), X, y_bin)
    add("noise_ordered_rmse", "regressor", dict(ordered, **noise), X, y_reg)
    add("noise_mc", "classifier", dict(mc, **noise), X, y_mc.astype(float))
    add_cat("noise_cat_logloss", dict(cat, **noise), y_cat)
    add_cat("noise_ordered_cat", dict(cat, boosting_type="Ordered", **noise), y_cat)
    add_cat("noise_combo", dict(combo, **noise), y_cat)
    add_cat("noise_combo_ordered", dict(combo, boosting_type="Ordered", **noise), y_cat)
    add_cat("noise_combo_mc", dict(combo, loss_function="MultiClass", depth=6, **noise), y_cat_mc)
    add_cat("noise_combo_seed11", dict(combo, depth=6, random_seed=11, **noise), y_cat_reg,
            kind="regressor")
    add_cat("noise_leaf10_combo", dict(combo, leaf_estimation_iterations=10, **noise), y_cat)

    # Bootstrap (CatBoost's default MVS with subsample 0.8; Bayesian for
    # MultiClass), drawn per tree: it weights and picks the scored rows.
    def boot(**kw):
        return dict(noise, **kw)
    add("boot_mvs_rmse", "regressor", dict(base, **boot(bootstrap_type="MVS")), X, y_reg)
    add("boot_mvs_logloss", "classifier", dict(base, **boot(bootstrap_type="MVS")), X, y_bin)
    add("boot_mvs_ordered", "regressor", dict(ordered, **boot(bootstrap_type="MVS")), X, y_reg)
    add("boot_mvs_half_reg", "regressor",
        dict(base, **boot(bootstrap_type="MVS", subsample=0.5, mvs_reg=0.1)), X, y_reg)
    add("boot_bayesian", "regressor", dict(base, **boot(bootstrap_type="Bayesian")), X, y_reg)
    add("boot_bayesian_t3", "classifier",
        dict(base, **boot(bootstrap_type="Bayesian", bagging_temperature=3.0)), X, y_bin)
    add("boot_bernoulli", "regressor",
        dict(base, **boot(bootstrap_type="Bernoulli", subsample=0.7)), X, y_reg)
    add("boot_bernoulli_ordered", "classifier",
        dict(ordered, **boot(bootstrap_type="Bernoulli")), X, y_bin)
    add("boot_mc_default", "classifier", dict(mc, **boot(bootstrap_type="Bayesian")), X,
        y_mc.astype(float))
    add("boot_weighted", "regressor", dict(base, **boot(bootstrap_type="MVS")), X, y_reg,
        sw=weights)
    add_cat("boot_combo", dict(combo, **boot(bootstrap_type="MVS")), y_cat)
    add_cat("boot_ordered_combo", dict(combo, boosting_type="Ordered",
                                       **boot(bootstrap_type="MVS")), y_cat)

    # CatBoost's automatic learning rate (no learning_rate, l2_leaf_reg,
    # leaf estimation settings given): from the row and iteration counts.
    auto = dict(iterations=60, depth=4, **COMMON)
    add("auto_lr_rmse", "regressor", auto, X, y_reg)
    add("auto_lr_rmse_no_average", "regressor", dict(auto, boost_from_average=False), X, y_reg)
    add("auto_lr_logloss", "classifier", auto, X, y_bin)
    add("auto_lr_mc", "classifier", dict(auto, loss_function="MultiClass"), X,
        y_mc.astype(float))
    add("auto_lr_l2_set", "regressor", dict(auto, l2_leaf_reg=1.0), X, y_reg)
    add_cat("auto_lr_defaults", dict(COMMON, iterations=250, cat_features=cat_cols,
                                      random_strength=1.0, has_time=False,
                                      bootstrap_type="MVS"), y_cat)

    # Class weights (#156): multiplied into the sample weights; the
    # automatic ones from the classes' weight sums (truncated to integers).
    y_imb = (eta + rng.logistic(size=n) > 1.2).astype(float)
    add("cw_explicit", "classifier", dict(base, class_weights=[1.0, 2.5]), X, y_imb)
    add("cw_balanced", "classifier", dict(base, auto_class_weights="Balanced"), X, y_imb)
    add("cw_sqrt_balanced", "classifier", dict(base, auto_class_weights="SqrtBalanced"), X,
        y_imb)
    add("cw_balanced_weighted", "classifier", dict(base, auto_class_weights="Balanced"), X,
        y_imb, sw=weights)
    add("cw_balanced_average", "classifier",
        dict(base, auto_class_weights="Balanced", boost_from_average=True), X, y_imb)
    add("cw_ordered", "classifier",
        dict(ordered, auto_class_weights="SqrtBalanced"), X, y_imb)
    add("cw_mc_balanced", "classifier", dict(mc, auto_class_weights="Balanced"), X,
        y_mc5.astype(float))
    add("cw_mc_explicit", "classifier", dict(mc, class_weights=[0.5, 1.0, 3.0]), X,
        y_mc.astype(float), sw=weights)
    add_cat("cw_cat_defaults", dict(COMMON, iterations=80, depth=4, cat_features=cat_cols,
                                     random_strength=1.0, has_time=False, bootstrap_type="MVS",
                                     auto_class_weights="Balanced"), y_cat)

    # Monotone constraints (#156): isotonic leaf values in the split scores
    # and the leaf estimation, and model shrinkage (default 0.01 with them).
    mono = [1, 1, -1, 0]
    add("mono_rmse", "regressor", dict(base, monotone_constraints=mono), X, y_reg)
    add("mono_rmse_l2_depth6", "regressor",
        dict(base, monotone_constraints=mono, score_function="L2", depth=6, iterations=15),
        X, y_reg)
    add("mono_rmse_gradient", "regressor",
        dict(base, monotone_constraints=mono, leaf_estimation_method="Gradient"), X, y_reg)
    add("mono_rmse_weighted", "regressor", dict(base, monotone_constraints=mono), X, y_reg,
        sw=weights)
    add("mono_rmse_no_shrink", "regressor",
        dict(base, monotone_constraints=mono, model_shrink_rate=0.0), X, y_reg)
    add("mono_rmse_decreasing", "regressor",
        dict(base, monotone_constraints=mono, model_shrink_mode="Decreasing"), X, y_reg)
    add("mono_ordered_rmse", "regressor", dict(ordered, monotone_constraints=mono), X, y_reg)
    add("mono_logloss", "classifier", dict(base, monotone_constraints=mono), X, y_bin)
    add("mono_leaf10_logloss", "classifier",
        dict(base, monotone_constraints=mono, leaf_estimation_iterations=10), X, y_bin)
    add("mono_ordered_logloss", "classifier", dict(ordered, monotone_constraints=[0, 1, 0, 1]),
        X, y_bin)
    add("shrink_only_rmse", "regressor", dict(base, model_shrink_rate=0.05), X, y_reg)
    add("shrink_decreasing_logloss", "classifier",
        dict(base, model_shrink_rate=0.3, model_shrink_mode="Decreasing"), X, y_bin)
    add_cat("mono_cat_logloss", dict(cat, monotone_constraints=[1, 0, 1, 0, 0]), y_cat)
    add("mono_defaults", "regressor",
        dict(COMMON, iterations=80, depth=4, monotone_constraints=mono, random_strength=1.0,
             has_time=False, bootstrap_type="MVS"), X, y_reg)

    # More regression losses (#153), with CatBoost's per-loss defaults:
    # Exact leaves for MAE / Quantile, boost_from_average for RMSE, MAE and
    # Quantile, Newton steps for Expectile / Poisson (log scale, stored
    # exp(a)) / Tweedie, Gradient for LogLinQuantile.
    mu = np.exp(0.5 * eta)
    y_count = rng.poisson(mu).astype(float)
    y_tweedie = np.where(rng.uniform(size=n) < 0.3, 0.0, rng.gamma(2.0, mu / 2.0))
    y_pos = np.exp(0.5 * eta + rng.normal(scale=0.3, size=n))
    q3 = dict(base, loss_function="Quantile:alpha=0.3")
    add("loss_mae", "regressor", dict(base, loss_function="MAE"), X, y_reg)
    add("loss_quantile_03", "regressor", q3, X, y_reg)
    add("loss_quantile_08_weighted", "regressor", dict(base, loss_function="Quantile:alpha=0.8"),
        X, y_reg, sw=weights)
    add("loss_quantile_ordered", "regressor", dict(q3, boosting_type="Ordered"), X, y_reg)
    add("loss_quantile_gradient", "regressor", dict(q3, leaf_estimation_method="Gradient"), X,
        y_reg)
    add("loss_quantile_small", "regressor", q3, X[:80], y_reg[:80])
    add("loss_quantile_no_average", "regressor", dict(q3, boost_from_average=False), X, y_reg)
    add("loss_expectile", "regressor", dict(base, loss_function="Expectile:alpha=0.3"), X, y_reg)
    add("loss_expectile_leaf5", "regressor",
        dict(base, loss_function="Expectile:alpha=0.7", leaf_estimation_iterations=5), X, y_reg)
    add("loss_poisson", "regressor", dict(base, loss_function="Poisson"), X, y_count)
    add("loss_poisson_leaf10", "regressor",
        dict(base, loss_function="Poisson", leaf_estimation_iterations=10), X, y_count)
    add("loss_poisson_weighted", "regressor", dict(base, loss_function="Poisson"), X, y_count,
        sw=weights)
    add("loss_poisson_ordered", "regressor",
        dict(base, loss_function="Poisson", boosting_type="Ordered"), X, y_count)
    add("loss_tweedie", "regressor", dict(base, loss_function="Tweedie:variance_power=1.5"), X,
        y_tweedie)
    add("loss_tweedie_weighted", "regressor",
        dict(base, loss_function="Tweedie:variance_power=1.2"), X, y_tweedie, sw=weights)
    add("loss_loglinquantile", "regressor", dict(base, loss_function="LogLinQuantile:alpha=0.4"),
        X, y_pos)
    add("loss_mae_defaults", "regressor",
        dict(COMMON, iterations=80, depth=4, loss_function="MAE", random_strength=1.0,
             has_time=False, bootstrap_type="MVS"), X, y_reg)
    add("loss_poisson_200", "regressor",
        dict(base, loss_function="Poisson", iterations=200, learning_rate=0.05), X, y_count)
    add_cat("loss_quantile_cat", dict(cat, loss_function="Quantile:alpha=0.6"), y_cat_reg,
            kind="regressor")

    # Eval sets (#155): the objective's metric per iteration, the
    # overfitting detector and use_best_model (on by default with an eval
    # set, which also switches the automatic learning rate's coefficients).
    Xv = np.column_stack([rng.normal(size=150), rng.uniform(-2, 2, size=150),
                          np.round(rng.normal(size=150) * 3), rng.exponential(size=150)])
    eta_v = np.sin(2 * Xv[:, 0]) + 0.6 * Xv[:, 1] - 0.2 * Xv[:, 2] + 0.3 * Xv[:, 3]
    yv_reg = eta_v + rng.normal(scale=0.4, size=150)
    yv_bin = (eta_v + rng.logistic(size=150) > 0).astype(float)
    yv_mc = np.digitize(eta_v + rng.logistic(size=150) * 0.5, [-0.5, 0.7]).astype(float)
    wv = rng.uniform(0.2, 3.0, size=150)

    def add_eval(name, kind, params, Xd, y, Xe, ye, sw=None, we=None, fit_kw=None):
        cls = cb.CatBoostRegressor if kind == "regressor" else cb.CatBoostClassifier
        m = cls(**params).fit(Xd, y, sample_weight=sw, eval_set=cb.Pool(Xe, ye, weight=we),
                              **(fit_kw or {}))
        X_test = Xd[::4]
        pred = m.predict(X_test) if kind == "regressor" else m.predict_proba(X_test)
        evals = m.get_evals_result()["validation"]
        metric = next(iter(evals))
        out["models"].append({
            "name": name, "kind": kind,
            "params": dict({k: v for k, v in params.items() if k not in COMMON or v != COMMON[k]},
                           **(fit_kw or {})),
            "X": nulls(Xd), "y": y.tolist(),
            "sample_weight": None if sw is None else sw.tolist(),
            "eval": {"X": nulls(Xe), "y": ye.tolist(),
                     "sample_weight": None if we is None else we.tolist(),
                     "metrics": evals[metric], "best_iteration": m.get_best_iteration(),
                     "tree_count": m.tree_count_},
            "X_test": nulls(X_test), "predictions": pred.tolist(),
        })

    over = dict(base, iterations=80, learning_rate=0.4)
    add_eval("eval_rmse_best", "regressor", over, X, y_reg, Xv, yv_reg)
    add_eval("eval_rmse_early", "regressor", over, X, y_reg, Xv, yv_reg,
             fit_kw=dict(early_stopping_rounds=5))
    add_eval("eval_rmse_no_best", "regressor", over, X, y_reg, Xv, yv_reg,
             fit_kw=dict(use_best_model=False))
    add_eval("eval_rmse_inctodec", "regressor", dict(over, od_type="IncToDec", od_pval=0.01,
                                                     od_wait=5), X, y_reg, Xv, yv_reg)
    add_eval("eval_rmse_weighted", "regressor", over, X, y_reg, Xv, yv_reg, sw=weights, we=wv,
             fit_kw=dict(early_stopping_rounds=8))
    add_eval("eval_rmse_auto_lr", "regressor", dict(COMMON, iterations=60, depth=4), X, y_reg,
             Xv, yv_reg)
    add_eval("eval_logloss", "classifier", over, X, y_bin, Xv, yv_bin,
             fit_kw=dict(early_stopping_rounds=6))
    add_eval("eval_logloss_auto_lr", "classifier", dict(COMMON, iterations=60, depth=4), X,
             y_bin, Xv, yv_bin)
    add_eval("eval_mc", "classifier", dict(mc, iterations=60, learning_rate=0.4), X,
             y_mc.astype(float), Xv, yv_mc, fit_kw=dict(early_stopping_rounds=5))
    add_eval("eval_quantile", "regressor", dict(over, loss_function="Quantile:alpha=0.3"), X,
             y_reg, Xv, yv_reg, fit_kw=dict(early_stopping_rounds=6))
    add_eval("eval_poisson", "regressor", dict(over, loss_function="Poisson"), X, y_count, Xv,
             np.round(np.exp(0.5 * eta_v)), fit_kw=dict(early_stopping_rounds=6))
    add_eval("eval_monotone_shrink", "regressor", dict(over, monotone_constraints=[1, 1, -1, 0]),
             X, y_reg, Xv, yv_reg, fit_kw=dict(early_stopping_rounds=5))
    add_eval("eval_class_weights", "classifier", dict(over, auto_class_weights="Balanced"), X,
             y_imb, Xv, yv_bin, fit_kw=dict(early_stopping_rounds=6))
    add_eval("eval_defaults", "regressor",
             dict(COMMON, iterations=120, depth=4, random_strength=1.0, has_time=False,
                  bootstrap_type="MVS"), X, y_reg, Xv, yv_reg,
             fit_kw=dict(early_stopping_rounds=10))

    # Depthwise trees (#154; plain boosting, numeric and one-hot features).
    dw = dict(base, grow_policy="Depthwise")
    add("dw_rmse", "regressor", dw, X, y_reg)
    add("dw_rmse_l2_depth6", "regressor", dict(dw, score_function="L2", depth=6), X, y_reg)
    add("dw_rmse_min_data", "regressor", dict(dw, depth=6, min_data_in_leaf=15), X, y_reg)
    add("dw_rmse_weighted", "regressor", dw, X, y_reg, sw=weights)
    add("dw_rmse_nan", "regressor", dw, X_nan, y_reg)
    add("dw_logloss", "classifier", dw, X, y_bin)
    add("dw_logloss_leaf10", "classifier", dict(dw, leaf_estimation_iterations=10), X, y_bin)
    add("dw_mc", "classifier", dict(mc, grow_policy="Depthwise"), X, y_mc.astype(float))
    add("dw_quantile", "regressor", dict(dw, loss_function="Quantile:alpha=0.3"), X, y_reg)
    add("dw_noise", "regressor", dict(dw, random_strength=1.0), X, y_reg)
    add("dw_mvs", "regressor", dict(dw, random_strength=1.0, bootstrap_type="MVS"), X, y_reg)
    add("dw_defaults", "regressor",
        dict(COMMON, iterations=80, grow_policy="Depthwise", random_strength=1.0,
             has_time=False, bootstrap_type="MVS"), X, y_reg)
    lg = dict(base, grow_policy="Lossguide")
    add("lg_rmse", "regressor", lg, X, y_reg)
    add("lg_rmse_leaves8_depth3", "regressor", dict(lg, max_leaves=8, depth=3), X, y_reg)
    add("lg_rmse_l2_min_data", "regressor", dict(lg, score_function="L2", min_data_in_leaf=10),
        X, y_reg)
    add("lg_rmse_weighted", "regressor", dict(lg, max_leaves=12), X, y_reg, sw=weights)
    add("lg_logloss", "classifier", lg, X, y_bin)
    add("lg_mc", "classifier", dict(mc, grow_policy="Lossguide", max_leaves=10), X,
        y_mc.astype(float))
    add("lg_noise_mvs", "regressor", dict(lg, random_strength=1.0, bootstrap_type="MVS"), X,
        y_reg)
    add("lg_defaults", "regressor",
        dict(COMMON, iterations=80, grow_policy="Lossguide", random_strength=1.0,
             has_time=False, bootstrap_type="MVS"), X, y_reg)
    add_cat("dw_onehot", dict(cat, grow_policy="Depthwise", one_hot_max_size=255), y_cat_reg,
            kind="regressor")
    return out


if __name__ == "__main__":
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "catboost_parity.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
