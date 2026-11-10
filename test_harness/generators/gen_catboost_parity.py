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

    base = dict(iterations=30, learning_rate=0.2, depth=4, boosting_type="Plain",
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
    add_cat("lg_ctr_rmse_perm", dict(cat, grow_policy="Lossguide", has_time=False, depth=5),
            y_cat_reg, kind="regressor")
    add_cat("lg_ctr", dict(cat, grow_policy="Lossguide", max_leaves=12), y_cat)
    add_cat("lg_ctr_defaults", dict(COMMON, iterations=60, grow_policy="Lossguide",
                                    cat_features=cat_cols, random_strength=1.0,
                                    has_time=False, bootstrap_type="MVS"), y_cat)
    add_cat("dw_onehot", dict(cat, grow_policy="Depthwise", one_hot_max_size=255), y_cat_reg,
            kind="regressor")

    # Huber, LogCosh and Lq (#209): Newton leaves for Huber and Lq with
    # q >= 2, Gradient below, Exact for LogCosh.
    y_heavy = y_reg + (rng.uniform(size=n) < 0.05) * rng.normal(scale=6.0, size=n)
    add("loss_huber", "regressor", dict(base, loss_function="Huber:delta=0.5"), X, y_heavy)
    add("loss_huber_weighted", "regressor", dict(base, loss_function="Huber:delta=1.5"), X,
        y_heavy, sw=weights)
    add("loss_logcosh", "regressor", dict(base, loss_function="LogCosh"), X, y_heavy)
    add("loss_logcosh_weighted", "regressor", dict(base, loss_function="LogCosh"), X, y_heavy,
        sw=weights)
    add("loss_lq_15", "regressor", dict(base, loss_function="Lq:q=1.5"), X, y_heavy)
    add("loss_lq_3", "regressor", dict(base, loss_function="Lq:q=3"), X, y_reg)
    add("loss_lq_2_ordered", "regressor",
        dict(base, loss_function="Lq:q=2", boosting_type="Ordered"), X, y_reg)
    add("loss_huber_defaults", "regressor",
        dict(COMMON, iterations=80, depth=4, loss_function="Huber:delta=1.0",
             random_strength=1.0, has_time=False, bootstrap_type="MVS"), X, y_heavy)
    add("loss_logcosh_defaults", "regressor",
        dict(COMMON, iterations=80, depth=4, loss_function="LogCosh", random_strength=1.0,
             has_time=False, bootstrap_type="MVS"), X, y_heavy)
    # CrossEntropy on probabilities (#209): the Logloss machinery with soft
    # targets, 10 Newton leaf steps with backtracking, learning rate 0.03.
    y_soft = 1 / (1 + np.exp(-eta)) * 0.9 + 0.05 * rng.uniform(size=n)
    add("loss_crossentropy", "classifier", dict(base, loss_function="CrossEntropy"), X, y_soft)
    add("loss_crossentropy_weighted", "classifier", dict(base, loss_function="CrossEntropy"),
        X, y_soft, sw=weights)
    add("loss_crossentropy_bfa", "classifier",
        dict(base, loss_function="CrossEntropy", boost_from_average=True), X, y_soft)
    add("loss_crossentropy_defaults", "classifier",
        dict(COMMON, iterations=60, depth=4, loss_function="CrossEntropy", random_strength=1.0,
             has_time=False, bootstrap_type="MVS"), X, y_soft)
    # MultiClassOneVsAll (#209): a sigmoid per class, diagonal Newton steps,
    # fmath's expd (scalar below four classes, its AVX2 blocks from four).
    ova = dict(base, loss_function="MultiClassOneVsAll")
    y5 = np.digitize(eta + rng.logistic(size=n) * 0.5, [-1.0, -0.2, 0.5, 1.2]).astype(float)
    add("loss_ova", "classifier", ova, X, y_mc.astype(float))
    add("loss_ova_weighted", "classifier", ova, X, y_mc.astype(float), sw=weights)
    add("loss_ova_5", "classifier", ova, X, y5)
    add("loss_ova_ordered", "classifier", dict(ova, boosting_type="Ordered"), X, y5)
    add("loss_ova_depthwise", "classifier", dict(ova, grow_policy="Depthwise"), X,
        y_mc.astype(float))
    add("loss_ova_defaults", "classifier",
        dict(COMMON, iterations=60, depth=4, loss_function="MultiClassOneVsAll",
             random_strength=1.0, has_time=False), X, y5)
    # RMSEWithUncertainty (#209): mean and log standard deviation, starting
    # from the target's mean and log std; predictions are [mean, variance].
    y_het = eta + rng.normal(size=n) * np.exp(0.4 * X[:, 1])
    unc = dict(base, loss_function="RMSEWithUncertainty")
    add("loss_uncertainty", "regressor", unc, X, y_het)
    add("loss_uncertainty_weighted", "regressor", unc, X, y_het, sw=weights)
    add("loss_uncertainty_ordered", "regressor", dict(unc, boosting_type="Ordered"), X, y_het)
    add("loss_uncertainty_defaults", "regressor",
        dict(COMMON, iterations=60, depth=4, loss_function="RMSEWithUncertainty",
             random_strength=1.0, has_time=False, bootstrap_type="MVS"), X, y_het)
    # MultiRMSE (#209): a target matrix, a diagonal Newton step per
    # target, each target starting from its mean.
    Y = np.column_stack([y_reg, y_het, 0.5 * eta - X[:, 3] + rng.normal(scale=0.3, size=n)])
    Yv = np.column_stack([yv_reg, yv_reg * 0.5 + rng.normal(size=150),
                          rng.normal(size=150)])
    multi = dict(base, loss_function="MultiRMSE")
    add("loss_multirmse", "regressor", multi, X, Y)
    add("loss_multirmse_2", "regressor", multi, X, Y[:, :2])
    add("loss_multirmse_weighted", "regressor", multi, X, Y, sw=weights)
    add("loss_multirmse_l2", "regressor", dict(multi, score_function="L2"), X, Y)
    add("loss_multirmse_no_average", "regressor", dict(multi, boost_from_average=False), X, Y)
    add("loss_multirmse_ordered", "regressor", dict(multi, boosting_type="Ordered"), X, Y)
    add("loss_multirmse_depthwise", "regressor", dict(multi, grow_policy="Depthwise"), X, Y)
    add("loss_multirmse_lossguide", "regressor", dict(multi, grow_policy="Lossguide"), X, Y)
    add("loss_multirmse_defaults", "regressor",
        dict(COMMON, iterations=60, depth=4, loss_function="MultiRMSE",
             random_strength=1.0, has_time=False, bootstrap_type="MVS"), X, Y)
    add_eval("eval_multirmse", "regressor", dict(over, loss_function="MultiRMSE"), X, Y, Xv, Yv,
             sw=weights, we=wv)

    # QueryRMSE (#210): consecutive query groups (CatBoost's group_id);
    # fold and data permutations shuffle whole queries.
    def groups_of(rows):
        sizes = []
        while sum(sizes) < rows:
            sizes.append(int(rng.integers(1, 13)))
        sizes[-1] -= sum(sizes) - rows
        return sizes

    def add_group(name, params, Xd, y, sizes, sw=None, eval_data=None):
        gid = np.repeat(np.arange(len(sizes)), sizes)
        cats = params.get("cat_features") or []

        def typed(A):
            if not cats:
                return A
            Ao = A.astype(object)
            for c in cats:
                Ao[:, c] = A[:, c].astype(int)
            return Ao

        pool = cb.Pool(typed(Xd), y, weight=sw, group_id=gid, cat_features=cats or None)
        fit_kw = {}
        if eval_data is not None:
            Xe, ye, ge = eval_data
            fit_kw["eval_set"] = cb.Pool(typed(Xe), ye,
                                         group_id=np.repeat(np.arange(len(ge)), ge),
                                         cat_features=cats or None)
        kw = {k: v for k, v in params.items() if k != "cat_features"}
        m = cb.CatBoostRanker(**kw).fit(pool, **fit_kw)
        X_test = Xd[::4]
        case = {
            "name": name, "kind": "regressor",
            "params": dict({k: v for k, v in params.items() if k not in COMMON or v != COMMON[k]},
                           # COMMON settings left at catboost's defaults.
                           **({"defaults": [k for k in ("random_strength", "bootstrap_type")
                                            if k not in params]}
                              if any(k not in params for k in ("random_strength",
                                                                "bootstrap_type")) else {})),
            "X": nulls(Xd), "y": y.tolist(), "group": sizes,
            "sample_weight": None if sw is None else sw.tolist(),
            "X_test": nulls(X_test), "predictions": m.predict(typed(X_test)).tolist(),
        }
        if eval_data is not None:
            evals = m.get_evals_result()["validation"]
            case["eval"] = {"X": nulls(Xe), "y": ye.tolist(), "group": ge,
                            "sample_weight": None, "metrics": evals[next(iter(evals))],
                            "best_iteration": m.get_best_iteration(),
                            "tree_count": m.tree_count_}
        out["models"].append(case)

    sizes = groups_of(n)
    y_q = y_reg + np.repeat(rng.normal(scale=2.0, size=len(sizes)), sizes)
    qr = dict(base, loss_function="QueryRMSE")
    add_group("loss_query_rmse", qr, X, y_q, sizes)
    add_group("loss_query_rmse_weighted", qr, X, y_q, sizes, sw=weights)
    add_group("loss_query_rmse_ordered", dict(qr, boosting_type="Ordered"), X, y_q, sizes)
    add_group("loss_query_rmse_permuted", dict(qr, boosting_type="Ordered", has_time=False),
              X, y_q, sizes)
    add_group("loss_query_rmse_depthwise", dict(qr, grow_policy="Depthwise"), X, y_q, sizes)
    # A groupwise loss defaults to one_hot_max_size=10.
    add_group("loss_query_rmse_cat", dict(qr, cat_features=cat_cols, has_time=False), Xc, y_q,
              sizes)
    add_group("loss_query_rmse_defaults", dict(COMMON, iterations=60, depth=4,
              loss_function="QueryRMSE", random_strength=1.0, has_time=False,
              bootstrap_type="MVS"), X, y_q, sizes)
    sizes_v = groups_of(150)
    yv_q = yv_reg + np.repeat(rng.normal(scale=2.0, size=len(sizes_v)), sizes_v)
    add_group("eval_query_rmse", dict(over, loss_function="QueryRMSE"), X, y_q, sizes,
              eval_data=(Xv, yv_q, sizes_v))
    # PairLogit (#210): every pair of a query's rows with different
    # relevance, on stored exp(a).
    y_rel = np.digitize(y_q - np.repeat([y_q[a:b].mean() for a, b in
                        zip(np.cumsum([0] + sizes[:-1]), np.cumsum(sizes))], sizes),
                        [-1.0, 0.0, 1.0]).astype(float)
    pl = dict(base, loss_function="PairLogit")
    add_group("loss_pair_logit", pl, X, y_rel, sizes)
    add_group("loss_pair_logit_l2", dict(pl, score_function="L2"), X, y_rel, sizes)
    add_group("loss_pair_logit_ordered", dict(pl, boosting_type="Ordered"), X, y_rel, sizes)
    add_group("loss_pair_logit_permuted", dict(pl, boosting_type="Ordered", has_time=False),
              X, y_rel, sizes)
    add_group("loss_pair_logit_depthwise", dict(pl, grow_policy="Depthwise"), X, y_rel, sizes)
    add_group("loss_pair_logit_lossguide", dict(pl, grow_policy="Lossguide"), X, y_rel, sizes)
    add_group("loss_pair_logit_defaults", dict(COMMON, iterations=60, depth=4,
              loss_function="PairLogit", random_strength=1.0, has_time=False,
              bootstrap_type="MVS"), X, y_rel, sizes)
    yv_rel = np.digitize(yv_q - yv_q.mean(), [-1.0, 0.0, 1.0]).astype(float)
    add_group("eval_pair_logit", dict(over, loss_function="PairLogit"), X, y_rel, sizes,
              eval_data=(Xv, yv_rel, sizes_v))
    # YetiRank (#210): PairLogit on pairs drawn every iteration from noisy
    # rankings by the current scores (CatBoost's default l2_leaf_reg 0).
    yr = dict(base, loss_function="YetiRank")
    add_group("loss_yetirank", yr, X, y_rel, sizes)
    add_group("loss_yetirank_l2", dict(yr, l2_leaf_reg=3.0), X, y_rel, sizes)
    add_group("loss_yetirank_params", dict(yr, loss_function="YetiRank:permutations=3;decay=0.9"),
              X, y_rel, sizes)
    add_group("loss_yetirank_depthwise", dict(yr, grow_policy="Depthwise"), X, y_rel, sizes)
    add_group("loss_yetirank_permuted", dict(yr, has_time=False, cat_features=cat_cols),
              Xc, y_rel, sizes)
    add_group("loss_yetirank_defaults", dict(COMMON, iterations=60, depth=4,
              loss_function="YetiRank", random_strength=1.0, has_time=False,
              bootstrap_type="MVS"), X, y_rel, sizes)
    # Its eval metric: NDCG, or PFound for targets in [0, 1] (maximized).
    yr_over = dict(over, loss_function="YetiRank")
    add_group("eval_yetirank_ndcg", yr_over, X, y_rel, sizes,
              eval_data=(Xv, yv_rel, sizes_v))
    add_group("eval_yetirank_pfound", yr_over, X, y_rel / 3, sizes,
              eval_data=(Xv, yv_rel / 3, sizes_v))
    # Pairwise scoring (#210): splits and leaves scored on the pairs
    # (Gradient leaves, border_count 32, Bernoulli 0.5 pairs by default);
    # X has a binary feature for the binary-pack scores.
    Xb = np.column_stack([X, (X[:, 0] > 0.3).astype(float)])
    plp = dict(base, loss_function="PairLogitPairwise")
    add_group("loss_pair_logit_pairwise", plp, Xb, y_rel, sizes)
    add_group("loss_pair_logit_pairwise_l2", dict(plp, l2_leaf_reg=1.0, border_count=16), Xb, y_rel, sizes)
    add_group("loss_pair_logit_pairwise_bernoulli",
              dict(plp, bootstrap_type="Bernoulli", subsample=0.5), Xb, y_rel, sizes)
    add_group("loss_pair_logit_pairwise_depth6", dict(plp, depth=6, iterations=10), Xb, y_rel,
              sizes)
    # From 200 iterations: 50 Gradient leaf steps with backtracking.
    add_group("loss_pair_logit_pairwise_steps", dict(plp, iterations=200, depth=3,
              learning_rate=0.1), Xb, y_rel, sizes)
    yrp = dict(base, loss_function="YetiRankPairwise")
    add_group("loss_yetirank_pairwise", yrp, Xb, y_rel, sizes)
    add_group("loss_yetirank_pairwise_bayesian",
              dict(yrp, bootstrap_type="Bayesian", bagging_temperature=1.0), Xb, y_rel, sizes)
    pw_defaults = {k: v for k, v in COMMON.items() if k not in ("random_strength", "bootstrap_type")}
    add_group("loss_pair_logit_pairwise_defaults", dict(pw_defaults, iterations=60, depth=4,
              loss_function="PairLogitPairwise", has_time=False), Xb, y_rel, sizes)
    add_group("loss_yetirank_pairwise_defaults", dict(pw_defaults, iterations=60, depth=4,
              loss_function="YetiRankPairwise", has_time=False), Xb, y_rel, sizes)

    # Text features (#211): the default processing for regression, BoW over
    # BiGram and Word dictionaries of space-split tokens.
    import pandas as pd
    vocab = ["red", "green", "blue", "big", "small", "fast", "slow", "old", "new", "cat"]
    def texts(rows, eff):
        out = []
        for _ in range(rows):
            k = int(rng.integers(0, 6))
            out.append(" ".join(vocab[int(i)] for i in rng.integers(0, len(vocab), size=k)))
        return out
    def add_text(name, params, Xd, cols, y, sw=None, kind="regressor"):
        frame = pd.DataFrame({f"f{j}": Xd[:, j] for j in range(Xd.shape[1])})
        for t, col in enumerate(cols):
            frame[f"t{t}"] = col
        text_idx = list(range(Xd.shape[1], Xd.shape[1] + len(cols)))
        cls = cb.CatBoostRegressor if kind == "regressor" else cb.CatBoostClassifier
        m = cls(**params).fit(cb.Pool(frame, y, weight=sw, text_features=text_idx))
        test = frame.iloc[::4]
        tp = cb.Pool(test, text_features=text_idx)
        out["models"].append({
            "name": name, "kind": kind,
            "params": dict({k: v for k, v in params.items() if k not in COMMON or v != COMMON[k]},
                           **({"defaults": [k for k in ("random_strength", "bootstrap_type")
                                            if k not in params]}
                              if any(k not in params for k in ("random_strength",
                                                                "bootstrap_type")) else {})),
            "X": nulls(Xd), "y": y.tolist(), "text": [list(c) for c in cols],
            "sample_weight": None if sw is None else sw.tolist(),
            "X_test": nulls(Xd[::4]), "text_test": [list(c[::4]) for c in cols],
            "predictions": (m.predict(tp) if kind == "regressor" else m.predict_proba(tp)).tolist(),
        })
    t0 = texts(n, None)
    t1 = texts(n, None)
    has = lambda col, w: np.array([w in s.split(" ") for s in col], float)
    y_text = 0.5 * X[:, 0] + has(t0, "red") - 0.7 * has(t0, "slow") + 0.4 * has(t1, "cat") \
        + rng.normal(scale=0.3, size=n)
    add_text("text_bow", base, X, [t0], y_text)
    add_text("text_bow_two_columns", base, X, [t0, t1], y_text)
    add_text("text_bow_weighted", base, X, [t0], y_text, sw=weights)
    add_text("text_bow_ordered", dict(base, boosting_type="Ordered", has_time=False), X, [t0],
             y_text)
    add_text("text_bow_defaults", dict({k: v for k, v in COMMON.items()
                                        if k not in ("random_strength", "bootstrap_type")},
                                       iterations=60, depth=4, has_time=False), X, [t0, t1], y_text)
    # Classification: BoW and the online NaiveBayes of the Word tokens.
    y_tb = (y_text > np.median(y_text)).astype(float)
    y_tm = np.digitize(y_text, np.quantile(y_text, [0.33, 0.66])).astype(float)
    add_text("text_nb_binary", base, X, [t0], y_tb, kind="classifier")
    add_text("text_nb_two_columns", base, X, [t0, t1], y_tb, kind="classifier")
    add_text("text_nb_multiclass", dict(base, iterations=20), X, [t0], y_tm, kind="classifier")
    add_text("text_nb_permuted", dict(base, has_time=False, boosting_type="Ordered"), X, [t0],
             y_tb, kind="classifier")
    add_text("text_nb_defaults", dict({k: v for k, v in COMMON.items()
                                       if k not in ("random_strength", "bootstrap_type")},
                                      iterations=60, depth=4, has_time=False), X, [t0, t1], y_tb,
             kind="classifier")

    # Custom text processing (`text_processing`) and BM25.
    def text_options(tp):
        """Our CatBoostTextOptions for a catboost text_processing dict."""
        tok = tp["tokenizers"][0]
        calcer = {"NaiveBayes": "NaiveBayes", "BM25": "Bm25"}
        def calcers(names):
            out_c = []
            for c in names:
                name, _, rest = c.partition(":")
                args = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
                out_c.append({"BoW": {"top_tokens_count": int(args.get("top_tokens_count", 2000))}}
                             if name == "BoW" else calcer[name])
            return out_c
        return {
            "delimiter": tok.get("delimiter", " "),
            "lowercasing": tok.get("lowercasing", "false") == "true",
            "dictionaries": [{"name": d["dictionary_id"], "gram_order": int(d.get("gram_order", 1)),
                              "occurrence_lower_bound": int(d.get("occurrence_lower_bound", 3)),
                              "max_dictionary_size": int(d.get("max_dictionary_size", 50000))}
                             for d in tp["dictionaries"]],
            "processing": [{"dictionaries": f["dictionaries_names"],
                            "calcers": calcers(f["feature_calcers"])}
                           for f in tp["feature_processing"]["default"]],
        }
    def add_custom(name, params, tp, cols, y, kind):
        add_text(name, dict(params, text_processing=tp), X, cols, y, kind=kind)
        case = out["models"][-1]
        case["params"].pop("text_processing")
        case["text_options"] = text_options(tp)
    def processing(dicts, feats, delimiter=" ", lowercasing="false"):
        return {"tokenizers": [{"tokenizer_id": "T", "separator_type": "ByDelimiter",
                                "delimiter": delimiter, "lowercasing": lowercasing}],
                "dictionaries": dicts,
                "feature_processing": {"default": [dict(f, tokenizers_names=["T"])
                                                   for f in feats]}}
    word = {"dictionary_id": "W", "gram_order": "1", "occurrence_lower_bound": "2"}
    add_custom("text_bm25_binary", base,
               processing([word], [{"dictionaries_names": ["W"],
                                    "feature_calcers": ["BoW", "BM25"]}]), [t0], y_tb,
               "classifier")
    add_custom("text_nb_bm25_multiclass", dict(base, iterations=20),
               processing([word], [{"dictionaries_names": ["W"],
                                    "feature_calcers": ["BM25", "NaiveBayes"]}]), [t0, t1], y_tm,
               "classifier")
    add_custom("text_bm25_permuted", dict(base, has_time=False, boosting_type="Ordered"),
               processing([word], [{"dictionaries_names": ["W"],
                                    "feature_calcers": ["BM25"]}]), [t0], y_tb, "classifier")
    # Rare and mixed-case words, comma separated: lowercasing, trigrams,
    # dictionary bounds and sizes, BoW top tokens.
    big = [f"w{i}" for i in range(60)]
    def rare_texts(rows):
        res = []
        for _ in range(rows):
            k = int(rng.integers(0, 7))
            words = [big[min(int(rng.zipf(1.3)) - 1, 59)] for _ in range(k)]
            res.append(",".join(w.upper() if rng.random() < 0.3 else w for w in words))
        return res
    t2 = rare_texts(n)
    y_rare = 0.5 * X[:, 0] + np.array([("w0" in s.lower().split(",")) for s in t2], float) \
        + rng.normal(scale=0.3, size=n)
    custom = processing(
        [{"dictionary_id": "Tri", "gram_order": "3", "occurrence_lower_bound": "2"},
         {"dictionary_id": "W", "gram_order": "1", "occurrence_lower_bound": "4",
          "max_dictionary_size": "12"}],
        [{"dictionaries_names": ["W", "Tri"], "feature_calcers": ["BoW:top_tokens_count=8"]}],
        delimiter=",", lowercasing="true")
    add_custom("text_custom_regressor", base, custom, [t2], y_rare, "regressor")
    add_custom("text_custom_classifier", base, dict(custom, feature_processing={"default": [
        dict(custom["feature_processing"]["default"][0]),
        {"dictionaries_names": ["W"], "feature_calcers": ["NaiveBayes", "BM25"],
         "tokenizers_names": ["T"]}]}), [t2], (y_rare > np.median(y_rare)).astype(float),
        "classifier")
    # A dictionary without occurrence_lower_bound: catboost's default.
    add_custom("text_default_bound", base,
               processing([{"dictionary_id": "W", "gram_order": "1"}],
                          [{"dictionaries_names": ["W"], "feature_calcers": ["BoW"]}],
                          delimiter=","), [t2], y_rare, "regressor")
    return out


if __name__ == "__main__":
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "catboost_parity.json")
    data = generate()
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {path}")
