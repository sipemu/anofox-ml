"""Golden data for LgbmRegressor / LgbmClassifier against LightGBM 4.7.0 (#127).

`deterministic=True, force_row_wise=True`, one thread. Bagging, GOSS and
the feature fractions reproduce LightGBM's own generators (#152).
The tweedie, cross_entropy and multiclassova objectives are #206.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np


COMMON = dict(deterministic=True, force_row_wise=True, verbose=-1, n_jobs=1)


def nulls(a):
    return np.where(np.isnan(a), None, a).tolist()


def generate():
    rng = np.random.default_rng(4600)
    n = 400
    X = np.column_stack([
        rng.normal(size=n),
        rng.uniform(-2, 2, size=n),
        rng.integers(0, 8, size=n).astype(float),
        rng.normal(size=n),
    ])
    X[rng.choice(n, 25, replace=False), 3] = np.nan
    eta = np.sin(2 * X[:, 0]) + 0.6 * X[:, 1] - 0.2 * X[:, 2] + 0.3 * np.nan_to_num(X[:, 3])
    weights = rng.uniform(0.2, 3.0, size=n)
    mu = np.exp(0.5 * eta)
    y_reg = eta + rng.normal(scale=0.4, size=n)
    targets = {
        "regression": y_reg,
        "regression_l1": eta + rng.standard_t(3, size=n) * 0.3,
        "huber": y_reg + (rng.uniform(size=n) < 0.05) * 8.0,
        "quantile": y_reg,
        "poisson": rng.poisson(mu).astype(float),
        "gamma": rng.gamma(2.0, mu / 2.0),
    }
    base = dict(n_estimators=25, learning_rate=0.2, num_leaves=8, **COMMON)
    X_test = X[::5]
    out = []

    def add(name, kind, params, y, sw=None, fit_kw=None, extra=None, X=X, X_test=X_test):
        cls = lgb.LGBMRegressor if kind == "regressor" else lgb.LGBMClassifier
        m = cls(**params).fit(X, y, sample_weight=sw, **(fit_kw or {}))
        pred = m.predict(X_test) if kind == "regressor" else m.predict_proba(X_test)
        case = {"name": name, "kind": kind,
                "params": {k: v for k, v in params.items() if k not in COMMON},
                "X": nulls(X), "y": y.tolist(),
                "sample_weight": None if sw is None else sw.tolist(),
                "X_test": nulls(X_test), "predictions": pred.tolist()}
        case.update(extra or {})
        out.append(case)

    for obj, y in targets.items():
        params = dict(base, objective=obj)
        if obj == "quantile":
            params["alpha"] = 0.7
        add(obj, "regressor", params, y)
        add(f"{obj}_weighted", "regressor", params, y, sw=weights)
    # LightGBM's default min_child_samples=20 with uneven hessians: its
    # count limit uses counts estimated from the hessians.
    add("gamma_min_child_20", "regressor", dict(base, objective="gamma", min_child_samples=20),
        targets["gamma"])
    add("regression_l2_l1_reg", "regressor",
        dict(base, reg_lambda=2.0, reg_alpha=0.5, min_split_gain=0.01, min_child_samples=5),
        y_reg)
    add("regression_monotone", "regressor",
        dict(base, monotone_constraints=[1, 1, -1, 0]), y_reg)
    add("regression_categorical", "regressor", dict(base, min_child_samples=5), y_reg,
        fit_kw=dict(categorical_feature=[2]))
    y_bin = (eta + rng.logistic(size=n) > 0).astype(float)
    y_multi = np.digitize(eta + rng.logistic(size=n) * 0.5, [-0.5, 0.7]).astype(float)
    add("binary", "classifier", dict(base), y_bin)
    add("binary_weighted", "classifier", dict(base), y_bin, sw=weights)
    add("binary_balanced", "classifier", dict(base, class_weight="balanced"), y_bin)
    add("multiclass", "classifier", dict(base), y_multi)
    add("multiclass_weighted", "classifier", dict(base, min_child_samples=5), y_multi,
        sw=weights)
    # DART (its dropping uses LightGBM's own LCG, seeded by drop_seed).
    dart = dict(base, boosting_type="dart", n_estimators=30)
    add("dart_regression", "regressor", dart, y_reg)
    add("dart_heavy", "regressor", dict(dart, drop_rate=0.4, skip_drop=0.1, drop_seed=7), y_reg)
    add("dart_uniform", "regressor", dict(dart, uniform_drop=True, drop_rate=0.3), y_reg)
    add("dart_xgboost", "regressor", dict(dart, xgboost_dart_mode=True, drop_rate=0.3), y_reg)
    add("dart_max_drop", "regressor", dict(dart, max_drop=2, drop_rate=0.5, skip_drop=0.0), y_reg)
    add("dart_l1_weighted", "regressor", dict(dart, objective="regression_l1", drop_rate=0.3),
        targets["regression_l1"], sw=weights)
    add("dart_poisson", "regressor", dict(dart, objective="poisson", drop_rate=0.3),
        targets["poisson"])
    add("dart_binary", "classifier", dict(dart, drop_rate=0.3), y_bin)
    add("dart_multiclass", "classifier", dict(dart, drop_rate=0.3), y_multi)
    # Exclusive feature bundling: a one-hot block, sparse columns (some with
    # missing values) and dense ones. Below 10 000 rows LightGBM bundles
    # exclusive features only, so bundling is lossless.
    level = rng.integers(0, 10, size=n)
    onehot = np.eye(10)[level]
    sparse = rng.normal(size=(n, 6)) * (rng.uniform(size=(n, 6)) < 0.08)
    sparse[rng.choice(n, 6, replace=False), 4] = np.nan
    Xs = np.column_stack([X[:, :2], onehot, sparse])
    eta_s = (0.6 * X[:, 0] + np.linspace(-1.5, 1.5, 10)[level]
             + 2.0 * np.nan_to_num(sparse).sum(axis=1))
    ys = eta_s + rng.normal(scale=0.3, size=n)
    sparse_kw = dict(X=Xs, X_test=Xs[::5])
    add("efb_regression", "regressor", dict(base, min_child_samples=5), ys, **sparse_kw)
    add("efb_regression_weighted", "regressor", dict(base, min_child_samples=5), ys,
        sw=weights, **sparse_kw)
    add("efb_no_bundle", "regressor", dict(base, min_child_samples=5, enable_bundle=False), ys,
        **sparse_kw)
    add("efb_binary", "classifier", dict(base, min_child_samples=5),
        (eta_s + rng.logistic(size=n) > 0).astype(float), **sparse_kw)
    add("efb_multiclass", "classifier", dict(base, min_child_samples=5),
        np.digitize(eta_s + rng.logistic(size=n) * 0.5, [-0.5, 0.7]).astype(float),
        **sparse_kw)
    # LightGBM's generators (#152): bagging (every `subsample_freq`
    # iterations, balanced for binary labels), GOSS (after 1 / learning_rate
    # iterations), feature fractions by tree and by node, and the seeds
    # derived from `random_state`.
    bag = dict(base, subsample=0.7, subsample_freq=1)
    add("bagging_regression", "regressor", bag, y_reg)
    add("bagging_freq3_seed", "regressor",
        dict(bag, subsample=0.6, subsample_freq=3, random_state=42), y_reg)
    add("bagging_weighted", "regressor", dict(bag, min_child_samples=5), y_reg, sw=weights)
    add("bagging_quantile", "regressor", dict(bag, objective="quantile", alpha=0.3), y_reg)
    add("bagging_binary", "classifier", dict(bag, random_state=3), y_bin)
    add("bagging_balanced", "classifier",
        dict(base, subsample_freq=1, pos_bagging_fraction=0.5, neg_bagging_fraction=0.8), y_bin)
    add("bagging_multiclass", "classifier", dict(bag, subsample=0.6, subsample_freq=2), y_multi)
    add("dart_bagging_seed", "regressor",
        dict(dart, subsample=0.8, subsample_freq=1, drop_rate=0.3, random_state=11), y_reg)
    goss = dict(base, boosting_type="goss")
    add("goss_regression", "regressor", goss, y_reg)
    add("goss_rates_seed", "regressor",
        dict(goss, top_rate=0.3, other_rate=0.2, random_state=5), y_reg, sw=weights)
    add("goss_binary", "classifier", goss, y_bin)
    add("goss_multiclass", "classifier", dict(goss, min_child_samples=5), y_multi)
    add("colsample_bytree", "regressor", dict(base, colsample_bytree=0.5), y_reg)
    add("colsample_bynode", "regressor", dict(base, feature_fraction_bynode=0.5), y_reg)
    add("colsample_both_seed", "regressor",
        dict(base, colsample_bytree=0.75, feature_fraction_bynode=0.6, random_state=7,
             min_child_samples=5), y_reg)
    add("colsample_multiclass", "classifier",
        dict(base, colsample_bytree=0.7, feature_fraction_bynode=0.7), y_multi)
    # A constant feature: LightGBM draws from the non-trivial ones only.
    Xc = np.column_stack([X[:, :2], np.full(n, 3.0), X[:, 2:]])
    add("colsample_constant_feature", "regressor",
        dict(base, colsample_bytree=0.5, feature_fraction_bynode=0.7), y_reg,
        X=Xc, X_test=Xc[::5])
    add("all_samplers_efb", "regressor",
        dict(base, subsample=0.8, subsample_freq=2, colsample_bytree=0.8,
             feature_fraction_bynode=0.8, min_child_samples=5, random_state=2), ys,
        **sparse_kw)
    # More than one generator block (1024 rows each).
    nb = 1300
    Xb = np.column_stack([rng.normal(size=nb), rng.uniform(-2, 2, size=nb),
                          rng.normal(size=nb)])
    yb = np.sin(2 * Xb[:, 0]) + 0.6 * Xb[:, 1] + rng.normal(scale=0.4, size=nb)
    big_kw = dict(X=Xb, X_test=Xb[::20])
    add("bagging_blocks", "regressor", dict(bag, subsample=0.5, num_leaves=15), yb, **big_kw)
    add("goss_blocks", "regressor", dict(goss, num_leaves=15), yb, **big_kw)
    # Tweedie, cross-entropy and one-vs-all objectives (#206).
    y_tweedie = np.where(rng.uniform(size=n) < 0.35, 0.0, rng.gamma(2.0, mu / 2.0))
    y_xent = 1 / (1 + np.exp(-eta)) * 0.8 + 0.2 * rng.uniform(size=n)
    for name, extra, y in [("tweedie", {}, y_tweedie),
                           ("tweedie_power_1_2", dict(tweedie_variance_power=1.2), y_tweedie),
                           ("cross_entropy", {}, y_xent)]:
        params = dict(base, objective=name.split("_power")[0], **extra)
        add(name, "regressor", params, y)
        add(f"{name}_weighted", "regressor", params, y, sw=weights)
    ova = dict(base, objective="multiclassova")
    add("multiclassova", "classifier", ova, y_multi)
    add("multiclassova_weighted", "classifier", dict(ova, min_child_samples=5), y_multi,
        sw=weights)
    add("multiclassova_balanced", "classifier", dict(ova, class_weight="balanced"), y_multi)
    # Many-vs-many categorical splits (#240): a strong categorical of 7 and
    # 30 levels, all well above min_data_per_group and cat_smooth.
    def store_data(rows, ncat):
        Xd, yd = [], []
        for i in range(rows):
            store, dow = i % ncat, i % 5
            l1, l2, l3 = ((i * 37) % 101) / 10, ((i * 53) % 89) / 10, (i * 11) % 23
            Xd.append([l1, l2, l3, dow, store])
            yd.append(0.6 * l1 + 0.3 * l2 + 0.5 * dow + store + ((i * 7) % 13) / 13)
        return np.array(Xd, dtype=np.float32).astype(float), np.array(yd)
    cat_base = dict(learning_rate=0.1, num_leaves=15, min_child_samples=20, **COMMON)
    for ncat, rows, iters in ((7, 3000, 1), (7, 3000, 50), (30, 6000, 1)):
        Xd, yd = store_data(rows, ncat)
        add(f"categorical_many_{ncat}_{iters}", "regressor", dict(cat_base, n_estimators=iters),
            yd, fit_kw=dict(categorical_feature=[4]), extra={"categorical": [4]}, X=Xd,
            X_test=Xd[::7])
    return out


if __name__ == "__main__":
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "lgbm_parity.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {path}")
