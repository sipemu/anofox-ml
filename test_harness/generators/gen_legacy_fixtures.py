"""Re-validation of golden fixtures committed without a generator (#113).

These files were added in 8921972 (scikit-learn 1.8.0) without their
generator scripts. Each case stores its inputs, so this script recomputes
every output from the committed inputs with the pinned scikit-learn and
writes the file back; the drift check then reports any value that changed.
Estimator parameters follow the Rust tests that consume the files.

Covered: the deterministic fixtures (metrics, model selection,
preprocessing, naive Bayes, univariate selection, linear models). The
ensemble / LGBM / SVM / calibration fixtures depend on seeded randomness or
are only compared loosely, and stay as they are.
"""

import json
import os

import numpy as np
from sklearn import feature_selection, linear_model, metrics, naive_bayes, preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, LeaveOneOut, TimeSeriesSplit

DIR = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "anofox-ml", "tests",
                   "golden_data")


def _load(name):
    with open(os.path.join(DIR, name)) as f:
        return json.load(f)


def _clean(v):
    """NaN as null, as the committed files have it (bare NaN is not JSON)."""
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def _save(name, data):
    with open(os.path.join(DIR, name), "w") as f:
        json.dump(_clean(data), f, indent=2)
    print(f"wrote {name}")


A = np.asarray


def metrics_extra(cases):
    for c in cases:
        yt, n = A(c.get("y_true", [])), c["name"]
        if n == "log_loss_binary":
            c["log_loss"] = metrics.log_loss(yt, A(c["y_prob"]))
        elif n.startswith("balanced_accuracy"):
            yp = A(c["y_pred"])
            c["balanced_accuracy"] = metrics.balanced_accuracy_score(yt, yp)
            if "accuracy" in c:
                c["accuracy"] = metrics.accuracy_score(yt, yp)
        elif n == "cohen_kappa":
            c["cohen_kappa"] = metrics.cohen_kappa_score(yt, A(c["y_pred"]))
        elif n == "silhouette_well_separated":
            c["silhouette_score"] = metrics.silhouette_score(A(c["X"]), A(c["labels"]))
        elif n == "regression_extra":
            yp = A(c["y_pred"])
            c["mean_squared_log_error"] = metrics.mean_squared_log_error(yt, yp)
            c["median_absolute_error"] = metrics.median_absolute_error(yt, yp)
        elif n == "extended_classification":
            ys, yp = A(c["y_scores"]), A(c["y_pred"])
            c["roc_auc"] = metrics.roc_auc_score(yt, ys)
            c["average_precision"] = metrics.average_precision_score(yt, ys)
            c["matthews_corrcoef"] = metrics.matthews_corrcoef(yt, yp)
        elif n == "regression_extended":
            yp = A(c["y_pred"])
            c["explained_variance"] = metrics.explained_variance_score(yt, yp)
            c["mape"] = metrics.mean_absolute_percentage_error(yt, yp)
            c["max_error"] = metrics.max_error(yt, yp)
    return cases


def metrics_round2(cases):
    for c in cases:
        n = c["name"]
        if n == "adjusted_rand_score":
            c["score"] = metrics.adjusted_rand_score(c["labels_true"], c["labels_pred"])
        elif n == "normalized_mutual_info_score":
            c["score"] = metrics.normalized_mutual_info_score(c["labels_true"], c["labels_pred"])
        elif n == "brier_score_loss":
            c["score"] = metrics.brier_score_loss(c["y_true"], c["y_prob"])
        elif n == "roc_curve":
            fpr, tpr, _ = metrics.roc_curve(c["y_true"], c["y_score"])
            c.update(fpr=fpr.tolist(), tpr=tpr.tolist(), n_points=len(fpr))
        elif n == "precision_recall_curve":
            p, r, _ = metrics.precision_recall_curve(c["y_true"], c["y_score"])
            c.update(precision=p.tolist(), recall=r.tolist(), n_points=len(p))
    return cases


def model_selection_extra(cases):
    for c in cases:
        n = c["n_samples"]
        X = np.zeros((n, 1))
        split = {"k_fold": lambda: KFold(c.get("k", 5)),
                 "time_series_split": lambda: TimeSeriesSplit(c.get("n_splits", 5)),
                 "leave_one_out": LeaveOneOut}[c["name"]]()
        c["folds"] = [[tr.tolist(), te.tolist()] for tr, te in split.split(X)]
    return cases


def preprocessing_extra(cases):
    for c in cases:
        X, n = A(c["X"]), c["name"]
        if n == "max_abs_scaler":
            t = preprocessing.MaxAbsScaler().fit(X)
            c["max_abs"] = t.max_abs_.tolist()
        elif n == "normalizer_l2":
            t = preprocessing.Normalizer()
        elif n == "normalizer_l1":
            t = preprocessing.Normalizer(norm="l1")
        elif n == "binarizer":
            t = preprocessing.Binarizer(threshold=c["threshold"])
        elif n == "polynomial_features":
            t = preprocessing.PolynomialFeatures(degree=c["degree"], include_bias=False)
        elif n == "robust_scaler":
            t = preprocessing.RobustScaler().fit(X)
            c["center"], c["scale"] = t.center_.tolist(), t.scale_.tolist()
        c["X_transformed"] = t.fit(X).transform(X).tolist()
    return cases


def preprocessing_round2(cases):
    for c in cases:
        X, n = A(c["X"], dtype=float), c["name"]
        if n == "power_transformer":
            # Yeo-Johnson's lambda comes from scipy's Brent optimiser, whose
            # result moves by ~1e-8 between scipy versions; no Rust test
            # reads this case. Left as committed.
            continue
        if n == "kbins_uniform_ordinal":
            t = preprocessing.KBinsDiscretizer(n_bins=c["n_bins"], encode="ordinal",
                                               strategy="uniform").fit(X)
        else:
            t = SimpleImputer(strategy=n.rsplit("_", 1)[1]).fit(X)
            c["fill_values"] = t.statistics_.tolist()
        c["X_transformed"] = t.transform(X).tolist()
    return cases


def naive_bayes_extra(cases):
    for c in cases:
        X, y = A(c["X"]), A(c["y"])
        m = {"multinomial_nb": naive_bayes.MultinomialNB,
             "bernoulli_nb": naive_bayes.BernoulliNB}[c["name"]]().fit(X, y)
        c["predictions"], c["predict_proba"] = m.predict(X).tolist(), m.predict_proba(X).tolist()
    return cases


def feature_selection_batch(cases):
    for c in cases:
        X, y = A(c["X"]), A(c["y"])
        func, k = {"select_k_best_f_classif": (feature_selection.f_classif, 2),
                   "select_k_best_f_regression": (feature_selection.f_regression, 1)}[c["name"]]
        s = feature_selection.SelectKBest(func, k=k).fit(X, y)
        c["selected_indices"] = np.flatnonzero(s.get_support()).tolist()
        c["scores"] = s.scores_.tolist()
    return cases


def regression_batch(cases):
    makers = {
        "ols": lambda c: linear_model.LinearRegression(),
        "ridge": lambda c: linear_model.Ridge(alpha=c["alpha"]),
        "lasso": lambda c: linear_model.Lasso(alpha=c["alpha"]),
        "elastic_net": lambda c: linear_model.ElasticNet(alpha=c["alpha"], l1_ratio=c["l1_ratio"]),
        "logistic_regression": lambda c: linear_model.LogisticRegression(C=1e6, max_iter=1000),
        "huber_regressor": lambda c: linear_model.HuberRegressor(epsilon=1.35, alpha=1e-4),
    }
    for c in cases:
        if c["name"] not in makers:  # sgd_regressor: seeded, checked loosely
            continue
        if c["name"] == "logistic_regression":
            # C = 1e6 leaves an intercept of ~1e-11 whose value is L-BFGS /
            # BLAS noise; matched locally (#120), left as committed.
            continue
        X, y = A(c["X"]), A(c["y"])
        m = makers[c["name"]](c).fit(X, y)
        c["predictions"] = m.predict(X).tolist()
        c["coef"] = np.ravel(m.coef_).tolist()
        c["intercept"] = float(np.ravel(m.intercept_)[0])
        if hasattr(m, "predict_proba"):
            c["predict_proba"] = m.predict_proba(X).tolist()
    return cases


def main():
    for fn in (metrics_extra, metrics_round2, model_selection_extra, preprocessing_extra,
               preprocessing_round2, naive_bayes_extra, feature_selection_batch,
               regression_batch):
        name = fn.__name__ + ".json"
        _save(name, fn(_load(name)))


if __name__ == "__main__":
    main()
