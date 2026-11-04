"""Golden data for anofox-ml-conformal, from MAPIE (mapie.regression).

The base estimator is ordinary least squares, so both sides start from the
same point predictions and only the conformal logic is compared. Folds are
passed explicitly (one fold index per row) and rebuilt in Rust.
"""

import json
import os

import numpy as np
from mapie.regression import CrossConformalRegressor, SplitConformalRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, PredefinedSplit

LEVELS = [0.9, 0.8, 0.5]


def data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    noise = rng.standard_normal(n) * (0.5 + np.abs(X[:, 0]))
    y = X @ np.array([1.5, -2.0, 0.5]) + 1.0 + noise
    return X, y


def intervals(y_pis):
    # (n, 2, n_levels) -> per level {lower, upper}; +-inf encoded as strings
    def enc(a):
        return [v if np.isfinite(v) else ("inf" if v > 0 else "-inf") for v in a.tolist()]

    return [{"lower": enc(y_pis[:, 0, j]), "upper": enc(y_pis[:, 1, j])}
            for j in range(y_pis.shape[2])]


def generate():
    cases = []
    X, y = data(80, 0)
    X_test, _ = data(15, 1)

    # Split conformal, prefit on rows 0..50, calibrated on rows 50..80. With
    # 8 calibration rows the 0.8 level needs rank 8 = n (still finite) and
    # 0.9 is unreachable: MAPIE raises (its symmetric split path ignores
    # allow_infinite_bounds), so that level is listed under error_levels.
    for n_cal, levels, error_levels in [(30, LEVELS, []), (8, [0.8, 0.5], [0.9])]:
        X_cal, y_cal = X[50:50 + n_cal], y[50:50 + n_cal]
        est = LinearRegression().fit(X[:50], y[:50])
        scr = SplitConformalRegressor(est, confidence_level=levels, prefit=True)
        scr.conformalize(X_cal, y_cal)
        y_pred, y_pis = scr.predict_interval(X_test)
        for level in error_levels:
            bad = SplitConformalRegressor(est, confidence_level=level, prefit=True)
            bad.conformalize(X_cal, y_cal)
            try:
                bad.predict_interval(X_test)
                raise AssertionError("expected MAPIE to raise")
            except ValueError:
                pass
        cases.append({
            "name": f"split_ncal{n_cal}",
            "kind": "split",
            "levels": levels,
            "error_levels": error_levels,
            "train": list(range(50)),
            "calibration": list(range(50, 50 + n_cal)),
            "y_pred": y_pred.tolist(),
            "intervals": intervals(y_pis),
        })

    # Cross-conformal with 7 unequal folds (80 rows), every method and point
    # prediction. "median" is left out: MAPIE 1.6.0 returns the same point
    # predictions for aggregate_point_predictions="median" as for "mean".
    test_fold = np.arange(80) % 7
    for method in ["plus", "minmax", "base"]:
        for agg in ["mean", None]:
            ccr = CrossConformalRegressor(LinearRegression(), confidence_level=LEVELS,
                                          method=method, cv=PredefinedSplit(test_fold))
            ccr.fit_conformalize(X, y)
            y_pred, y_pis = ccr.predict_interval(X_test, aggregate_point_predictions=agg)
            cases.append({
                "name": f"cross_{method}_{agg}",
                "kind": "cross",
                "method": method,
                "point": agg or "full",
                "levels": LEVELS,
                "error_levels": [],
                "fold": test_fold.tolist(),
                "y_pred": y_pred.tolist(),
                "intervals": intervals(y_pis),
            })

    # Jackknife+ (leave-one-out) on 25 rows.
    ccr = CrossConformalRegressor(LinearRegression(), confidence_level=LEVELS,
                                  method="plus", cv=LeaveOneOut())
    ccr.fit_conformalize(X[:25], y[:25])
    y_pred, y_pis = ccr.predict_interval(X_test)
    cases.append({
        "name": "jackknife_plus",
        "kind": "cross",
        "method": "plus",
        "point": "mean",
        "levels": LEVELS,
        "error_levels": [],
        "fold": list(range(25)),
        "y_pred": y_pred.tolist(),
        "intervals": intervals(y_pis),
    })

    return {"X": X.tolist(), "y": y.tolist(),
            "X_test": X_test.tolist(), "cases": cases}


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "..", "crates",
                       "anofox-ml-conformal", "tests", "golden_data", "mapie.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {out}")
