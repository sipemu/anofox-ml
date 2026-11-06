"""Gradient boosting: anofox-ml against scikit-learn and LightGBM (#127).

Fit and predict times on standard sizes, same data and parameters, every
library on all cores. anofox-ml runs through its Python binding (the
`anofox_ml_python` wheel), so the data hand-over is part of its times.

    python benchmarks/boosting_benchmark.py [--quick] [--json out.json]

Prints a Markdown table (median of `REPEATS` fits) and each model's score,
so that a faster model is also seen to be the same model.
"""

import argparse
import json
import os
import platform
import time

import numpy as np

import anofox_ml_python as am
import lightgbm as lgb
import sklearn
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

SEED = 42
REPEATS = 3


def dense(n, p, kind, rng):
    X = rng.normal(size=(n, p))
    eta = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.5 * X[:, 3:8].sum(axis=1)
    if kind == "regression":
        return X, eta + rng.normal(scale=0.5, size=n)
    return X, (eta + rng.logistic(size=n) > 0).astype(float)


def one_hot(n, levels, rng):
    level = rng.integers(0, levels, size=n)
    Xd = rng.normal(size=(n, 10))
    X = np.column_stack([Xd, np.eye(levels)[level]])
    y = Xd[:, 0] + np.sin(level) + rng.normal(scale=0.3, size=n)
    return X, y


def timed(fn):
    times = []
    out = None
    for _ in range(REPEATS):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times)), out


def score(kind, y, pred):
    if kind == "regression":
        return 1 - np.mean((y - pred) ** 2) / np.var(y)
    return np.mean((pred > 0.5) == y)


def lgbm_pair(kind):
    kw = dict(n_estimators=100, num_leaves=31, learning_rate=0.1, min_child_samples=20)
    if kind == "regression":
        return (lambda: am.LGBMRegressor(**kw),
                lambda: lgb.LGBMRegressor(**kw, verbose=-1))
    return (lambda: am.LGBMClassifier(**kw),
            lambda: lgb.LGBMClassifier(**kw, verbose=-1))


def hgb_pair(kind):
    kw = dict(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, early_stopping=False)
    if kind == "regression":
        return (lambda: am.HistGradientBoostingRegressor(**kw),
                lambda: HistGradientBoostingRegressor(**kw))
    return (lambda: am.HistGradientBoostingClassifier(**kw),
            lambda: HistGradientBoostingClassifier(**kw))


def gb_pair(kind):
    kw = dict(n_estimators=100, max_depth=3, learning_rate=0.1)
    return (lambda: am.GradientBoostingRegressor(**kw),
            lambda: GradientBoostingRegressor(**kw))


def run(quick):
    rng = np.random.default_rng(SEED)
    sizes = [(10_000, 20), (100_000, 50)] if not quick else [(10_000, 20)]
    cases = []
    for n, p in sizes:
        for kind in ("regression", "binary"):
            X, y = dense(n, p, kind, rng)
            cases.append((f"{kind} {n}×{p}", kind, X, y, ["LGBM", "HistGB"]))
    n_oh = 100_000 if not quick else 10_000
    X, y = one_hot(n_oh, 200, rng)
    cases.append((f"one-hot {n_oh}×210", "regression", X, y, ["LGBM", "HistGB"]))
    X, y = dense(10_000, 20, "regression", rng)
    cases.append(("regression 10000×20", "regression", X, y, ["GradientBoosting"]))

    pairs = {"LGBM": (lgbm_pair, "LightGBM"), "HistGB": (hgb_pair, "sklearn"),
             "GradientBoosting": (gb_pair, "sklearn")}
    rows = []
    for name, kind, X, y, models in cases:
        for model in models:
            make, ref_name = pairs[model]
            ours_f, ref_f = make(kind)
            row = {"data": name, "model": model, "reference": ref_name}
            for who, factory in (("anofox", ours_f), ("ref", ref_f)):
                fit_t, m = timed(lambda: factory().fit(X, y))
                if kind == "regression":
                    pred_t, pred = timed(lambda: m.predict(X))
                else:
                    pred_t, proba = timed(lambda: m.predict_proba(X))
                    pred = proba[:, 1]
                row[f"{who}_fit_s"] = fit_t
                row[f"{who}_predict_s"] = pred_t
                row[f"{who}_score"] = float(score(kind, y, pred))
            rows.append(row)
            print(f"  {name:24s} {model:16s} fit {row['anofox_fit_s']:.3f}s vs "
                  f"{row['ref_fit_s']:.3f}s", flush=True)
    return rows


def table(rows):
    out = ["| Model | Data | Reference | fit (s) anofox / ref | predict (s) anofox / ref "
           "| score anofox / ref |",
           "|---|---|---|--:|--:|--:|"]
    for r in rows:
        out.append(
            f"| {r['model']} | {r['data']} | {r['reference']} "
            f"| {r['anofox_fit_s']:.3f} / {r['ref_fit_s']:.3f} "
            f"| {r['anofox_predict_s']:.3f} / {r['ref_predict_s']:.3f} "
            f"| {r['anofox_score']:.4f} / {r['ref_score']:.4f} |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    print(f"{platform.processor() or platform.machine()}, {os.cpu_count()} threads; "
          f"sklearn {sklearn.__version__}, lightgbm {lgb.__version__}")
    rows = run(args.quick)
    print()
    print(table(rows))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
