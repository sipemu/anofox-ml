"""Golden data for the gap batch 3 (#113): OneVsRestClassifier,
OneVsOneClassifier, FixedThresholdClassifier and TunedThresholdClassifierCV
(scikit-learn 1.8.0), over LogisticRegression (decision_function path) and
GaussianNB (predict_proba path)."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier, TunedThresholdClassifierCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB


def _data(seed, n=150, k=3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2, (k, 3))
    y = np.arange(n) % k
    X = centers[y] + rng.normal(0, 1.6, (n, 3))
    return X, y.astype(float)


BASES = {
    "logistic": (lambda: LogisticRegression(tol=1e-12, max_iter=100000), "LogisticRegressor", {}),
    "gnb": (lambda: GaussianNB(), "GaussianNB", {}),
}


def generate():
    X, y = _data(11)
    Xb, yb = _data(12, n=160, k=2)
    yb = np.where(yb == 1, 5.0, 2.0)  # labels 2 / 5
    yb[::7] = 5.0                     # unbalanced
    out = {"ovr": [], "ovo": [], "fixed": [], "tuned": []}
    for base, (make, kind, params) in BASES.items():
        for yy, tag in ((y, "multi"), (yb, "binary")):
            XX = X if tag == "multi" else Xb
            m = OneVsRestClassifier(make()).fit(XX, yy)
            out["ovr"].append({"name": f"{base}_{tag}", "kind": kind, "params": params,
                               "X": XX.tolist(), "y": yy.tolist(),
                               "predict": m.predict(XX).tolist(),
                               "predict_proba": m.predict_proba(XX).tolist()})
            m = OneVsOneClassifier(make()).fit(XX, yy)
            out["ovo"].append({"name": f"{base}_{tag}", "kind": kind, "params": params,
                               "X": XX.tolist(), "y": yy.tolist(),
                               "predict": m.predict(XX).tolist(),
                               "decision_function": np.atleast_2d(
                                   m.decision_function(XX).T).T.reshape(len(XX), -1).tolist()})
        for thr in (0.3, 0.7):
            m = FixedThresholdClassifier(make(), threshold=thr).fit(Xb, yb)
            out["fixed"].append({"name": f"{base}_{thr}", "kind": kind, "params": params,
                                 "threshold": thr, "pos_label": None,
                                 "X": Xb.tolist(), "y": yb.tolist(),
                                 "predict": m.predict(Xb).tolist()})
        m = FixedThresholdClassifier(make(), threshold=0.4, pos_label=2.0).fit(Xb, yb)
        out["fixed"].append({"name": f"{base}_pos2", "kind": kind, "params": params,
                             "threshold": 0.4, "pos_label": 2.0,
                             "X": Xb.tolist(), "y": yb.tolist(),
                             "predict": m.predict(Xb).tolist()})
        for scoring in ("balanced_accuracy", "accuracy", "f1", "recall"):
            kw = {"pos_label": 5.0} if scoring in ("f1", "recall") else {}
            sc = scoring
            if kw:
                from sklearn.metrics import make_scorer, f1_score, recall_score
                sc = make_scorer({"f1": f1_score, "recall": recall_score}[scoring], **kw)
            m = TunedThresholdClassifierCV(make(), scoring=sc).fit(Xb, yb)
            out["tuned"].append({"name": f"{base}_{scoring}", "kind": kind, "params": params,
                                 "scoring": scoring, "X": Xb.tolist(), "y": yb.tolist(),
                                 "best_threshold": float(m.best_threshold_),
                                 "best_score": float(m.best_score_),
                                 "predict": m.predict(Xb).tolist()})
    return out


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "meta.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
