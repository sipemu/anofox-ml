"""Golden data for the gap batch 4 (#113): EmpiricalCovariance,
ShrunkCovariance, LedoitWolf, OAS, MinCovDet and EllipticEnvelope
(scikit-learn 1.8.0). The MCD data has well-separated outliers, so FastMCD
reaches the same optimal subset whatever its random starts."""

import numpy as np
from sklearn.covariance import (OAS, EllipticEnvelope, EmpiricalCovariance, LedoitWolf,
                                MinCovDet, ShrunkCovariance)


def _contaminated(seed, n, p, n_out):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, p))
    X = rng.normal(size=(n, p)) @ A + 3.0
    X[:n_out] += rng.normal(12.0, 1.0, size=(n_out, p))
    return X


def generate():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(60, 4)) @ rng.normal(size=(4, 4)) + 1.5
    Xt = rng.normal(size=(15, 4)) + 1.5
    plain = []
    for name, est in [("empirical", EmpiricalCovariance()),
                      ("empirical_centered", EmpiricalCovariance(assume_centered=True)),
                      ("shrunk", ShrunkCovariance(shrinkage=0.3)),
                      ("ledoit_wolf", LedoitWolf()),
                      ("ledoit_wolf_centered", LedoitWolf(assume_centered=True)),
                      ("oas", OAS()), ("oas_centered", OAS(assume_centered=True))]:
        est.fit(X)
        plain.append({"name": name, "params": {k: v for k, v in est.get_params().items()
                                               if k in ("assume_centered", "shrinkage")},
                      "X": X.tolist(), "X_test": Xt.tolist(),
                      "location": est.location_.tolist(),
                      "covariance": est.covariance_.tolist(),
                      "precision": est.precision_.tolist(),
                      "shrinkage": float(getattr(est, "shrinkage_", getattr(est, "shrinkage", 0.0))),
                      "mahalanobis": est.mahalanobis(Xt).tolist(),
                      "score": float(est.score(Xt))})
    mcd = []
    for name, XX, kw in [("p3", _contaminated(1, 120, 3, 12), {}),
                         ("p3_frac", _contaminated(2, 150, 3, 15), {"support_fraction": 0.75}),
                         ("p1", _contaminated(3, 80, 1, 8), {}),
                         ("p2_big", _contaminated(4, 700, 2, 60), {})]:
        m = MinCovDet(random_state=0, **kw).fit(XX)
        env = EllipticEnvelope(random_state=0, contamination=0.1, **kw).fit(XX)
        mcd.append({"name": name, "params": kw, "X": XX.tolist(),
                    "raw_location": m.raw_location_.tolist(),
                    "raw_covariance": m.raw_covariance_.tolist(),
                    "raw_support": np.flatnonzero(m.raw_support_).tolist(),
                    "location": m.location_.tolist(), "covariance": m.covariance_.tolist(),
                    "support": np.flatnonzero(m.support_).tolist(), "dist": m.dist_.tolist(),
                    "envelope_offset": float(env.offset_),
                    "envelope_predict": env.predict(XX).tolist(),
                    "envelope_decision": env.decision_function(XX).tolist()})
    return {"plain": plain, "mcd": mcd}


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "crates", "anofox-ml", "tests", "golden_data", "covariance.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
