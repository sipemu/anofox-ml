"""Golden data for LatentDirichletAllocation (#113) against scikit-learn
1.8.0: seeded batch and online fits (numpy RandomState gamma draws in
sklearn's order, sklearn's digamma)."""

import json
import os

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation


def generate():
    r = np.random.RandomState(5)
    X = r.poisson(np.outer(r.gamma(2.0, 1.0, 30), r.dirichlet(np.ones(12))) * 6).astype(float)
    X[3] = 0.0  # an empty document
    cases = []
    for params in [
        {"n_components": 3, "random_state": 0},
        {"n_components": 4, "max_iter": 25, "doc_topic_prior": 0.3, "topic_word_prior": 0.05,
         "random_state": 7},
        {"n_components": 3, "learning_method": "online", "batch_size": 8, "max_iter": 6,
         "random_state": 2},
        {"n_components": 2, "evaluate_every": 2, "max_iter": 40, "perp_tol": 0.5,
         "random_state": 1},
    ]:
        m = LatentDirichletAllocation(**params).fit(X)
        cases.append({"params": params, "components": m.components_.tolist(),
                      "n_iter": int(m.n_iter_), "bound": float(m.bound_),
                      "transform": m.transform(X[:6]).tolist(),
                      "score": float(m.score(X)), "perplexity": float(m.perplexity(X))})
    return {"X": X.tolist(), "cases": cases}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "lda_topics.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
