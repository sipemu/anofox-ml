"""Golden data for CountVectorizer / TfidfVectorizer.

sklearn's vocabulary order is alphabetical (when sorted), which matches ours.
Both implementations should produce identical count matrices (modulo
tokenisation choices: ours uses `[A-Za-z]{2,}` lowercase, which is the sklearn
default ``token_pattern=r"(?u)\\b\\w\\w+\\b"`` for ASCII-only input).
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def generate():
    docs = [
        "the cat sat on the mat",
        "the dog ran on the mat",
        "the cat and the dog",
        "a cat and a mat",
    ]
    cv = CountVectorizer(token_pattern=r"(?u)\b[A-Za-z]{2,}\b", lowercase=True)
    Xc = cv.fit_transform(docs).toarray()

    tv = TfidfVectorizer(token_pattern=r"(?u)\b[A-Za-z]{2,}\b", lowercase=True, norm="l2")
    Xt = tv.fit_transform(docs).toarray()

    return [{
        "name": "small_corpus",
        "docs": docs,
        "vocab": list(cv.get_feature_names_out()),
        "count_matrix": Xc.tolist(),
        "tfidf_matrix": Xt.tolist(),
    }]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "text.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
