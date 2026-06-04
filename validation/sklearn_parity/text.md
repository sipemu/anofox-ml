# Text vectorizers — sklearn parity

Issue: [#24](https://github.com/sipemu/rustml/issues/24)

## What

New crate `rustml-text`:

- **CountVectorizer**: tokenises with `[A-Za-z]{2,}` (lowercased), builds an
  alphabetically-sorted vocabulary, returns a dense count matrix.
- **TfidfVectorizer**: count matrix → IDF with sklearn's `smooth_idf=True`
  (`log((1+n)/(1+df)) + 1`), then optional `l2` row normalisation.
- **HashingVectorizer**: fixed `n_features`, FNV-1a hashing, optional `l2`
  normalisation.

## Reference

`sklearn.feature_extraction.text.{CountVectorizer, TfidfVectorizer, HashingVectorizer}` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_text.py`
- Fixture:   `crates/rustml/tests/golden_data/text.json`
- Rust test: `crates/rustml/tests/golden_text.rs`

4-document corpus. CountVectorizer's vocabulary and count matrix match
sklearn **exactly**; TfidfVectorizer matches to `1e-6` element-wise. sklearn
is configured with `token_pattern=r"(?u)\b[A-Za-z]{2,}\b"`, `lowercase=True`
to match our tokeniser.

## Differences from sklearn

- Token pattern is fixed to `[A-Za-z]{2,}` (no Unicode word chars; no
  configurability).
- No `stop_words`, `ngram_range`, `max_features`, `analyzer='char'`.
- Output is dense `ndarray::Array2<f64>` — sklearn returns CSR. For large
  corpora this is the main scaling limitation.
- HashingVectorizer's hash is FNV-1a, not sklearn's MurmurHash3 — bucket
  assignments differ.
