//! Text feature extraction.
//!
//! Mirrors `sklearn.feature_extraction.text.{CountVectorizer, TfidfVectorizer,
//! HashingVectorizer}` in their default configuration:
//!
//! * Unicode lowercasing of the whole document, then sklearn's default
//!   `token_pattern = r"(?u)\b\w\w+\b"` — maximal runs of Unicode
//!   alphanumerics / underscore that are at least two characters long
//!   (`"Müller"` → `"müller"`, `"café"` → `"café"`, `"2024"`, `"foo_bar"`).
//! * Word unigrams, no stop words, no n-grams.
//!
//! `CountVectorizer` / `TfidfVectorizer` follow sklearn's fit / transform
//! split: [`CountVectorizer::fit`] returns a [`FittedCountVectorizer`] holding
//! the vocabulary (and [`TfidfVectorizer::fit`] a [`FittedTfidfVectorizer`]
//! that also holds the IDF weights), which can then encode unseen documents.
//! Tokens that are not in the fitted vocabulary are ignored at transform time.
//!
//! `HashingVectorizer` and `FeatureHasher` use sklearn's signed 32-bit
//! MurmurHash3 (seed 0), so column indices and signs are identical to
//! sklearn's.

pub mod tfidf_transformer;
pub use tfidf_transformer::{FittedTfidfTransformer, TfidfNorm, TfidfTransformer};

use anofox_ml_core::{CsrMatrix, Result, RustMlError};
use ndarray::Array2;
use std::collections::{HashMap, HashSet};

/// sklearn-compatible analyzer: lowercase, then `(?u)\b\w\w+\b`.
///
/// `\w` is approximated by `char::is_alphanumeric() || c == '_'`. For the
/// vast majority of scripts this matches Python's `re` exactly; it can
/// differ for a handful of combining marks that Unicode classifies as
/// alphabetic but Python's `str.isalnum()` does not.
fn tokenize(s: &str) -> Vec<String> {
    let lowered = s.to_lowercase();
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    let mut n_chars = 0usize;
    for (pos, c) in lowered.char_indices() {
        if c.is_alphanumeric() || c == '_' {
            if start.is_none() {
                start = Some(pos);
                n_chars = 0;
            }
            n_chars += 1;
        } else if let Some(st) = start.take() {
            if n_chars >= 2 {
                out.push(lowered[st..pos].to_string());
            }
        }
    }
    if let Some(st) = start {
        if n_chars >= 2 {
            out.push(lowered[st..].to_string());
        }
    }
    out
}

/// Build a CSR row block from per-document column lists (duplicates are
/// summed, indices sorted ascending within each row).
fn csr_from_rows(rows: Vec<Vec<(usize, f64)>>, n_cols: usize) -> CsrMatrix<f64> {
    let n_rows = rows.len();
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for mut row in rows {
        row.sort_unstable_by_key(|&(c, _)| c);
        let mut iter = row.into_iter().peekable();
        while let Some((c, mut v)) = iter.next() {
            while let Some(&(c2, v2)) = iter.peek() {
                if c2 != c {
                    break;
                }
                v += v2;
                iter.next();
            }
            if v != 0.0 {
                indices.push(c);
                data.push(v);
            }
        }
        indptr.push(indices.len());
    }
    CsrMatrix {
        indptr,
        indices,
        data,
        n_rows,
        n_cols,
    }
}

/// In-place L2 normalisation of every CSR row (zero rows stay zero).
fn l2_normalize_rows(csr: &mut CsrMatrix<f64>) {
    for i in 0..csr.n_rows {
        let (start, end) = (csr.indptr[i], csr.indptr[i + 1]);
        let s: f64 = csr.data[start..end].iter().map(|v| v * v).sum();
        if s > 0.0 {
            let norm = s.sqrt();
            for v in &mut csr.data[start..end] {
                *v /= norm;
            }
        }
    }
}

/// Validate `min_df` / `max_df_frac` against the corpus size and return the
/// inclusive document-frequency bounds, like sklearn's `_limit_features`
/// pre-check (`ValueError: max_df corresponds to < documents than min_df`).
fn df_bounds(min_df: usize, max_df_frac: f64, n_docs: usize) -> Result<(usize, usize)> {
    if !(0.0..=1.0).contains(&max_df_frac) {
        return Err(RustMlError::InvalidParameter(format!(
            "max_df_frac must be in [0, 1], got {max_df_frac}"
        )));
    }
    let max_df = (max_df_frac * n_docs as f64).floor() as usize;
    if max_df < min_df {
        return Err(RustMlError::InvalidParameter(format!(
            "max_df corresponds to < documents than min_df \
             (max_df_frac={max_df_frac} -> {max_df} of {n_docs} documents, min_df={min_df})"
        )));
    }
    Ok((min_df, max_df))
}

// ---------------------------------------------------------------------------
// CountVectorizer
// ---------------------------------------------------------------------------

/// Bag-of-words term counts (sklearn's `CountVectorizer`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CountVectorizer {
    /// Ignore terms that appear in fewer than `min_df` training documents.
    pub min_df: usize,
    /// Ignore terms that appear in more than `max_df_frac × n_docs`
    /// training documents.
    pub max_df_frac: f64,
}

/// A fitted [`CountVectorizer`]: the learned (sorted) vocabulary.
/// The fitted vocabulary (serialised as the sorted term list; the lookup
/// table is rebuilt on load).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "Vocabulary", into = "Vocabulary")]
pub struct FittedCountVectorizer {
    vocabulary: Vec<String>,
    term_to_col: HashMap<String, usize>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Vocabulary {
    vocabulary: Vec<String>,
}

impl From<Vocabulary> for FittedCountVectorizer {
    fn from(v: Vocabulary) -> Self {
        Self::from_vocabulary(v.vocabulary)
    }
}

impl From<FittedCountVectorizer> for Vocabulary {
    fn from(f: FittedCountVectorizer) -> Self {
        Self {
            vocabulary: f.vocabulary,
        }
    }
}

impl CountVectorizer {
    pub fn new() -> Self {
        Self {
            min_df: 1,
            max_df_frac: 1.0,
        }
    }
    pub fn with_min_df(mut self, m: usize) -> Self {
        self.min_df = m;
        self
    }
    pub fn with_max_df_frac(mut self, f: f64) -> Self {
        self.max_df_frac = f;
        self
    }

    /// Learn the vocabulary from `docs`.
    ///
    /// Errors when `docs` is empty, when `max_df_frac` is outside `[0, 1]`,
    /// or when `max_df_frac × n_docs < min_df` (sklearn raises a
    /// `ValueError` for the latter as well).
    pub fn fit(&self, docs: &[&str]) -> Result<FittedCountVectorizer> {
        let tokenised: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
        self.fit_tokens(&tokenised)
    }

    fn fit_tokens(&self, tokenised: &[Vec<String>]) -> Result<FittedCountVectorizer> {
        if tokenised.is_empty() {
            return Err(RustMlError::EmptyInput("no documents".into()));
        }
        let (min_df, max_df) = df_bounds(self.min_df, self.max_df_frac, tokenised.len())?;
        // Document frequency per term.
        let mut df: HashMap<&str, usize> = HashMap::new();
        for tokens in tokenised {
            let seen: HashSet<&str> = tokens.iter().map(String::as_str).collect();
            for t in seen {
                *df.entry(t).or_default() += 1;
            }
        }
        let mut vocabulary: Vec<String> = df
            .into_iter()
            .filter(|&(_, c)| c >= min_df && c <= max_df)
            .map(|(k, _)| k.to_string())
            .collect();
        vocabulary.sort();
        Ok(FittedCountVectorizer::from_vocabulary(vocabulary))
    }

    /// Fit on `docs` and return `(vocabulary, dense counts)`.
    pub fn fit_transform(&self, docs: &[&str]) -> Result<(Vec<String>, Array2<f64>)> {
        let (vocab, csr) = self.fit_transform_sparse(docs)?;
        Ok((vocab, csr.to_dense()))
    }

    /// Sparse-output counterpart. For high-vocab corpora the dense
    /// `fit_transform` blows up memory (~n_docs × vocab × 8 bytes);
    /// `fit_transform_sparse` stays at O(total_token_occurrences).
    pub fn fit_transform_sparse(&self, docs: &[&str]) -> Result<(Vec<String>, CsrMatrix<f64>)> {
        let tokenised: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
        let fitted = self.fit_tokens(&tokenised)?;
        let csr = fitted.transform_tokens(&tokenised);
        Ok((fitted.vocabulary, csr))
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl FittedCountVectorizer {
    fn from_vocabulary(vocabulary: Vec<String>) -> Self {
        let term_to_col = vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();
        Self {
            vocabulary,
            term_to_col,
        }
    }

    /// The learned vocabulary, sorted; position = output column
    /// (sklearn's `get_feature_names_out()`).
    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    /// Output column of `term`, if it is in the vocabulary
    /// (sklearn's `vocabulary_[term]`).
    pub fn column_of(&self, term: &str) -> Option<usize> {
        self.term_to_col.get(term).copied()
    }

    /// Number of output columns (vocabulary size).
    pub fn n_features(&self) -> usize {
        self.vocabulary.len()
    }

    fn transform_tokens(&self, tokenised: &[Vec<String>]) -> CsrMatrix<f64> {
        let rows: Vec<Vec<(usize, f64)>> = tokenised
            .iter()
            .map(|tokens| {
                tokens
                    .iter()
                    .filter_map(|t| self.term_to_col.get(t.as_str()).map(|&c| (c, 1.0)))
                    .collect()
            })
            .collect();
        csr_from_rows(rows, self.vocabulary.len())
    }

    /// Encode `docs` as sparse term counts. Tokens outside the fitted
    /// vocabulary are ignored.
    pub fn transform_sparse(&self, docs: &[&str]) -> Result<CsrMatrix<f64>> {
        let tokenised: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
        Ok(self.transform_tokens(&tokenised))
    }

    /// Dense counterpart of [`transform_sparse`](Self::transform_sparse).
    pub fn transform(&self, docs: &[&str]) -> Result<Array2<f64>> {
        Ok(self.transform_sparse(docs)?.to_dense())
    }
}

// ---------------------------------------------------------------------------
// TfidfVectorizer (sklearn's smooth_idf=True, sublinear_tf=False, l2-norm)
// ---------------------------------------------------------------------------

/// TF-IDF features (sklearn's `TfidfVectorizer` with `smooth_idf=True`,
/// `sublinear_tf=False`, `norm="l2"` when `norm_l2`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TfidfVectorizer {
    pub min_df: usize,
    pub max_df_frac: f64,
    pub norm_l2: bool,
}

/// A fitted [`TfidfVectorizer`]: vocabulary plus per-term IDF weights.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedTfidfVectorizer {
    counts: FittedCountVectorizer,
    idf: Vec<f64>,
    norm_l2: bool,
}

impl TfidfVectorizer {
    pub fn new() -> Self {
        Self {
            min_df: 1,
            max_df_frac: 1.0,
            norm_l2: true,
        }
    }
    pub fn with_min_df(mut self, m: usize) -> Self {
        self.min_df = m;
        self
    }
    pub fn with_max_df_frac(mut self, f: f64) -> Self {
        self.max_df_frac = f;
        self
    }
    pub fn with_norm_l2(mut self, norm_l2: bool) -> Self {
        self.norm_l2 = norm_l2;
        self
    }

    fn count_vectorizer(&self) -> CountVectorizer {
        CountVectorizer {
            min_df: self.min_df,
            max_df_frac: self.max_df_frac,
        }
    }

    /// Learn vocabulary and IDF weights
    /// `idf(t) = ln((1 + n) / (1 + df(t))) + 1` from `docs`.
    pub fn fit(&self, docs: &[&str]) -> Result<FittedTfidfVectorizer> {
        let tokenised: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
        Ok(self.fit_tokens(&tokenised)?.0)
    }

    /// Fit and return the training count matrix too (avoids re-tokenising
    /// in `fit_transform`).
    fn fit_tokens(
        &self,
        tokenised: &[Vec<String>],
    ) -> Result<(FittedTfidfVectorizer, CsrMatrix<f64>)> {
        let counts = self.count_vectorizer().fit_tokens(tokenised)?;
        let csr = counts.transform_tokens(tokenised);
        let n = tokenised.len();
        let mut df_t = vec![0usize; counts.n_features()];
        for &c in &csr.indices {
            df_t[c] += 1;
        }
        let idf: Vec<f64> = df_t
            .iter()
            .map(|&df| ((1.0 + n as f64) / (1.0 + df as f64)).ln() + 1.0)
            .collect();
        Ok((
            FittedTfidfVectorizer {
                counts,
                idf,
                norm_l2: self.norm_l2,
            },
            csr,
        ))
    }

    /// Fit on `docs` and return `(vocabulary, dense tf-idf)`.
    pub fn fit_transform(&self, docs: &[&str]) -> Result<(Vec<String>, Array2<f64>)> {
        let (vocab, csr) = self.fit_transform_sparse(docs)?;
        Ok((vocab, csr.to_dense()))
    }

    /// Sparse-output TF-IDF. IDF is computed once per term, then applied
    /// element-wise to the sparse count matrix; optional L2-normalisation
    /// runs over each row's non-zero slice.
    pub fn fit_transform_sparse(&self, docs: &[&str]) -> Result<(Vec<String>, CsrMatrix<f64>)> {
        let tokenised: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
        let (fitted, counts) = self.fit_tokens(&tokenised)?;
        let csr = fitted.weight(counts);
        Ok((fitted.counts.vocabulary, csr))
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl FittedTfidfVectorizer {
    /// The learned vocabulary, sorted; position = output column.
    pub fn vocabulary(&self) -> &[String] {
        self.counts.vocabulary()
    }

    /// Output column of `term`, if it is in the vocabulary.
    pub fn column_of(&self, term: &str) -> Option<usize> {
        self.counts.column_of(term)
    }

    /// Per-term IDF weights (sklearn's `idf_`), aligned with `vocabulary()`.
    pub fn idf(&self) -> &[f64] {
        &self.idf
    }

    /// Number of output columns (vocabulary size).
    pub fn n_features(&self) -> usize {
        self.counts.n_features()
    }

    fn weight(&self, mut counts: CsrMatrix<f64>) -> CsrMatrix<f64> {
        for (v, &c) in counts.data.iter_mut().zip(counts.indices.iter()) {
            *v *= self.idf[c];
        }
        if self.norm_l2 {
            l2_normalize_rows(&mut counts);
        }
        counts
    }

    /// Encode `docs` as sparse TF-IDF using the fitted vocabulary and IDF.
    /// Tokens outside the vocabulary are ignored; a document with no known
    /// token yields an all-zero row.
    pub fn transform_sparse(&self, docs: &[&str]) -> Result<CsrMatrix<f64>> {
        Ok(self.weight(self.counts.transform_sparse(docs)?))
    }

    /// Dense counterpart of [`transform_sparse`](Self::transform_sparse).
    pub fn transform(&self, docs: &[&str]) -> Result<Array2<f64>> {
        Ok(self.transform_sparse(docs)?.to_dense())
    }
}

// ---------------------------------------------------------------------------
// HashingVectorizer (sklearn-compatible signed MurmurHash3)
// ---------------------------------------------------------------------------

/// Stateless hashed bag-of-words (sklearn's `HashingVectorizer`).
///
/// Each token is hashed with signed 32-bit MurmurHash3 (seed 0) exactly like
/// sklearn: column `|h| % n_features`, and with `alternate_sign` the value is
/// `+1` for `h >= 0` and `-1` otherwise. The sign comes from the hash's sign
/// bit while the column comes from its magnitude, so colliding tokens cancel
/// in expectation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct HashingVectorizer {
    pub n_features: usize,
    pub alternate_sign: bool,
    pub norm_l2: bool,
}

/// sklearn's default `n_features = 2**20`.
impl Default for HashingVectorizer {
    fn default() -> Self {
        Self::new(1 << 20)
    }
}

impl HashingVectorizer {
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            alternate_sign: true,
            norm_l2: true,
        }
    }

    /// Column and signed value for one token (sklearn's `_hashing_fast`).
    fn hash_token(&self, token: &str) -> (usize, f64) {
        let h = murmurhash3_32(token.as_bytes(), 0) as i32;
        let n = self.n_features as i64;
        let col = if h == i32::MIN {
            // sklearn special-cases abs(INT_MIN) overflow.
            (i32::MAX as i64 - (n - 1)).rem_euclid(n)
        } else {
            (h as i64).abs() % n
        } as usize;
        let value = if self.alternate_sign && h < 0 {
            -1.0
        } else {
            1.0
        };
        (col, value)
    }

    /// Sparse output, shape `(n_docs, n_features)`.
    pub fn transform_sparse(&self, docs: &[&str]) -> CsrMatrix<f64> {
        let rows: Vec<Vec<(usize, f64)>> = docs
            .iter()
            .map(|d| {
                if self.n_features == 0 {
                    return Vec::new();
                }
                tokenize(d).iter().map(|t| self.hash_token(t)).collect()
            })
            .collect();
        let mut csr = csr_from_rows(rows, self.n_features);
        if self.norm_l2 {
            l2_normalize_rows(&mut csr);
        }
        csr
    }

    /// Dense output, shape `(n_docs, n_features)`.
    pub fn transform(&self, docs: &[&str]) -> Array2<f64> {
        self.transform_sparse(docs).to_dense()
    }
}

// ---------------------------------------------------------------------------
// FeatureHasher
// ---------------------------------------------------------------------------

/// The hashing trick on named features (sklearn's
/// `feature_extraction.FeatureHasher`): each `(name, value)` pair adds
/// `±value` to column `|h| % n_features` of its row, `h` the signed 32-bit
/// MurmurHash3 of the name's UTF-8 bytes (seed 0), the sign that of `h`
/// with `alternate_sign`. Zero values are skipped; collisions are summed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FeatureHasher {
    pub n_features: usize,
    pub alternate_sign: bool,
}

/// sklearn's default `n_features = 2**20`.
impl Default for FeatureHasher {
    fn default() -> Self {
        Self {
            n_features: 1 << 20,
            alternate_sign: true,
        }
    }
}

impl FeatureHasher {
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            ..Self::default()
        }
    }

    fn hashing(&self) -> Result<HashingVectorizer> {
        if self.n_features == 0 || self.n_features > i32::MAX as usize {
            return Err(RustMlError::InvalidParameter(format!(
                "n_features must be in 1..=2^31-1, got {}",
                self.n_features
            )));
        }
        Ok(HashingVectorizer {
            n_features: self.n_features,
            alternate_sign: self.alternate_sign,
            norm_l2: false,
        })
    }

    /// Sparse output, shape `(samples.len(), n_features)`.
    pub fn transform_sparse<S: AsRef<str>>(
        &self,
        samples: &[Vec<(S, f64)>],
    ) -> Result<CsrMatrix<f64>> {
        let h = self.hashing()?;
        let rows = samples
            .iter()
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(_, v)| *v != 0.0)
                    .map(|(name, v)| {
                        let (col, sign) = h.hash_token(name.as_ref());
                        (col, sign * v)
                    })
                    .collect()
            })
            .collect();
        Ok(csr_from_rows(rows, self.n_features))
    }

    /// Dense output, shape `(samples.len(), n_features)`.
    pub fn transform<S: AsRef<str>>(&self, samples: &[Vec<(S, f64)>]) -> Result<Array2<f64>> {
        Ok(self.transform_sparse(samples)?.to_dense())
    }
}

/// MurmurHash3 x86 32-bit (Austin Appleby) — the hash behind
/// `sklearn.utils.murmurhash3_32`. Interpret the result as `i32` for
/// sklearn's default `positive=False`.
fn murmurhash3_32(key: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mut h = seed;
    let chunks = key.chunks_exact(4);
    let tail = chunks.remainder();
    for c in chunks {
        let mut k = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    if !tail.is_empty() {
        let mut k = 0u32;
        for (i, &b) in tail.iter().enumerate() {
            k ^= (b as u32) << (8 * i);
        }
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
    }
    h ^= key.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

#[cfg(test)]
mod tests {
    /// `FeatureHasher(16).transform([{"dog": 1, "cat": 2.5, "elephant": 4},
    /// {"run": 5, "colour=brown": 1}])` from scikit-learn 1.8.
    #[test]
    fn feature_hasher_matches_sklearn() {
        let samples = vec![
            vec![("dog", 1.0), ("cat", 2.5), ("elephant", 4.0)],
            vec![("run", 5.0), ("colour=brown", 1.0)],
        ];
        let got = super::FeatureHasher::new(16).transform(&samples).unwrap();
        let mut want = ndarray::Array2::<f64>::zeros((2, 16));
        want[[0, 5]] = -1.0;
        want[[0, 7]] = 2.5;
        want[[0, 14]] = -4.0;
        want[[1, 8]] = -5.0;
        want[[1, 14]] = 1.0;
        assert_eq!(got, want);
        assert!(super::FeatureHasher::new(0).transform(&samples).is_err());
    }

    use super::*;

    #[test]
    fn test_count_vectorizer_basic() {
        let docs = ["the cat sat", "the dog sat", "cat dog"];
        let cv = CountVectorizer::new();
        let (vocab, x) = cv.fit_transform(&docs).unwrap();
        assert!(vocab.contains(&"cat".to_string()));
        assert!(vocab.contains(&"dog".to_string()));
        assert!(vocab.contains(&"sat".to_string()));
        assert!(vocab.contains(&"the".to_string()));
        let cat_col = vocab.iter().position(|w| w == "cat").unwrap();
        assert_eq!(x[[0, cat_col]], 1.0);
        assert_eq!(x[[1, cat_col]], 0.0);
        assert_eq!(x[[2, cat_col]], 1.0);
    }

    #[test]
    fn test_tfidf_vectorizer_norm() {
        let docs = ["the cat sat", "the dog sat"];
        let tv = TfidfVectorizer::new();
        let (_, x) = tv.fit_transform(&docs).unwrap();
        for i in 0..2 {
            let s: f64 = (0..x.ncols()).map(|j| x[[i, j]].powi(2)).sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_count_vectorizer_sparse_matches_dense() {
        let docs = ["the cat sat on the mat", "the dog sat", "cat dog mat"];
        let cv = CountVectorizer::new();
        let (vocab_d, dense) = cv.fit_transform(&docs).unwrap();
        let (vocab_s, sparse) = cv.fit_transform_sparse(&docs).unwrap();
        assert_eq!(vocab_d, vocab_s);
        let dense_from_sparse = sparse.to_dense();
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert_eq!(dense[[i, j]], dense_from_sparse[[i, j]]);
            }
        }
        // "the" in doc 0 appears twice → expect a 2 somewhere on row 0.
        assert!(sparse.row_iter(0).any(|(_, v)| (v - 2.0).abs() < 1e-9));
    }

    #[test]
    fn test_tfidf_vectorizer_sparse_matches_dense() {
        let docs = ["the cat sat", "the dog sat", "cat dog"];
        let tv = TfidfVectorizer::new();
        let (_, dense) = tv.fit_transform(&docs).unwrap();
        let (_, sparse) = tv.fit_transform_sparse(&docs).unwrap();
        let dense_from_sparse = sparse.to_dense();
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert!(
                    (dense[[i, j]] - dense_from_sparse[[i, j]]).abs() < 1e-9,
                    "mismatch at [{i},{j}]: dense {} vs sparse {}",
                    dense[[i, j]],
                    dense_from_sparse[[i, j]]
                );
            }
        }
        // Sparse L2-row norms must equal 1.
        for i in 0..sparse.n_rows {
            let s: f64 = sparse.row_iter(i).map(|(_, v)| v * v).sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_hashing_vectorizer_no_oov() {
        let docs = ["unseenword wordone", "wordone wordtwo"];
        let hv = HashingVectorizer::new(8);
        let x = hv.transform(&docs);
        // Both documents produce nonzero rows.
        for i in 0..2 {
            let s: f64 = (0..x.ncols()).map(|j| x[[i, j]].abs()).sum();
            assert!(s > 0.0);
        }
    }

    // ---- Regression tests for issue #52. Reference values: sklearn 1.8.0. ----

    const TRAIN: [&str; 4] = [
        "the cat sat on the mat",
        "the dog ran on the mat",
        "the cat and the dog",
        "a cat and a mat",
    ];
    const TEST: [&str; 3] = ["the zebra ate the cat", "unknown words only", "dog dog mat"];

    #[test]
    fn test_count_fit_then_transform_unseen_docs() {
        let fitted = CountVectorizer::new().fit(&TRAIN).unwrap();
        assert_eq!(
            fitted.vocabulary(),
            &["and", "cat", "dog", "mat", "on", "ran", "sat", "the"]
        );
        assert_eq!(fitted.column_of("dog"), Some(2));
        assert_eq!(fitted.column_of("zebra"), None);
        let x = fitted.transform(&TEST).unwrap();
        let expected = [
            [0., 1., 0., 0., 0., 0., 0., 2.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 2., 1., 0., 0., 0., 0.],
        ];
        assert_eq!(x.dim(), (3, 8));
        for (i, row) in expected.iter().enumerate() {
            for (j, &e) in row.iter().enumerate() {
                assert_eq!(x[[i, j]], e, "[{i},{j}]");
            }
        }
        // fit_transform == fit + transform on the training docs.
        let (vocab, ft) = CountVectorizer::new().fit_transform(&TRAIN).unwrap();
        assert_eq!(vocab, fitted.vocabulary());
        assert_eq!(ft, fitted.transform(&TRAIN).unwrap());
        // Sparse variant agrees.
        let sp = fitted.transform_sparse(&TEST).unwrap();
        assert_eq!(sp.to_dense(), x);
        assert_eq!(sp.row_iter(1).count(), 0);
    }

    #[test]
    fn test_tfidf_fit_then_transform_unseen_docs() {
        let fitted = TfidfVectorizer::new().fit(&TRAIN).unwrap();
        let sk_idf = [
            1.5108256237659907,
            1.2231435513142097,
            1.5108256237659907,
            1.2231435513142097,
            1.5108256237659907,
            1.916290731874155,
            1.916290731874155,
            1.2231435513142097,
        ];
        for (a, e) in fitted.idf().iter().zip(sk_idf) {
            assert!((a - e).abs() < 1e-12, "idf {a} vs {e}");
        }
        let x = fitted.transform(&TEST).unwrap();
        let mut expected = Array2::<f64>::zeros((3, 8));
        expected[[0, 1]] = 0.4472135954999579;
        expected[[0, 7]] = 0.8944271909999159;
        expected[[2, 2]] = 0.9269367589632833;
        expected[[2, 3]] = 0.3752175967124194;
        for ((i, j), &e) in expected.indexed_iter() {
            assert!(
                (x[[i, j]] - e).abs() < 1e-12,
                "[{i},{j}]: {} vs {e}",
                x[[i, j]]
            );
        }
        let (_, ft) = TfidfVectorizer::new().fit_transform(&TRAIN).unwrap();
        let t = fitted.transform(&TRAIN).unwrap();
        for (a, b) in ft.iter().zip(t.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
        assert_eq!(fitted.transform_sparse(&TEST).unwrap().to_dense(), x);
    }

    #[test]
    fn test_min_df_greater_than_max_df_errors() {
        // sklearn: ValueError("max_df corresponds to < documents than min_df").
        let err = CountVectorizer::new()
            .with_min_df(3)
            .with_max_df_frac(0.5)
            .fit(&TRAIN);
        assert!(matches!(err, Err(RustMlError::InvalidParameter(_))));
        assert!(TfidfVectorizer::new()
            .with_min_df(3)
            .with_max_df_frac(0.5)
            .fit_transform(&TRAIN)
            .is_err());
        // Valid combination still filters.
        let fitted = CountVectorizer::new()
            .with_min_df(2)
            .with_max_df_frac(0.75)
            .fit(&TRAIN)
            .unwrap();
        // df: the=3 (<=3 ok), cat=3, mat=3, and=2, dog=2, on=2; ran/sat=1 dropped.
        assert_eq!(
            fitted.vocabulary(),
            &["and", "cat", "dog", "mat", "on", "the"]
        );
    }

    #[test]
    fn test_unicode_tokenizer_matches_sklearn() {
        assert_eq!(tokenize("Müller"), vec!["müller"]);
        assert_eq!(tokenize("café"), vec!["café"]);
        let docs = [
            "The cat sat on the mat",
            "Müller café STRASSE straße 2024 foo_bar x y",
            "ĞÜŞ İstanbul naïve façade",
        ];
        let (vocab, _) = CountVectorizer::new().fit_transform(&docs).unwrap();
        // sklearn 1.8.0 CountVectorizer().fit(docs).get_feature_names_out()
        let sk = [
            "2024", "café", "cat", "façade", "foo_bar", "mat", "müller", "naïve", "on", "sat",
            "stanbul", "strasse", "straße", "the", "ğüş",
        ];
        assert_eq!(vocab, sk);
    }

    #[test]
    fn test_murmurhash3_matches_sklearn() {
        // sklearn.utils.murmurhash3_32(w, seed) (signed).
        let cases: [(&str, i32, i32); 11] = [
            ("", 0, 142593372),
            ("a", 1009084850, -1293573533),
            ("ab", -1681926305, -684913081),
            ("abc", -1277324294, 1313807976),
            ("abcd", 1139631978, -396302900),
            ("abcde", -392455434, -1361433616),
            ("cat", 1751422759, -77977549),
            ("the", -1132748958, -1772783983),
            ("müller", 1613901631, -1594273775),
            ("café", 605818632, 1312538061),
            ("hello world", 1586663183, -368272391),
        ];
        for (w, h0, h42) in cases {
            assert_eq!(murmurhash3_32(w.as_bytes(), 0) as i32, h0, "{w:?} seed 0");
            assert_eq!(
                murmurhash3_32(w.as_bytes(), 42) as i32,
                h42,
                "{w:?} seed 42"
            );
        }
    }

    #[test]
    fn test_hashing_vectorizer_matches_sklearn() {
        let docs = [
            "The cat sat on the mat",
            "Müller café STRASSE straße 2024 foo_bar x y",
            "ĞÜŞ İstanbul naïve façade",
        ];
        let mut hv = HashingVectorizer::new(16);
        hv.norm_l2 = false;
        let x = hv.transform(&docs);
        // sklearn HashingVectorizer(n_features=16, norm=None).transform(docs)
        let sk: [[f64; 16]; 3] = [
            [
                0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., -1., 0., 0., -2., 0.,
            ],
            [
                0., 0., -1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
            ],
            [
                0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
            ],
        ];
        for (i, row) in sk.iter().enumerate() {
            for (j, &e) in row.iter().enumerate() {
                assert_eq!(x[[i, j]], e, "[{i},{j}]");
            }
        }
        // Default l2 norm.
        let xn = HashingVectorizer::new(16).transform(&docs);
        assert!((xn[[0, 14]] + 0.8164965809277261).abs() < 1e-12);
        assert_eq!(
            HashingVectorizer::new(16)
                .transform_sparse(&docs)
                .to_dense(),
            xn
        );
    }

    #[test]
    fn test_hashing_sign_independent_of_column_and_cancels() {
        // The old implementation used column = h % n and sign = h & 1, so for
        // even n_features the sign was a function of the column and colliding
        // tokens could never cancel. Find two distinct tokens that collide
        // into one column with opposite signs and check they cancel.
        let hv = HashingVectorizer {
            n_features: 4,
            alternate_sign: true,
            norm_l2: false,
        };
        let words: Vec<String> = (0..200).map(|i| format!("tok{i}")).collect();
        let mut pair = None;
        'outer: for a in &words {
            for b in &words {
                let (ca, va) = hv.hash_token(a);
                let (cb, vb) = hv.hash_token(b);
                if a != b && ca == cb && va != vb {
                    pair = Some((a.clone(), b.clone(), ca));
                    break 'outer;
                }
            }
        }
        let (a, b, col) = pair.expect("an opposite-sign collision exists");
        let doc = format!("{a} {b}");
        let x = hv.transform(&[doc.as_str()]);
        assert_eq!(
            x[[0, col]],
            0.0,
            "{a} and {b} should cancel in column {col}"
        );
        // Both signs occur in every column across many tokens.
        for c in 0..4 {
            let signs: HashSet<i32> = words
                .iter()
                .map(|w| hv.hash_token(w))
                .filter(|&(cc, _)| cc == c)
                .map(|(_, v)| v as i32)
                .collect();
            assert_eq!(signs.len(), 2, "column {c} only ever gets one sign");
        }
        // alternate_sign = false → all non-negative.
        let hv_pos = HashingVectorizer {
            alternate_sign: false,
            ..hv
        };
        assert!(hv_pos.transform(&[doc.as_str()]).iter().all(|&v| v >= 0.0));
    }
}
