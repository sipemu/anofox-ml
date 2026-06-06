//! Text feature extraction.
//!
//! Mirrors `sklearn.feature_extraction.text.{CountVectorizer, TfidfVectorizer,
//! HashingVectorizer}` in their simplest form: lowercase + token pattern
//! `[A-Za-z]{2,}`, no stop-words, dense output.

use ndarray::Array2;
use rustml_core::{CsrMatrix, Result, RustMlError};
use std::collections::HashMap;

fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for c in s.chars() {
        if c.is_ascii_alphabetic() {
            buf.push(c.to_ascii_lowercase());
        } else if !buf.is_empty() {
            if buf.len() >= 2 {
                out.push(buf.clone());
            }
            buf.clear();
        }
    }
    if buf.len() >= 2 {
        out.push(buf);
    }
    out
}

// ---------------------------------------------------------------------------
// CountVectorizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CountVectorizer {
    pub min_df: usize,
    pub max_df_frac: f64,
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

    pub fn fit_transform(&self, docs: &[&str]) -> Result<(Vec<String>, Array2<f64>)> {
        let (vocab, csr) = self.fit_transform_sparse(docs)?;
        Ok((vocab, csr.to_dense()))
    }

    /// Sparse-output counterpart. For high-vocab corpora the dense
    /// `fit_transform` blows up memory (~n_docs × vocab × 8 bytes);
    /// `fit_transform_sparse` stays at O(total_token_occurrences).
    pub fn fit_transform_sparse(&self, docs: &[&str]) -> Result<(Vec<String>, CsrMatrix<f64>)> {
        if docs.is_empty() {
            return Err(RustMlError::EmptyInput("no documents".into()));
        }
        // Pass 1: document frequency per term.
        let mut df: HashMap<String, usize> = HashMap::new();
        let tokenised: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
        for tokens in &tokenised {
            let mut seen = std::collections::HashSet::new();
            for t in tokens {
                if seen.insert(t.clone()) {
                    *df.entry(t.clone()).or_default() += 1;
                }
            }
        }
        let n = docs.len();
        let max_df = (self.max_df_frac * n as f64).floor() as usize;
        let mut vocab: Vec<String> = df
            .iter()
            .filter(|(_, &c)| c >= self.min_df && c <= max_df.max(self.min_df))
            .map(|(k, _)| k.clone())
            .collect();
        vocab.sort();
        let term_to_col: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();

        // Aggregate counts per (doc, col) and emit triplets.
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for (i, tokens) in tokenised.iter().enumerate() {
            let mut row_counts: HashMap<usize, f64> = HashMap::new();
            for t in tokens {
                if let Some(&c) = term_to_col.get(t) {
                    *row_counts.entry(c).or_default() += 1.0;
                }
            }
            for (c, v) in row_counts {
                triplets.push((i, c, v));
            }
        }
        let csr = CsrMatrix::from_triplets(n, vocab.len(), triplets);
        Ok((vocab, csr))
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TfidfVectorizer (sklearn's smooth_idf=True, sublinear_tf=False, l2-norm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TfidfVectorizer {
    pub min_df: usize,
    pub max_df_frac: f64,
    pub norm_l2: bool,
}

impl TfidfVectorizer {
    pub fn new() -> Self {
        Self {
            min_df: 1,
            max_df_frac: 1.0,
            norm_l2: true,
        }
    }

    pub fn fit_transform(&self, docs: &[&str]) -> Result<(Vec<String>, Array2<f64>)> {
        let (vocab, csr) = self.fit_transform_sparse(docs)?;
        Ok((vocab, csr.to_dense()))
    }

    /// Sparse-output TF-IDF. IDF is computed once per term, then applied
    /// element-wise to the sparse count matrix; optional L2-normalisation
    /// runs over each row's non-zero slice.
    pub fn fit_transform_sparse(&self, docs: &[&str]) -> Result<(Vec<String>, CsrMatrix<f64>)> {
        let cv = CountVectorizer {
            min_df: self.min_df,
            max_df_frac: self.max_df_frac,
        };
        let (vocab, counts) = cv.fit_transform_sparse(docs)?;
        let n = counts.n_rows;
        let d = counts.n_cols;

        // IDF: smooth, +1.
        let mut df_t = vec![0usize; d];
        for i in 0..n {
            for (c, _) in counts.row_iter(i) {
                df_t[c] += 1;
            }
        }
        let idf: Vec<f64> = df_t
            .iter()
            .map(|&df| ((1.0 + n as f64) / (1.0 + df as f64)).ln() + 1.0)
            .collect();

        // Apply IDF to each non-zero and optionally L2-normalise the row.
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(counts.nnz());
        let mut data = Vec::with_capacity(counts.nnz());
        indptr.push(0);
        for i in 0..n {
            let start = counts.indptr[i];
            let end = counts.indptr[i + 1];
            let mut row_vals: Vec<(usize, f64)> = counts.indices[start..end]
                .iter()
                .zip(counts.data[start..end].iter())
                .map(|(&c, &v)| (c, v * idf[c]))
                .collect();
            if self.norm_l2 {
                let s: f64 = row_vals.iter().map(|&(_, v)| v * v).sum();
                let norm = s.sqrt().max(1e-12);
                for entry in row_vals.iter_mut() {
                    entry.1 /= norm;
                }
            }
            for (c, v) in row_vals {
                indices.push(c);
                data.push(v);
            }
            indptr.push(indices.len());
        }
        let csr = CsrMatrix {
            indptr,
            indices,
            data,
            n_rows: n,
            n_cols: d,
        };
        Ok((vocab, csr))
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HashingVectorizer (fixed n_features, signed hash for stable signs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HashingVectorizer {
    pub n_features: usize,
    pub alternate_sign: bool,
    pub norm_l2: bool,
}

impl HashingVectorizer {
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            alternate_sign: true,
            norm_l2: true,
        }
    }

    pub fn transform(&self, docs: &[&str]) -> Array2<f64> {
        let n = docs.len();
        let mut x = Array2::<f64>::zeros((n, self.n_features));
        for (i, d) in docs.iter().enumerate() {
            for t in tokenize(d) {
                let h = fxhash(&t);
                let col = (h as usize) % self.n_features;
                let sign = if self.alternate_sign && (h & 1) == 0 {
                    1.0
                } else {
                    -1.0
                };
                let sign = if self.alternate_sign { sign } else { 1.0 };
                x[[i, col]] += sign;
            }
            if self.norm_l2 {
                let mut s = 0.0;
                for j in 0..self.n_features {
                    s += x[[i, j]] * x[[i, j]];
                }
                let nrm = s.sqrt().max(1e-12);
                for j in 0..self.n_features {
                    x[[i, j]] /= nrm;
                }
            }
        }
        x
    }
}

fn fxhash(s: &str) -> u64 {
    // Simple FNV-1a — stable across runs.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
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
}
