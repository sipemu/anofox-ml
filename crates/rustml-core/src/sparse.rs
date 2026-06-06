//! Compressed Sparse Row matrix type, intended for high-vocab text
//! vectorisation output and downstream sparse-friendly estimators.
//!
//! Layout matches scipy.sparse.csr_matrix:
//!
//! - `indptr`: length `n_rows + 1`. Row `i` occupies the slice
//!   `data[indptr[i]..indptr[i+1]]` / `indices[indptr[i]..indptr[i+1]]`.
//! - `indices`: column indices for each non-zero, sorted ascending within a
//!   row (callers must maintain this invariant for predictable behaviour).
//! - `data`: parallel values.
//!
//! Operations are kept minimal: `nnz`, `density`, `to_dense`, `row_iter`.

use ndarray::Array2;

use crate::float::Float;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct CsrMatrix<F: Float> {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<F>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl<F: Float> CsrMatrix<F> {
    /// Build from a list of `(row, col, value)` triplets. Triplets do not
    /// need to be sorted; this constructor sorts within each row by column.
    /// Duplicate `(row, col)` entries are summed.
    pub fn from_triplets(n_rows: usize, n_cols: usize, triplets: Vec<(usize, usize, F)>) -> Self {
        // Bucket per row.
        let mut buckets: Vec<Vec<(usize, F)>> = vec![Vec::new(); n_rows];
        for (r, c, v) in triplets {
            buckets[r].push((c, v));
        }
        // Sort + dedup-by-column within each row.
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for row in buckets.iter_mut() {
            row.sort_by(|a, b| a.0.cmp(&b.0));
            // Sum duplicates.
            let mut last_col: Option<usize> = None;
            for &(c, v) in row.iter() {
                if Some(c) == last_col {
                    let n = data.len();
                    data[n - 1] = data[n - 1] + v;
                } else {
                    indices.push(c);
                    data.push(v);
                    last_col = Some(c);
                }
            }
            indptr.push(indices.len());
        }
        Self {
            indptr,
            indices,
            data,
            n_rows,
            n_cols,
        }
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn density(&self) -> f64 {
        if self.n_rows == 0 || self.n_cols == 0 {
            return 0.0;
        }
        self.nnz() as f64 / (self.n_rows as f64 * self.n_cols as f64)
    }

    /// Iterate non-zeros of row `i` as `(col, value)` pairs.
    pub fn row_iter(&self, i: usize) -> impl Iterator<Item = (usize, F)> + '_ {
        let start = self.indptr[i];
        let end = self.indptr[i + 1];
        self.indices[start..end]
            .iter()
            .copied()
            .zip(self.data[start..end].iter().copied())
    }

    pub fn to_dense(&self) -> Array2<F> {
        let mut out = Array2::<F>::zeros((self.n_rows, self.n_cols));
        for i in 0..self.n_rows {
            for (c, v) in self.row_iter(i) {
                out[[i, c]] = v;
            }
        }
        out
    }

    /// Sparse-dense matrix-vector multiply: `y = A x`. Returns a dense
    /// vector of length `n_rows`.
    pub fn matvec(&self, x: &[F]) -> Vec<F> {
        assert_eq!(x.len(), self.n_cols, "matvec: dimension mismatch");
        let mut y = vec![F::zero(); self.n_rows];
        for i in 0..self.n_rows {
            let mut s = F::zero();
            for (c, v) in self.row_iter(i) {
                s = s + v * x[c];
            }
            y[i] = s;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_from_triplets_basic() {
        // 3×4 matrix:
        // [[1, 0, 0, 2],
        //  [0, 3, 0, 0],
        //  [0, 0, 4, 5]]
        let csr = CsrMatrix::<f64>::from_triplets(
            3,
            4,
            vec![
                (0, 0, 1.0),
                (0, 3, 2.0),
                (1, 1, 3.0),
                (2, 2, 4.0),
                (2, 3, 5.0),
            ],
        );
        assert_eq!(csr.nnz(), 5);
        let dense = csr.to_dense();
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[0, 3]], 2.0);
        assert_eq!(dense[[1, 1]], 3.0);
        assert_eq!(dense[[2, 2]], 4.0);
        assert_eq!(dense[[2, 3]], 5.0);
        assert_eq!(dense[[1, 0]], 0.0);
    }

    #[test]
    fn test_csr_duplicate_triplets_sum() {
        let csr =
            CsrMatrix::<f64>::from_triplets(1, 3, vec![(0, 1, 1.0), (0, 1, 2.0), (0, 1, 3.0)]);
        assert_eq!(csr.nnz(), 1);
        assert_eq!(csr.to_dense()[[0, 1]], 6.0);
    }

    #[test]
    fn test_csr_matvec() {
        // [[1, 0], [0, 2]] * [3, 4] = [3, 8]
        let csr = CsrMatrix::<f64>::from_triplets(2, 2, vec![(0, 0, 1.0), (1, 1, 2.0)]);
        let y = csr.matvec(&[3.0, 4.0]);
        assert_eq!(y, vec![3.0, 8.0]);
    }

    #[test]
    fn test_csr_density() {
        let csr = CsrMatrix::<f64>::from_triplets(2, 2, vec![(0, 0, 1.0)]);
        assert!((csr.density() - 0.25).abs() < 1e-12);
    }
}
