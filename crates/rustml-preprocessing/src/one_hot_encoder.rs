use ndarray::Array2;
use rustml_core::{Result, RustMlError};

/// One-hot encoder for integer-encoded categorical features.
///
/// Transforms integer-encoded columns into binary indicator columns.
/// For example, a column with values [0, 1, 2] becomes three binary columns:
/// ```text
/// [0] -> [1, 0, 0]
/// [1] -> [0, 1, 0]
/// [2] -> [0, 0, 1]
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OneHotEncoder;

impl OneHotEncoder {
    /// Create a new `OneHotEncoder`.
    pub fn new() -> Self {
        Self
    }

    /// Fit the encoder on integer-encoded data.
    ///
    /// Learns the number of unique categories per column.
    pub fn fit(&self, x: &Array2<usize>) -> Result<FittedOneHotEncoder> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        let ncols = x.ncols();
        let mut categories = Vec::with_capacity(ncols);

        for j in 0..ncols {
            let col = x.column(j);
            let max_val = col.iter().copied().max().unwrap_or(0);
            categories.push(max_val + 1);
        }

        Ok(FittedOneHotEncoder { categories })
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted OneHotEncoder — holds the number of unique categories per column.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedOneHotEncoder {
    categories: Vec<usize>,
}

impl FittedOneHotEncoder {
    /// Transform integer-encoded data into one-hot encoded data.
    ///
    /// Each input column of `k` categories is expanded into `k` binary columns.
    pub fn transform(&self, x: &Array2<usize>) -> Result<Array2<f64>> {
        if x.ncols() != self.categories.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} columns, got {}",
                self.categories.len(),
                x.ncols()
            )));
        }

        let total_out_cols: usize = self.categories.iter().sum();
        let nrows = x.nrows();
        let mut result = Array2::<f64>::zeros((nrows, total_out_cols));

        for i in 0..nrows {
            let mut col_offset = 0;
            for j in 0..x.ncols() {
                let val = x[[i, j]];
                if val >= self.categories[j] {
                    return Err(RustMlError::InvalidParameter(format!(
                        "value {} in column {} exceeds number of categories {}",
                        val, j, self.categories[j]
                    )));
                }
                result[[i, col_offset + val]] = 1.0;
                col_offset += self.categories[j];
            }
        }

        Ok(result)
    }

    /// Return the number of categories per original column.
    pub fn categories(&self) -> &[usize] {
        &self.categories
    }

    /// Return the total number of output columns after one-hot encoding.
    pub fn n_output_features(&self) -> usize {
        self.categories.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_single_column() {
        let x = array![[0usize], [1], [2]];
        let encoder = OneHotEncoder::new();
        let fitted = encoder.fit(&x).unwrap();
        let encoded = fitted.transform(&x).unwrap();

        assert_eq!(encoded.shape(), &[3, 3]);
        // Row 0: [1, 0, 0]
        assert_abs_diff_eq!(encoded[[0, 0]], 1.0);
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0);
        assert_abs_diff_eq!(encoded[[0, 2]], 0.0);
        // Row 1: [0, 1, 0]
        assert_abs_diff_eq!(encoded[[1, 0]], 0.0);
        assert_abs_diff_eq!(encoded[[1, 1]], 1.0);
        assert_abs_diff_eq!(encoded[[1, 2]], 0.0);
        // Row 2: [0, 0, 1]
        assert_abs_diff_eq!(encoded[[2, 0]], 0.0);
        assert_abs_diff_eq!(encoded[[2, 1]], 0.0);
        assert_abs_diff_eq!(encoded[[2, 2]], 1.0);
    }

    #[test]
    fn test_multiple_columns() {
        // Column 0 has 2 categories (0, 1), column 1 has 3 categories (0, 1, 2)
        let x = array![[0usize, 2], [1, 0], [0, 1]];
        let encoder = OneHotEncoder::new();
        let fitted = encoder.fit(&x).unwrap();
        let encoded = fitted.transform(&x).unwrap();

        assert_eq!(encoded.shape(), &[3, 5]); // 2 + 3 = 5 output columns
        assert_eq!(fitted.n_output_features(), 5);

        // Row 0: col0=0 -> [1,0], col1=2 -> [0,0,1] => [1,0,0,0,1]
        assert_abs_diff_eq!(encoded[[0, 0]], 1.0);
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0);
        assert_abs_diff_eq!(encoded[[0, 2]], 0.0);
        assert_abs_diff_eq!(encoded[[0, 3]], 0.0);
        assert_abs_diff_eq!(encoded[[0, 4]], 1.0);

        // Row 1: col0=1 -> [0,1], col1=0 -> [1,0,0] => [0,1,1,0,0]
        assert_abs_diff_eq!(encoded[[1, 0]], 0.0);
        assert_abs_diff_eq!(encoded[[1, 1]], 1.0);
        assert_abs_diff_eq!(encoded[[1, 2]], 1.0);
        assert_abs_diff_eq!(encoded[[1, 3]], 0.0);
        assert_abs_diff_eq!(encoded[[1, 4]], 0.0);
    }

    #[test]
    fn test_binary_column() {
        let x = array![[0usize], [1], [1], [0]];
        let encoder = OneHotEncoder::new();
        let fitted = encoder.fit(&x).unwrap();
        let encoded = fitted.transform(&x).unwrap();

        assert_eq!(encoded.shape(), &[4, 2]);
        assert_eq!(fitted.categories(), &[2]);
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<usize> = Array2::zeros((0, 0));
        let encoder = OneHotEncoder::new();
        assert!(encoder.fit(&x).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let x_train = array![[0usize, 1], [1, 0]];
        let encoder = OneHotEncoder::new();
        let fitted = encoder.fit(&x_train).unwrap();

        let x_wrong = array![[0usize, 1, 2]];
        assert!(fitted.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_unknown_category_in_transform() {
        let x_train = array![[0usize], [1]];
        let encoder = OneHotEncoder::new();
        let fitted = encoder.fit(&x_train).unwrap();

        // Value 5 was never seen during fit (max was 1, so categories = 2)
        let x_test = array![[5usize]];
        assert!(fitted.transform(&x_test).is_err());
    }

    #[test]
    fn test_all_zeros() {
        let x = array![[0usize, 0], [0, 0], [0, 0]];
        let encoder = OneHotEncoder::new();
        let fitted = encoder.fit(&x).unwrap();
        let encoded = fitted.transform(&x).unwrap();

        // 1 category per column -> 2 output columns
        assert_eq!(encoded.shape(), &[3, 2]);
        // Every row: [1, 1]
        for i in 0..3 {
            assert_abs_diff_eq!(encoded[[i, 0]], 1.0);
            assert_abs_diff_eq!(encoded[[i, 1]], 1.0);
        }
    }

    #[test]
    fn test_row_sums() {
        // Each one-hot block should have exactly one 1 per row per original column
        let x = array![[0usize, 2, 1], [2, 0, 0], [1, 1, 2]];
        let encoder = OneHotEncoder::new();
        let fitted = encoder.fit(&x).unwrap();
        let encoded = fitted.transform(&x).unwrap();

        // Total output columns = 3 + 3 + 3 = 9
        assert_eq!(encoded.shape(), &[3, 9]);

        // Each row should sum to number of original columns (3)
        for i in 0..3 {
            let row_sum: f64 = encoded.row(i).sum();
            assert_abs_diff_eq!(row_sum, 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_default() {
        let encoder = OneHotEncoder::default();
        let x = array![[0usize], [1]];
        let fitted = encoder.fit(&x).unwrap();
        assert_eq!(fitted.categories(), &[2]);
    }
}
