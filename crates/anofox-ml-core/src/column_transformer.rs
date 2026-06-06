//! ColumnTransformer — apply different transformers to different column subsets.
//!
//! Similar to sklearn's `ColumnTransformer`, this allows building pipelines
//! where different feature groups receive different preprocessing.

use ndarray::Array2;

use crate::error::{Result, RustMlError};
use crate::float::Float;
use crate::pipeline::{FitTransform, TransformStep};
use crate::traits::{FitUnsupervised, Transform};

/// Specifies which columns a transformer branch operates on.
#[derive(Debug, Clone)]
pub enum ColumnSelector {
    /// Select columns by index.
    Indices(Vec<usize>),
    /// Select all columns.
    All,
}

/// Applies different transformers to different column subsets and
/// concatenates the results horizontally.
///
/// # Example
///
/// ```ignore
/// use anofox_ml_core::{ColumnTransformer, ColumnSelector};
/// use anofox_ml_preprocessing::{StandardScaler, OneHotEncoder};
///
/// let ct = ColumnTransformer::new()
///     .push("numeric", ColumnSelector::Indices(vec![0, 1, 2]), StandardScaler::new())
///     .push("categorical", ColumnSelector::Indices(vec![3, 4]), StandardScaler::new());
/// ```
pub struct ColumnTransformer<F: Float> {
    branches: Vec<Branch<F>>,
    /// How to handle columns not assigned to any branch.
    remainder: Remainder,
}

struct Branch<F: Float> {
    name: String,
    selector: ColumnSelector,
    transformer: Box<dyn FitTransform<F>>,
}

/// What to do with columns not selected by any branch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Remainder {
    /// Drop unselected columns (default).
    Drop,
    /// Pass through unselected columns unchanged.
    Passthrough,
}

impl Default for Remainder {
    fn default() -> Self {
        Remainder::Drop
    }
}

impl<F: Float> ColumnTransformer<F> {
    /// Create a new empty ColumnTransformer.
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
            remainder: Remainder::Drop,
        }
    }

    /// Add a transformer branch operating on the specified columns.
    pub fn push(
        mut self,
        name: impl Into<String>,
        selector: ColumnSelector,
        transformer: impl FitTransform<F> + 'static,
    ) -> Self {
        self.branches.push(Branch {
            name: name.into(),
            selector,
            transformer: Box::new(transformer),
        });
        self
    }

    /// Set the remainder strategy (default: Drop).
    pub fn with_remainder(mut self, remainder: Remainder) -> Self {
        self.remainder = remainder;
        self
    }
}

impl<F: Float> std::fmt::Debug for ColumnTransformer<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnTransformer")
            .field("n_branches", &self.branches.len())
            .field("remainder", &self.remainder)
            .finish()
    }
}

/// Resolve a ColumnSelector to concrete column indices.
fn resolve_columns(selector: &ColumnSelector, n_cols: usize) -> Vec<usize> {
    match selector {
        ColumnSelector::Indices(indices) => indices.clone(),
        ColumnSelector::All => (0..n_cols).collect(),
    }
}

/// Select specific columns from a matrix, returning a new C-contiguous array.
fn select_columns<F: Float>(x: &Array2<F>, cols: &[usize]) -> Array2<F> {
    let n_rows = x.nrows();
    let n_cols = cols.len();
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for &c in cols {
            data.push(x[[i, c]]);
        }
    }
    Array2::from_shape_vec((n_rows, n_cols), data).expect("shape matches data length")
}

/// Fitted ColumnTransformer.
pub struct FittedColumnTransformer<F: Float> {
    fitted_branches: Vec<FittedBranch<F>>,
    /// Columns to passthrough (only if remainder=Passthrough).
    passthrough_cols: Vec<usize>,
    n_features_in: usize,
}

struct FittedBranch<F: Float> {
    name: String,
    cols: Vec<usize>,
    fitted: Box<dyn TransformStep<F>>,
}

// Safety: all TransformStep<F> impls are already Send + Sync
unsafe impl<F: Float> Send for FittedColumnTransformer<F> {}
unsafe impl<F: Float> Sync for FittedColumnTransformer<F> {}

impl<F: Float> Transform<F> for FittedColumnTransformer<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features_in {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features_in,
                x.ncols()
            )));
        }

        let n_rows = x.nrows();
        let mut parts: Vec<Array2<F>> = Vec::with_capacity(self.fitted_branches.len() + 1);

        for branch in &self.fitted_branches {
            let sub_x = select_columns(x, &branch.cols);
            let transformed = branch.fitted.transform(&sub_x)?;
            parts.push(transformed);
        }

        if !self.passthrough_cols.is_empty() {
            parts.push(select_columns(x, &self.passthrough_cols));
        }

        concat_horizontal(n_rows, &parts)
    }
}

impl<F: Float + 'static> FitUnsupervised<F> for ColumnTransformer<F> {
    type Fitted = FittedColumnTransformer<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let n_cols = x.ncols();

        let mut used_cols = std::collections::HashSet::new();
        let mut fitted_branches = Vec::with_capacity(self.branches.len());

        for branch in &self.branches {
            let cols = resolve_columns(&branch.selector, n_cols);

            for &c in &cols {
                if c >= n_cols {
                    return Err(RustMlError::InvalidParameter(format!(
                        "column index {} out of range for data with {} columns",
                        c, n_cols
                    )));
                }
                used_cols.insert(c);
            }

            let sub_x = select_columns(x, &cols);
            let (fitted_step, _) = branch.transformer.fit_transform(&sub_x)?;

            fitted_branches.push(FittedBranch {
                name: branch.name.clone(),
                cols,
                fitted: fitted_step,
            });
        }

        let passthrough_cols: Vec<usize> = if self.remainder == Remainder::Passthrough {
            (0..n_cols).filter(|c| !used_cols.contains(c)).collect()
        } else {
            Vec::new()
        };

        Ok(FittedColumnTransformer {
            fitted_branches,
            passthrough_cols,
            n_features_in: n_cols,
        })
    }
}

/// Concatenate arrays horizontally.
fn concat_horizontal<F: Float>(n_rows: usize, parts: &[Array2<F>]) -> Result<Array2<F>> {
    if parts.is_empty() {
        return Ok(Array2::zeros((n_rows, 0)));
    }

    let total_cols: usize = parts.iter().map(|p| p.ncols()).sum();
    let mut result = Array2::zeros((n_rows, total_cols));
    let mut col_offset = 0;
    for part in parts {
        for j in 0..part.ncols() {
            for i in 0..n_rows {
                result[[i, col_offset + j]] = part[[i, j]];
            }
        }
        col_offset += part.ncols();
    }

    Ok(result)
}

impl<F: Float> FittedColumnTransformer<F> {
    /// Return the branch names.
    pub fn branch_names(&self) -> Vec<&str> {
        self.fitted_branches
            .iter()
            .map(|b| b.name.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{FitUnsupervised, Transform};
    use ndarray::array;

    /// Dummy scaler that multiplies by 2 (for testing column selection).
    #[derive(Debug, Clone)]
    struct DoubleScaler;

    struct FittedDoubleScaler;

    impl<F: Float> FitUnsupervised<F> for DoubleScaler {
        type Fitted = FittedDoubleScaler;
        fn fit(&self, _x: &Array2<F>) -> Result<Self::Fitted> {
            Ok(FittedDoubleScaler)
        }
    }

    impl<F: Float> Transform<F> for FittedDoubleScaler {
        fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
            Ok(x.mapv(|v| v + v))
        }
    }

    /// Identity transformer for passthrough testing.
    #[derive(Debug, Clone)]
    struct IdentityTransformer;

    struct FittedIdentity;

    impl<F: Float> FitUnsupervised<F> for IdentityTransformer {
        type Fitted = FittedIdentity;
        fn fit(&self, _x: &Array2<F>) -> Result<Self::Fitted> {
            Ok(FittedIdentity)
        }
    }

    impl<F: Float> Transform<F> for FittedIdentity {
        fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
            Ok(x.to_owned())
        }
    }

    #[test]
    fn test_column_transformer_basic() {
        let x = array![[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]];

        let ct = ColumnTransformer::<f64>::new()
            .push(
                "double_01",
                ColumnSelector::Indices(vec![0, 1]),
                DoubleScaler,
            )
            .push(
                "identity_2",
                ColumnSelector::Indices(vec![2]),
                IdentityTransformer,
            );

        let fitted = FitUnsupervised::fit(&ct, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed, array![[2.0, 20.0, 100.0], [4.0, 40.0, 200.0]]);
    }

    #[test]
    fn test_column_transformer_passthrough() {
        let x = array![[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]];

        let ct = ColumnTransformer::<f64>::new()
            .push("double_0", ColumnSelector::Indices(vec![0]), DoubleScaler)
            .with_remainder(Remainder::Passthrough);

        let fitted = FitUnsupervised::fit(&ct, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.ncols(), 3);
        assert_eq!(transformed[[0, 0]], 2.0);
        assert_eq!(transformed[[0, 1]], 10.0);
        assert_eq!(transformed[[0, 2]], 100.0);
    }

    #[test]
    fn test_column_transformer_drop_remainder() {
        let x = array![[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]];

        let ct = ColumnTransformer::<f64>::new().push(
            "double_0",
            ColumnSelector::Indices(vec![0]),
            DoubleScaler,
        );

        let fitted = FitUnsupervised::fit(&ct, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.ncols(), 1);
        assert_eq!(transformed[[0, 0]], 2.0);
        assert_eq!(transformed[[1, 0]], 4.0);
    }

    #[test]
    fn test_column_transformer_all_selector() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let ct =
            ColumnTransformer::<f64>::new().push("double_all", ColumnSelector::All, DoubleScaler);

        let fitted = FitUnsupervised::fit(&ct, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed, array![[2.0, 4.0], [6.0, 8.0]]);
    }

    #[test]
    fn test_column_transformer_shape_mismatch_predict() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let ct = ColumnTransformer::<f64>::new().push(
            "a",
            ColumnSelector::Indices(vec![0]),
            IdentityTransformer,
        );

        let fitted = FitUnsupervised::fit(&ct, &x).unwrap();
        let x_bad = array![[1.0, 2.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_column_transformer_invalid_column() {
        let x = array![[1.0, 2.0]];

        let ct = ColumnTransformer::<f64>::new().push(
            "bad",
            ColumnSelector::Indices(vec![5]),
            IdentityTransformer,
        );

        assert!(FitUnsupervised::fit(&ct, &x).is_err());
    }
}
