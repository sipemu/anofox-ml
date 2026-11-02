//! Conversion helpers between ndarray and faer types.

use anofox_ml_core::{Result, RustMlError};
use anofox_regression::core::RegressionResult;
use anofox_regression::solvers::RegressionError;
use faer::{Col, Mat, MatRef};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix1, Ix2};

/// Convert an ndarray `Array2<f64>` to a faer `Mat<f64>`.
///
/// Upstream solvers take an owned column-major `Mat`, so one copy is
/// unavoidable; for a standard-layout (row-major) array it goes through a
/// strided faer view instead of bounds-checked element indexing.
pub fn ndarray_to_mat<'a>(x: impl AsArray<'a, f64, Ix2>) -> Mat<f64> {
    let x: ArrayView2<f64> = x.into();
    let (nrows, ncols) = x.dim();
    match x.as_slice() {
        Some(s) => MatRef::from_row_major_slice(s, nrows, ncols).to_owned(),
        None => Mat::from_fn(nrows, ncols, |i, j| x[[i, j]]),
    }
}

/// Convert an ndarray `Array1<f64>` to a faer `Col<f64>`.
pub fn ndarray_to_col<'a>(y: impl AsArray<'a, f64, Ix1>) -> Col<f64> {
    let y: ArrayView1<f64> = y.into();
    match y.as_slice() {
        Some(s) => Col::from_fn(s.len(), |i| s[i]),
        None => Col::from_fn(y.len(), |i| y[i]),
    }
}

/// Convert a faer `Col<f64>` to an ndarray `Array1<f64>`.
pub fn col_to_ndarray(c: &Col<f64>) -> Array1<f64> {
    Array1::from_vec((0..c.nrows()).map(|i| c[i]).collect())
}

/// Convert a faer `Mat<f64>` to an ndarray `Array2<f64>`.
pub fn mat_to_ndarray(m: &Mat<f64>) -> Array2<f64> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
}

/// Map an upstream `anofox-regression` error into this crate's error type.
pub fn upstream_err(e: RegressionError) -> RustMlError {
    RustMlError::InvalidParameter(e.to_string())
}

/// Validate the `(X, y)` pair every regressor needs: matching rows, non-empty.
pub fn check_xy<'a, 'b>(
    x: impl AsArray<'a, f64, Ix2>,
    y: impl AsArray<'b, f64, Ix1>,
) -> Result<()> {
    let x: ArrayView2<f64> = x.into();
    let y: ArrayView1<f64> = y.into();
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(RustMlError::EmptyInput("training data is empty".into()));
    }
    Ok(())
}

/// Validate the feature count of a prediction matrix.
pub fn check_n_features<'a>(x: impl AsArray<'a, f64, Ix2>, expected: usize) -> Result<()> {
    let x: ArrayView2<f64> = x.into();
    if x.ncols() != expected {
        return Err(RustMlError::ShapeMismatch(format!(
            "expected {expected} features, got {}",
            x.ncols()
        )));
    }
    Ok(())
}

/// `X·coef + intercept`, with the feature-count check.
pub fn linear_predict<'a>(
    x: impl AsArray<'a, f64, Ix2>,
    coef: &Array1<f64>,
    intercept: f64,
) -> Result<Array1<f64>> {
    let x: ArrayView2<f64> = x.into();
    check_n_features(x, coef.len())?;
    Ok(x.dot(coef).mapv(|v| v + intercept))
}

/// Upstream reports aliased (collinear) coefficients as `NaN`; at the
/// sklearn-shaped boundary they are dropped from the model (coefficient 0).
pub fn coef_from_col(c: &Col<f64>) -> Array1<f64> {
    Array1::from_shape_fn(c.nrows(), |i| if c[i].is_nan() { 0.0 } else { c[i] })
}

/// `(coef, intercept)` cached from an upstream result for prediction in
/// ndarray: aliased or `NaN` coefficients contribute nothing, exactly as in
/// upstream `predict`, and a missing intercept is 0.
pub(crate) fn predict_params(r: &RegressionResult) -> (Array1<f64>, f64) {
    let coef = Array1::from_shape_fn(r.coefficients.nrows(), |j| {
        let c = r.coefficients[j];
        if c.is_nan() || r.aliased.get(j).copied().unwrap_or(false) {
            0.0
        } else {
            c
        }
    });
    (coef, r.intercept.unwrap_or(0.0))
}

/// Validate optional sample weights (length `n`, finite, non-negative, not
/// all zero) and return them, or a vector of ones when absent.
pub(crate) fn resolve_sample_weight(
    sample_weight: Option<ArrayView1<f64>>,
    n: usize,
) -> anofox_ml_core::Result<Array1<f64>> {
    use anofox_ml_core::RustMlError;
    match sample_weight {
        None => Ok(Array1::ones(n)),
        Some(w) => {
            if w.len() != n {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight has {} elements but X has {} rows",
                    w.len(),
                    n
                )));
            }
            if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(RustMlError::InvalidParameter(
                    "sample_weight must be finite and non-negative".into(),
                ));
            }
            if w.sum() <= 0.0 {
                return Err(RustMlError::InvalidParameter(
                    "sample_weight must not sum to zero".into(),
                ));
            }
            Ok(w.to_owned())
        }
    }
}

/// Validate an optional per-row vector (offset / exposure) of length `n`.
pub(crate) fn check_row_vector(
    v: Option<&Array1<f64>>,
    n: usize,
    name: &str,
) -> anofox_ml_core::Result<()> {
    use anofox_ml_core::RustMlError;
    if let Some(v) = v {
        if v.len() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "{name} has {} elements but X has {} rows",
                v.len(),
                n
            )));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(RustMlError::InvalidParameter(format!(
                "{name} must be finite"
            )));
        }
    }
    Ok(())
}

/// `ln(exposure)` after checking that every exposure is positive.
pub(crate) fn exposure_to_offset(exposure: &Array1<f64>) -> anofox_ml_core::Result<Array1<f64>> {
    if exposure.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(anofox_ml_core::RustMlError::InvalidParameter(
            "exposure must be finite and strictly positive".into(),
        ));
    }
    Ok(exposure.mapv(f64::ln))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    #[test]
    fn test_roundtrip_mat() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mat = ndarray_to_mat(x.view());
        assert_eq!(mat.nrows(), 3);
        assert_eq!(mat.ncols(), 2);
        assert_abs_diff_eq!(mat[(0, 0)], 1.0);
        assert_abs_diff_eq!(mat[(2, 1)], 6.0);
    }

    #[test]
    fn test_mat_from_non_standard_layout() {
        use ndarray::ShapeBuilder;
        let c = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let f = Array2::from_shape_vec((3, 2).f(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]).unwrap();
        assert!(f.as_slice().is_none());
        let (mc, mf) = (ndarray_to_mat(c.view()), ndarray_to_mat(f.view()));
        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(mc[(i, j)], c[[i, j]]);
                assert_abs_diff_eq!(mf[(i, j)], c[[i, j]]);
            }
        }
    }

    #[test]
    fn test_roundtrip_col() {
        let y = array![1.0, 2.0, 3.0];
        let col = ndarray_to_col(y.view());
        let back = col_to_ndarray(&col);
        assert_abs_diff_eq!(back[0], 1.0);
        assert_abs_diff_eq!(back[2], 3.0);
    }
}
