//! Conversion helpers between ndarray and faer types.

use faer::{Col, Mat};
use ndarray::{Array1, Array2};

/// Convert an ndarray Array2<f64> to a faer Mat<f64>.
pub fn ndarray_to_mat(x: &Array2<f64>) -> Mat<f64> {
    let (nrows, ncols) = x.dim();
    Mat::from_fn(nrows, ncols, |i, j| x[[i, j]])
}

/// Convert an ndarray Array1<f64> to a faer Col<f64>.
pub fn ndarray_to_col(y: &Array1<f64>) -> Col<f64> {
    Col::from_fn(y.len(), |i| y[i])
}

/// Convert a faer Col<f64> to an ndarray Array1<f64>.
pub fn col_to_ndarray(c: &Col<f64>) -> Array1<f64> {
    Array1::from_vec((0..c.nrows()).map(|i| c[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    #[test]
    fn test_roundtrip_mat() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mat = ndarray_to_mat(&x);
        assert_eq!(mat.nrows(), 3);
        assert_eq!(mat.ncols(), 2);
        assert_abs_diff_eq!(mat[(0, 0)], 1.0);
        assert_abs_diff_eq!(mat[(2, 1)], 6.0);
    }

    #[test]
    fn test_roundtrip_col() {
        let y = array![1.0, 2.0, 3.0];
        let col = ndarray_to_col(&y);
        let back = col_to_ndarray(&col);
        assert_abs_diff_eq!(back[0], 1.0);
        assert_abs_diff_eq!(back[2], 3.0);
    }
}
