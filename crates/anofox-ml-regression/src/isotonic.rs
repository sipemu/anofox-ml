//! Isotonic (monotonic) regression wrapper.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::IsotonicRegressor as InnerIsotonic;
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Handling of inputs outside the training range at prediction time
/// (sklearn `out_of_bounds`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum OutOfBounds {
    /// Predict NaN (sklearn's default).
    #[default]
    Nan,
    /// Predict the value at the nearest end.
    Clip,
    /// Return an error.
    Raise,
}

/// Isotonic regression estimator.
///
/// Fits a monotonic (non-decreasing or non-increasing) function to 1D data.
/// Requires exactly one feature column. As sklearn's
/// `IsotonicRegression.predict`, predictions interpolate linearly between
/// the fitted points; outside the training range they follow
/// `out_of_bounds`.
#[derive(Debug, Clone)]
pub struct IsotonicRegressor {
    increasing: bool,
    out_of_bounds: OutOfBounds,
}

impl IsotonicRegressor {
    pub fn new() -> Self {
        Self {
            increasing: true,
            out_of_bounds: OutOfBounds::Nan,
        }
    }

    /// Set the handling of out-of-range inputs.
    pub fn with_out_of_bounds(mut self, oob: OutOfBounds) -> Self {
        self.out_of_bounds = oob;
        self
    }

    pub fn with_increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }
}

impl Default for IsotonicRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted isotonic regression model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedIsotonicRegressor {
    /// Training R² of the upstream fit.
    r_squared: f64,
    /// Sorted distinct training inputs (sklearn `X_thresholds_`).
    x_thresholds: Vec<f64>,
    /// Fitted values at `x_thresholds` (sklearn `y_thresholds_`).
    y_thresholds: Vec<f64>,
    out_of_bounds: OutOfBounds,
}

impl FittedIsotonicRegressor {
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }
}

impl Fit<f64> for IsotonicRegressor {
    type Fitted = FittedIsotonicRegressor;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
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
        if x.ncols() != 1 {
            return Err(RustMlError::InvalidParameter(
                "isotonic regression requires exactly one feature column".into(),
            ));
        }

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let inner_model = InnerIsotonic::builder().increasing(self.increasing).build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        // Fitted value at every distinct training input: the knots of the
        // interpolating prediction function.
        let mut xs: Vec<f64> = x.column(0).to_vec();
        xs.sort_by(f64::total_cmp);
        xs.dedup();
        let knots = Array2::from_shape_vec((xs.len(), 1), xs.clone()).expect("one column");
        let ys = col_to_ndarray(&fitted.predict(&ndarray_to_mat(knots.view()))).to_vec();
        Ok(FittedIsotonicRegressor {
            r_squared: fitted.r_squared(),
            x_thresholds: xs,
            y_thresholds: ys,
            out_of_bounds: self.out_of_bounds,
        })
    }
}

impl Predict<f64> for FittedIsotonicRegressor {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != 1 {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected 1 feature, got {}",
                x.ncols()
            )));
        }

        let (xs, ys) = (&self.x_thresholds, &self.y_thresholds);
        let (lo, hi) = (xs[0], xs[xs.len() - 1]);
        x.column(0)
            .iter()
            .map(|&v| {
                if v < lo || v > hi {
                    match self.out_of_bounds {
                        OutOfBounds::Nan => return Ok(f64::NAN),
                        OutOfBounds::Raise => {
                            return Err(RustMlError::InvalidParameter(format!(
                                "{v} is outside the training range [{lo}, {hi}]"
                            )))
                        }
                        OutOfBounds::Clip => {}
                    }
                }
                let v = v.clamp(lo, hi);
                // np.interp
                let j = xs.partition_point(|&t| t < v);
                Ok(if j == 0 {
                    ys[0]
                } else if xs[j.min(xs.len() - 1)] == v {
                    ys[j]
                } else {
                    let (x0, x1, y0, y1) = (xs[j - 1], xs[j], ys[j - 1], ys[j]);
                    y0 + (v - x0) / (x1 - x0) * (y1 - y0)
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_isotonic_increasing() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        // Noisy increasing data with a violation: 3.0 > 2.5
        let y = array![1.0, 2.0, 3.0, 2.5, 4.0, 5.0];

        let fitted = IsotonicRegressor::new().fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // After isotonic fit, predictions should be non-decreasing
        for i in 1..preds.len() {
            assert!(
                preds[i] >= preds[i - 1] - 1e-10,
                "predictions should be non-decreasing: preds[{}]={} < preds[{}]={}",
                i,
                preds[i],
                i - 1,
                preds[i - 1]
            );
        }
    }

    #[test]
    fn test_isotonic_multi_feature_error() {
        let x = Array2::from_shape_vec((5, 2), vec![0.0; 10]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!(IsotonicRegressor::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_predict_interpolates_like_sklearn() {
        // sklearn 1.8 IsotonicRegression().fit(x, y).predict(q):
        //   [2.5, 4.0625, 7.75, nan]; out_of_bounds='clip' -> last = 10.0.
        let x = Array2::from_shape_fn((12, 1), |(i, _)| i as f64);
        let y = array![1.0, 3.0, 2.0, 4.0, 3.5, 5.0, 6.5, 6.0, 7.0, 9.0, 8.0, 10.0];
        let q = array![[1.5], [4.25], [8.5], [20.0]];
        let p = IsotonicRegressor::new()
            .fit(&x, &y)
            .unwrap()
            .predict(&q)
            .unwrap();
        for (a, b) in p.iter().take(3).zip([2.5, 4.0625, 7.75]) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        assert!(p[3].is_nan());
        let p = IsotonicRegressor::new()
            .with_out_of_bounds(OutOfBounds::Clip)
            .fit(&x, &y)
            .unwrap()
            .predict(&q)
            .unwrap();
        assert_eq!(p[3], 10.0);
        assert!(IsotonicRegressor::new()
            .with_out_of_bounds(OutOfBounds::Raise)
            .fit(&x, &y)
            .unwrap()
            .predict(&q)
            .is_err());
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedIsotonicRegressor {}
