//! sklearn's `LabelEncoder` and `OrdinalEncoder` over numeric categories
//! (the Python package codes other labels before they get here).

use anofox_ml_core::{Result, RustMlError};
use anofox_ml_preprocessing as pre;
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// The columns of `x`, as the encoders take them.
pub(crate) fn columns(x: ArrayView2<f64>) -> Vec<Vec<f64>> {
    x.columns().into_iter().map(|c| c.to_vec()).collect()
}

/// Matrix from per-column values (`None` as NaN).
pub(crate) fn from_columns(cols: Vec<Vec<Option<f64>>>, n_rows: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_rows, cols.len()), |(i, j)| {
        cols[j][i].unwrap_or(f64::NAN)
    })
}

fn single(x: ArrayView2<f64>) -> Result<Vec<f64>> {
    if x.ncols() != 1 {
        return Err(RustMlError::ShapeMismatch(format!(
            "LabelEncoder takes the labels as one column, got {}",
            x.ncols()
        )));
    }
    Ok(x.column(0).to_vec())
}

/// `LabelEncoder` (no parameters); the labels are the one column of `x`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct LabelEncoderConfig {}

impl LabelEncoderConfig {
    pub fn fit(&self, x: ArrayView2<f64>) -> Result<pre::FittedLabelEncoder<f64>> {
        pre::LabelEncoder::new().fit(&single(x)?)
    }
}

pub(crate) fn label_transform(
    m: &pre::FittedLabelEncoder<f64>,
    x: ArrayView2<f64>,
) -> Result<Array1<f64>> {
    Ok(m.transform(&single(x)?)?.mapv(|c| c as f64))
}

pub(crate) fn label_inverse(
    m: &pre::FittedLabelEncoder<f64>,
    x: ArrayView2<f64>,
) -> Result<Array1<f64>> {
    let codes = single(x)?
        .into_iter()
        .map(|c| {
            if c >= 0.0 && c.fract() == 0.0 {
                Ok(c as usize)
            } else {
                Err(RustMlError::InvalidParameter(format!(
                    "y contains previously unseen labels: {c}"
                )))
            }
        })
        .collect::<Result<Array1<usize>>>()?;
    Ok(Array1::from_vec(m.inverse_transform(&codes)?))
}
