//! TF-IDF reweighting of a count matrix (sklearn's `TfidfTransformer`).

use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Row normalisation of a [`TfidfTransformer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TfidfNorm {
    #[default]
    L2,
    L1,
    None,
}

/// Turns term counts (rows = documents) into TF-IDF weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TfidfTransformer {
    pub norm: TfidfNorm,
    pub use_idf: bool,
    /// `idf = ln((1 + n) / (1 + df)) + 1` instead of `ln(n / df) + 1`.
    pub smooth_idf: bool,
    /// `tf = 1 + ln(tf)` for non-zero counts.
    pub sublinear_tf: bool,
}

impl Default for TfidfTransformer {
    fn default() -> Self {
        Self {
            norm: TfidfNorm::L2,
            use_idf: true,
            smooth_idf: true,
            sublinear_tf: false,
        }
    }
}

impl TfidfTransformer {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted [`TfidfTransformer`]: the IDF weights (when `use_idf`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedTfidfTransformer {
    config: TfidfTransformer,
    idf: Option<Array1<f64>>,
    n_features: usize,
}

impl FittedTfidfTransformer {
    pub fn idf(&self) -> Option<&Array1<f64>> {
        self.idf.as_ref()
    }
}

impl FitUnsupervised<f64> for TfidfTransformer {
    type Fitted = FittedTfidfTransformer;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedTfidfTransformer> {
        let (n, d) = x.dim();
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("x has no rows or columns".into()));
        }
        if x.iter().any(|&v| !v.is_finite() || v < 0.0) {
            return Err(RustMlError::InvalidParameter(
                "counts must be finite and non-negative".into(),
            ));
        }
        let idf = self.use_idf.then(|| {
            let s = if self.smooth_idf { 1.0 } else { 0.0 };
            Array1::from_shape_fn(d, |j| {
                let df = x.column(j).iter().filter(|&&v| v != 0.0).count() as f64;
                ((n as f64 + s) / (df + s)).ln() + 1.0
            })
        });
        Ok(FittedTfidfTransformer {
            config: self.clone(),
            idf,
            n_features: d,
        })
    }
}

impl Transform<f64> for FittedTfidfTransformer {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let mut out = x.to_owned();
        if self.config.sublinear_tf {
            out.mapv_inplace(|v| if v != 0.0 { v.ln() + 1.0 } else { 0.0 });
        }
        if let Some(idf) = &self.idf {
            out *= idf;
        }
        for mut row in out.rows_mut() {
            let norm = match self.config.norm {
                TfidfNorm::L2 => row.dot(&row).sqrt(),
                TfidfNorm::L1 => row.iter().map(|v| v.abs()).sum(),
                TfidfNorm::None => 1.0,
            };
            if norm != 0.0 {
                row /= norm;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn smooth_idf_and_l2_rows() {
        let x = array![
            [3.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 2.0, 0.0],
            [3.0, 0.0, 2.0]
        ];
        let f = TfidfTransformer::new().fit(&x).unwrap();
        // sklearn's documented example: idf = [1, 2.25276297, 1.84729786].
        let idf = f.idf().unwrap();
        assert!((idf[1] - 2.252_762_968_495_368).abs() < 1e-12);
        assert!((idf[2] - 1.847_297_860_387_203_6).abs() < 1e-12);
        let t = f.transform(&x).unwrap();
        assert!((t[[0, 0]] - 0.851_513_347_210_46).abs() < 1e-12);
        for r in t.rows() {
            assert!((r.dot(&r) - 1.0).abs() < 1e-12);
        }
    }
}
