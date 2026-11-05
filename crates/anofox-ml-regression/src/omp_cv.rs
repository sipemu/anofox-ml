//! Orthogonal matching pursuit with the number of non-zero coefficients
//! chosen by cross-validation (sklearn's `OrthogonalMatchingPursuitCV`):
//! unshuffled `cv`-fold, mean squared error along the path `1..max_iter`,
//! refit at the best size.

use anofox_ml_core::{k_fold, Fit, Predict, Result, RustMlError};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::{FittedOrthogonalMatchingPursuit, OrthogonalMatchingPursuit};

/// OMP with cross-validated sparsity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OrthogonalMatchingPursuitCV {
    /// Longest path (`None`: `min(max(⌊0.1 p⌋, 5), p)`).
    pub max_iter: Option<usize>,
    pub cv: usize,
    pub fit_intercept: bool,
}

impl Default for OrthogonalMatchingPursuitCV {
    fn default() -> Self {
        Self {
            max_iter: None,
            cv: 5,
            fit_intercept: true,
        }
    }
}

/// Fitted [`OrthogonalMatchingPursuitCV`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedOrthogonalMatchingPursuitCV {
    model: FittedOrthogonalMatchingPursuit,
    n_nonzero_coefs: usize,
    /// Mean squared error per path length (averaged over the folds).
    mse_path: Vec<f64>,
}

impl FittedOrthogonalMatchingPursuitCV {
    pub fn model(&self) -> &FittedOrthogonalMatchingPursuit {
        &self.model
    }

    /// The chosen sparsity (sklearn's `n_nonzero_coefs_`).
    pub fn n_nonzero_coefs(&self) -> usize {
        self.n_nonzero_coefs
    }

    pub fn mse_path(&self) -> &[f64] {
        &self.mse_path
    }
}

impl Fit<f64> for OrthogonalMatchingPursuitCV {
    type Fitted = FittedOrthogonalMatchingPursuitCV;

    fn fit_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<FittedOrthogonalMatchingPursuitCV> {
        let (n, p) = x.dim();
        if y.len() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "x has {n} rows but y has {}",
                y.len()
            )));
        }
        let max_iter = self
            .max_iter
            .unwrap_or_else(|| ((p as f64 * 0.1) as usize).max(5).min(p))
            .min(p);
        if max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "max_iter must be at least 1".into(),
            ));
        }
        let omp = |k: usize| {
            OrthogonalMatchingPursuit::new()
                .with_n_nonzero_coefs(k)
                .with_fit_intercept(self.fit_intercept)
        };
        let mut folds: Vec<Vec<f64>> = Vec::new();
        for (train, test) in k_fold(n, self.cv)? {
            let (xt, yt) = (x.select(Axis(0), &train), y.select(Axis(0), &train));
            let (xv, yv) = (x.select(Axis(0), &test), y.select(Axis(0), &test));
            let mut path = Vec::new();
            for k in 1..=max_iter.min(train.len()) {
                // The path stops early when the next atom is linearly
                // dependent, as sklearn's does.
                let Ok(m) = omp(k).fit(&xt, &yt) else { break };
                let r = m.predict(&xv)? - &yv;
                path.push(r.mapv(|v| v * v).mean().expect("non-empty fold"));
            }
            folds.push(path);
        }
        let len = folds.iter().map(Vec::len).min().unwrap_or(0);
        if len == 0 {
            return Err(RustMlError::InvalidParameter(
                "no OMP path could be computed".into(),
            ));
        }
        let mse_path: Vec<f64> = (0..len)
            .map(|k| folds.iter().map(|f| f[k]).sum::<f64>() / folds.len() as f64)
            .collect();
        let best = mse_path
            .iter()
            .enumerate()
            .fold(0, |b, (k, &v)| if v < mse_path[b] { k } else { b });
        let model = omp(best + 1).fit(&x, &y)?;
        Ok(FittedOrthogonalMatchingPursuitCV {
            model,
            n_nonzero_coefs: best + 1,
            mse_path,
        })
    }
}

impl Predict<f64> for FittedOrthogonalMatchingPursuitCV {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.model.predict_view(x)
    }
}
