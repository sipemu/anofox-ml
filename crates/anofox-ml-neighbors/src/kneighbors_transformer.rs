//! k-nearest-neighbour graphs as a transformer (sklearn's
//! `KNeighborsTransformer`), returned dense: row `i` holds the
//! euclidean distances (`Distance`) or ones (`Connectivity`) to its
//! nearest training rows and zeros elsewhere.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Entries of a [`KNeighborsTransformer`] graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GraphMode {
    /// Distances to `n_neighbors + 1` neighbours (sklearn adds one so that
    /// a training row's own zero distance does not use up a neighbour).
    #[default]
    Distance,
    /// Ones for the `n_neighbors` nearest rows.
    Connectivity,
}

/// k-nearest-neighbour graph transformer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KNeighborsTransformer {
    pub mode: GraphMode,
    pub n_neighbors: usize,
}

impl Default for KNeighborsTransformer {
    fn default() -> Self {
        Self {
            mode: GraphMode::Distance,
            n_neighbors: 5,
        }
    }
}

/// Fitted [`KNeighborsTransformer`]: the training rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedKNeighborsTransformer {
    mode: GraphMode,
    n_neighbors: usize,
    x: Array2<f64>,
}

impl FitUnsupervised<f64> for KNeighborsTransformer {
    type Fitted = FittedKNeighborsTransformer;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedKNeighborsTransformer> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(RustMlError::EmptyInput("x has no rows or columns".into()));
        }
        let k = self.n_neighbors + (self.mode == GraphMode::Distance) as usize;
        if self.n_neighbors == 0 || k > x.nrows() {
            return Err(RustMlError::InvalidParameter(format!(
                "need 1 <= n_neighbors (+1 in distance mode) <= {} rows",
                x.nrows()
            )));
        }
        Ok(FittedKNeighborsTransformer {
            mode: self.mode,
            n_neighbors: self.n_neighbors,
            x: x.to_owned(),
        })
    }
}

impl Transform<f64> for FittedKNeighborsTransformer {
    fn transform_view(&self, q: ArrayView2<f64>) -> Result<Array2<f64>> {
        if q.ncols() != self.x.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x.ncols(),
                q.ncols()
            )));
        }
        let k = self.n_neighbors + (self.mode == GraphMode::Distance) as usize;
        let mut out = Array2::zeros((q.nrows(), self.x.nrows()));
        for (i, r) in q.rows().into_iter().enumerate() {
            let mut d: Vec<(f64, usize)> = self
                .x
                .rows()
                .into_iter()
                .enumerate()
                .map(|(j, t)| {
                    (
                        r.iter()
                            .zip(t)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt(),
                        j,
                    )
                })
                .collect();
            d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for &(dist, j) in d.iter().take(k) {
                out[[i, j]] = match self.mode {
                    GraphMode::Distance => dist,
                    GraphMode::Connectivity => 1.0,
                };
            }
        }
        Ok(out)
    }
}
