//! sklearn's unsupervised `NearestNeighbors` (an index answering
//! `kneighbors`) and `RadiusNeighborsTransformer` (the radius graph as a
//! transformer, returned dense like
//! [`KNeighborsTransformer`](crate::KNeighborsTransformer)).

use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::distance::DistanceMetric;
use crate::index::{NeighborAlgorithm, NeighborIndex};
use crate::kneighbors_transformer::GraphMode;

/// sklearn's `NearestNeighbors`: the training rows indexed for neighbour
/// queries. `n_neighbors` is the default `k` of
/// [`kneighbors`](FittedNearestNeighbors::kneighbors); ties in distance go
/// to the lower training index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NearestNeighbors {
    pub n_neighbors: usize,
    pub metric: DistanceMetric,
    pub algorithm: NeighborAlgorithm,
}

impl Default for NearestNeighbors {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            metric: DistanceMetric::Euclidean,
            algorithm: NeighborAlgorithm::Auto,
        }
    }
}

/// A fitted [`NearestNeighbors`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedNearestNeighbors {
    index: NeighborIndex<f64>,
    n_neighbors: usize,
}

fn check_x(x: ArrayView2<f64>) -> Result<()> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(RustMlError::EmptyInput("x has no rows or columns".into()));
    }
    Ok(())
}

impl FitUnsupervised<f64> for NearestNeighbors {
    type Fitted = FittedNearestNeighbors;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedNearestNeighbors> {
        check_x(x)?;
        if self.n_neighbors == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_neighbors must be at least 1".into(),
            ));
        }
        Ok(FittedNearestNeighbors {
            index: NeighborIndex::new(x, self.metric, self.algorithm, Some(self.n_neighbors))?,
            n_neighbors: self.n_neighbors,
        })
    }
}

impl FittedNearestNeighbors {
    /// Distances and training indices of the `k` (default `n_neighbors`)
    /// nearest training rows of every row of `x`, nearest first.
    pub fn kneighbors(
        &self,
        x: ArrayView2<f64>,
        k: Option<usize>,
    ) -> Result<(Array2<f64>, Array2<usize>)> {
        self.index.kneighbors(x, k.unwrap_or(self.n_neighbors))
    }

    pub fn n_samples(&self) -> usize {
        self.index.n_samples()
    }
}

/// sklearn's `RadiusNeighborsTransformer`: row `i` holds the distances
/// (`Distance`) or ones (`Connectivity`) to the training rows within
/// `radius` (inclusive) of row `i`, zeros elsewhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RadiusNeighborsTransformer {
    pub mode: GraphMode,
    pub radius: f64,
    pub metric: DistanceMetric,
    pub algorithm: NeighborAlgorithm,
}

impl Default for RadiusNeighborsTransformer {
    fn default() -> Self {
        Self {
            mode: GraphMode::Distance,
            radius: 1.0,
            metric: DistanceMetric::Euclidean,
            algorithm: NeighborAlgorithm::Auto,
        }
    }
}

/// A fitted [`RadiusNeighborsTransformer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedRadiusNeighborsTransformer {
    index: NeighborIndex<f64>,
    mode: GraphMode,
    radius: f64,
}

impl FitUnsupervised<f64> for RadiusNeighborsTransformer {
    type Fitted = FittedRadiusNeighborsTransformer;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedRadiusNeighborsTransformer> {
        check_x(x)?;
        if !(self.radius >= 0.0 && self.radius.is_finite()) {
            return Err(RustMlError::InvalidParameter(format!(
                "radius must be finite and >= 0, got {}",
                self.radius
            )));
        }
        Ok(FittedRadiusNeighborsTransformer {
            index: NeighborIndex::new(x, self.metric, self.algorithm, None)?,
            mode: self.mode,
            radius: self.radius,
        })
    }
}

impl Transform<f64> for FittedRadiusNeighborsTransformer {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let rows = self.index.radius_neighbors(x, self.radius)?;
        let mut out = Array2::zeros((x.nrows(), self.index.n_samples()));
        for (i, row) in rows.iter().enumerate() {
            for &(j, d) in row {
                out[[i, j]] = match self.mode {
                    GraphMode::Distance => d,
                    GraphMode::Connectivity => 1.0,
                };
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
    fn radius_graph_and_kneighbors() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]];
        let t = RadiusNeighborsTransformer {
            radius: 1.0,
            ..Default::default()
        }
        .fit(&x)
        .unwrap();
        // Row 0: itself (0) and row 1 at exactly the radius (inclusive).
        assert_eq!(
            t.transform(&x).unwrap().row(0).to_vec(),
            vec![0.0, 1.0, 0.0]
        );
        let c = RadiusNeighborsTransformer {
            radius: 1.0,
            mode: GraphMode::Connectivity,
            ..Default::default()
        }
        .fit(&x)
        .unwrap();
        assert_eq!(
            c.transform(&x).unwrap().row(0).to_vec(),
            vec![1.0, 1.0, 0.0]
        );
        let nn = NearestNeighbors {
            n_neighbors: 2,
            ..Default::default()
        }
        .fit(&x)
        .unwrap();
        let (d, i) = nn.kneighbors(array![[2.9, 0.0]].view(), None).unwrap();
        assert_eq!(i.row(0).to_vec(), vec![2, 1]);
        assert!((d[[0, 0]] - 0.1).abs() < 1e-12);
    }
}
