use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::distance::{compute_distance, DistanceMetric};
use crate::knn_classifier::WeightFunction;

/// KNN regressor parameters (unfitted state).
#[derive(Debug, Clone)]
pub struct KnnRegressor {
    pub n_neighbors: usize,
    pub metric: DistanceMetric,
    pub weights: WeightFunction,
}

impl Default for KnnRegressor {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            metric: DistanceMetric::default(),
            weights: WeightFunction::default(),
        }
    }
}

/// Fitted KNN regressor — stores training data (lazy learner).
#[derive(Debug, Clone)]
pub struct FittedKnnRegressor<F: Float> {
    x_train: Array2<F>,
    y_train: Array1<F>,
    n_neighbors: usize,
    metric: DistanceMetric,
    weights: WeightFunction,
}

impl<F: Float> Fit<F> for KnnRegressor {
    type Fitted = FittedKnnRegressor<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
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
        if self.n_neighbors == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_neighbors must be > 0".into(),
            ));
        }
        if self.n_neighbors > x.nrows() {
            return Err(RustMlError::InvalidParameter(format!(
                "n_neighbors ({}) > number of training samples ({})",
                self.n_neighbors,
                x.nrows()
            )));
        }

        Ok(FittedKnnRegressor {
            x_train: x.to_owned(),
            y_train: y.to_owned(),
            n_neighbors: self.n_neighbors,
            metric: self.metric,
            weights: self.weights,
        })
    }
}

impl<F: Float> Predict<F> for FittedKnnRegressor<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.x_train.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }

        let mut predictions = Array1::<F>::zeros(x.nrows());

        for (i, row) in x.rows().into_iter().enumerate() {
            let mut distances: Vec<(F, F)> = self
                .x_train
                .rows()
                .into_iter()
                .zip(self.y_train.iter())
                .map(|(train_row, &target)| {
                    let dist = compute_distance(&row, &train_row, self.metric);
                    (dist, target)
                })
                .collect();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let neighbors = &distances[..self.n_neighbors];

            predictions[i] = weighted_mean(neighbors, self.weights);
        }

        Ok(predictions)
    }
}

/// Weighted mean of neighbor targets.
fn weighted_mean<F: Float>(neighbors: &[(F, F)], weights: WeightFunction) -> F {
    match weights {
        WeightFunction::Uniform => {
            let sum: F = neighbors.iter().map(|&(_, y)| y).fold(F::zero(), |a, b| a + b);
            sum / F::from_usize(neighbors.len()).unwrap()
        }
        WeightFunction::Distance => {
            let mut weight_sum = F::zero();
            let mut value_sum = F::zero();

            for &(dist, y) in neighbors {
                let w = if dist < F::from_f64(1e-15).unwrap() {
                    F::from_f64(1e15).unwrap()
                } else {
                    F::one() / dist
                };
                weight_sum += w;
                value_sum += w * y;
            }

            value_sum / weight_sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_knn_regressor_simple() {
        let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y_train = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let knn = KnnRegressor {
            n_neighbors: 3,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test = array![[3.0]];
        let preds = fitted.predict(&x_test).unwrap();
        // Nearest to 3.0 are 2.0, 3.0, 4.0 with targets 4.0, 6.0, 8.0
        assert_abs_diff_eq!(preds[0], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_regressor_distance_weights() {
        let x_train = array![[1.0], [3.0], [5.0]];
        let y_train = array![10.0, 20.0, 30.0];

        let knn = KnnRegressor {
            n_neighbors: 3,
            weights: WeightFunction::Distance,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // Point at 2.9: nearest is 3.0 (dist=0.1), then 1.0 (dist=1.9), then 5.0 (dist=2.1)
        let x_test = array![[2.9]];
        let preds = fitted.predict(&x_test).unwrap();
        // Should be heavily weighted towards 20.0
        assert!(preds[0] > 18.0 && preds[0] < 22.0);
    }
}
