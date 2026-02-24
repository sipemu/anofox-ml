use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::distance::{compute_distance, DistanceMetric};

/// Weighting strategy for KNN.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum WeightFunction {
    /// All neighbors contribute equally.
    #[default]
    Uniform,
    /// Closer neighbors have more influence (weight = 1/distance).
    Distance,
}

/// KNN classifier parameters (unfitted state).
#[derive(Debug, Clone)]
pub struct KnnClassifier {
    pub n_neighbors: usize,
    pub metric: DistanceMetric,
    pub weights: WeightFunction,
}

impl Default for KnnClassifier {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            metric: DistanceMetric::default(),
            weights: WeightFunction::default(),
        }
    }
}

/// Fitted KNN classifier — stores training data (lazy learner).
#[derive(Debug, Clone)]
pub struct FittedKnnClassifier<F: Float> {
    x_train: Array2<F>,
    y_train: Array1<F>,
    n_neighbors: usize,
    metric: DistanceMetric,
    weights: WeightFunction,
}

impl<F: Float> Fit<F> for KnnClassifier {
    type Fitted = FittedKnnClassifier<F>;

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

        Ok(FittedKnnClassifier {
            x_train: x.to_owned(),
            y_train: y.to_owned(),
            n_neighbors: self.n_neighbors,
            metric: self.metric,
            weights: self.weights,
        })
    }
}

impl<F: Float> Predict<F> for FittedKnnClassifier<F> {
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
            // Compute distances to all training points
            let mut distances: Vec<(F, F)> = self
                .x_train
                .rows()
                .into_iter()
                .zip(self.y_train.iter())
                .map(|(train_row, &label)| {
                    let dist = compute_distance(&row, &train_row, self.metric);
                    (dist, label)
                })
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Take k nearest
            let neighbors = &distances[..self.n_neighbors];

            // Weighted majority vote
            predictions[i] = weighted_majority_vote(neighbors, self.weights);
        }

        Ok(predictions)
    }
}

/// Weighted majority vote among neighbors.
fn weighted_majority_vote<F: Float>(neighbors: &[(F, F)], weights: WeightFunction) -> F {
    // Collect unique classes
    let mut classes: Vec<F> = neighbors.iter().map(|&(_, c)| c).collect();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    classes.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-9).unwrap());

    let mut best_class = classes[0];
    let mut best_weight = F::neg_infinity();

    for &class in &classes {
        let weight: F = neighbors
            .iter()
            .filter(|&&(_, c)| (c - class).abs() < F::from_f64(1e-9).unwrap())
            .map(|&(dist, _)| match weights {
                WeightFunction::Uniform => F::one(),
                WeightFunction::Distance => {
                    if dist < F::from_f64(1e-15).unwrap() {
                        F::from_f64(1e15).unwrap()
                    } else {
                        F::one() / dist
                    }
                }
            })
            .fold(F::zero(), |a, b| a + b);

        if weight > best_weight {
            best_weight = weight;
            best_class = class;
        }
    }

    best_class
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_knn_simple() {
        // Two clear clusters
        let x_train = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let knn = KnnClassifier {
            n_neighbors: 3,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test = array![[0.05, 0.05], [10.05, 10.05]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_distance_weights() {
        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 0.0, 1.0];

        let knn = KnnClassifier {
            n_neighbors: 3,
            weights: WeightFunction::Distance,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // Point at 1.9 is closest to class 1 (distance 0.1) vs class 0 (distances 1.9 and 0.9)
        let x_test = array![[1.9]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0]];
        let y = array![0.0, 1.0]; // mismatched length
        let knn = KnnClassifier::default();
        assert!(Fit::fit(&knn, &x, &y).is_err());
    }
}
