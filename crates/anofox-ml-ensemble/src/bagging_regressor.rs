use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use anofox_ml_trees::{DecisionTreeRegressor, FittedDecisionTreeRegressor};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Bagging (Bootstrap Aggregating) regressor parameters (unfitted state).
///
/// Trains an ensemble of decision tree regressors, each on a bootstrap sample
/// of the data using the **full** feature set. Unlike `RandomForestRegressor`,
/// bagging does not perform random feature subsampling at the tree level --
/// every tree sees all features.
///
/// Predictions are the average of individual tree predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaggingRegressor {
    /// Number of trees in the ensemble.
    pub n_estimators: usize,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Fraction of samples to draw for each tree (with replacement when
    /// `bootstrap=true`). If `None`, draws `n_samples`. Value in (0, 1].
    pub max_samples: Option<f64>,
    /// Whether to use bootstrap sampling. Default: true.
    pub bootstrap: bool,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl BaggingRegressor {
    /// Create a new `BaggingRegressor` with the given number of trees and default parameters.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            max_depth: None,
            max_samples: None,
            bootstrap: true,
            seed: 0,
        }
    }

    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }
    pub fn with_max_samples(mut self, max_samples: Option<f64>) -> Self {
        self.max_samples = max_samples;
        self
    }
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for BaggingRegressor {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Fitted bagging regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedBaggingRegressor<F: Float> {
    trees: Vec<FittedDecisionTreeRegressor<F>>,
    n_features: usize,
}

impl<F: Float> Fit<F> for BaggingRegressor {
    type Fitted = FittedBaggingRegressor<F>;

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
        if self.n_estimators == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_estimators must be > 0".into(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Compute bootstrap sample size
        let draw_size = if let Some(frac) = self.max_samples {
            if frac <= 0.0 || frac > 1.0 {
                return Err(RustMlError::InvalidParameter(
                    "max_samples must be in (0, 1]".into(),
                ));
            }
            (n_samples as f64 * frac).ceil() as usize
        } else {
            n_samples
        };

        let tree_params = DecisionTreeRegressor {
            max_depth: self.max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            sample_weight: None,
        };

        // Pre-generate bootstrap row indices for determinism
        let sample_plans: Vec<Vec<usize>> = (0..self.n_estimators)
            .map(|_| {
                if self.bootstrap {
                    (0..draw_size)
                        .map(|_| rng.gen_range(0..n_samples))
                        .collect()
                } else {
                    (0..n_samples).collect()
                }
            })
            .collect();

        // Train trees in parallel -- no feature subsampling
        let trees: Result<Vec<FittedDecisionTreeRegressor<F>>> = sample_plans
            .into_par_iter()
            .map(|row_indices| {
                let x_sub = build_sub_matrix_rows(x, &row_indices);
                let y_sub = Array1::from_vec(row_indices.iter().map(|&i| y[i]).collect::<Vec<F>>());
                tree_params.fit(&x_sub, &y_sub)
            })
            .collect();
        let trees = trees?;

        Ok(FittedBaggingRegressor { trees, n_features })
    }
}

impl<F: Float> Predict<F> for FittedBaggingRegressor<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let n_trees_f = F::from_usize(self.trees.len()).unwrap();

        // Collect all tree predictions in parallel
        let all_preds: Result<Vec<Array1<F>>> =
            self.trees.par_iter().map(|tree| tree.predict(x)).collect();
        let all_preds = all_preds?;

        // Average predictions across trees
        let mut predictions = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut sum = F::zero();
            for tree_pred in &all_preds {
                sum += tree_pred[i];
            }
            predictions.push(sum / n_trees_f);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl<F: Float> FittedBaggingRegressor<F> {
    /// Feature importances averaged across all trees and normalized to sum to 1.
    pub fn feature_importances(&self) -> Array1<F> {
        let mut importances = vec![F::zero(); self.n_features];
        let n_trees = F::from_usize(self.trees.len()).unwrap();

        for tree in &self.trees {
            let tree_importances = tree.feature_importances();
            for (idx, &imp) in tree_importances.iter().enumerate() {
                importances[idx] += imp / n_trees;
            }
        }

        // Normalize so importances sum to 1
        let sum: F = importances.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            Array1::from_vec(importances.into_iter().map(|v| v / sum).collect())
        } else {
            Array1::zeros(self.n_features)
        }
    }

    /// Number of trees in the ensemble.
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Compute R-squared score on the given data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<f64> {
        let preds = self.predict(x)?;
        let n = y.len();
        let y_mean = y.iter().copied().fold(F::zero(), |a, b| a + b) / F::from_usize(n).unwrap();
        let ss_res: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).to_f64().unwrap().powi(2))
            .sum();
        let ss_tot: f64 = y
            .iter()
            .map(|&t| (t - y_mean).to_f64().unwrap().powi(2))
            .sum();
        Ok(if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        })
    }
}

/// Build a sub-matrix selecting specific rows (all columns) from `x`.
fn build_sub_matrix_rows<F: Float>(x: &Array2<F>, row_indices: &[usize]) -> Array2<F> {
    let n_rows = row_indices.len();
    let n_cols = x.ncols();
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for &ri in row_indices {
        for ci in 0..n_cols {
            data.push(x[[ri, ci]]);
        }
    }
    Array2::from_shape_vec((n_rows, n_cols), data).expect("shape matches data length")
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_basic_regression() {
        // y = 2*x, ensemble should learn a good approximation
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let br = BaggingRegressor::new(50).with_seed(42);
        let fitted: FittedBaggingRegressor<f64> = br.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 2.0);
        }
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let br = BaggingRegressor::new(10).with_seed(123);

        let fitted1: FittedBaggingRegressor<f64> = br.fit(&x, &y).unwrap();
        let fitted2: FittedBaggingRegressor<f64> = br.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_feature_importances_sum_to_one() {
        let x = array![
            [1.0, 100.0],
            [2.0, 200.0],
            [3.0, 300.0],
            [4.0, 400.0],
            [5.0, 500.0],
            [6.0, 600.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let br = BaggingRegressor::new(20).with_seed(7);
        let fitted: FittedBaggingRegressor<f64> = br.fit(&x, &y).unwrap();

        let importances = fitted.feature_importances();
        let sum: f64 = importances.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_score() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let br = BaggingRegressor::new(50).with_seed(42);
        let fitted: FittedBaggingRegressor<f64> = br.fit(&x, &y).unwrap();

        let r2 = fitted.score(&x, &y).unwrap();
        // R-squared on training data should be high
        assert!(r2 > 0.8, "R2={r2} is too low");
    }

    #[test]
    fn test_n_estimators() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let br = BaggingRegressor::new(7).with_seed(0);
        let fitted: FittedBaggingRegressor<f64> = br.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 7);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let br = BaggingRegressor::default();
        let result: std::result::Result<FittedBaggingRegressor<f64>, _> = br.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let br = BaggingRegressor::new(5).with_seed(0);
        let fitted: FittedBaggingRegressor<f64> = br.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let br = BaggingRegressor::default();
        let result: std::result::Result<FittedBaggingRegressor<f64>, _> = br.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_estimators_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let br = BaggingRegressor::new(0);
        let result: std::result::Result<FittedBaggingRegressor<f64>, _> = br.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_samples() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let br = BaggingRegressor::new(30)
            .with_max_samples(Some(0.5))
            .with_seed(42);
        let fitted: FittedBaggingRegressor<f64> = br.fit(&x, &y).unwrap();

        // Should still produce valid predictions
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
    }

    #[test]
    fn test_default() {
        let br = BaggingRegressor::default();
        assert_eq!(br.n_estimators, 10);
        assert!(br.bootstrap);
        assert!(br.max_depth.is_none());
        assert!(br.max_samples.is_none());
        assert_eq!(br.seed, 0);
    }
}
