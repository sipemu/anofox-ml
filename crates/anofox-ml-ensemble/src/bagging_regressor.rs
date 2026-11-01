use anofox_ml_core::{Fit, FitWeighted, Float, Predict, Result, RustMlError};
use anofox_ml_trees::{DecisionTreeRegressor, FittedDecisionTreeRegressor, MaxFeatures};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::bagging_common::{add_importances, out_of_bag, select_columns, BagSampling};

/// Bagging (Bootstrap Aggregating) regressor parameters (unfitted state).
///
/// Trains an ensemble of decision tree regressors (sklearn's
/// `BaggingRegressor` with its default tree estimator), each on a bootstrap
/// sample of the rows and, with `max_features`, on a random subset of the
/// features drawn once per estimator (`bootstrap_features` draws it with
/// replacement). Unlike a random forest, the features are not re-drawn at
/// every split.
///
/// Predictions are the average of individual tree predictions. With
/// `oob_score`, each training sample is also predicted by the trees that
/// did not see it ([`FittedBaggingRegressor::oob_prediction`],
/// [`FittedBaggingRegressor::oob_score`]).
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
    /// Features drawn for each tree (`None` = all).
    #[serde(default)]
    pub max_features: Option<MaxFeatures>,
    /// Draw each tree's features with replacement.
    #[serde(default)]
    pub bootstrap_features: bool,
    /// Compute the out-of-bag prediction and R² (needs `bootstrap`).
    #[serde(default)]
    pub oob_score: bool,
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
            max_features: None,
            bootstrap_features: false,
            oob_score: false,
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
    /// Features drawn per tree, with replacement if `bootstrap_features`.
    pub fn with_max_features(
        mut self,
        max_features: Option<MaxFeatures>,
        bootstrap_features: bool,
    ) -> Self {
        self.max_features = max_features;
        self.bootstrap_features = bootstrap_features;
        self
    }
    /// Compute the out-of-bag prediction and score.
    pub fn with_oob_score(mut self, oob_score: bool) -> Self {
        self.oob_score = oob_score;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn sampling(&self) -> BagSampling {
        BagSampling {
            n_estimators: self.n_estimators,
            max_samples: self.max_samples,
            bootstrap: self.bootstrap,
            max_features: self.max_features,
            bootstrap_features: self.bootstrap_features,
            oob_score: self.oob_score,
            seed: self.seed,
        }
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
    /// Feature subset of each tree (empty = all features).
    #[serde(default)]
    features: Vec<Vec<usize>>,
    n_features: usize,
    #[serde(default)]
    oob_prediction: Option<Array1<F>>,
    #[serde(default)]
    oob_score: Option<f64>,
}

impl<F: Float> Fit<F> for BaggingRegressor {
    type Fitted = FittedBaggingRegressor<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.fit_impl(x, y, None)
    }
}

impl<F: Float> FitWeighted<F> for BaggingRegressor {
    type Fitted = FittedBaggingRegressor<F>;

    /// Per-sample weights (sklearn's `sample_weight`), applied on top of
    /// the bootstrap counts.
    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        self.fit_impl(x, y, sample_weight)
    }
}

impl BaggingRegressor {
    fn fit_impl<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedBaggingRegressor<F>> {
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
        if let Some(w) = sample_weight {
            anofox_ml_core::check_sample_weight(w, x.nrows())?;
        }
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let plans = self.sampling().plans(n_samples, n_features)?;

        let tree_params = DecisionTreeRegressor {
            max_depth: self.max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            sample_weight: None,
            seed: 0,
            ..DecisionTreeRegressor::new()
        };

        let trees: Vec<FittedDecisionTreeRegressor<F>> = plans
            .par_iter()
            .map(|plan| {
                let xs = select_columns(x, &plan.features);
                crate::forest_common::fit_tree_on_rows(
                    &tree_params.clone().with_seed(plan.seed),
                    &xs,
                    y,
                    &plan.rows,
                    sample_weight,
                )
            })
            .collect::<Result<_>>()?;
        let features: Vec<Vec<usize>> = plans.iter().map(|p| p.features.clone()).collect();

        let (oob_prediction, oob_score) = if self.oob_score {
            let mut sum = Array1::<F>::zeros(n_samples);
            let mut count = vec![0usize; n_samples];
            for (plan, tree) in plans.iter().zip(&trees) {
                let oob = out_of_bag(n_samples, &plan.rows);
                if oob.is_empty() {
                    continue;
                }
                let xs = select_columns(x, &plan.features).select(ndarray::Axis(0), &oob);
                for (&i, p) in oob.iter().zip(tree.predict(&xs)?) {
                    sum[i] += p;
                    count[i] += 1;
                }
            }
            let pred = Array1::from_shape_fn(n_samples, |i| {
                if count[i] > 0 {
                    sum[i] / F::from_usize(count[i]).unwrap()
                } else {
                    F::nan()
                }
            });
            let score = r2_where_defined(y, &pred);
            (Some(pred), Some(score))
        } else {
            (None, None)
        };

        Ok(FittedBaggingRegressor {
            trees,
            features,
            n_features,
            oob_prediction,
            oob_score,
        })
    }
}

/// R² over the samples whose prediction is not NaN.
fn r2_where_defined<F: Float>(y: &Array1<F>, pred: &Array1<F>) -> f64 {
    let pairs: Vec<(f64, f64)> = y
        .iter()
        .zip(pred)
        .filter(|(_, p)| !p.is_nan())
        .map(|(t, p)| (t.to_f64().unwrap(), p.to_f64().unwrap()))
        .collect();
    if pairs.is_empty() {
        return f64::NAN;
    }
    let mean = pairs.iter().map(|p| p.0).sum::<f64>() / pairs.len() as f64;
    let ss_res: f64 = pairs.iter().map(|(t, p)| (t - p).powi(2)).sum();
    let ss_tot: f64 = pairs.iter().map(|(t, _)| (t - mean).powi(2)).sum();
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
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

        let n_trees_f = F::from_usize(self.trees.len()).unwrap();
        let all_preds: Vec<Array1<F>> = self
            .trees
            .par_iter()
            .enumerate()
            .map(|(t, tree)| tree.predict(&select_columns(x, self.tree_features(t))))
            .collect::<Result<_>>()?;

        let mut predictions = Array1::<F>::zeros(x.nrows());
        for p in &all_preds {
            predictions += p;
        }
        Ok(predictions / n_trees_f)
    }
}

impl<F: Float> FittedBaggingRegressor<F> {
    fn tree_features(&self, t: usize) -> &[usize] {
        self.features.get(t).map_or(&[], Vec::as_slice)
    }

    /// Feature importances averaged across all trees and normalized to sum to 1.
    pub fn feature_importances(&self) -> Array1<F> {
        let mut importances = vec![F::zero(); self.n_features];
        for (t, tree) in self.trees.iter().enumerate() {
            add_importances(
                &mut importances,
                &tree.feature_importances(),
                self.tree_features(t),
            );
        }

        // Normalize so importances sum to 1
        let sum: F = importances.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            Array1::from_vec(importances.into_iter().map(|v| v / sum).collect())
        } else {
            Array1::zeros(self.n_features)
        }
    }

    /// Feature subset each tree was trained on (empty = all features).
    pub fn estimators_features(&self) -> &[Vec<usize>] {
        &self.features
    }

    /// Out-of-bag prediction of every training sample (NaN for samples that
    /// every tree saw); `None` unless fitted with `oob_score`.
    pub fn oob_prediction(&self) -> Option<&Array1<F>> {
        self.oob_prediction.as_ref()
    }

    /// Out-of-bag R² over the samples with an out-of-bag prediction; `None`
    /// unless fitted with `oob_score`.
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
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
