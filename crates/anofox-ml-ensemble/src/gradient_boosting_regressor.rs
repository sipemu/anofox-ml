use anofox_ml_core::{Fit, FitWeighted, Float, Predict, Result, RustMlError};
use anofox_ml_trees::{DecisionTreeRegressor, FittedDecisionTreeRegressor};
use ndarray::{Array1, ArrayView1, ArrayView2, AsArray, Ix1, Ix2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Gradient boosting regressor parameters (unfitted state).
///
/// Builds an ensemble of decision tree regressors sequentially, where each
/// tree fits the negative gradient (residuals) of the squared-error loss.
/// Predictions are the sum of the initial value plus a learning-rate-scaled
/// contribution from each tree.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradientBoostingRegressor {
    /// Number of boosting rounds (trees).
    pub n_estimators: usize,
    /// Shrinkage applied to each tree's contribution.
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node.
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Fraction of training samples used per tree (stochastic gradient boosting).
    pub subsample: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Loss to minimise (sklearn's `loss`).
    #[serde(default)]
    pub loss: GbLoss,
    /// Quantile of `Quantile` and the Huber threshold quantile (sklearn's
    /// `alpha`, default 0.9).
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// Features drawn per split of every tree (`None` = all).
    #[serde(default)]
    pub max_features: Option<anofox_ml_trees::MaxFeatures>,
    /// Early stopping: stop when the validation loss has not improved by
    /// `tol` for this many rounds (`None` = off).
    #[serde(default)]
    pub n_iter_no_change: Option<usize>,
    /// Fraction of the training data held out for early stopping.
    #[serde(default = "default_validation_fraction")]
    pub validation_fraction: f64,
    /// Minimum validation-loss improvement for early stopping.
    #[serde(default = "default_tol")]
    pub tol: f64,
}

/// Loss of [`GradientBoostingRegressor`] (sklearn's `loss`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum GbLoss {
    /// Squared error; leaves are mean residuals.
    #[default]
    SquaredError,
    /// Absolute error; leaves are median residuals.
    AbsoluteError,
    /// Huber loss with the threshold at the `alpha` quantile of |residual|.
    Huber,
    /// Pinball loss for the `alpha` quantile.
    Quantile,
}

pub(crate) fn default_alpha() -> f64 {
    0.9
}
pub(crate) fn default_validation_fraction() -> f64 {
    0.1
}
pub(crate) fn default_tol() -> f64 {
    1e-4
}

/// sklearn's `_weighted_percentile` (the lower weighted percentile) over
/// `idx`; `None` means unit weights. sklearn's gradient boosting always
/// passes (unit) sample weights, so this is also its unweighted rule.
pub(crate) fn percentile<F: Float>(
    values: &[F],
    idx: &[usize],
    weights: Option<ArrayView1<F>>,
    q: f64,
) -> F {
    let mut order: Vec<usize> = idx.to_vec();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let w_of = |i: usize| weights.map_or(F::one(), |w| w[i]);
    {
        {
            let total = order.iter().fold(F::zero(), |a, &i| a + w_of(i));
            let target = F::from_f64(q / 100.0).unwrap() * total;
            let mut cum = F::zero();
            for &i in &order {
                cum += w_of(i);
                if cum >= target {
                    return values[i];
                }
            }
            values[*order.last().unwrap()]
        }
    }
}

/// NumPy's linear-interpolation percentile over `idx` (sklearn's init
/// without sample weights).
pub(crate) fn percentile_interp<F: Float>(values: &[F], idx: &[usize], q: f64) -> F {
    let mut v: Vec<F> = idx.iter().map(|&i| values[i]).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * F::from_f64(pos - lo as f64).unwrap()
}

impl GradientBoostingRegressor {
    /// Create a new `GradientBoostingRegressor` with default parameters.
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            seed: 0,
            loss: GbLoss::SquaredError,
            alpha: 0.9,
            max_features: None,
            n_iter_no_change: None,
            validation_fraction: 0.1,
            tol: 1e-4,
        }
    }

    /// Set the loss (`SquaredError`, `AbsoluteError`, `Huber`, `Quantile`).
    pub fn with_loss(mut self, loss: GbLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Quantile for `Quantile`, threshold quantile for `Huber`.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Features drawn per split.
    pub fn with_max_features(mut self, mf: Option<anofox_ml_trees::MaxFeatures>) -> Self {
        self.max_features = mf;
        self
    }

    /// Enable early stopping on a held-out `validation_fraction`.
    pub fn with_early_stopping(
        mut self,
        n_iter_no_change: Option<usize>,
        validation_fraction: f64,
        tol: f64,
    ) -> Self {
        self.n_iter_no_change = n_iter_no_change;
        self.validation_fraction = validation_fraction;
        self.tol = tol;
        self
    }

    /// Set the number of boosting rounds.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate (shrinkage).
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum depth of each tree.
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split a node.
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in a leaf node.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the fraction of samples used per boosting round.
    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for GradientBoostingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted gradient boosting regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedGradientBoostingRegressor<F: Float> {
    /// Initial prediction (mean of training targets).
    initial_prediction: F,
    /// Sequence of fitted regression trees.
    trees: Vec<FittedDecisionTreeRegressor<F>>,
    /// Learning rate used during training.
    learning_rate: F,
    /// Number of features expected at prediction time.
    n_features: usize,
}

impl<F: Float> Fit<F> for GradientBoostingRegressor {
    type Fitted = FittedGradientBoostingRegressor<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        self.fit_impl(x, y, None, None)
    }
}

impl<F: Float> FitWeighted<F> for GradientBoostingRegressor {
    type Fitted = FittedGradientBoostingRegressor<F>;

    /// Per-sample weights (sklearn's `sample_weight`).
    fn fit_weighted_view(
        &self,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        sample_weight: Option<ArrayView1<F>>,
    ) -> Result<Self::Fitted> {
        self.fit_impl(x, y, sample_weight, None)
    }
}

impl GradientBoostingRegressor {
    /// sklearn's `warm_start=True`: keep the trees of `prev` and add more
    /// until there are `n_estimators`, fitted on the same `(X, y)`. The
    /// random draws of the earlier rounds are replayed, so without early
    /// stopping the result equals a single fit with `n_estimators` trees.
    /// The early-stopping history restarts, as in sklearn. Asking for fewer
    /// trees than `prev` has is an error; as many returns `prev` unchanged.
    pub fn fit_warm_start<'a, 'b, F: Float>(
        &self,
        prev: FittedGradientBoostingRegressor<F>,
        x: impl AsArray<'a, F, Ix2>,
        y: impl AsArray<'b, F, Ix1>,
        sample_weight: Option<ArrayView1<F>>,
    ) -> Result<FittedGradientBoostingRegressor<F>> {
        let x: ArrayView2<F> = x.into();
        let y: ArrayView1<F> = y.into();
        self.fit_impl(x.view(), y.view(), sample_weight, Some(prev))
    }

    fn fit_impl<F: Float>(
        &self,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        sample_weight: Option<ArrayView1<F>>,
        prev: Option<FittedGradientBoostingRegressor<F>>,
    ) -> Result<FittedGradientBoostingRegressor<F>> {
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
        if self.learning_rate <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "learning_rate must be > 0".into(),
            ));
        }
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err(RustMlError::InvalidParameter(
                "subsample must be in (0, 1]".into(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let lr = F::from_f64(self.learning_rate).unwrap();

        if let Some(w) = sample_weight {
            anofox_ml_core::check_sample_weight(w, n_samples)?;
        }

        if !(0.0..1.0).contains(&self.alpha) || self.alpha <= 0.0 {
            if matches!(self.loss, GbLoss::Huber | GbLoss::Quantile) {
                return Err(RustMlError::InvalidParameter(
                    "alpha must be in (0, 1)".into(),
                ));
            }
        }
        if let Some(p) = &prev {
            if p.n_features != n_features {
                return Err(RustMlError::ShapeMismatch(format!(
                    "warm start: model has {} features, X has {n_features}",
                    p.n_features
                )));
            }
            if p.learning_rate != lr {
                return Err(RustMlError::InvalidParameter(
                    "warm start needs the learning_rate of the previous fit".into(),
                ));
            }
            if p.trees.len() > self.n_estimators {
                return Err(RustMlError::InvalidParameter(format!(
                    "warm start: n_estimators={} is less than the {} trees already fitted",
                    self.n_estimators,
                    p.trees.len()
                )));
            }
        }
        if let Some(p) = prev.as_ref().filter(|p| p.trees.len() == self.n_estimators) {
            return Ok(p.clone());
        }
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Early stopping: hold out a validation split.
        let (train_rows, val_rows): (Vec<usize>, Vec<usize>) = match self.n_iter_no_change {
            Some(_) => {
                if !(self.validation_fraction > 0.0 && self.validation_fraction < 1.0) {
                    return Err(RustMlError::InvalidParameter(
                        "validation_fraction must be in (0, 1)".into(),
                    ));
                }
                let mut perm: Vec<usize> = (0..n_samples).collect();
                perm.shuffle(&mut rng);
                let n_val = ((self.validation_fraction * n_samples as f64).ceil() as usize)
                    .clamp(1, n_samples - 1);
                let mut val = perm[..n_val].to_vec();
                let mut train = perm[n_val..].to_vec();
                val.sort_unstable();
                train.sort_unstable();
                (train, val)
            }
            None => ((0..n_samples).collect(), Vec::new()),
        };

        let yv: Vec<F> = y.to_vec();
        let q = self.alpha * 100.0;
        // Step 1: the loss-minimising constant (sklearn's `init`).
        let initial_prediction = match &prev {
            Some(p) => p.initial_prediction,
            None => match self.loss {
                GbLoss::SquaredError => {
                    let (num, den) =
                        train_rows
                            .iter()
                            .fold((F::zero(), F::zero()), |(n, d), &i| {
                                let wi = sample_weight.map_or(F::one(), |w| w[i]);
                                (n + wi * y[i], d + wi)
                            });
                    num / den
                }
                GbLoss::AbsoluteError | GbLoss::Huber => match sample_weight {
                    Some(_) => percentile(&yv, &train_rows, sample_weight, 50.0),
                    None => percentile_interp(&yv, &train_rows, 50.0),
                },
                GbLoss::Quantile => match sample_weight {
                    Some(_) => percentile(&yv, &train_rows, sample_weight, q),
                    None => percentile_interp(&yv, &train_rows, q),
                },
            },
        };

        // Current predictions for every training sample.
        let mut predictions = Array1::from_elem(n_samples, initial_prediction);

        let tree_params = DecisionTreeRegressor {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            sample_weight: None,
            seed: 0,
            ..DecisionTreeRegressor::new()
        };

        // Per-tree seeds for feature sampling, from a stream of their own so
        // the subsample draws do not depend on `max_features`.
        let mut tree_seeds =
            StdRng::seed_from_u64(self.seed ^ crate::gradient_boosting_classifier::TREE_SEED_SALT);
        let mut trees = Vec::with_capacity(self.n_estimators);
        let n_train = train_rows.len();
        let subsample_size = ((self.subsample * n_train as f64).round() as usize).max(1);
        // Warm start: take over the previous trees and replay their random
        // draws so the streams continue where the previous fit stopped.
        if let Some(p) = prev {
            for tree in p.trees {
                if subsample_size < n_train {
                    train_rows.clone().shuffle(&mut rng);
                }
                let _: u64 = tree_seeds.gen();
                predictions.scaled_add(lr, &tree.predict(x)?);
                trees.push(tree);
            }
        }
        let begin = trees.len();
        let half = F::from_f64(0.5).unwrap();
        let alpha = F::from_f64(self.alpha).unwrap();
        let mut loss_history = vec![F::infinity(); self.n_iter_no_change.unwrap_or(0)];

        for iteration in begin..self.n_estimators {
            let diff = &y - &predictions;
            // Huber threshold: the alpha quantile of |residual| (in-bag).
            let delta = match self.loss {
                GbLoss::Huber => {
                    let abs: Vec<F> = diff.iter().map(|d| d.abs()).collect();
                    percentile(&abs, &train_rows, sample_weight, q)
                }
                _ => F::zero(),
            };
            // Negative gradient.
            let grad = diff.mapv(|d| match self.loss {
                GbLoss::SquaredError => d,
                // NumPy's sign: 0 at 0 (Rust's `signum` would give 1).
                GbLoss::AbsoluteError => {
                    if d == F::zero() {
                        F::zero()
                    } else {
                        d.signum()
                    }
                }
                GbLoss::Huber => {
                    if d.abs() <= delta {
                        d
                    } else {
                        delta * d.signum()
                    }
                }
                GbLoss::Quantile => {
                    if d >= F::zero() {
                        alpha
                    } else {
                        alpha - F::one()
                    }
                }
            });

            let in_bag: Vec<usize> = if subsample_size < n_train {
                let mut rows = train_rows.clone();
                rows.shuffle(&mut rng);
                rows.truncate(subsample_size);
                rows.sort_unstable();
                rows
            } else {
                train_rows.clone()
            };
            // Every round gets its own tree seed (feature order, feature subset).
            let round_params = tree_params.clone().with_seed(tree_seeds.gen());
            let params = &round_params;
            let mut fitted_tree: FittedDecisionTreeRegressor<F> =
                crate::forest_common::fit_tree_on_rows(
                    params,
                    x,
                    grad.view(),
                    &in_bag,
                    sample_weight,
                )?;

            // Terminal-region update (sklearn's line search) for the
            // non-squared losses: the leaf's loss-minimising constant of the
            // in-bag residuals.
            if self.loss != GbLoss::SquaredError {
                let leaves = fitted_tree.apply(x)?;
                let mut members: Vec<Vec<usize>> = vec![Vec::new(); fitted_tree.n_leaves()];
                for &i in &in_bag {
                    members[leaves[i]].push(i);
                }
                let dv: Vec<F> = diff.to_vec();
                let values: Vec<F> = members
                    .iter()
                    .map(|m| {
                        if m.is_empty() {
                            return F::zero();
                        }
                        match self.loss {
                            GbLoss::AbsoluteError => percentile(&dv, m, sample_weight, 50.0),
                            GbLoss::Quantile => percentile(&dv, m, sample_weight, q),
                            GbLoss::Huber => {
                                let median = percentile(&dv, m, sample_weight, 50.0);
                                let (num, den) =
                                    m.iter().fold((F::zero(), F::zero()), |(n, d), &i| {
                                        let r = dv[i] - median;
                                        let sign = if r == F::zero() {
                                            F::zero()
                                        } else {
                                            r.signum()
                                        };
                                        let term = sign * r.abs().min(delta);
                                        let wi = sample_weight.map_or(F::one(), |w| w[i]);
                                        (n + wi * term, d + wi)
                                    });
                                median + num / den
                            }
                            GbLoss::SquaredError => unreachable!(),
                        }
                    })
                    .collect();
                fitted_tree.set_leaf_values(&values)?;
            }

            // Update predictions on all rows (training and validation).
            let tree_preds = fitted_tree.predict(x)?;
            predictions.scaled_add(lr, &tree_preds);
            trees.push(fitted_tree);

            if let Some(n_no_change) = self.n_iter_no_change {
                let (mut num, mut den) = (F::zero(), F::zero());
                for &i in &val_rows {
                    let d = y[i] - predictions[i];
                    let l = match self.loss {
                        GbLoss::SquaredError => half * d * d,
                        GbLoss::AbsoluteError => d.abs(),
                        GbLoss::Huber => {
                            if d.abs() <= delta {
                                half * d * d
                            } else {
                                delta * (d.abs() - half * delta)
                            }
                        }
                        GbLoss::Quantile => {
                            if d > F::zero() {
                                alpha * d
                            } else {
                                (alpha - F::one()) * d
                            }
                        }
                    };
                    let wi = sample_weight.map_or(F::one(), |w| w[i]);
                    num += wi * l;
                    den += wi;
                }
                let val_loss = num / den;
                let tol = F::from_f64(self.tol).unwrap();
                if loss_history.iter().any(|&h| val_loss + tol < h) {
                    loss_history[iteration % n_no_change] = val_loss;
                } else {
                    break;
                }
            }
        }

        Ok(FittedGradientBoostingRegressor {
            initial_prediction,
            trees,
            learning_rate: lr,
            n_features,
        })
    }
}

impl<F: Float> Predict<F> for FittedGradientBoostingRegressor<F> {
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::from_elem(n_samples, self.initial_prediction);

        for tree in &self.trees {
            let tree_preds = tree.predict(x)?;
            predictions += &(tree_preds * self.learning_rate);
        }

        Ok(predictions)
    }
}

impl<F: Float> FittedGradientBoostingRegressor<F> {
    /// Predictions after each tree (sklearn's `staged_predict`); the last
    /// equals [`Predict::predict`].
    pub fn staged_predict<'a>(
        &self,
        x: impl AsArray<'a, F, Ix2>,
    ) -> Result<impl Iterator<Item = Array1<F>> + '_> {
        let x: ArrayView2<F> = x.into();
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let tree_preds = self
            .trees
            .iter()
            .map(|t| t.predict(x))
            .collect::<Result<Vec<_>>>()?;
        let mut predictions = Array1::from_elem(x.nrows(), self.initial_prediction);
        Ok(tree_preds.into_iter().map(move |tree_preds| {
            predictions += &(tree_preds * self.learning_rate);
            predictions.clone()
        }))
    }

    /// Number of trees in the ensemble.
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// The initial prediction (mean of training targets).
    pub fn initial_prediction(&self) -> F {
        self.initial_prediction
    }

    /// Feature importances averaged across all trees, normalized to sum to 1.
    pub fn feature_importances(&self) -> Array1<F> {
        let mut importances = vec![F::zero(); self.n_features];
        let n_trees = self.trees.len();

        if n_trees == 0 {
            return Array1::zeros(self.n_features);
        }

        let n_trees_f = F::from_usize(n_trees).unwrap();
        for tree in &self.trees {
            let tree_imp = tree.feature_importances();
            for (j, &imp) in tree_imp.iter().enumerate() {
                importances[j] += imp / n_trees_f;
            }
        }

        // Normalize to sum to 1
        let sum: F = importances.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            Array1::from_vec(importances.into_iter().map(|v| v / sum).collect())
        } else {
            Array1::zeros(self.n_features)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    #[test]
    fn test_basic_regression() {
        // y = 2*x, gradient boosting should learn a good approximation.
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

        let gb = GradientBoostingRegressor {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1.0);
        }
    }

    #[test]
    fn test_training_error_decreases() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];

        let gb_few = GradientBoostingRegressor {
            n_estimators: 5,
            learning_rate: 0.1,
            max_depth: Some(2),
            seed: 0,
            ..Default::default()
        };
        let gb_many = GradientBoostingRegressor {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(2),
            seed: 0,
            ..Default::default()
        };

        let fitted_few: FittedGradientBoostingRegressor<f64> = gb_few.fit(&x, &y).unwrap();
        let fitted_many: FittedGradientBoostingRegressor<f64> = gb_many.fit(&x, &y).unwrap();

        let preds_few = fitted_few.predict(&x).unwrap();
        let preds_many = fitted_many.predict(&x).unwrap();

        let mse_few: f64 = (&y - &preds_few).mapv(|v| v * v).mean().unwrap();
        let mse_many: f64 = (&y - &preds_many).mapv(|v| v * v).mean().unwrap();

        assert!(
            mse_many < mse_few,
            "more estimators should reduce training error: mse_many={mse_many} >= mse_few={mse_few}"
        );
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let gb = GradientBoostingRegressor {
            n_estimators: 20,
            seed: 123,
            ..Default::default()
        };

        let fitted1: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();
        let fitted2: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_subsample() {
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

        let gb = GradientBoostingRegressor {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            subsample: 0.8,
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // With subsample < 1.0, predictions should still be reasonable.
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 2.0);
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let gb = GradientBoostingRegressor::default();
        let result: std::result::Result<FittedGradientBoostingRegressor<f64>, _> = gb.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let gb = GradientBoostingRegressor {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_initial_prediction_is_mean() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![10.0, 20.0, 30.0, 40.0];

        let gb = GradientBoostingRegressor {
            n_estimators: 1,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();
        assert_abs_diff_eq!(fitted.initial_prediction(), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_parameters() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        // n_estimators = 0
        let gb = GradientBoostingRegressor {
            n_estimators: 0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());

        // learning_rate <= 0
        let gb = GradientBoostingRegressor {
            learning_rate: 0.0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());

        // subsample out of range
        let gb = GradientBoostingRegressor {
            subsample: 0.0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());

        let gb = GradientBoostingRegressor {
            subsample: 1.5,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());
    }

    #[test]
    fn test_predictions_finite() {
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

        let gb = GradientBoostingRegressor {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction must be finite, got {p}");
        }
    }

    #[test]
    fn test_staged_predict_ends_at_predict() {
        let x = Array2::from_shape_fn((50, 2), |(i, j)| ((i * (j + 3)) as f64 * 0.29).sin());
        let y = x.column(0).mapv(|v| 2.0 * v) + x.column(1);
        let f: FittedGradientBoostingRegressor<f64> = GradientBoostingRegressor {
            n_estimators: 9,
            ..Default::default()
        }
        .fit(&x, &y)
        .unwrap();
        let stages: Vec<_> = f.staged_predict(&x).unwrap().collect();
        assert_eq!(stages.len(), 9);
        assert_eq!(stages[8], f.predict(&x).unwrap());
    }

    #[test]
    fn test_n_estimators_one_regressor() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let gb = GradientBoostingRegressor {
            n_estimators: 1,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 1);

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
        // All predictions should be finite even with a single tree.
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_warm_start_equals_single_fit() {
        let x = Array2::from_shape_fn((80, 3), |(i, j)| ((i * (j + 2)) as f64 * 0.37).sin());
        let y = x.column(0).mapv(|v| 3.0 * v) + x.column(1).mapv(|v| v * v);
        for loss in [GbLoss::SquaredError, GbLoss::Huber, GbLoss::Quantile] {
            let gb = |n| GradientBoostingRegressor {
                n_estimators: n,
                subsample: 0.7,
                loss,
                max_depth: Some(2),
                max_features: Some(anofox_ml_trees::MaxFeatures::Fixed(2)),
                seed: 5,
                ..GradientBoostingRegressor::new()
            };
            let full = gb(12).fit(&x, &y).unwrap();
            let part = gb(5).fit(&x, &y).unwrap();
            let warm = gb(12).fit_warm_start(part, &x, &y, None).unwrap();
            assert_eq!(warm.n_estimators(), 12);
            assert_eq!(warm.predict(&x).unwrap(), full.predict(&x).unwrap());
        }
        let gb = GradientBoostingRegressor::new().with_seed(1);
        let f = Fit::<f64>::fit(&gb, &x, &y).unwrap();
        let n = f.n_estimators();
        // Same size: unchanged; fewer trees: error.
        assert_eq!(
            gb.fit_warm_start(f.clone(), &x, &y, None)
                .unwrap()
                .n_estimators(),
            n
        );
        let fewer = GradientBoostingRegressor {
            n_estimators: n - 1,
            ..gb.clone()
        };
        assert!(fewer.fit_warm_start(f, &x, &y, None).is_err());
    }
}
