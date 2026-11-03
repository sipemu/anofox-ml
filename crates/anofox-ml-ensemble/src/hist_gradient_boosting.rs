//! Histogram-based gradient boosting with sklearn's
//! `HistGradientBoostingRegressor` / `HistGradientBoostingClassifier` names
//! and defaults.
//!
//! Both are thin facades over the [`lgbm`](crate::lgbm) engine: features are
//! binned into at most `max_bins` (≤ 255) bins, trees grow leaf-wise up to
//! `max_leaf_nodes` leaves (depth-limited by `max_depth` if set), missing
//! values (`NaN`) get their own bin and a learned direction at every split,
//! and leaves are Newton steps with an L2 penalty. Sample weights, feature
//! importances and the engine's histogram subtraction come with it.

use anofox_ml_core::{Fit, FitWeighted, Predict, Result};
use ndarray::{Array1, Array2};

use crate::lgbm::{
    BinningMethod, FittedLgbmClassifier, FittedLgbmRegressor, LgbmClassifier, LgbmRegressor,
};

/// Shared hyperparameters and builder methods of the two facades.
macro_rules! hist_gb_config {
    ($name:ident) => {
        impl $name {
            pub fn new() -> Self {
                Self {
                    n_estimators: 100,
                    learning_rate: 0.1,
                    max_leaf_nodes: Some(31),
                    max_depth: None,
                    min_samples_leaf: 20,
                    l2_regularization: 0.0,
                    max_bins: 255,
                }
            }

            /// Number of boosting rounds (sklearn's `max_iter`).
            pub fn with_n_estimators(mut self, n: usize) -> Self {
                self.n_estimators = n;
                self
            }
            pub fn with_learning_rate(mut self, lr: f64) -> Self {
                self.learning_rate = lr;
                self
            }
            /// Maximum leaves per tree (`None` = unlimited, depth-bound).
            pub fn with_max_leaf_nodes(mut self, n: Option<usize>) -> Self {
                self.max_leaf_nodes = n;
                self
            }
            pub fn with_max_depth(mut self, d: usize) -> Self {
                self.max_depth = Some(d);
                self
            }
            pub fn with_min_samples_leaf(mut self, m: usize) -> Self {
                self.min_samples_leaf = m;
                self
            }
            pub fn with_l2_regularization(mut self, l2: f64) -> Self {
                self.l2_regularization = l2;
                self
            }
            pub fn with_max_bins(mut self, b: usize) -> Self {
                self.max_bins = b;
                self
            }

            fn validate(&self) -> Result<()> {
                if self.max_leaf_nodes.is_none() && self.max_depth.is_none() {
                    return Err(anofox_ml_core::RustMlError::InvalidParameter(
                        "max_leaf_nodes and max_depth cannot both be None".into(),
                    ));
                }
                if self.max_leaf_nodes.is_some_and(|n| n < 2) {
                    return Err(anofox_ml_core::RustMlError::InvalidParameter(
                        "max_leaf_nodes must be at least 2".into(),
                    ));
                }
                Ok(())
            }

            /// Leaf budget for the engine: unlimited leaves are bounded by
            /// the depth (a depth-`d` tree has at most `2^d` leaves).
            fn num_leaves(&self) -> usize {
                self.max_leaf_nodes.unwrap_or_else(|| {
                    1usize << self.max_depth.unwrap_or(0).min(usize::BITS as usize - 2)
                })
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Histogram-based gradient boosting regressor (squared error).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistGradientBoostingRegressor {
    pub n_estimators: usize,
    pub learning_rate: f64,
    /// Maximum leaves per tree (sklearn default 31).
    #[serde(default = "default_leaves")]
    pub max_leaf_nodes: Option<usize>,
    /// Maximum depth (sklearn default `None`).
    pub max_depth: Option<usize>,
    pub min_samples_leaf: usize,
    pub l2_regularization: f64,
    pub max_bins: usize,
}

/// Histogram-based gradient boosting classifier (binary log-loss or
/// multinomial softmax).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistGradientBoostingClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    /// Maximum leaves per tree (sklearn default 31).
    #[serde(default = "default_leaves")]
    pub max_leaf_nodes: Option<usize>,
    /// Maximum depth (sklearn default `None`).
    pub max_depth: Option<usize>,
    pub min_samples_leaf: usize,
    pub l2_regularization: f64,
    pub max_bins: usize,
}

fn default_leaves() -> Option<usize> {
    Some(31)
}

hist_gb_config!(HistGradientBoostingRegressor);
hist_gb_config!(HistGradientBoostingClassifier);

impl HistGradientBoostingRegressor {
    fn engine(&self) -> Result<LgbmRegressor> {
        self.validate()?;
        Ok(LgbmRegressor::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_num_leaves(self.num_leaves())
            .with_max_depth(self.max_depth)
            .with_min_child_samples(self.min_samples_leaf)
            .with_reg_lambda(self.l2_regularization)
            .with_max_bins(self.max_bins)
            .with_binning(BinningMethod::Sklearn))
    }
}

impl HistGradientBoostingClassifier {
    fn engine(&self) -> Result<LgbmClassifier> {
        self.validate()?;
        let mut engine = LgbmClassifier::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_num_leaves(self.num_leaves())
            .with_max_depth(self.max_depth)
            .with_min_child_samples(self.min_samples_leaf)
            .with_reg_lambda(self.l2_regularization)
            .with_max_bins(self.max_bins)
            .with_binning(BinningMethod::Sklearn);
        // sklearn's multinomial hessian is p(1 - p), without LightGBM's
        // K / (K - 1) factor.
        engine.softmax_hessian_factor = false;
        Ok(engine)
    }
}

/// Fitted histogram-based gradient boosting regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedHistGradientBoostingRegressor {
    inner: FittedLgbmRegressor,
}

impl FittedHistGradientBoostingRegressor {
    pub fn n_estimators(&self) -> usize {
        self.inner.n_estimators()
    }

    /// Feature importances by total split gain (normalised to sum to 1).
    pub fn feature_importances(&self) -> Array1<f64> {
        self.inner.feature_importances()
    }
}

/// Fitted histogram-based gradient boosting classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedHistGradientBoostingClassifier {
    inner: FittedLgbmClassifier,
}

impl FittedHistGradientBoostingClassifier {
    pub fn classes(&self) -> &[f64] {
        self.inner.classes()
    }

    pub fn n_estimators(&self) -> usize {
        self.inner.n_estimators()
    }

    /// Feature importances by total split gain (normalised to sum to 1).
    pub fn feature_importances(&self) -> Array1<f64> {
        self.inner.feature_importances()
    }

    /// Class probabilities, columns in [`classes`](Self::classes) order.
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.inner.predict_proba(x)
    }
}

impl Fit<f64> for HistGradientBoostingRegressor {
    type Fitted = FittedHistGradientBoostingRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.fit_weighted(x, y, None)
    }
}

impl FitWeighted<f64> for HistGradientBoostingRegressor {
    type Fitted = FittedHistGradientBoostingRegressor;

    fn fit_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Self::Fitted> {
        Ok(FittedHistGradientBoostingRegressor {
            inner: self.engine()?.fit_weighted(x, y, sample_weight)?,
        })
    }
}

impl Fit<f64> for HistGradientBoostingClassifier {
    type Fitted = FittedHistGradientBoostingClassifier;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.fit_weighted(x, y, None)
    }
}

impl FitWeighted<f64> for HistGradientBoostingClassifier {
    type Fitted = FittedHistGradientBoostingClassifier;

    fn fit_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Self::Fitted> {
        Ok(FittedHistGradientBoostingClassifier {
            inner: self.engine()?.fit_weighted(x, y, sample_weight)?,
        })
    }
}

impl Predict<f64> for FittedHistGradientBoostingRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.inner.predict(x)
    }
}

impl Predict<f64> for FittedHistGradientBoostingClassifier {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.inner.predict(x)
    }
}

impl anofox_ml_core::PredictProba<f64> for FittedHistGradientBoostingClassifier {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.inner.predict_proba(x)
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedHistGradientBoostingRegressor {}
impl anofox_ml_core::ClassifierScore<f64> for FittedHistGradientBoostingClassifier {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn data() -> (Array2<f64>, Array1<f64>) {
        let n = 200;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| ((i * (j + 2) * 37) % 101) as f64 / 10.0);
        let y = x.column(0).mapv(|v| v * 2.0) - x.column(1);
        (x, y)
    }

    #[test]
    fn regressor_fits_and_matches_the_engine() {
        let (x, y) = data();
        let hgb = HistGradientBoostingRegressor::new().with_n_estimators(30);
        let f = hgb.fit(&x, &y).unwrap();
        let direct = hgb.engine().unwrap().fit(&x, &y).unwrap();
        assert_eq!(f.predict(&x).unwrap(), direct.predict(&x).unwrap());
        let r2 = anofox_ml_core::RegressorScore::score(&f, &x, &y).unwrap();
        assert!(r2 > 0.95, "r2 = {r2}");
        assert_eq!(f.feature_importances().len(), 3);
    }

    #[test]
    fn classifier_binary_and_multiclass_with_nan() {
        let (mut x, y) = data();
        x[[3, 1]] = f64::NAN;
        x[[50, 0]] = f64::NAN;
        for yc in [
            y.mapv(|v| if v > 5.0 { 1.0 } else { 0.0 }),
            y.mapv(|v| (v.abs() as i64 % 3) as f64),
        ] {
            let f = HistGradientBoostingClassifier::new()
                .with_n_estimators(30)
                .with_min_samples_leaf(5)
                .fit(&x, &yc)
                .unwrap();
            let proba = f.predict_proba(&x).unwrap();
            assert_eq!(proba.ncols(), f.classes().len());
            assert!(proba
                .rows()
                .into_iter()
                .all(|r| (r.sum() - 1.0).abs() < 1e-9));
            let acc = anofox_ml_core::ClassifierScore::score(&f, &x, &yc).unwrap();
            assert!(acc > 0.8, "accuracy = {acc}");
        }
    }

    #[test]
    fn depth_only_trees_and_validation() {
        let (x, y) = data();
        let f = HistGradientBoostingRegressor::new()
            .with_max_leaf_nodes(None)
            .with_max_depth(3)
            .fit(&x, &y)
            .unwrap();
        assert!(f.n_estimators() > 0);
        assert!(HistGradientBoostingRegressor::new()
            .with_max_leaf_nodes(None)
            .fit(&x, &y)
            .is_err());
        assert!(HistGradientBoostingClassifier::new()
            .fit(&array![[1.0], [2.0]], &array![0.0, 0.0])
            .is_err());
    }

    #[test]
    fn sample_weights_change_the_fit() {
        let (x, y) = data();
        let w = Array1::from_shape_fn(y.len(), |i| if i < 100 { 5.0 } else { 0.2 });
        let hgb = HistGradientBoostingRegressor::new().with_n_estimators(10);
        let a = hgb.fit(&x, &y).unwrap().predict(&x).unwrap();
        let b = hgb
            .fit_weighted(&x, &y, Some(&w))
            .unwrap()
            .predict(&x)
            .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn regressor_matches_sklearn_binning() {
        // sklearn 1.8 HistGradientBoostingRegressor(max_iter=20,
        // min_samples_leaf=5[, max_bins=16]).fit(X, y).predict(X[rows]).
        // max_bins=16 takes the percentile path of sklearn's binning. sklearn
        // keeps gradients in float32, hence the 1e-6 tolerance.
        let x = Array2::from_shape_fn((120, 3), |(i, j)| {
            let t = i as f64;
            match j {
                0 => (t * 0.37).sin() * 3.0 + t * 0.01,
                1 => ((t * 0.11).cos() * 4.0).round(),
                _ => (t * 0.7) % 5.3,
            }
        });
        let y = Array1::from_shape_fn(120, |i| {
            x[[i, 0]].powi(2) - 2.0 * x[[i, 1]] + x[[i, 2]].sin()
        });
        let rows = [0, 7, 33, 90, 119];
        let cases: [(usize, [f64; 5]); 2] = [
            (
                255,
                [
                    -5.777_367_549_331_124_5,
                    -3.408_584_907_578_834_6,
                    8.455_614_568_356_298,
                    17.890_444_108_490_723,
                    -3.509_011_691_821_183,
                ],
            ),
            (
                16,
                [
                    -5.885_602_490_798_531,
                    -3.395_267_404_999_707,
                    8.690_401_533_423_369,
                    17.968_895_264_862_134,
                    -3.352_575_566_258_404_5,
                ],
            ),
        ];
        for (max_bins, expected) in cases {
            let m = HistGradientBoostingRegressor::new()
                .with_n_estimators(20)
                .with_min_samples_leaf(5)
                .with_max_bins(max_bins)
                .fit(&x, &y)
                .unwrap();
            let p = m.predict(&x.select(ndarray::Axis(0), &rows)).unwrap();
            for (a, b) in p.iter().zip(expected) {
                assert!((a - b).abs() < 1e-6, "max_bins={max_bins}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn sklearn_thresholds_match_numpy() {
        // <= max_bins distinct values: midpoints.
        assert_eq!(
            crate::lgbm::sklearn_thresholds_for_test(&[1.0, 1.0, 2.0, 4.0], 4),
            vec![1.5, 3.0]
        );
        // np.percentile(arange(10), [25, 50, 75], method="midpoint")
        //   -> [2.5, 4.5, 6.5]
        let v: Vec<f64> = (0..10).map(f64::from).collect();
        assert_eq!(
            crate::lgbm::sklearn_thresholds_for_test(&v, 4),
            vec![2.5, 4.5, 6.5]
        );
    }
}
