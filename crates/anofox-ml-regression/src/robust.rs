//! Robust regression: TheilSen and RANSAC.
//!
//! Mirrors `sklearn.linear_model.{TheilSenRegressor, RANSACRegressor}`,
//! delegating to `anofox_regression::solvers::{TheilSenRegressor,
//! RansacRegressor}` and converting at the ndarray boundary.
//!
//! - Theil-Sen enumerates every subset when `C(n, n_subsamples) <=
//!   max_subpopulation` (deterministic, as sklearn) and samples otherwise;
//!   the estimate is the spatial median of the subset least-squares solutions
//!   (also for one feature, as sklearn — the classical median-of-pairwise-
//!   slopes estimator is available via `with_pairwise_slopes(true)`).
//! - RANSAC uses the MAD of `y` as default threshold, `<=` inlier test,
//!   Fischler–Bolles dynamic trial bound (`stop_probability`), and an OLS refit
//!   on the consensus set.
//!
//! Random subsets come from upstream's seeded generator, so sampled (not
//! enumerated) fits are reproducible per seed but not bit-identical to
//! NumPy's.

use crate::convert::{check_xy, coef_from_col, linear_predict, ndarray_to_col, ndarray_to_mat};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::solvers::{
    FittedRegressor as _, RansacRegressor as InnerRansac, Regressor as _,
    TheilSenRegressor as InnerTheilSen,
};
use ndarray::{Array1, Array2};

fn check_finite(x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    if x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
        return Err(RustMlError::InvalidParameter(
            "X and y must not contain NaN or infinite values".into(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// RANSACRegressor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RansacRegressor {
    /// Subset size; default `n_features + fit_intercept`.
    pub min_samples: Option<usize>,
    /// Inlier threshold on `|y − ŷ|`; default MAD(y).
    pub residual_threshold: Option<f64>,
    pub max_trials: usize,
    /// Stop as soon as a consensus set has at least this many inliers.
    pub stop_n_inliers: usize,
    /// Stop once an outlier-free subset has been drawn with this
    /// probability (sklearn `stop_probability`, default 0.99).
    pub stop_probability: f64,
    pub seed: u64,
    pub fit_intercept: bool,
}

impl RansacRegressor {
    pub fn new() -> Self {
        Self {
            min_samples: None,
            residual_threshold: None,
            max_trials: 100,
            stop_n_inliers: usize::MAX,
            stop_probability: 0.99,
            seed: 0,
            fit_intercept: true,
        }
    }
    pub fn with_min_samples(mut self, m: usize) -> Self {
        self.min_samples = Some(m);
        self
    }
    pub fn with_residual_threshold(mut self, t: f64) -> Self {
        self.residual_threshold = Some(t);
        self
    }
    pub fn with_max_trials(mut self, n: usize) -> Self {
        self.max_trials = n;
        self
    }
    pub fn with_stop_n_inliers(mut self, n: usize) -> Self {
        self.stop_n_inliers = n;
        self
    }
    pub fn with_stop_probability(mut self, p: f64) -> Self {
        self.stop_probability = p;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl Default for RansacRegressor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedRansacRegressor {
    pub coef: Array1<f64>,
    pub intercept: f64,
    pub inlier_mask: Vec<bool>,
    pub n_trials: usize,
    n_features: usize,
}

impl Fit<f64> for RansacRegressor {
    type Fitted = FittedRansacRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        check_finite(x, y)?;
        let (n, d) = x.dim();
        let min_samples = self
            .min_samples
            .unwrap_or(d + usize::from(self.fit_intercept));
        if min_samples == 0 || min_samples > n {
            return Err(RustMlError::InvalidParameter(format!(
                "min_samples must be in 1..={n}, got {min_samples}"
            )));
        }
        if !(self.stop_probability > 0.0 && self.stop_probability <= 1.0) {
            return Err(RustMlError::InvalidParameter(format!(
                "stop_probability must be in (0, 1], got {}",
                self.stop_probability
            )));
        }
        let mut b = InnerRansac::builder()
            .with_intercept(self.fit_intercept)
            .min_samples(min_samples)
            .max_trials(self.max_trials)
            .stop_probability(self.stop_probability)
            .random_state(self.seed);
        if let Some(t) = self.residual_threshold {
            b = b.residual_threshold(t);
        }
        if self.stop_n_inliers != usize::MAX {
            b = b.stop_n_inliers(self.stop_n_inliers);
        }
        let fitted = b
            .build()
            .fit(&ndarray_to_mat(x), &ndarray_to_col(y))
            .map_err(|e| {
                RustMlError::InvalidParameter(format!(
                    "RANSAC could not find a valid consensus set: {e}"
                ))
            })?;
        let res = fitted.result();
        Ok(FittedRansacRegressor {
            coef: coef_from_col(&res.coefficients),
            intercept: res.intercept.unwrap_or(0.0),
            inlier_mask: fitted.inlier_mask().to_vec(),
            n_trials: fitted.n_trials(),
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedRansacRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        linear_predict(x, &self.coef, self.intercept)
    }
}

// ---------------------------------------------------------------------------
// TheilSenRegressor (spatial median of subset least-squares fits)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TheilSenRegressor {
    pub max_subpopulation: usize,
    /// Subset size. Only the minimal size `n_features + fit_intercept`
    /// (sklearn's default) is supported.
    pub n_subsamples: Option<usize>,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
    pub fit_intercept: bool,
    /// With one feature and an intercept, use the classical Theil-Sen
    /// estimator (median of pairwise slopes, Sen 1968) instead of sklearn's
    /// spatial median of `(intercept, slope)` pairs.
    pub pairwise_slopes: bool,
}

impl TheilSenRegressor {
    pub fn new() -> Self {
        Self {
            max_subpopulation: 10_000,
            n_subsamples: None,
            max_iter: 300,
            tol: 1e-3,
            seed: 0,
            fit_intercept: true,
            pairwise_slopes: false,
        }
    }
    pub fn with_max_subpopulation(mut self, m: usize) -> Self {
        self.max_subpopulation = m;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    pub fn with_pairwise_slopes(mut self, enabled: bool) -> Self {
        self.pairwise_slopes = enabled;
        self
    }
}

impl Default for TheilSenRegressor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedTheilSenRegressor {
    pub coef: Array1<f64>,
    pub intercept: f64,
    n_features: usize,
}

impl Fit<f64> for TheilSenRegressor {
    type Fitted = FittedTheilSenRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        check_finite(x, y)?;
        let (n, d) = x.dim();
        let k = d + usize::from(self.fit_intercept);
        if let Some(m) = self.n_subsamples {
            if m != k {
                return Err(RustMlError::InvalidParameter(format!(
                    "n_subsamples must equal n_features + fit_intercept = {k} (got {m}); \
                     larger subsets are not supported"
                )));
            }
        }
        if n < k {
            return Err(RustMlError::InvalidParameter(format!(
                "n_subsamples={k} > n={n}"
            )));
        }
        let fitted = InnerTheilSen::builder()
            .with_intercept(self.fit_intercept)
            .max_subpopulation(self.max_subpopulation)
            .max_iter(self.max_iter)
            .tolerance(self.tol)
            .random_state(self.seed)
            .univariate_pairwise(self.pairwise_slopes)
            .build()
            .fit(&ndarray_to_mat(x), &ndarray_to_col(y))
            .map_err(|e| {
                RustMlError::InvalidParameter(format!("TheilSen could not fit any subset: {e}"))
            })?;
        let res = fitted.result();
        Ok(FittedTheilSenRegressor {
            coef: coef_from_col(&res.coefficients),
            intercept: res.intercept.unwrap_or(0.0),
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedTheilSenRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        linear_predict(x, &self.coef, self.intercept)
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedRansacRegressor {}
impl anofox_ml_core::RegressorScore<f64> for FittedTheilSenRegressor {}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_with_outliers() -> (Array2<f64>, Array1<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        // 30 inliers on y = 2x + 1.
        for i in 0..30 {
            x.push(i as f64);
            y.push(2.0 * (i as f64) + 1.0);
        }
        // 5 wild outliers.
        for k in 0..5 {
            x.push(k as f64);
            y.push(100.0);
        }
        (
            Array2::from_shape_vec((x.len(), 1), x).unwrap(),
            Array1::from_vec(y),
        )
    }

    /// 12×2 design with two gross outliers (`data.make(12, 2)` in the
    /// sklearn reference script).
    fn ts_data() -> (Array2<f64>, Array1<f64>) {
        let x = vec![
            -2.0, -0.75, -0.75, -2.0, 1.0, 1.5, -1.0, 1.25, -1.0, -1.25, -0.5, 1.0, 0.5, -0.5, 2.0,
            -1.5, -0.25, -2.0, -0.5, -0.5, -0.25, 1.5, 0.5, -0.25,
        ];
        let y = vec![
            -1.4,
            4.775,
            0.09999999999999998,
            12.3,
            2.0,
            -2.05,
            2.15,
            -3.9000000000000004,
            5.525,
            1.85,
            -2.075,
            2.25,
        ];
        (
            Array2::from_shape_vec((12, 2), x).unwrap(),
            Array1::from_vec(y),
        )
    }

    #[test]
    fn test_ransac_ignores_outliers() {
        let (x, y) = line_with_outliers();
        let fitted = RansacRegressor::new()
            .with_min_samples(2)
            .with_residual_threshold(0.5)
            .with_max_trials(200)
            .with_seed(1)
            .fit(&x, &y)
            .unwrap();
        assert!((fitted.coef[0] - 2.0).abs() < 1e-9);
        assert!((fitted.intercept - 1.0).abs() < 1e-9);
        assert_eq!(fitted.inlier_mask.iter().filter(|&&b| b).count(), 30);
    }

    #[test]
    fn test_ransac_default_threshold_matches_sklearn() {
        // sklearn RANSACRegressor(random_state=0): coef 2, intercept 1, 17 inliers.
        let x = Array2::from_shape_fn((20, 1), |(i, _)| i as f64);
        let mut y = x.column(0).mapv(|v| 2.0 * v + 1.0);
        y[2] += 30.0;
        y[5] -= 25.0;
        y[11] += 40.0;
        let f = RansacRegressor::new().fit(&x, &y).unwrap();
        assert!((f.coef[0] - 2.0).abs() < 1e-9);
        assert!((f.intercept - 1.0).abs() < 1e-9);
        assert_eq!(f.inlier_mask.iter().filter(|&&b| b).count(), 17);
    }

    #[test]
    fn test_ransac_constant_y_all_inliers() {
        // MAD = 0 for constant y; sklearn's `<=` test keeps every row.
        let x = Array2::from_shape_fn((20, 1), |(i, _)| i as f64);
        let y = Array1::from_elem(20, 3.0);
        let f = RansacRegressor::new().fit(&x, &y).unwrap();
        assert!(f.coef[0].abs() < 1e-12);
        assert!((f.intercept - 3.0).abs() < 1e-12);
        assert!(f.inlier_mask.iter().all(|&b| b));
    }

    #[test]
    fn test_ransac_rejects_bad_input_without_panicking() {
        let (x, y) = line_with_outliers();
        // min_samples larger than n used to panic on the slice.
        assert!(RansacRegressor::new()
            .with_min_samples(1000)
            .fit(&x, &y)
            .is_err());
        let mut y_nan = y.clone();
        y_nan[3] = f64::NAN;
        assert!(RansacRegressor::new().fit(&x, &y_nan).is_err());
    }

    #[test]
    fn test_theil_sen_matches_sklearn_enumerated() {
        // C(12, 3) = 220 <= max_subpopulation → all subsets, deterministic.
        // sklearn TheilSenRegressor(random_state=0).
        let (x, y) = ts_data();
        let f = TheilSenRegressor::new().fit(&x, &y).unwrap();
        assert!(
            (f.coef[0] - 0.844130614456644).abs() < 1e-3,
            "{}",
            f.coef[0]
        );
        assert!(
            (f.coef[1] - -1.7264133144872977).abs() < 1e-3,
            "{}",
            f.coef[1]
        );
        assert!((f.intercept - 1.2278163535707047).abs() < 1e-3);

        let f = TheilSenRegressor::new()
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();
        assert!((f.coef[0] - 0.5903998713958836).abs() < 1e-3);
        assert!((f.coef[1] - -1.8172298044483608).abs() < 1e-3);
        assert_eq!(f.intercept, 0.0);
    }

    #[test]
    fn test_theil_sen_one_feature_is_spatial_median_like_sklearn() {
        // Distinct x values: with tied x, sklearn keeps the minimum-norm
        // lstsq solution of the degenerate pairs while upstream skips them.
        let (_, y) = ts_data();
        let x1 = Array2::from_shape_fn((12, 1), |(i, _)| (i * 7 % 12) as f64 / 3.0);
        let f = TheilSenRegressor::new().fit(&x1, &y).unwrap();
        assert!(
            (f.coef[0] - 2.255912988310535).abs() < 1e-3,
            "{}",
            f.coef[0]
        );
        assert!((f.intercept - -1.3758373967663822).abs() < 1e-3);
    }

    #[test]
    fn test_theil_sen_robust_slope() {
        let (x, y) = line_with_outliers();
        let fitted = TheilSenRegressor::new()
            .with_max_subpopulation(500)
            .with_seed(0)
            .fit(&x, &y)
            .unwrap();
        assert!((fitted.coef[0] - 2.0).abs() < 0.5);
        let pairwise = TheilSenRegressor::new()
            .with_pairwise_slopes(true)
            .fit(&x, &y)
            .unwrap();
        assert!((pairwise.coef[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_theil_sen_rejects_unsupported_n_subsamples() {
        let (x, y) = ts_data();
        let mut ts = TheilSenRegressor::new();
        ts.n_subsamples = Some(5);
        assert!(ts.fit(&x, &y).is_err());
    }
}
