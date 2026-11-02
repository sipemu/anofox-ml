use crate::special::{brent_minimize, fminbound};
use crate::standard_scaler::is_constant_feature;
use anofox_ml_core::{FitUnsupervised, Float, InverseTransform, Result, RustMlError, Transform};
use ndarray::{Array1, Array2, ArrayView2};

/// Power-transform family used by [`PowerTransformer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PowerMethod {
    /// Yeo-Johnson transform (works with positive and negative data).
    #[default]
    YeoJohnson,
    /// Box-Cox transform (strictly positive data only).
    BoxCox,
}

/// Parameters for PowerTransformer (unfitted state).
///
/// Applies a Yeo-Johnson or Box-Cox power transform to each feature to make
/// the data more Gaussian-like, then optionally standardizes to zero mean
/// and unit variance.
///
/// The λ per feature maximises the profile log-likelihood, found with the
/// same optimisers scipy uses (`yeojohnson_normmax` → bounded Brent,
/// `boxcox_normmax` → Brent with bracket `(-2, 2)`), so λ matches
/// `sklearn.preprocessing.PowerTransformer` to optimiser tolerance.
/// NaNs are ignored at fit time and passed through by `transform`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerTransformer {
    /// If true, standardize the transformed data to zero mean, unit variance.
    pub standardize: bool,
    /// Transform family (default Yeo-Johnson).
    #[serde(default)]
    pub method: PowerMethod,
}

impl PowerTransformer {
    /// Create a new `PowerTransformer` with defaults (Yeo-Johnson,
    /// standardization enabled).
    pub fn new() -> Self {
        Self {
            standardize: true,
            method: PowerMethod::YeoJohnson,
        }
    }

    /// Set whether to standardize after the power transform.
    pub fn standardize(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }

    /// Set the transform family.
    pub fn with_method(mut self, method: PowerMethod) -> Self {
        self.method = method;
        self
    }
}

impl Default for PowerTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted PowerTransformer -- holds learned lambdas per feature and
/// optional standardization parameters (mean and std).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedPowerTransformer<F: Float> {
    lambdas: Array1<F>,
    means: Array1<F>,
    stds: Array1<F>,
    standardize: bool,
    #[serde(default)]
    method: PowerMethod,
}

/// Apply the Yeo-Johnson transform to a single value (scipy's numerically
/// stable `expm1`/`log1p` form).
fn yeo_johnson<F: Float>(x: F, lam: F) -> F {
    let zero = F::zero();
    let two = F::from_f64(2.0).unwrap();
    let eps = F::epsilon();

    if x >= zero {
        if lam.abs() < eps {
            x.ln_1p()
        } else {
            (lam * x.ln_1p()).exp_m1() / lam
        }
    } else if (lam - two).abs() > eps {
        -((two - lam) * (-x).ln_1p()).exp_m1() / (two - lam)
    } else {
        -(-x).ln_1p()
    }
}

/// Apply the inverse Yeo-Johnson transform to a single value.
fn yeo_johnson_inverse<F: Float>(y: F, lam: F) -> F {
    let zero = F::zero();
    let one = F::one();
    let two = F::from_f64(2.0).unwrap();
    let eps = F::epsilon();

    if y >= zero {
        if lam.abs() < eps {
            y.exp_m1()
        } else {
            // x = (y * lambda + 1)^(1/lambda) - 1
            ((y * lam).ln_1p() / lam).exp_m1()
        }
    } else if (lam - two).abs() > eps {
        // x = 1 - (-(2 - lambda) * y + 1)^(1/(2-lambda))
        -((-(two - lam) * y).ln_1p() / (two - lam)).exp_m1()
    } else {
        // x = 1 - exp(-y)
        one - (-y).exp()
    }
}

/// Box-Cox transform of a single (positive) value.
fn box_cox<F: Float>(x: F, lam: F) -> F {
    if lam == F::zero() {
        x.ln()
    } else {
        (lam * x.ln()).exp_m1() / lam
    }
}

/// Inverse Box-Cox transform.
fn box_cox_inverse<F: Float>(y: F, lam: F) -> F {
    if lam == F::zero() {
        y.exp()
    } else {
        ((y * lam).ln_1p() / lam).exp()
    }
}

/// Population variance of a slice (two-pass).
fn var_f64(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    v.iter().map(|&t| (t - m) * (t - m)).sum::<f64>() / n
}

/// Negative Yeo-Johnson profile log-likelihood (`-scipy.stats.yeojohnson_llf`).
fn yj_neg_llf(col: &[f64], lam: f64) -> f64 {
    let n = col.len() as f64;
    let t: Vec<f64> = col.iter().map(|&x| yeo_johnson(x, lam)).collect();
    let var = var_f64(&t);
    if var.is_nan() || var < f64::MIN_POSITIVE {
        return f64::INFINITY;
    }
    let jac: f64 = col.iter().map(|&x| x.signum() * x.abs().ln_1p()).sum();
    let llf = -n / 2.0 * var.ln() + (lam - 1.0) * jac;
    if llf.is_infinite() {
        f64::INFINITY
    } else {
        -llf
    }
}

/// `scipy.stats.yeojohnson_normmax(x)` (bounded Brent on safe λ bounds).
fn yeo_johnson_normmax(col: &[f64]) -> Result<f64> {
    if col.iter().any(|v| !v.is_finite()) {
        return Err(RustMlError::InvalidParameter(
            "Yeo-Johnson input must be finite".into(),
        ));
    }
    if col.iter().all(|&v| v == 0.0) {
        return Ok(1.0);
    }
    let max_abs = col.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let log1p_max_x = (20.0 * max_abs).ln_1p();
    let log_eps = f64::EPSILON.ln();
    let log_tiny = (f64::MIN_POSITIVE.ln() - log_eps) / 2.0;
    let log_max = (f64::MAX.ln() + log_eps) / 2.0;
    let mut lb = log_tiny / log1p_max_x;
    let mut ub = log_max / log1p_max_x;
    if col.iter().all(|&v| v < 0.0) {
        (lb, ub) = (2.0 - ub, 2.0 - lb);
    } else if col.iter().any(|&v| v < 0.0) {
        (lb, ub) = ((2.0 - ub).max(lb), (2.0 - lb).min(ub));
    }
    Ok(fminbound(|l| yj_neg_llf(col, l), lb, ub, 1.48e-8))
}

/// log of the variance of `exp(logx)`, computed in log space like scipy's
/// `_log_var` (avoids overflow for large |λ|).
fn log_var_from_log(logx: &[f64]) -> f64 {
    let n = logx.len() as f64;
    let lse = |v: &mut dyn Iterator<Item = f64>| -> f64 {
        let vals: Vec<f64> = v.collect();
        let m = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            return m;
        }
        m + vals.iter().map(|&t| (t - m).exp()).sum::<f64>().ln()
    };
    let log_mean = lse(&mut logx.iter().copied()) - n.ln();
    // log|x - mean| for each element.
    let logxmu: Vec<f64> = logx
        .iter()
        .map(|&lx| {
            let (hi, lo) = if lx >= log_mean {
                (lx, log_mean)
            } else {
                (log_mean, lx)
            };
            if hi == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                hi + (-(lo - hi).exp()).ln_1p()
            }
        })
        .collect();
    lse(&mut logxmu.iter().map(|&v| 2.0 * v)) - n.ln()
}

/// Negative Box-Cox profile log-likelihood (`-scipy.stats.boxcox_llf`).
fn bc_neg_llf(logdata: &[f64], lam: f64) -> f64 {
    let n = logdata.len() as f64;
    let logvar = if lam == 0.0 {
        var_f64(logdata).ln()
    } else {
        let logx: Vec<f64> = logdata.iter().map(|&l| lam * l).collect();
        log_var_from_log(&logx) - 2.0 * lam.abs().ln()
    };
    let llf = (lam - 1.0) * logdata.iter().sum::<f64>() - n / 2.0 * logvar;
    -llf
}

/// `scipy.stats.boxcox_normmax(x, method="mle")` (Brent, bracket (-2, 2)).
fn box_cox_normmax(col: &[f64]) -> f64 {
    let logdata: Vec<f64> = col.iter().map(|v| v.ln()).collect();
    brent_minimize(|l| bc_neg_llf(&logdata, l), -2.0, 2.0)
}

impl<F: Float> FitUnsupervised<F> for PowerTransformer {
    type Fitted = FittedPowerTransformer<F>;

    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        let ncols = x.ncols();
        let mut lambdas = Array1::<F>::zeros(ncols);
        let mut means = Array1::<F>::zeros(ncols);
        let mut stds = Array1::<F>::ones(ncols);

        for j in 0..ncols {
            let col: Vec<f64> = x
                .column(j)
                .iter()
                .map(|v| v.to_f64().unwrap())
                .filter(|v| !v.is_nan())
                .collect();
            if col.is_empty() {
                return Err(RustMlError::InvalidParameter(format!(
                    "feature {j} contains only NaN values"
                )));
            }
            let lam = match self.method {
                PowerMethod::YeoJohnson => yeo_johnson_normmax(&col)?,
                PowerMethod::BoxCox => {
                    if col.iter().any(|&v| v <= 0.0) {
                        return Err(RustMlError::InvalidParameter(
                            "The Box-Cox transformation can only be applied to strictly \
                             positive data"
                                .into(),
                        ));
                    }
                    box_cox_normmax(&col)
                }
            };
            let lam_f = F::from_f64(lam).unwrap();
            lambdas[j] = lam_f;

            if self.standardize {
                let t: Vec<F> = col
                    .iter()
                    .map(|&v| {
                        let v = F::from_f64(v).unwrap();
                        match self.method {
                            PowerMethod::YeoJohnson => yeo_johnson(v, lam_f),
                            PowerMethod::BoxCox => box_cox(v, lam_f),
                        }
                    })
                    .collect();
                let n = F::from_usize(t.len()).unwrap();
                let mean = t.iter().fold(F::zero(), |a, &b| a + b) / n;
                let var = t
                    .iter()
                    .fold(F::zero(), |a, &b| a + (b - mean) * (b - mean))
                    / n;
                means[j] = mean;
                stds[j] = if is_constant_feature(var, mean, F::from_usize(t.len()).unwrap()) {
                    F::one()
                } else {
                    var.sqrt()
                };
            }
        }

        Ok(FittedPowerTransformer {
            lambdas,
            means,
            stds,
            standardize: self.standardize,
            method: self.method,
        })
    }
}

impl<F: Float> Transform<F> for FittedPowerTransformer<F> {
    fn transform_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.lambdas.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.lambdas.len(),
                x.ncols()
            )));
        }
        if self.method == PowerMethod::BoxCox && x.iter().any(|&v| v <= F::zero()) {
            return Err(RustMlError::InvalidParameter(
                "The Box-Cox transformation can only be applied to strictly positive data".into(),
            ));
        }

        let mut result = Array2::<F>::zeros(x.raw_dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let v = x[[i, j]];
                let mut val = match self.method {
                    PowerMethod::YeoJohnson => yeo_johnson(v, self.lambdas[j]),
                    PowerMethod::BoxCox => box_cox(v, self.lambdas[j]),
                };
                if self.standardize {
                    val -= self.means[j];
                    if self.stds[j] != F::zero() {
                        val /= self.stds[j];
                    }
                }
                result[[i, j]] = val;
            }
        }
        Ok(result)
    }
}

impl<F: Float> InverseTransform<F> for FittedPowerTransformer<F> {
    fn inverse_transform_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.lambdas.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.lambdas.len(),
                x.ncols()
            )));
        }

        let mut result = Array2::<F>::zeros(x.raw_dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let mut val = x[[i, j]];
                // Undo standardization
                if self.standardize {
                    if self.stds[j] != F::zero() {
                        val *= self.stds[j];
                    }
                    val += self.means[j];
                }
                result[[i, j]] = match self.method {
                    PowerMethod::YeoJohnson => yeo_johnson_inverse(val, self.lambdas[j]),
                    PowerMethod::BoxCox => box_cox_inverse(val, self.lambdas[j]),
                };
            }
        }
        Ok(result)
    }
}

impl<F: Float> FittedPowerTransformer<F> {
    /// Return the fitted lambda per feature.
    pub fn lambdas(&self) -> &Array1<F> {
        &self.lambdas
    }

    /// Return the mean per feature (after the power transform, before
    /// standardization).
    pub fn means(&self) -> &Array1<F> {
        &self.means
    }

    /// Return the scale per feature (after the power transform, before
    /// standardization; 1 for constant features).
    pub fn stds(&self) -> &Array1<F> {
        &self.stds
    }

    /// Transform family used.
    pub fn method(&self) -> PowerMethod {
        self.method
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn sk_x() -> Array2<f64> {
        array![
            [4.101, 3.2085],
            [1.3773, 0.3844],
            [2.188, 1.3454],
            [6.0057, 0.9271],
            [4.4551, 12.6867],
            [0.4576, 0.041],
            [2.1384, 0.86],
            [0.886, -0.071],
            [0.9207, -13.8446],
            [1.3889, 2.4564],
            [1.1221, 4.0642],
            [3.2009, -0.0227]
        ]
    }

    #[test]
    fn test_yeo_johnson_lambda_matches_sklearn() {
        // sklearn 1.8: PowerTransformer().fit(X).lambdas_
        let x = sk_x();
        let fitted = FitUnsupervised::<f64>::fit(&PowerTransformer::new(), &x).unwrap();
        assert_abs_diff_eq!(fitted.lambdas()[0], -0.4117946600025421, epsilon = 1e-6);
        assert_abs_diff_eq!(fitted.lambdas()[1], 1.1322761638509586, epsilon = 1e-6);
        let t = fitted.transform(&x).unwrap();
        let expected = [
            [1.13191499150288, 0.3399313597980212],
            [-0.4283133339103107, -0.22295521546240757],
            [0.2299594160221772, -0.04068475198649925],
        ];
        for (i, row) in expected.iter().enumerate() {
            for (j, &e) in row.iter().enumerate() {
                assert_abs_diff_eq!(t[[i, j]], e, epsilon = 1e-6);
            }
        }
        let fitted =
            FitUnsupervised::<f64>::fit(&PowerTransformer::new().standardize(false), &x).unwrap();
        let t = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(t[[0, 0]], 1.1869980174247963, epsilon = 1e-6);
        assert_abs_diff_eq!(t[[0, 1]], 3.6118484864534937, epsilon = 1e-6);
        assert_abs_diff_eq!(t[[1, 1]], 0.3932466633282193, epsilon = 1e-6);
    }

    #[test]
    fn test_box_cox_matches_sklearn() {
        // sklearn 1.8: PowerTransformer(method="box-cox").fit(X[:, :1])
        let x = sk_x().slice(ndarray::s![.., 0..1]).to_owned();
        let pt = PowerTransformer::new().with_method(PowerMethod::BoxCox);
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();
        assert_abs_diff_eq!(fitted.lambdas()[0], 0.04783575148165765, epsilon = 1e-6);
        let t = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(t[[0, 0]], 1.1055988628273683, epsilon = 1e-6);
        assert_abs_diff_eq!(t[[1, 0]], -0.3925928551972952, epsilon = 1e-6);
        assert_abs_diff_eq!(t[[2, 0]], 0.2334248239723684, epsilon = 1e-6);
        let back = fitted.inverse_transform(&t).unwrap();
        for (a, b) in x.iter().zip(back.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_box_cox_rejects_non_positive() {
        let x = array![[1.0], [0.0], [2.0]];
        let pt = PowerTransformer::new().with_method(PowerMethod::BoxCox);
        assert!(FitUnsupervised::<f64>::fit(&pt, &x).is_err());
        let ok = FitUnsupervised::<f64>::fit(&pt, &array![[1.0], [2.0], [5.0]]).unwrap();
        assert!(ok.transform(&array![[-1.0]]).is_err());
    }

    #[test]
    fn test_nan_ignored_at_fit_and_passed_through() {
        let mut x = sk_x();
        let full = FitUnsupervised::<f64>::fit(&PowerTransformer::new(), &x).unwrap();
        // Append a NaN row: λ must not change, NaN passes through.
        x.push_row(ndarray::ArrayView1::from(&[f64::NAN, f64::NAN]))
            .unwrap();
        let with_nan = FitUnsupervised::<f64>::fit(&PowerTransformer::new(), &x).unwrap();
        assert_abs_diff_eq!(full.lambdas()[0], with_nan.lambdas()[0], epsilon = 1e-12);
        let t = with_nan.transform(&x).unwrap();
        assert!(t[[12, 0]].is_nan());
    }

    #[test]
    fn test_fit_transform_basic() {
        let x = array![
            [1.0, -1.0],
            [2.0, -2.0],
            [3.0, -3.0],
            [4.0, -4.0],
            [5.0, -5.0],
        ];
        let pt = PowerTransformer::default();
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Each column should have mean ~0 and std ~1
        let n = x.nrows() as f64;
        for j in 0..x.ncols() {
            let col_mean: f64 = transformed.column(j).sum() / n;
            assert_abs_diff_eq!(col_mean, 0.0, epsilon = 1e-8);

            let col_std: f64 = (transformed
                .column(j)
                .iter()
                .map(|&v| (v - col_mean).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
            assert_abs_diff_eq!(col_std, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let x = array![[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0], [4.5, 5.0],];
        let pt = PowerTransformer::default();
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_without_standardize() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let pt = PowerTransformer::new().standardize(false);
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Without standardization, the transform is just Yeo-Johnson
        let lam = fitted.lambdas()[0];
        for i in 0..x.nrows() {
            let expected = yeo_johnson(x[[i, 0]], lam);
            assert_abs_diff_eq!(transformed[[i, 0]], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_yeo_johnson_identity_at_lambda_one() {
        // When lambda = 1, Yeo-Johnson for x >= 0 is ((x+1)^1 - 1)/1 = x
        let one = 1.0_f64;
        for &x in &[0.0, 1.0, 2.0, 5.0, 10.0] {
            let result = yeo_johnson(x, one);
            assert_abs_diff_eq!(result, x, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_yeo_johnson_lambda_zero() {
        // When lambda = 0, Yeo-Johnson for x >= 0 is ln(x + 1)
        let zero = 0.0_f64;
        for &x in &[0.0, 1.0, 2.0, 5.0] {
            let result = yeo_johnson(x, zero);
            let expected = (x + 1.0).ln();
            assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_yeo_johnson_negative_values() {
        // Verify transform and inverse for negative values
        let lam = 0.5_f64;
        for &x in &[-1.0, -2.0, -5.0, -0.5] {
            let y = yeo_johnson(x, lam);
            let x_back = yeo_johnson_inverse(y, lam);
            assert_abs_diff_eq!(x, x_back, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_lambda_two_negative_branch() {
        // When lambda = 2 and x < 0: -ln(-x + 1)
        let two = 2.0_f64;
        for &x in &[-1.0, -2.0, -0.5] {
            let result = yeo_johnson(x, two);
            let expected = -(-x + 1.0).ln();
            assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 0));
        let pt = PowerTransformer::default();
        assert!(FitUnsupervised::<f64>::fit(&pt, &x).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let pt = PowerTransformer::default();
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_wrong).is_err());
        assert!(fitted.inverse_transform(&x_wrong).is_err());
    }

    #[test]
    fn test_mixed_positive_negative() {
        let x = array![
            [-3.0, 10.0],
            [-1.0, 20.0],
            [0.0, 30.0],
            [1.0, 40.0],
            [3.0, 50.0],
        ];
        let pt = PowerTransformer::default();
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }
}
