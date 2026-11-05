//! Kernel density estimation (sklearn's `KernelDensity`), computed exactly
//! (sklearn's tree with `atol = rtol = 0`), with sklearn's six kernels and
//! their normalisations, sample weights and `scott` / `silverman`
//! bandwidths.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Smoothing kernel of a [`KernelDensity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DensityKernel {
    #[default]
    Gaussian,
    Tophat,
    Epanechnikov,
    Exponential,
    Linear,
    Cosine,
}

/// Bandwidth of a [`KernelDensity`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Bandwidth {
    Fixed(f64),
    /// `n^(−1 / (d + 4))`.
    Scott,
    /// `(n (d + 2) / 4)^(−1 / (d + 4))`.
    Silverman,
}

impl Default for Bandwidth {
    fn default() -> Self {
        Bandwidth::Fixed(1.0)
    }
}

/// Kernel density estimator (euclidean distance).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct KernelDensity {
    pub bandwidth: Bandwidth,
    pub kernel: DensityKernel,
}

/// Fitted [`KernelDensity`]: the training points and weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedKernelDensity {
    kernel: DensityKernel,
    bandwidth: f64,
    data: Array2<f64>,
    weights: Option<Array1<f64>>,
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation (g = 7, n = 9), accurate to ~1e-15.
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Log volume of the unit `n`-ball (`n` may be −1, as sklearn evaluates
/// it for the 0-sphere).
fn log_vn(n: i64) -> f64 {
    0.5 * n as f64 * PI.ln() - ln_gamma(0.5 * n as f64 + 1.0)
}

/// Log surface of the unit `n`-sphere.
fn log_sn(n: i64) -> f64 {
    (2.0 * PI).ln() + log_vn(n - 1)
}

/// sklearn's `_log_kernel_norm(h, d, kernel)`.
fn log_kernel_norm(h: f64, d: usize, kernel: DensityKernel) -> f64 {
    let df = d as f64;
    let di = d as i64;
    let factor = match kernel {
        DensityKernel::Gaussian => 0.5 * df * (2.0 * PI).ln(),
        DensityKernel::Tophat => log_vn(di),
        DensityKernel::Epanechnikov => log_vn(di) + (2.0 / (df + 2.0)).ln(),
        DensityKernel::Exponential => log_sn(di - 1) + ln_gamma(df),
        DensityKernel::Linear => log_vn(di) - (df + 1.0).ln(),
        DensityKernel::Cosine => {
            let mut f = 0.0;
            let mut tmp = 2.0 / PI;
            let mut k = 1;
            while k <= d {
                f += tmp;
                tmp *= -((d - k) as f64) * ((d - k) as f64 - 1.0) * (2.0 / PI).powi(2);
                k += 2;
            }
            f.ln() + log_sn(di - 1)
        }
    };
    -factor - df * h.ln()
}

/// Log kernel at distance `dist` (normalised so that `K(0) = 1`).
fn log_kernel(dist: f64, h: f64, kernel: DensityKernel) -> f64 {
    match kernel {
        DensityKernel::Gaussian => -0.5 * dist * dist / (h * h),
        DensityKernel::Exponential => -dist / h,
        _ if dist >= h => f64::NEG_INFINITY,
        DensityKernel::Tophat => 0.0,
        DensityKernel::Epanechnikov => (1.0 - dist * dist / (h * h)).ln(),
        DensityKernel::Linear => (1.0 - dist / h).ln(),
        DensityKernel::Cosine => (0.5 * PI * dist / h).cos().ln(),
    }
}

impl KernelDensity {
    pub fn new(bandwidth: Bandwidth, kernel: DensityKernel) -> Self {
        Self { bandwidth, kernel }
    }

    /// Fit with optional non-negative sample weights.
    pub fn fit_weighted(
        &self,
        x: ArrayView2<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedKernelDensity> {
        let (n, d) = x.dim();
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("x has no rows or columns".into()));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter("x must be finite".into()));
        }
        let bandwidth = match self.bandwidth {
            Bandwidth::Fixed(h) => h,
            Bandwidth::Scott => (n as f64).powf(-1.0 / (d as f64 + 4.0)),
            Bandwidth::Silverman => {
                (n as f64 * (d as f64 + 2.0) / 4.0).powf(-1.0 / (d as f64 + 4.0))
            }
        };
        if !(bandwidth > 0.0 && bandwidth.is_finite()) {
            return Err(RustMlError::InvalidParameter(format!(
                "bandwidth must be positive, got {bandwidth}"
            )));
        }
        let weights = match sample_weight {
            None => None,
            Some(w) => {
                if w.len() != n || w.iter().any(|&v| !v.is_finite() || v < 0.0) || w.sum() <= 0.0 {
                    return Err(RustMlError::InvalidParameter(
                        "sample_weight must have one finite non-negative entry per row and a positive sum"
                            .into(),
                    ));
                }
                Some(w.to_owned())
            }
        };
        Ok(FittedKernelDensity {
            kernel: self.kernel,
            bandwidth,
            data: x.to_owned(),
            weights,
        })
    }
}

impl FitUnsupervised<f64> for KernelDensity {
    type Fitted = FittedKernelDensity;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedKernelDensity> {
        self.fit_weighted(x, None)
    }
}

impl FittedKernelDensity {
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Log density at each row of `x`.
    pub fn score_samples(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let d = self.data.ncols();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {d} features, got {}",
                x.ncols()
            )));
        }
        let h = self.bandwidth;
        let total = self
            .weights
            .as_ref()
            .map_or(self.data.nrows() as f64, |w| w.sum());
        let norm = log_kernel_norm(h, d, self.kernel) - total.ln();
        Ok(x.rows()
            .into_iter()
            .map(|q| {
                let terms: Vec<f64> = self
                    .data
                    .rows()
                    .into_iter()
                    .enumerate()
                    .map(|(j, r)| {
                        let dist = q
                            .iter()
                            .zip(r)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let lw = self.weights.as_ref().map_or(0.0, |w| w[j].ln());
                        log_kernel(dist, h, self.kernel) + lw
                    })
                    .collect();
                let m = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if m == f64::NEG_INFINITY {
                    return f64::NEG_INFINITY;
                }
                m + terms.iter().map(|t| (t - m).exp()).sum::<f64>().ln() + norm
            })
            .collect())
    }

    /// Total log-likelihood of `x` (sklearn's `score`).
    pub fn score(&self, x: ArrayView2<f64>) -> Result<f64> {
        Ok(self.score_samples(x)?.sum())
    }

    /// Draw `n` points from the density (Gaussian and tophat kernels only).
    pub fn sample(&self, n: usize, seed: u64) -> Result<Array2<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = |rng: &mut StdRng| {
            let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
            let u2: f64 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        };
        let (m, d) = self.data.dim();
        let pick = |rng: &mut StdRng| -> usize {
            let u: f64 = rng.gen();
            match &self.weights {
                None => ((u * m as f64) as usize).min(m - 1),
                Some(w) => {
                    let target = u * w.sum();
                    let mut acc = 0.0;
                    for (i, &v) in w.iter().enumerate() {
                        acc += v;
                        if acc >= target {
                            return i;
                        }
                    }
                    m - 1
                }
            }
        };
        let mut out = Array2::zeros((n, d));
        for i in 0..n {
            let c = pick(&mut rng);
            match self.kernel {
                DensityKernel::Gaussian => {
                    for j in 0..d {
                        out[[i, j]] = self.data[[c, j]] + self.bandwidth * normal(&mut rng);
                    }
                }
                DensityKernel::Tophat => {
                    // Uniform in the ball: a random direction, radius h·U^(1/d).
                    let z: Vec<f64> = (0..d).map(|_| normal(&mut rng)).collect();
                    let norm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let u: f64 = rng.gen();
                    let r = self.bandwidth * u.powf(1.0 / d as f64);
                    for j in 0..d {
                        out[[i, j]] = self.data[[c, j]] + z[j] / norm * r;
                    }
                }
                _ => {
                    return Err(RustMlError::InvalidParameter(
                        "sampling needs the gaussian or tophat kernel".into(),
                    ))
                }
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
    fn densities_integrate_to_one_in_1d() {
        let x = array![[0.0], [0.5], [2.0]];
        for kernel in [
            DensityKernel::Gaussian,
            DensityKernel::Tophat,
            DensityKernel::Epanechnikov,
            DensityKernel::Exponential,
            DensityKernel::Linear,
            DensityKernel::Cosine,
        ] {
            let f = KernelDensity::new(Bandwidth::Fixed(0.7), kernel)
                .fit(&x)
                .unwrap();
            let grid = Array2::from_shape_fn((40001, 1), |(i, _)| -20.0 + i as f64 * 1e-3);
            let dens = f.score_samples(grid.view()).unwrap().mapv(f64::exp);
            let integral = dens.sum() * 1e-3;
            assert!((integral - 1.0).abs() < 1e-3, "{kernel:?}: {integral}");
        }
    }

    #[test]
    fn ln_gamma_values() {
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-13);
        assert!((ln_gamma(0.5) - PI.sqrt().ln()).abs() < 1e-13);
    }
}
