//! Kernel approximation and random projections (sklearn's
//! `kernel_approximation` and `random_projection`): [`Nystroem`],
//! [`RbfSampler`], [`KernelCenterer`], [`GaussianRandomProjection`],
//! [`SparseRandomProjection`] and [`johnson_lindenstrauss_min_dim`].
//!
//! The random ones draw from their own seeded generator, so their features
//! are not numerically sklearn's; what they approximate (the kernel, the
//! pairwise distances) is the same.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

fn check_x(x: ArrayView2<f64>) -> Result<()> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(RustMlError::EmptyInput("x has no rows or columns".into()));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(RustMlError::InvalidParameter("x must be finite".into()));
    }
    Ok(())
}

fn check_cols(expected: usize, x: ArrayView2<f64>) -> Result<()> {
    if x.ncols() != expected {
        return Err(RustMlError::ShapeMismatch(format!(
            "expected {expected} features, got {}",
            x.ncols()
        )));
    }
    Ok(())
}

/// Standard normal draw (Box–Muller).
fn normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ============================================================
// Nystroem
// ============================================================

/// Kernel of a [`Nystroem`] approximation. `gamma: None` is
/// `1 / n_features`, as in sklearn's pairwise kernels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NystroemKernel {
    Rbf {
        gamma: Option<f64>,
    },
    Laplacian {
        gamma: Option<f64>,
    },
    Polynomial {
        degree: f64,
        gamma: Option<f64>,
        coef0: f64,
    },
    Sigmoid {
        gamma: Option<f64>,
        coef0: f64,
    },
    Linear,
}

impl Default for NystroemKernel {
    fn default() -> Self {
        NystroemKernel::Rbf { gamma: None }
    }
}

impl NystroemKernel {
    fn matrix(&self, a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
        let g = |gamma: Option<f64>| gamma.unwrap_or(1.0 / a.ncols() as f64);
        match *self {
            NystroemKernel::Linear => a.dot(&b.t()),
            NystroemKernel::Polynomial {
                degree,
                gamma,
                coef0,
            } => (a.dot(&b.t()) * g(gamma) + coef0).mapv(|v| v.powf(degree)),
            NystroemKernel::Sigmoid { gamma, coef0 } => {
                (a.dot(&b.t()) * g(gamma) + coef0).mapv(f64::tanh)
            }
            NystroemKernel::Rbf { gamma } => {
                let gm = g(gamma);
                Array2::from_shape_fn((a.nrows(), b.nrows()), |(i, j)| {
                    let d2: f64 = a
                        .row(i)
                        .iter()
                        .zip(b.row(j))
                        .map(|(x, y)| (x - y).powi(2))
                        .sum();
                    (-gm * d2).exp()
                })
            }
            NystroemKernel::Laplacian { gamma } => {
                let gm = g(gamma);
                Array2::from_shape_fn((a.nrows(), b.nrows()), |(i, j)| {
                    let d1: f64 = a
                        .row(i)
                        .iter()
                        .zip(b.row(j))
                        .map(|(x, y)| (x - y).abs())
                        .sum();
                    (-gm * d1).exp()
                })
            }
        }
    }
}

/// Low-rank kernel feature map from `n_components` random training rows
/// (sklearn's `Nystroem`): `Φ(x) = k(x, B) K_BB^{−1/2}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Nystroem {
    pub kernel: NystroemKernel,
    pub n_components: usize,
    pub seed: u64,
}

impl Default for Nystroem {
    fn default() -> Self {
        Self {
            kernel: NystroemKernel::default(),
            n_components: 100,
            seed: 0,
        }
    }
}

/// Fitted [`Nystroem`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedNystroem {
    kernel: NystroemKernel,
    components: Array2<f64>,
    component_indices: Vec<usize>,
    normalization: Array2<f64>,
}

impl FittedNystroem {
    /// The basis rows (sklearn's `components_`).
    pub fn components(&self) -> &Array2<f64> {
        &self.components
    }

    pub fn component_indices(&self) -> &[usize] {
        &self.component_indices
    }

    pub fn normalization(&self) -> &Array2<f64> {
        &self.normalization
    }
}

impl FitUnsupervised<f64> for Nystroem {
    type Fitted = FittedNystroem;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedNystroem> {
        check_x(x)?;
        if self.n_components == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_components must be at least 1".into(),
            ));
        }
        let n = x.nrows();
        let k = self.n_components.min(n);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);
        perm.truncate(k);
        let basis = x.select(Axis(0), &perm);
        let kb = self.kernel.matrix(basis.view(), basis.view());
        let svd = Svd::new_thin(Mat::<f64>::from_fn(k, k, |i, j| kb[[i, j]]).as_ref())
            .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
        let (u, s, v) = (svd.U(), svd.S().column_vector(), svd.V());
        // sklearn: (U / sqrt(max(S, 1e-12))) @ Vh.
        let normalization = Array2::from_shape_fn((k, k), |(i, j)| {
            (0..k)
                .map(|c| u[(i, c)] / s[c].max(1e-12).sqrt() * v[(j, c)])
                .sum()
        });
        Ok(FittedNystroem {
            kernel: self.kernel,
            components: basis,
            component_indices: perm,
            normalization,
        })
    }
}

impl Transform<f64> for FittedNystroem {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        check_cols(self.components.ncols(), x)?;
        Ok(self
            .kernel
            .matrix(x, self.components.view())
            .dot(&self.normalization.t()))
    }
}

// ============================================================
// Random Fourier features
// ============================================================

/// Random Fourier features for the RBF kernel `exp(−γ‖x − y‖²)` (sklearn's
/// `RBFSampler`): `Φ(x) = √(2 / D) cos(x W + b)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RbfSampler {
    /// `None`: sklearn's `"scale"`, `1 / (n_features · Var(X))`.
    pub gamma: Option<f64>,
    pub n_components: usize,
    pub seed: u64,
}

impl Default for RbfSampler {
    fn default() -> Self {
        Self {
            gamma: Some(1.0),
            n_components: 100,
            seed: 0,
        }
    }
}

/// Fitted [`RbfSampler`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedRbfSampler {
    weights: Array2<f64>,
    offset: Array1<f64>,
}

impl FittedRbfSampler {
    pub fn random_weights(&self) -> &Array2<f64> {
        &self.weights
    }

    pub fn random_offset(&self) -> &Array1<f64> {
        &self.offset
    }
}

impl FitUnsupervised<f64> for RbfSampler {
    type Fitted = FittedRbfSampler;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedRbfSampler> {
        check_x(x)?;
        if self.n_components == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_components must be at least 1".into(),
            ));
        }
        let p = x.ncols();
        let gamma = match self.gamma {
            Some(g) => g,
            None => {
                let mean = x.mean().expect("non-empty");
                let var = x.mapv(|v| (v - mean).powi(2)).mean().expect("non-empty");
                if var != 0.0 {
                    1.0 / (p as f64 * var)
                } else {
                    1.0
                }
            }
        };
        let mut rng = StdRng::seed_from_u64(self.seed);
        let scale = (2.0 * gamma).sqrt();
        let weights =
            Array2::from_shape_simple_fn((p, self.n_components), || scale * normal(&mut rng));
        let offset = Array1::from_shape_simple_fn(self.n_components, || {
            rng.gen_range(0.0..2.0 * std::f64::consts::PI)
        });
        Ok(FittedRbfSampler { weights, offset })
    }
}

impl Transform<f64> for FittedRbfSampler {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        check_cols(self.weights.nrows(), x)?;
        let d = self.weights.ncols() as f64;
        let factor = (2.0 / d).sqrt();
        Ok((x.dot(&self.weights) + &self.offset).mapv(|v| v.cos() * factor))
    }
}

// ============================================================
// Kernel centering
// ============================================================

/// Centres a kernel matrix in feature space (sklearn's `KernelCenterer`):
/// fit on the training kernel `K(X, X)`, transform `K(Y, X)`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct KernelCenterer;

/// Fitted [`KernelCenterer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedKernelCenterer {
    fit_rows: Array1<f64>,
    fit_all: f64,
}

impl FitUnsupervised<f64> for KernelCenterer {
    type Fitted = FittedKernelCenterer;

    fn fit_view(&self, k: ArrayView2<f64>) -> Result<FittedKernelCenterer> {
        check_x(k)?;
        if k.nrows() != k.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "the training kernel must be square, got {} × {}",
                k.nrows(),
                k.ncols()
            )));
        }
        let fit_rows = k.mean_axis(Axis(0)).expect("non-empty");
        let fit_all = fit_rows.mean().expect("non-empty");
        Ok(FittedKernelCenterer { fit_rows, fit_all })
    }
}

impl Transform<f64> for FittedKernelCenterer {
    fn transform_view(&self, k: ArrayView2<f64>) -> Result<Array2<f64>> {
        check_cols(self.fit_rows.len(), k)?;
        let pred_cols = k
            .mean_axis(Axis(1))
            .expect("non-empty")
            .insert_axis(Axis(1));
        Ok(&k - &self.fit_rows - &pred_cols + self.fit_all)
    }
}

// ============================================================
// Random projections
// ============================================================

/// Smallest dimension that preserves pairwise distances of `n_samples`
/// points within a factor `1 ± eps` (sklearn's
/// `johnson_lindenstrauss_min_dim`).
pub fn johnson_lindenstrauss_min_dim(n_samples: usize, eps: f64) -> Result<usize> {
    if !(eps > 0.0 && eps < 1.0) {
        return Err(RustMlError::InvalidParameter(format!(
            "eps must be in (0, 1), got {eps}"
        )));
    }
    if n_samples == 0 {
        return Err(RustMlError::InvalidParameter(
            "n_samples must be positive".into(),
        ));
    }
    let denominator = eps * eps / 2.0 - eps.powi(3) / 3.0;
    Ok((4.0 * (n_samples as f64).ln() / denominator) as usize)
}

/// Target dimension of a random projection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProjectionComponents {
    /// [`johnson_lindenstrauss_min_dim`] of the training rows at `eps`.
    Auto {
        eps: f64,
    },
    Fixed(usize),
}

impl Default for ProjectionComponents {
    fn default() -> Self {
        ProjectionComponents::Auto { eps: 0.1 }
    }
}

fn n_components(c: ProjectionComponents, n: usize, p: usize) -> Result<usize> {
    let k = match c {
        ProjectionComponents::Fixed(k) => k,
        ProjectionComponents::Auto { eps } => {
            let k = johnson_lindenstrauss_min_dim(n, eps)?;
            if k > p {
                return Err(RustMlError::InvalidParameter(format!(
                    "eps = {eps} and n_samples = {n} lead to a target dimension of {k}, \
                     which is larger than the {p} features"
                )));
            }
            k
        }
    };
    if k == 0 {
        return Err(RustMlError::InvalidParameter(
            "n_components must be at least 1".into(),
        ));
    }
    Ok(k)
}

/// Projection onto `N(0, 1 / n_components)` random directions (sklearn's
/// `GaussianRandomProjection`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GaussianRandomProjection {
    pub n_components: ProjectionComponents,
    pub seed: u64,
}

/// Sparse projection with entries `±√(1 / (density · n_components))` and
/// zeros (Achlioptas / Li et al.; sklearn's `SparseRandomProjection`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SparseRandomProjection {
    pub n_components: ProjectionComponents,
    /// Fraction of non-zero entries; `None` is `1 / √n_features`.
    pub density: Option<f64>,
    pub seed: u64,
}

/// A fitted random projection: `x ↦ x · componentsᵀ`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedRandomProjection {
    components: Array2<f64>,
}

impl FittedRandomProjection {
    /// `n_components × n_features`.
    pub fn components(&self) -> &Array2<f64> {
        &self.components
    }
}

impl FitUnsupervised<f64> for GaussianRandomProjection {
    type Fitted = FittedRandomProjection;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedRandomProjection> {
        check_x(x)?;
        let (n, p) = x.dim();
        let k = n_components(self.n_components, n, p)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let sd = 1.0 / (k as f64).sqrt();
        let components = Array2::from_shape_simple_fn((k, p), || sd * normal(&mut rng));
        Ok(FittedRandomProjection { components })
    }
}

impl FitUnsupervised<f64> for SparseRandomProjection {
    type Fitted = FittedRandomProjection;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedRandomProjection> {
        check_x(x)?;
        let (n, p) = x.dim();
        let k = n_components(self.n_components, n, p)?;
        let density = self.density.unwrap_or(1.0 / (p as f64).sqrt());
        if !(density > 0.0 && density <= 1.0) {
            return Err(RustMlError::InvalidParameter(format!(
                "density must be in (0, 1], got {density}"
            )));
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let v = (1.0 / density).sqrt() / (k as f64).sqrt();
        let components = Array2::from_shape_simple_fn((k, p), || {
            let u: f64 = rng.gen();
            if u < density / 2.0 {
                -v
            } else if u < density {
                v
            } else {
                0.0
            }
        });
        Ok(FittedRandomProjection { components })
    }
}

impl Transform<f64> for FittedRandomProjection {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        check_cols(self.components.ncols(), x)?;
        Ok(x.dot(&self.components.t()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, p: usize) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(9);
        Array2::from_shape_simple_fn((n, p), || normal(&mut rng))
    }

    fn sq_dist(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
    }

    #[test]
    fn rbf_features_approximate_the_kernel() {
        let x = data(30, 4);
        let f = RbfSampler {
            gamma: Some(0.2),
            n_components: 20000,
            seed: 1,
        }
        .fit(&x)
        .unwrap();
        let z = f.transform(&x).unwrap();
        let approx = z.dot(&z.t());
        let mut worst = 0.0f64;
        for i in 0..30 {
            for j in 0..30 {
                let exact = (-0.2 * sq_dist(x.row(i), x.row(j))).exp();
                worst = worst.max((approx[[i, j]] - exact).abs());
            }
        }
        assert!(worst < 0.05, "max error {worst}");
    }

    #[test]
    fn nystroem_with_all_rows_reproduces_the_kernel() {
        let x = data(25, 3);
        let kernel = NystroemKernel::Rbf { gamma: Some(0.5) };
        let f = Nystroem {
            kernel,
            n_components: 100,
            seed: 0,
        }
        .fit(&x)
        .unwrap();
        let z = f.transform(&x).unwrap();
        let exact = kernel.matrix(x.view(), x.view());
        let d = (&z.dot(&z.t()) - &exact)
            .mapv(f64::abs)
            .fold(0.0f64, |m, &v| m.max(v));
        assert!(d < 1e-8, "{d}");
    }

    #[test]
    fn random_projections_preserve_distances() {
        let x = data(40, 2000);
        for f in [
            GaussianRandomProjection {
                n_components: ProjectionComponents::Fixed(800),
                seed: 2,
            }
            .fit(&x)
            .unwrap(),
            SparseRandomProjection {
                n_components: ProjectionComponents::Fixed(800),
                seed: 2,
                ..Default::default()
            }
            .fit(&x)
            .unwrap(),
        ] {
            let z = f.transform(&x).unwrap();
            for (i, j) in [(0, 1), (2, 7), (10, 30)] {
                let ratio = sq_dist(z.row(i), z.row(j)) / sq_dist(x.row(i), x.row(j));
                assert!((ratio - 1.0).abs() < 0.2, "{ratio}");
            }
        }
        assert_eq!(johnson_lindenstrauss_min_dim(1_000_000, 0.5).unwrap(), 663);
        assert!(GaussianRandomProjection::default()
            .fit(&data(1000, 50))
            .is_err());
    }
}
