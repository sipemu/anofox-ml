//! Gaussian-process kernels as log-space hyperparameter vectors with bounds
//! and analytic gradients (mirrors `sklearn.gaussian_process.kernels`).
//!
//! Every kernel exposes its hyperparameters in a fixed order (see the
//! variant docs). [`GpKernel::theta`] returns the natural logarithm of the
//! *free* (non-fixed) hyperparameters and [`GpKernel::theta_bounds`] their
//! log-space bounds — exactly sklearn's `kernel.theta` / `kernel.bounds`.
//! [`GpKernel::gram_with_gradient`] returns `K(X, X)` together with
//! `∂K/∂θ_k` for every free `θ_k`.

use anofox_ml_core::{Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};

/// Default hyperparameter bounds, `(1e-5, 1e5)` — sklearn's default for
/// every kernel hyperparameter.
pub const DEFAULT_BOUNDS: ParamBounds = ParamBounds::Bounded {
    lower: 1e-5,
    upper: 1e5,
};

/// Bounds of a single (linear-scale, positive) hyperparameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamBounds {
    /// Excluded from hyperparameter optimisation (sklearn `"fixed"`).
    Fixed,
    /// Optimised within `[lower, upper]` (linear scale, `0 < lower <= upper`).
    Bounded { lower: f64, upper: f64 },
}

impl Default for ParamBounds {
    fn default() -> Self {
        DEFAULT_BOUNDS
    }
}

/// Composable covariance function.
///
/// Hyperparameter order (used by [`GpKernel::params`],
/// [`GpKernel::param_bounds`] and [`GpKernel::theta`]):
///
/// | variant | parameters |
/// |---|---|
/// | `Rbf` | `[length_scale, signal_var]` |
/// | `RbfArd` | `[length_scales[0..d], signal_var]` |
/// | `Matern` | `[length_scale, signal_var]` |
/// | `MaternArd` | `[length_scales[0..d], signal_var]` |
/// | `RationalQuadratic` | `[length_scale, signal_var, alpha]` |
/// | `White` | `[noise_level]` |
/// | `Constant` | `[value]` |
/// | `Sum(a, b)` / `Product(a, b)` | params of `a`, then params of `b` |
/// | `Bounded { kernel, .. }` | params of `kernel` |
///
/// Every parameter defaults to bounds `(1e-5, 1e5)`; wrap a kernel with
/// [`GpKernel::fixed`], [`GpKernel::with_bounds`] or
/// [`GpKernel::with_param_bounds`] to change that.
#[derive(Debug, Clone, PartialEq)]
pub enum GpKernel {
    /// Squared exponential `σ² exp(-||x-x'||² / (2ℓ²))`
    /// (sklearn `ConstantKernel(σ²) * RBF(ℓ)`).
    Rbf {
        length_scale: f64,
        signal_var: f64,
    },
    /// Anisotropic (ARD) squared exponential with one length scale per
    /// feature: `σ² exp(-½ Σ_d (x_d-x'_d)² / ℓ_d²)`.
    RbfArd {
        length_scales: Vec<f64>,
        signal_var: f64,
    },
    /// Matern kernel; `nu ∈ {0.5, 1.5, 2.5, ∞}` (closed forms). Other
    /// values are rejected at fit time.
    Matern {
        length_scale: f64,
        signal_var: f64,
        nu: f64,
    },
    /// Anisotropic (ARD) Matern kernel, `nu ∈ {0.5, 1.5, 2.5, ∞}`.
    MaternArd {
        length_scales: Vec<f64>,
        signal_var: f64,
        nu: f64,
    },
    /// `σ² (1 + ||x-x'||² / (2αℓ²))^(-α)`.
    RationalQuadratic {
        length_scale: f64,
        signal_var: f64,
        alpha: f64,
    },
    /// Adds `noise_level` to the diagonal of `K(X, X)`. Like sklearn's
    /// `WhiteKernel`, the cross-covariance `K(X, Y)` between two different
    /// point sets is zero — even for coinciding points.
    White {
        noise_level: f64,
    },
    /// `σ²` everywhere.
    Constant {
        value: f64,
    },
    Sum(Box<GpKernel>, Box<GpKernel>),
    Product(Box<GpKernel>, Box<GpKernel>),
    /// Overrides the hyperparameter bounds of `kernel` (one entry per
    /// parameter of `kernel`, see the table above). An outer `Bounded`
    /// wins over inner ones.
    Bounded {
        kernel: Box<GpKernel>,
        bounds: Vec<ParamBounds>,
    },
}

fn matern_kind(nu: f64) -> Option<u8> {
    if nu.is_infinite() && nu > 0.0 {
        Some(3)
    } else if (nu - 0.5).abs() < 1e-9 {
        Some(0)
    } else if (nu - 1.5).abs() < 1e-9 {
        Some(1)
    } else if (nu - 2.5).abs() < 1e-9 {
        Some(2)
    } else {
        None
    }
}

/// Unit-variance Matern value at distance `r` and `g(r)` such that
/// `∂k/∂log ℓ_d = g(r) · D_d` with `D_d = (Δ_d/ℓ_d)²` (isotropic:
/// `D = r²`).
#[inline]
fn matern_eval(kind: u8, r: f64) -> (f64, f64) {
    match kind {
        0 => {
            let e = (-r).exp();
            (e, if r > 0.0 { e / r } else { 0.0 })
        }
        1 => {
            let s = 3.0_f64.sqrt() * r;
            let e = (-s).exp();
            ((1.0 + s) * e, 3.0 * e)
        }
        2 => {
            let s = 5.0_f64.sqrt() * r;
            let e = (-s).exp();
            ((1.0 + s + s * s / 3.0) * e, 5.0 / 3.0 * (1.0 + s) * e)
        }
        _ => {
            let e = (-0.5 * r * r).exp();
            (e, e)
        }
    }
}

/// Squared (scaled) Euclidean distances via GEMM. `b = None` → `a` vs `a`
/// with an exactly-zero diagonal.
fn sq_dists(a: &ArrayView2<f64>, b: Option<&ArrayView2<f64>>) -> Array2<f64> {
    let na: Array1<f64> = a.map_axis(Axis(1), |r| r.dot(&r));
    match b {
        None => {
            let mut d = a.dot(&a.t());
            let n = d.nrows();
            for i in 0..n {
                for j in 0..n {
                    let v = (na[i] + na[j] - 2.0 * d[[i, j]]).max(0.0);
                    d[[i, j]] = v;
                }
            }
            for i in 0..n {
                d[[i, i]] = 0.0;
                for j in 0..i {
                    let v = 0.5 * (d[[i, j]] + d[[j, i]]);
                    d[[i, j]] = v;
                    d[[j, i]] = v;
                }
            }
            d
        }
        Some(b) => {
            let nb: Array1<f64> = b.map_axis(Axis(1), |r| r.dot(&r));
            let mut d = a.dot(&b.t());
            Zip::indexed(&mut d).for_each(|(i, j), v| {
                *v = (na[i] + nb[j] - 2.0 * *v).max(0.0);
            });
            d
        }
    }
}

fn scale_cols(x: &ArrayView2<f64>, ls: &[f64]) -> Array2<f64> {
    let mut out = x.to_owned();
    for (k, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
        let l = if ls.len() == 1 { ls[0] } else { ls[k] };
        col.mapv_inplace(|v| v / l);
    }
    out
}

/// Per-dimension scaled squared differences `((a_ik - b_jk)/ℓ_k)²`.
fn per_dim_sq(a: &ArrayView2<f64>, b: &ArrayView2<f64>, ls: &[f64], k: usize) -> Array2<f64> {
    let ca = a.column(k);
    let cb = b.column(k);
    let l2 = ls[k] * ls[k];
    Array2::from_shape_fn((a.nrows(), b.nrows()), |(i, j)| {
        let d = ca[i] - cb[j];
        d * d / l2
    })
}

impl GpKernel {
    /// `self * other`.
    pub fn product(self, other: GpKernel) -> GpKernel {
        GpKernel::Product(Box::new(self), Box::new(other))
    }

    /// `self + other`.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: GpKernel) -> GpKernel {
        GpKernel::Sum(Box::new(self), Box::new(other))
    }

    /// Exclude every hyperparameter of this kernel from optimisation.
    pub fn fixed(self) -> GpKernel {
        let n = self.n_params();
        GpKernel::Bounded {
            kernel: Box::new(self),
            bounds: vec![ParamBounds::Fixed; n],
        }
    }

    /// Use bounds `[lower, upper]` for every hyperparameter of this kernel.
    pub fn with_bounds(self, lower: f64, upper: f64) -> GpKernel {
        let n = self.n_params();
        GpKernel::Bounded {
            kernel: Box::new(self),
            bounds: vec![ParamBounds::Bounded { lower, upper }; n],
        }
    }

    /// Per-parameter bounds (order as in the type-level table).
    pub fn with_param_bounds(self, bounds: Vec<ParamBounds>) -> GpKernel {
        GpKernel::Bounded {
            kernel: Box::new(self),
            bounds,
        }
    }

    /// Total number of hyperparameters (fixed and free).
    pub fn n_params(&self) -> usize {
        match self {
            GpKernel::Rbf { .. } | GpKernel::Matern { .. } => 2,
            GpKernel::RbfArd { length_scales, .. } | GpKernel::MaternArd { length_scales, .. } => {
                length_scales.len() + 1
            }
            GpKernel::RationalQuadratic { .. } => 3,
            GpKernel::White { .. } | GpKernel::Constant { .. } => 1,
            GpKernel::Sum(a, b) | GpKernel::Product(a, b) => a.n_params() + b.n_params(),
            GpKernel::Bounded { kernel, .. } => kernel.n_params(),
        }
    }

    /// All hyperparameter values (linear scale).
    pub fn params(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_params());
        self.collect_params(&mut out);
        out
    }

    fn collect_params(&self, out: &mut Vec<f64>) {
        match self {
            GpKernel::Rbf {
                length_scale,
                signal_var,
            }
            | GpKernel::Matern {
                length_scale,
                signal_var,
                ..
            } => out.extend([*length_scale, *signal_var]),
            GpKernel::RbfArd {
                length_scales,
                signal_var,
            }
            | GpKernel::MaternArd {
                length_scales,
                signal_var,
                ..
            } => {
                out.extend_from_slice(length_scales);
                out.push(*signal_var);
            }
            GpKernel::RationalQuadratic {
                length_scale,
                signal_var,
                alpha,
            } => out.extend([*length_scale, *signal_var, *alpha]),
            GpKernel::White { noise_level } => out.push(*noise_level),
            GpKernel::Constant { value } => out.push(*value),
            GpKernel::Sum(a, b) | GpKernel::Product(a, b) => {
                a.collect_params(out);
                b.collect_params(out);
            }
            GpKernel::Bounded { kernel, .. } => kernel.collect_params(out),
        }
    }

    /// Set all hyperparameters (linear scale); returns the number consumed.
    fn assign_params(&mut self, p: &[f64]) -> usize {
        match self {
            GpKernel::Rbf {
                length_scale,
                signal_var,
            }
            | GpKernel::Matern {
                length_scale,
                signal_var,
                ..
            } => {
                *length_scale = p[0];
                *signal_var = p[1];
                2
            }
            GpKernel::RbfArd {
                length_scales,
                signal_var,
            }
            | GpKernel::MaternArd {
                length_scales,
                signal_var,
                ..
            } => {
                let d = length_scales.len();
                length_scales.copy_from_slice(&p[..d]);
                *signal_var = p[d];
                d + 1
            }
            GpKernel::RationalQuadratic {
                length_scale,
                signal_var,
                alpha,
            } => {
                *length_scale = p[0];
                *signal_var = p[1];
                *alpha = p[2];
                3
            }
            GpKernel::White { noise_level } => {
                *noise_level = p[0];
                1
            }
            GpKernel::Constant { value } => {
                *value = p[0];
                1
            }
            GpKernel::Sum(a, b) | GpKernel::Product(a, b) => {
                let k = a.assign_params(p);
                k + b.assign_params(&p[k..])
            }
            GpKernel::Bounded { kernel, .. } => kernel.assign_params(p),
        }
    }

    /// Bounds of every hyperparameter (fixed ones included).
    pub fn param_bounds(&self) -> Vec<ParamBounds> {
        match self {
            GpKernel::Sum(a, b) | GpKernel::Product(a, b) => {
                let mut v = a.param_bounds();
                v.extend(b.param_bounds());
                v
            }
            GpKernel::Bounded { kernel, bounds } => {
                if bounds.len() == kernel.n_params() {
                    bounds.clone()
                } else {
                    // Invalid; reported by `validate`.
                    kernel.param_bounds()
                }
            }
            _ => vec![DEFAULT_BOUNDS; self.n_params()],
        }
    }

    fn free_mask(&self) -> Vec<bool> {
        self.param_bounds()
            .iter()
            .map(|b| !matches!(b, ParamBounds::Fixed))
            .collect()
    }

    /// Number of free (optimised) hyperparameters, i.e. `theta().len()`.
    pub fn n_free_params(&self) -> usize {
        self.free_mask().iter().filter(|&&f| f).count()
    }

    /// Natural log of the free hyperparameters (sklearn `kernel.theta`).
    pub fn theta(&self) -> Vec<f64> {
        self.params()
            .into_iter()
            .zip(self.free_mask())
            .filter(|(_, f)| *f)
            .map(|(p, _)| p.ln())
            .collect()
    }

    /// Log-space bounds of the free hyperparameters (sklearn `kernel.bounds`).
    pub fn theta_bounds(&self) -> Vec<(f64, f64)> {
        self.param_bounds()
            .into_iter()
            .filter_map(|b| match b {
                ParamBounds::Fixed => None,
                ParamBounds::Bounded { lower, upper } => Some((lower.ln(), upper.ln())),
            })
            .collect()
    }

    /// Copy of this kernel with the free hyperparameters set to `exp(theta)`.
    pub fn with_theta(&self, theta: &[f64]) -> Result<GpKernel> {
        let mask = self.free_mask();
        let n_free = mask.iter().filter(|&&f| f).count();
        if theta.len() != n_free {
            return Err(RustMlError::ShapeMismatch(format!(
                "theta has length {} but the kernel has {n_free} free hyperparameters",
                theta.len()
            )));
        }
        let mut p = self.params();
        let mut it = theta.iter();
        for (v, f) in p.iter_mut().zip(mask) {
            if f {
                *v = it.next().unwrap().exp();
            }
        }
        let mut k = self.clone();
        k.assign_params(&p);
        Ok(k)
    }

    /// Check parameter validity (positivity, supported Matern `nu`, ARD
    /// dimension, bounds lengths) for data with `n_features` columns.
    pub fn validate(&self, n_features: usize) -> Result<()> {
        let bad = |m: String| Err(RustMlError::InvalidParameter(m));
        match self {
            GpKernel::Matern { nu, .. } | GpKernel::MaternArd { nu, .. }
                if matern_kind(*nu).is_none() =>
            {
                return bad(format!(
                    "Matern kernel supports nu in {{0.5, 1.5, 2.5, inf}}, got {nu}"
                ));
            }
            GpKernel::RbfArd { length_scales, .. } | GpKernel::MaternArd { length_scales, .. }
                if length_scales.len() != n_features =>
            {
                return bad(format!(
                    "ARD kernel has {} length scales but X has {n_features} features",
                    length_scales.len()
                ));
            }
            GpKernel::Sum(a, b) | GpKernel::Product(a, b) => {
                a.validate(n_features)?;
                return b.validate(n_features);
            }
            GpKernel::Bounded { kernel, bounds } => {
                if bounds.len() != kernel.n_params() {
                    return bad(format!(
                        "Bounded kernel: {} bounds for {} hyperparameters",
                        bounds.len(),
                        kernel.n_params()
                    ));
                }
                for b in bounds {
                    if let ParamBounds::Bounded { lower, upper } = *b {
                        if !(lower > 0.0 && lower <= upper) {
                            return bad(format!(
                                "hyperparameter bounds must satisfy 0 < lower <= upper, got ({lower}, {upper})"
                            ));
                        }
                    }
                }
                return kernel.validate(n_features);
            }
            _ => {}
        }
        for p in self.params() {
            if !(p > 0.0 && p.is_finite()) {
                return bad(format!(
                    "kernel hyperparameters must be positive and finite, got {p}"
                ));
            }
        }
        Ok(())
    }

    /// `K(X, X)` (White noise on the diagonal).
    pub fn gram(&self, x: &Array2<f64>) -> Array2<f64> {
        self.eval(&x.view(), None, None)
    }

    /// Cross-covariance `K(A, B)` between two point sets (White ≡ 0).
    pub fn cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        self.eval(&a.view(), Some(&b.view()), None)
    }

    /// `K(X, X)` and `∂K/∂θ_k` for every free log-hyperparameter `θ_k`.
    pub fn gram_with_gradient(&self, x: &Array2<f64>) -> (Array2<f64>, Vec<Array2<f64>>) {
        let mut grads = Vec::with_capacity(self.n_params());
        let k = self.eval(&x.view(), None, Some(&mut grads));
        let grads = grads
            .into_iter()
            .zip(self.free_mask())
            .filter(|(_, f)| *f)
            .map(|(g, _)| g)
            .collect();
        (k, grads)
    }

    /// Diagonal of `K(X, X)` without forming the matrix.
    pub fn diag(&self, x: &Array2<f64>) -> Array1<f64> {
        let n = x.nrows();
        match self {
            GpKernel::Rbf { signal_var, .. }
            | GpKernel::RbfArd { signal_var, .. }
            | GpKernel::Matern { signal_var, .. }
            | GpKernel::MaternArd { signal_var, .. }
            | GpKernel::RationalQuadratic { signal_var, .. } => Array1::from_elem(n, *signal_var),
            GpKernel::White { noise_level } => Array1::from_elem(n, *noise_level),
            GpKernel::Constant { value } => Array1::from_elem(n, *value),
            GpKernel::Sum(a, b) => a.diag(x) + b.diag(x),
            GpKernel::Product(a, b) => a.diag(x) * b.diag(x),
            GpKernel::Bounded { kernel, .. } => kernel.diag(x),
        }
    }

    /// Core evaluator. `b = None` means `K(a, a)`. When `grads` is given,
    /// pushes `∂K/∂log p` for every parameter of this subtree, in order.
    fn eval(
        &self,
        a: &ArrayView2<f64>,
        b: Option<&ArrayView2<f64>>,
        grads: Option<&mut Vec<Array2<f64>>>,
    ) -> Array2<f64> {
        let nb = b.map_or(a.nrows(), |b| b.nrows());
        let shape = (a.nrows(), nb);
        match self {
            GpKernel::Rbf {
                length_scale,
                signal_var,
            } => {
                let sa = scale_cols(a, &[*length_scale]);
                let sb = b.map(|b| scale_cols(b, &[*length_scale]));
                let d = sq_dists(&sa.view(), sb.as_ref().map(|s| s.view()).as_ref());
                let k = d.mapv(|v| signal_var * (-0.5 * v).exp());
                if let Some(g) = grads {
                    g.push(&k * &d);
                    g.push(k.clone());
                }
                k
            }
            GpKernel::RbfArd {
                length_scales,
                signal_var,
            } => {
                let sa = scale_cols(a, length_scales);
                let sb = b.map(|b| scale_cols(b, length_scales));
                let d = sq_dists(&sa.view(), sb.as_ref().map(|s| s.view()).as_ref());
                let k = d.mapv(|v| signal_var * (-0.5 * v).exp());
                if let Some(g) = grads {
                    let bb: ArrayView2<f64> = match b {
                        Some(v) => v.view(),
                        None => a.view(),
                    };
                    for dim in 0..length_scales.len() {
                        g.push(&k * &per_dim_sq(a, &bb, length_scales, dim));
                    }
                    g.push(k.clone());
                }
                k
            }
            GpKernel::Matern {
                length_scale,
                signal_var,
                nu,
            } => {
                let kind = matern_kind(*nu).unwrap_or(3);
                let sa = scale_cols(a, &[*length_scale]);
                let sb = b.map(|b| scale_cols(b, &[*length_scale]));
                let d = sq_dists(&sa.view(), sb.as_ref().map(|s| s.view()).as_ref());
                let mut k = Array2::zeros(shape);
                let mut gl = grads.as_ref().map(|_| Array2::zeros(shape));
                Zip::indexed(&d).for_each(|(i, j), &d2| {
                    let (v, gr) = matern_eval(kind, d2.sqrt());
                    k[[i, j]] = signal_var * v;
                    if let Some(gl) = gl.as_mut() {
                        gl[[i, j]] = signal_var * gr * d2;
                    }
                });
                if let (Some(g), Some(gl)) = (grads, gl) {
                    g.push(gl);
                    g.push(k.clone());
                }
                k
            }
            GpKernel::MaternArd {
                length_scales,
                signal_var,
                nu,
            } => {
                let kind = matern_kind(*nu).unwrap_or(3);
                let sa = scale_cols(a, length_scales);
                let sb = b.map(|b| scale_cols(b, length_scales));
                let d = sq_dists(&sa.view(), sb.as_ref().map(|s| s.view()).as_ref());
                let mut k = Array2::zeros(shape);
                let mut gfac = Array2::zeros(shape);
                Zip::indexed(&d).for_each(|(i, j), &d2| {
                    let (v, gr) = matern_eval(kind, d2.sqrt());
                    k[[i, j]] = signal_var * v;
                    gfac[[i, j]] = signal_var * gr;
                });
                if let Some(g) = grads {
                    let bb: ArrayView2<f64> = match b {
                        Some(v) => v.view(),
                        None => a.view(),
                    };
                    for dim in 0..length_scales.len() {
                        g.push(&gfac * &per_dim_sq(a, &bb, length_scales, dim));
                    }
                    g.push(k.clone());
                }
                k
            }
            GpKernel::RationalQuadratic {
                length_scale,
                signal_var,
                alpha,
            } => {
                let sa = scale_cols(a, &[*length_scale]);
                let sb = b.map(|b| scale_cols(b, &[*length_scale]));
                // d = ||x-x'||² / ℓ²
                let d = sq_dists(&sa.view(), sb.as_ref().map(|s| s.view()).as_ref());
                let base = d.mapv(|v| 1.0 + v / (2.0 * alpha));
                let k = base.mapv(|bv| signal_var * bv.powf(-alpha));
                if let Some(g) = grads {
                    // ∂K/∂log ℓ = K · d / base
                    let mut gl = k.clone();
                    Zip::from(&mut gl)
                        .and(&d)
                        .and(&base)
                        .for_each(|gv, &dv, &bv| *gv *= dv / bv);
                    // ∂K/∂log α = K · (-α ln base + d / (2 base))
                    let mut ga = k.clone();
                    Zip::from(&mut ga)
                        .and(&d)
                        .and(&base)
                        .for_each(|gv, &dv, &bv| *gv *= -alpha * bv.ln() + dv / (2.0 * bv));
                    g.push(gl);
                    g.push(k.clone());
                    g.push(ga);
                }
                k
            }
            GpKernel::White { noise_level } => {
                let mut k = Array2::zeros(shape);
                if b.is_none() {
                    k.diag_mut().fill(*noise_level);
                }
                if let Some(g) = grads {
                    g.push(k.clone());
                }
                k
            }
            GpKernel::Constant { value } => {
                let k = Array2::from_elem(shape, *value);
                if let Some(g) = grads {
                    g.push(k.clone());
                }
                k
            }
            GpKernel::Sum(k1, k2) => match grads {
                Some(g) => {
                    let a1 = k1.eval(a, b, Some(g));
                    let a2 = k2.eval(a, b, Some(g));
                    a1 + a2
                }
                None => k1.eval(a, b, None) + k2.eval(a, b, None),
            },
            GpKernel::Product(k1, k2) => match grads {
                Some(g) => {
                    let mut g1 = Vec::new();
                    let mut g2 = Vec::new();
                    let a1 = k1.eval(a, b, Some(&mut g1));
                    let a2 = k2.eval(a, b, Some(&mut g2));
                    for mut gi in g1 {
                        gi *= &a2;
                        g.push(gi);
                    }
                    for mut gi in g2 {
                        gi *= &a1;
                        g.push(gi);
                    }
                    a1 * a2
                }
                None => k1.eval(a, b, None) * k2.eval(a, b, None),
            },
            GpKernel::Bounded { kernel, .. } => kernel.eval(a, b, grads),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Array2<f64> {
        Array2::from_shape_fn((7, 2), |(i, j)| {
            ((i * 3 + j * 5) % 7) as f64 * 0.37 - 0.4 * j as f64
        })
    }

    fn kernels() -> Vec<GpKernel> {
        vec![
            GpKernel::Rbf {
                length_scale: 0.8,
                signal_var: 1.7,
            },
            GpKernel::RbfArd {
                length_scales: vec![0.6, 1.9],
                signal_var: 0.9,
            },
            GpKernel::Matern {
                length_scale: 1.1,
                signal_var: 1.3,
                nu: 0.5,
            },
            GpKernel::Matern {
                length_scale: 1.1,
                signal_var: 1.3,
                nu: 1.5,
            },
            GpKernel::Matern {
                length_scale: 1.1,
                signal_var: 1.3,
                nu: 2.5,
            },
            GpKernel::Matern {
                length_scale: 1.1,
                signal_var: 1.3,
                nu: f64::INFINITY,
            },
            GpKernel::MaternArd {
                length_scales: vec![0.7, 1.4],
                signal_var: 1.2,
                nu: 1.5,
            },
            GpKernel::MaternArd {
                length_scales: vec![0.7, 1.4],
                signal_var: 1.2,
                nu: 0.5,
            },
            GpKernel::RationalQuadratic {
                length_scale: 0.9,
                signal_var: 1.1,
                alpha: 0.7,
            },
            GpKernel::Constant { value: 2.0 }
                .product(GpKernel::Rbf {
                    length_scale: 1.2,
                    signal_var: 1.0,
                })
                .add(GpKernel::White { noise_level: 0.3 }),
            GpKernel::Rbf {
                length_scale: 1.2,
                signal_var: 1.0,
            }
            .with_param_bounds(vec![ParamBounds::default(), ParamBounds::Fixed])
            .add(GpKernel::White { noise_level: 0.3 }.fixed()),
        ]
    }

    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let x = data();
        for kern in kernels() {
            kern.validate(2).unwrap();
            let theta = kern.theta();
            let (k0, grads) = kern.gram_with_gradient(&x);
            assert_eq!(grads.len(), theta.len());
            assert!((&k0 - &kern.gram(&x)).iter().all(|v| v.abs() < 1e-14));
            let h = 1e-6;
            for p in 0..theta.len() {
                let mut tp = theta.clone();
                tp[p] += h;
                let mut tm = theta.clone();
                tm[p] -= h;
                let kp = kern.with_theta(&tp).unwrap().gram(&x);
                let km = kern.with_theta(&tm).unwrap().gram(&x);
                let fd = (kp - km) / (2.0 * h);
                for (a, b) in fd.iter().zip(grads[p].iter()) {
                    assert!(
                        (a - b).abs() < 1e-6,
                        "{kern:?} param {p}: fd {a} vs analytic {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn cross_matches_gram_and_diag() {
        let x = data();
        for kern in kernels() {
            let without_white = kern.cross(&x, &x);
            let k = kern.gram(&x);
            let d = kern.diag(&x);
            for i in 0..x.nrows() {
                assert!((k[[i, i]] - d[i]).abs() < 1e-12);
                for j in 0..x.nrows() {
                    if i != j {
                        assert!((k[[i, j]] - without_white[[i, j]]).abs() < 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn white_kernel_is_zero_for_cross_covariance() {
        let x = data();
        let w = GpKernel::White { noise_level: 0.5 };
        assert!(w.cross(&x, &x).iter().all(|&v| v == 0.0));
        assert_eq!(w.gram(&x)[[3, 3]], 0.5);
        assert_eq!(w.diag(&x)[0], 0.5);
    }

    #[test]
    fn unsupported_matern_nu_is_an_error() {
        let k = GpKernel::Matern {
            length_scale: 1.0,
            signal_var: 1.0,
            nu: 0.7,
        };
        assert!(k.validate(1).is_err());
        assert!(GpKernel::Matern {
            length_scale: 1.0,
            signal_var: 1.0,
            nu: f64::INFINITY,
        }
        .validate(1)
        .is_ok());
    }

    #[test]
    fn theta_roundtrip_and_bounds() {
        let k = GpKernel::Rbf {
            length_scale: 2.0,
            signal_var: 3.0,
        }
        .with_param_bounds(vec![
            ParamBounds::Bounded {
                lower: 0.1,
                upper: 10.0,
            },
            ParamBounds::Fixed,
        ])
        .add(GpKernel::White { noise_level: 0.5 });
        assert_eq!(k.n_params(), 3);
        assert_eq!(k.n_free_params(), 2);
        let th = k.theta();
        assert!((th[0] - 2.0_f64.ln()).abs() < 1e-15);
        assert!((th[1] - 0.5_f64.ln()).abs() < 1e-15);
        let b = k.theta_bounds();
        assert!((b[0].0 - 0.1_f64.ln()).abs() < 1e-15);
        assert!((b[1].1 - 1e5_f64.ln()).abs() < 1e-12);
        let k2 = k.with_theta(&[0.0, 0.0]).unwrap();
        assert_eq!(k2.params(), vec![1.0, 3.0, 1.0]);
    }

    #[test]
    fn ard_with_equal_scales_equals_isotropic() {
        let x = data();
        let iso = GpKernel::Matern {
            length_scale: 0.9,
            signal_var: 1.4,
            nu: 2.5,
        };
        let ard = GpKernel::MaternArd {
            length_scales: vec![0.9, 0.9],
            signal_var: 1.4,
            nu: 2.5,
        };
        let d = iso.gram(&x) - ard.gram(&x);
        assert!(d.iter().all(|v| v.abs() < 1e-12));
    }
}
