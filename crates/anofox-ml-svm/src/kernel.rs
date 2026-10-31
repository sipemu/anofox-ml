use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::{Array2, ArrayView1};

/// Kernel coefficient `γ` for the RBF, polynomial and sigmoid kernels.
///
/// Mirrors sklearn's `gamma` parameter: `'scale'` (the default) uses
/// `1 / (n_features · X.var())`, `'auto'` uses `1 / n_features`, and a value
/// is used as given. `Scale` / `Auto` are resolved from the training data at
/// fit time; fitted models always store a [`Gamma::Value`].
///
/// Serialises as a number for `Value` and as the strings `"scale"` /
/// `"auto"` otherwise, so models stored with the former bare-`f64` gamma
/// still deserialise.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Gamma {
    /// `1 / (n_features · Var(X))` where the variance is over all entries of
    /// `X` (`1.0` if that variance is zero).
    #[default]
    Scale,
    /// `1 / n_features`.
    Auto,
    /// A fixed value (must be `> 0`).
    Value(f64),
}

impl From<f64> for Gamma {
    fn from(v: f64) -> Self {
        Gamma::Value(v)
    }
}

impl Gamma {
    fn one() -> Self {
        Gamma::Value(1.0)
    }

    /// The fixed value, if already resolved.
    pub fn value(&self) -> Option<f64> {
        match self {
            Gamma::Value(v) => Some(*v),
            _ => None,
        }
    }

    /// Resolve `Scale` / `Auto` against training data.
    pub fn resolve<F: Float>(&self, x: &Array2<F>) -> f64 {
        let n_features = x.ncols().max(1) as f64;
        match self {
            Gamma::Value(v) => *v,
            Gamma::Auto => 1.0 / n_features,
            Gamma::Scale => {
                let n = x.len();
                if n == 0 {
                    return 1.0;
                }
                let nf = n as f64;
                let mean = x.iter().map(|v| v.to_f64().unwrap()).sum::<f64>() / nf;
                let var = x
                    .iter()
                    .map(|v| {
                        let d = v.to_f64().unwrap() - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / nf;
                if var != 0.0 {
                    1.0 / (n_features * var)
                } else {
                    1.0
                }
            }
        }
    }

    fn validate(&self) -> Result<()> {
        if let Gamma::Value(v) = self {
            if !(v.is_finite() && *v > 0.0) {
                return Err(RustMlError::InvalidParameter(format!(
                    "gamma must be positive, got {v}"
                )));
            }
        }
        Ok(())
    }
}

/// Representation in non-self-describing formats (bincode), which cannot
/// decode the number-or-string form.
#[derive(serde::Serialize, serde::Deserialize)]
enum GammaBin {
    Value(f64),
    Scale,
    Auto,
}

impl serde::Serialize for Gamma {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        if !s.is_human_readable() {
            let bin = match self {
                Gamma::Value(v) => GammaBin::Value(*v),
                Gamma::Scale => GammaBin::Scale,
                Gamma::Auto => GammaBin::Auto,
            };
            return serde::Serialize::serialize(&bin, s);
        }
        match self {
            Gamma::Value(v) => s.serialize_f64(*v),
            Gamma::Scale => s.serialize_str("scale"),
            Gamma::Auto => s.serialize_str("auto"),
        }
    }
}

impl<'de> serde::Deserialize<'de> for Gamma {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        if !d.is_human_readable() {
            return Ok(match <GammaBin as serde::Deserialize>::deserialize(d)? {
                GammaBin::Value(v) => Gamma::Value(v),
                GammaBin::Scale => Gamma::Scale,
                GammaBin::Auto => Gamma::Auto,
            });
        }
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Num(f64),
            Str(String),
        }
        match <Repr as serde::Deserialize>::deserialize(d)? {
            Repr::Num(v) => Ok(Gamma::Value(v)),
            Repr::Str(s) => match s.as_str() {
                "scale" => Ok(Gamma::Scale),
                "auto" => Ok(Gamma::Auto),
                other => Err(serde::de::Error::custom(format!(
                    "unknown gamma '{other}', expected a number, 'scale' or 'auto'"
                ))),
            },
        }
    }
}

/// Kernel functions for the Support Vector Machines.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SvmKernel {
    /// Linear kernel: K(x, y) = x · y
    Linear,
    /// Radial Basis Function kernel: K(x, y) = exp(-γ ‖x − y‖²)
    Rbf {
        /// Controls how far the influence of a single training example reaches.
        gamma: Gamma,
    },
    /// Polynomial kernel: K(x, y) = (γ x · y + coef0)^degree
    Polynomial {
        /// Degree of the polynomial kernel function.
        degree: usize,
        /// Kernel coefficient (models serialised before it existed load with
        /// `γ = 1`, the old behaviour).
        #[serde(default = "Gamma::one")]
        gamma: Gamma,
        /// Independent term in the polynomial kernel function.
        coef0: f64,
    },
    /// Sigmoid kernel: K(x, y) = tanh(γ x · y + coef0)
    Sigmoid {
        /// Kernel coefficient.
        gamma: Gamma,
        /// Independent term.
        coef0: f64,
    },
    /// Precomputed kernel: `X` passed to `fit` is the `(n, n)` Gram matrix
    /// of the training samples and `X` passed to `predict` /
    /// `decision_function` is the `(m, n_train)` kernel between test and
    /// training samples. Only supported by [`Svc`](crate::Svc) and
    /// [`NuSvc`](crate::NuSvc).
    Precomputed,
}

impl Default for SvmKernel {
    fn default() -> Self {
        SvmKernel::Rbf {
            gamma: Gamma::Scale,
        }
    }
}

impl SvmKernel {
    /// RBF kernel with the given gamma.
    pub fn rbf(gamma: impl Into<Gamma>) -> Self {
        SvmKernel::Rbf {
            gamma: gamma.into(),
        }
    }

    /// Polynomial kernel `(γ x·y + coef0)^degree`.
    pub fn poly(degree: usize, gamma: impl Into<Gamma>, coef0: f64) -> Self {
        SvmKernel::Polynomial {
            degree,
            gamma: gamma.into(),
            coef0,
        }
    }

    /// Sigmoid kernel `tanh(γ x·y + coef0)`.
    pub fn sigmoid(gamma: impl Into<Gamma>, coef0: f64) -> Self {
        SvmKernel::Sigmoid {
            gamma: gamma.into(),
            coef0,
        }
    }

    /// Whether this is the precomputed kernel.
    pub fn is_precomputed(&self) -> bool {
        matches!(self, SvmKernel::Precomputed)
    }

    /// Validate the kernel parameters. `allow_precomputed` is `false` for
    /// estimators that do not support a precomputed Gram matrix.
    pub fn validate(&self, allow_precomputed: bool) -> Result<()> {
        match self {
            SvmKernel::Linear => Ok(()),
            SvmKernel::Rbf { gamma } | SvmKernel::Sigmoid { gamma, .. } => gamma.validate(),
            SvmKernel::Polynomial { degree, gamma, .. } => {
                if *degree == 0 {
                    return Err(RustMlError::InvalidParameter(
                        "degree must be at least 1 for polynomial kernel".into(),
                    ));
                }
                gamma.validate()
            }
            SvmKernel::Precomputed if allow_precomputed => Ok(()),
            SvmKernel::Precomputed => Err(RustMlError::InvalidParameter(
                "the precomputed kernel is not supported by this estimator".into(),
            )),
        }
    }

    /// Return a copy with `Gamma::Scale` / `Gamma::Auto` replaced by their
    /// values for the training data `x` (sklearn resolves `gamma` at fit).
    pub fn resolve<F: Float>(&self, x: &Array2<F>) -> SvmKernel {
        match self {
            SvmKernel::Rbf { gamma } => SvmKernel::Rbf {
                gamma: Gamma::Value(gamma.resolve(x)),
            },
            SvmKernel::Polynomial {
                degree,
                gamma,
                coef0,
            } => SvmKernel::Polynomial {
                degree: *degree,
                gamma: Gamma::Value(gamma.resolve(x)),
                coef0: *coef0,
            },
            SvmKernel::Sigmoid { gamma, coef0 } => SvmKernel::Sigmoid {
                gamma: Gamma::Value(gamma.resolve(x)),
                coef0: *coef0,
            },
            other => other.clone(),
        }
    }

    /// Compute the kernel value between two feature vectors.
    ///
    /// # Panics
    ///
    /// Panics for [`SvmKernel::Precomputed`] and for a `Gamma::Scale` that
    /// has not been [`resolve`](Self::resolve)d (estimators resolve gamma at
    /// fit time). `Gamma::Auto` is resolved on the fly as `1 / len(x)`.
    pub fn compute<F: Float>(&self, x: &ArrayView1<F>, y: &ArrayView1<F>) -> F {
        let g = |gamma: &Gamma| -> F {
            let v = match gamma {
                Gamma::Value(v) => *v,
                Gamma::Auto => 1.0 / x.len().max(1) as f64,
                Gamma::Scale => panic!(
                    "SvmKernel::compute called with unresolved gamma='scale'; \
                     call SvmKernel::resolve(&x_train) first"
                ),
            };
            F::from_f64(v).unwrap()
        };
        match self {
            SvmKernel::Linear => x.dot(y),
            SvmKernel::Rbf { gamma } => {
                let gamma_f = g(gamma);
                let sq_dist = x.iter().zip(y.iter()).fold(F::zero(), |acc, (&a, &b)| {
                    let d = a - b;
                    acc + d * d
                });
                (-gamma_f * sq_dist).exp()
            }
            SvmKernel::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                let base = g(gamma) * x.dot(y) + F::from_f64(*coef0).unwrap();
                let mut result = F::one();
                for _ in 0..*degree {
                    result *= base;
                }
                result
            }
            SvmKernel::Sigmoid { gamma, coef0 } => {
                (g(gamma) * x.dot(y) + F::from_f64(*coef0).unwrap()).tanh()
            }
            SvmKernel::Precomputed => panic!(
                "SvmKernel::compute is undefined for the precomputed kernel; \
                 pass the Gram matrix as X instead"
            ),
        }
    }
}

/// Kernel matrix `K[i, j] = k(a_i, b_j)`. For the precomputed kernel `a`
/// already *is* the kernel matrix and is returned as-is.
pub(crate) fn kernel_matrix<F: Float>(
    a: &Array2<F>,
    b: &Array2<F>,
    kernel: &SvmKernel,
) -> Array2<F> {
    if kernel.is_precomputed() {
        return a.to_owned();
    }
    kernel.matrix(a, b)
}

impl SvmKernel {
    /// Kernel matrix `K[i, j] = k(a_i, b_j)` built from one GEMM `A Bᵀ`
    /// (RBF via `‖a‖² + ‖b‖² − 2 a·b`, clamped at 0, as sklearn does).
    ///
    /// # Panics
    ///
    /// Same conditions as [`compute`](Self::compute).
    pub fn matrix<F: Float>(&self, a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
        self.matrix_impl(a, b, false)
    }

    /// Symmetric Gram matrix `k(X, X)`. The RBF diagonal is exactly 1.
    ///
    /// # Panics
    ///
    /// Same conditions as [`compute`](Self::compute).
    pub fn gram<F: Float>(&self, x: &Array2<F>) -> Array2<F> {
        self.matrix_impl(x, x, true)
    }

    fn matrix_impl<F: Float>(&self, a: &Array2<F>, b: &Array2<F>, same: bool) -> Array2<F> {
        let g = |gamma: &Gamma| -> F {
            let v = match gamma {
                Gamma::Value(v) => *v,
                Gamma::Auto => 1.0 / a.ncols().max(1) as f64,
                Gamma::Scale => panic!(
                    "SvmKernel::matrix called with unresolved gamma='scale'; \
                     call SvmKernel::resolve(&x_train) first"
                ),
            };
            F::from_f64(v).unwrap()
        };
        if self.is_precomputed() {
            panic!(
                "SvmKernel::matrix is undefined for the precomputed kernel; \
                 pass the Gram matrix as X instead"
            );
        }
        let mut k = a.dot(&b.t());
        match self {
            SvmKernel::Linear | SvmKernel::Precomputed => {}
            SvmKernel::Rbf { gamma } => {
                let gamma = g(gamma);
                let na: Vec<F> = a.rows().into_iter().map(|r| r.dot(&r)).collect();
                let nb: Vec<F> = if same {
                    na.clone()
                } else {
                    b.rows().into_iter().map(|r| r.dot(&r)).collect()
                };
                let two = F::from_f64(2.0).unwrap();
                for ((i, j), v) in k.indexed_iter_mut() {
                    let d = if same && i == j {
                        F::zero()
                    } else {
                        (na[i] + nb[j] - two * *v).max(F::zero())
                    };
                    *v = (-gamma * d).exp();
                }
            }
            SvmKernel::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                let (gamma, coef0) = (g(gamma), F::from_f64(*coef0).unwrap());
                let degree = i32::try_from(*degree).unwrap_or(i32::MAX);
                k.mapv_inplace(|v| (gamma * v + coef0).powi(degree));
            }
            SvmKernel::Sigmoid { gamma, coef0 } => {
                let (gamma, coef0) = (g(gamma), F::from_f64(*coef0).unwrap());
                k.mapv_inplace(|v| (gamma * v + coef0).tanh());
            }
        }
        k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn gamma_roundtrips_through_bincode_and_json() {
        for g in [Gamma::Value(0.25), Gamma::Scale, Gamma::Auto] {
            let k = SvmKernel::Rbf { gamma: g };
            let bytes = bincode::serialize(&k).unwrap();
            assert_eq!(bincode::deserialize::<SvmKernel>(&bytes).unwrap(), k);
            let json = serde_json::to_string(&k).unwrap();
            assert_eq!(serde_json::from_str::<SvmKernel>(&json).unwrap(), k);
        }
    }

    #[test]
    fn test_matrix_matches_pointwise() {
        let a = array![[0.3, -1.2, 2.0], [1.5, 0.1, -0.7], [0.0, 0.4, 0.9]];
        let b = array![[1.0, 1.0, -1.0], [0.2, -0.3, 0.5]];
        for kernel in [
            SvmKernel::Linear,
            SvmKernel::rbf(0.4),
            SvmKernel::Rbf { gamma: Gamma::Auto },
            SvmKernel::poly(3, 0.5, 1.0),
            SvmKernel::sigmoid(0.2, -0.1),
        ] {
            let k = kernel.matrix(&a, &b);
            let g = kernel.gram(&a);
            for i in 0..3 {
                for j in 0..2 {
                    let e: f64 = kernel.compute(&a.row(i), &b.row(j));
                    assert_abs_diff_eq!(k[[i, j]], e, epsilon = 1e-12);
                }
                for j in 0..3 {
                    let e: f64 = kernel.compute(&a.row(i), &a.row(j));
                    assert_abs_diff_eq!(g[[i, j]], e, epsilon = 1e-12);
                    assert_eq!(g[[i, j]], g[[j, i]]);
                }
            }
        }
    }

    #[test]
    fn test_linear_kernel() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];
        let kernel = SvmKernel::Linear;
        let result: f64 = kernel.compute(&x.view(), &y.view());
        // 1*4 + 2*5 + 3*6 = 32
        assert_abs_diff_eq!(result, 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rbf_kernel_same_point() {
        let x = array![1.0, 2.0];
        let kernel = SvmKernel::rbf(0.5);
        let result: f64 = kernel.compute(&x.view(), &x.view());
        // exp(-0.5 * 0) = 1.0
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rbf_kernel_different_points() {
        let x = array![0.0, 0.0];
        let y = array![1.0, 1.0];
        let kernel = SvmKernel::rbf(0.5);
        let result: f64 = kernel.compute(&x.view(), &y.view());
        // exp(-0.5 * 2) = exp(-1)
        assert_abs_diff_eq!(result, (-1.0_f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_kernel() {
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];
        let kernel = SvmKernel::poly(2, 1.0, 1.0);
        let result: f64 = kernel.compute(&x.view(), &y.view());
        // (1*3 + 2*4 + 1)^2 = (12)^2 = 144
        assert_abs_diff_eq!(result, 144.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_kernel_gamma() {
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];
        let kernel = SvmKernel::poly(3, 0.5, 2.0);
        let result: f64 = kernel.compute(&x.view(), &y.view());
        // (0.5 * 11 + 2)^3 = 7.5^3
        assert_abs_diff_eq!(result, 421.875, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_kernel_degree_one() {
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];
        let kernel = SvmKernel::poly(1, 1.0, 0.0);
        let result: f64 = kernel.compute(&x.view(), &y.view());
        // (1*3 + 2*4 + 0)^1 = 11
        assert_abs_diff_eq!(result, 11.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel() {
        let x = array![1.0, 2.0];
        let y = array![0.5, -1.0];
        let kernel = SvmKernel::sigmoid(0.1, 0.3);
        let result: f64 = kernel.compute(&x.view(), &y.view());
        assert_abs_diff_eq!(result, (0.1_f64 * -1.5 + 0.3).tanh(), epsilon = 1e-14);
    }

    #[test]
    fn test_linear_kernel_f32() {
        let x = array![1.0_f32, 2.0, 3.0];
        let y = array![4.0_f32, 5.0, 6.0];
        let kernel = SvmKernel::Linear;
        let result: f32 = kernel.compute(&x.view(), &y.view());
        assert_abs_diff_eq!(result, 32.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_rbf_kernel_f32() {
        let x = array![0.0_f32, 0.0];
        let y = array![1.0_f32, 1.0];
        let kernel = SvmKernel::rbf(0.5);
        let result: f32 = kernel.compute(&x.view(), &y.view());
        assert_abs_diff_eq!(result, (-1.0_f32).exp(), epsilon = 1e-5);
    }

    #[test]
    fn gamma_scale_and_auto_resolution() {
        // sklearn: 1 / (n_features * X.var()) with the variance over all entries.
        let x = array![[0.0, 1.0], [2.0, 3.0], [4.0, 6.0]];
        let mean = 16.0 / 6.0;
        let var = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
            .iter()
            .map(|v: &f64| (v - mean).powi(2))
            .sum::<f64>()
            / 6.0;
        assert_abs_diff_eq!(Gamma::Scale.resolve(&x), 1.0 / (2.0 * var), epsilon = 1e-14);
        assert_abs_diff_eq!(Gamma::Auto.resolve(&x), 0.5, epsilon = 1e-14);
        assert_abs_diff_eq!(Gamma::Value(0.3).resolve(&x), 0.3, epsilon = 1e-14);
        // Constant data: sklearn falls back to gamma = 1.
        let c = array![[2.0, 2.0], [2.0, 2.0]];
        assert_abs_diff_eq!(Gamma::Scale.resolve(&c), 1.0, epsilon = 1e-14);
        assert_eq!(
            SvmKernel::default().resolve(&x),
            SvmKernel::Rbf {
                gamma: Gamma::Value(1.0 / (2.0 * var))
            }
        );
    }

    #[test]
    fn gamma_serde_is_backward_compatible() {
        let k: SvmKernel = serde_json::from_str(r#"{"Rbf":{"gamma":0.5}}"#).unwrap();
        assert_eq!(k, SvmKernel::rbf(0.5));
        let k: SvmKernel =
            serde_json::from_str(r#"{"Polynomial":{"degree":2,"coef0":1.0}}"#).unwrap();
        assert_eq!(k, SvmKernel::poly(2, 1.0, 1.0));
        let s = serde_json::to_string(&SvmKernel::default()).unwrap();
        assert_eq!(s, r#"{"Rbf":{"gamma":"scale"}}"#);
        let back: SvmKernel = serde_json::from_str(&s).unwrap();
        assert_eq!(back, SvmKernel::default());
    }

    #[test]
    fn default_scale_gamma_is_resolved_by_all_kernel_estimators() {
        use anofox_ml_core::{Fit, FitUnsupervised, Predict};
        let x = array![[0.0, 1.0], [1.0, 0.5], [2.0, 2.5], [3.0, 1.0], [4.0, 3.5]];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let k = SvmKernel::default().resolve(&x);
        let svr: crate::FittedSvr<f64> = crate::Svr::new().fit(&x, &y).unwrap();
        assert_eq!(svr.predict(&x).unwrap().len(), 5);
        let nusvr: crate::FittedNuSvr<f64> = crate::NuSvr::new().fit(&x, &y).unwrap();
        assert_eq!(nusvr.predict(&x).unwrap().len(), 5);
        let oc: crate::FittedOneClassSvm<f64> =
            FitUnsupervised::fit(&crate::OneClassSvm::new(), &x).unwrap();
        assert_eq!(oc.predict(&x).unwrap().len(), 5);
        assert!(matches!(
            k,
            SvmKernel::Rbf {
                gamma: Gamma::Value(_)
            }
        ));
        // Precomputed is rejected by the estimators that do not support it.
        let r: Result<crate::FittedSvr<f64>> = crate::Svr::new()
            .with_kernel(SvmKernel::Precomputed)
            .fit(&x, &y);
        assert!(r.is_err());
    }

    #[test]
    fn invalid_gamma_rejected() {
        assert!(SvmKernel::rbf(-1.0).validate(true).is_err());
        assert!(SvmKernel::rbf(0.0).validate(true).is_err());
        assert!(SvmKernel::sigmoid(-1.0, 0.0).validate(true).is_err());
        assert!(SvmKernel::Precomputed.validate(false).is_err());
        assert!(SvmKernel::Precomputed.validate(true).is_ok());
    }
}
