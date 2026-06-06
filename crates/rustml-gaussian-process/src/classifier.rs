//! Gaussian Process binary classifier with Laplace approximation.
//!
//! Mirrors `sklearn.gaussian_process.GaussianProcessClassifier` for the
//! binary case. Multi-class would wrap this in one-vs-rest (not yet done).
//!
//! Algorithm (Rasmussen & Williams §3.4, Algorithm 3.1):
//!
//! 1. Fix a kernel `k` and binary labels `y ∈ {0, 1}`.
//! 2. Find the mode `f̂` of the latent posterior by Newton-Raphson with
//!    Cholesky of `B = I + Wˢ K Wˢ` (`Wˢ = W^{1/2}`) for stability.
//! 3. Posterior covariance `Σ = (K⁻¹ + W)⁻¹`.
//! 4. Predict via probit approximation:
//!      `p(y*=1|x*) ≈ σ(f̄* / √(1 + π/8 · V[f*]))`.

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, PredictProba, Result, RustMlError};

use crate::{build_gram, GpKernel};

pub struct GaussianProcessClassifier {
    pub kernel: GpKernel,
    pub max_iter: usize,
    pub tol: f64,
}

impl GaussianProcessClassifier {
    pub fn new(kernel: GpKernel) -> Self {
        Self {
            kernel,
            max_iter: 100,
            tol: 1e-6,
        }
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }
}

pub struct FittedGaussianProcessClassifier {
    pub x_train: Array2<f64>,
    /// `y_train - π̂` (the dual coefficient vector at the posterior mode).
    pub alpha: Array1<f64>,
    /// Cholesky factor `L` of `I + Wˢ K Wˢ`.
    pub l_lower: Mat<f64>,
    /// `Wˢ = W^{1/2}` at the posterior mode.
    pub w_sqrt: Array1<f64>,
    pub kernel: GpKernel,
    pub classes: [f64; 2],
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn clone_kernel(k: &GpKernel) -> GpKernel {
    match k {
        GpKernel::Rbf {
            length_scale,
            signal_var,
        } => GpKernel::Rbf {
            length_scale: *length_scale,
            signal_var: *signal_var,
        },
        GpKernel::Matern {
            length_scale,
            signal_var,
            nu,
        } => GpKernel::Matern {
            length_scale: *length_scale,
            signal_var: *signal_var,
            nu: *nu,
        },
        GpKernel::RationalQuadratic {
            length_scale,
            signal_var,
            alpha,
        } => GpKernel::RationalQuadratic {
            length_scale: *length_scale,
            signal_var: *signal_var,
            alpha: *alpha,
        },
        GpKernel::White { noise_level } => GpKernel::White {
            noise_level: *noise_level,
        },
        GpKernel::Constant { value } => GpKernel::Constant { value: *value },
        GpKernel::Sum(a, b) => GpKernel::Sum(Box::new(clone_kernel(a)), Box::new(clone_kernel(b))),
        GpKernel::Product(a, b) => {
            GpKernel::Product(Box::new(clone_kernel(a)), Box::new(clone_kernel(b)))
        }
    }
}

impl Fit<f64> for GaussianProcessClassifier {
    type Fitted = FittedGaussianProcessClassifier;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if y.len() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}",
                n,
                y.len()
            )));
        }
        // Determine classes.
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        if classes.len() != 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "GPC expects 2 classes, found {}",
                classes.len()
            )));
        }
        let neg = classes[0];
        let pos = classes[1];
        // Encode labels as 0 / 1.
        let yb: Vec<f64> = y
            .iter()
            .map(|v| if *v == pos { 1.0 } else { 0.0 })
            .collect();

        let k = build_gram(x, x, &self.kernel);
        let mut f = Array1::<f64>::zeros(n);

        // Newton-Raphson on the Laplace objective.
        let mut prev_obj = f64::NEG_INFINITY;
        let mut alpha = Array1::<f64>::zeros(n);
        let mut l_lower = Mat::<f64>::zeros(n, n);
        let mut w_sqrt = Array1::<f64>::zeros(n);

        for _ in 0..self.max_iter {
            // π = sigmoid(f); W = diag(π(1-π))
            let pi: Vec<f64> = f.iter().map(|&v| sigmoid(v)).collect();
            let w: Vec<f64> = pi.iter().map(|&p| p * (1.0 - p)).collect();
            let ws: Vec<f64> = w.iter().map(|&v| v.sqrt()).collect();

            // B = I + Wˢ K Wˢ
            let mut b = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    b[[i, j]] = ws[i] * k[[i, j]] * ws[j];
                }
                b[[i, i]] += 1.0;
            }
            let bm = Mat::<f64>::from_fn(n, n, |i, j| b[[i, j]]);
            let llt = faer::linalg::solvers::Llt::new(bm.as_ref(), Side::Lower)
                .map_err(|e| RustMlError::InvalidParameter(format!("Cholesky failed: {e:?}")))?;
            let lower = llt.L();
            l_lower = Mat::<f64>::from_fn(n, n, |i, j| lower[(i, j)]);

            // b_vec = W f + (y - π)
            let mut b_vec = Array1::<f64>::zeros(n);
            for i in 0..n {
                b_vec[i] = w[i] * f[i] + (yb[i] - pi[i]);
            }
            // a = b - Wˢ L'^{-1} L^{-1} Wˢ K b
            // Equivalently: solve B v = Wˢ K b, then a = b - Wˢ v.
            let mut k_b = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += k[[i, j]] * b_vec[j];
                }
                k_b[i] = s;
            }
            let ws_kb: Vec<f64> = (0..n).map(|i| ws[i] * k_b[i]).collect();
            let rhs = Mat::<f64>::from_fn(n, 1, |i, _| ws_kb[i]);
            let v_mat = llt.solve(&rhs);
            let mut a = Array1::<f64>::zeros(n);
            for i in 0..n {
                a[i] = b_vec[i] - ws[i] * v_mat[(i, 0)];
            }
            // f = K a
            let mut new_f = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += k[[i, j]] * a[j];
                }
                new_f[i] = s;
            }

            // Objective: Ψ(f) = -0.5 fᵀ a + Σ log p(y_i | f_i)
            let mut obj = 0.0;
            for i in 0..n {
                obj -= 0.5 * new_f[i] * a[i];
                // log p(y_i | f_i) = y log σ(f) + (1-y) log σ(-f)
                let lp = if yb[i] > 0.5 {
                    -(-new_f[i]).ln_1p().min(0.0)
                        - if new_f[i] >= 0.0 {
                            (-new_f[i]).exp().ln_1p()
                        } else {
                            -new_f[i] + new_f[i].exp().ln_1p()
                        }
                } else {
                    if new_f[i] >= 0.0 {
                        -new_f[i] - (-new_f[i]).exp().ln_1p()
                    } else {
                        -new_f[i].exp().ln_1p()
                    }
                };
                obj += lp;
            }

            f = new_f;
            alpha = a;
            for i in 0..n {
                w_sqrt[i] = ws[i];
            }

            if (obj - prev_obj).abs() < self.tol {
                break;
            }
            prev_obj = obj;
        }

        Ok(FittedGaussianProcessClassifier {
            x_train: x.clone(),
            alpha,
            l_lower,
            w_sqrt,
            kernel: clone_kernel(&self.kernel),
            classes: [neg, pos],
        })
    }
}

impl FittedGaussianProcessClassifier {
    /// Latent posterior mean and variance at query points.
    fn latent_predict(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let n_train = self.x_train.nrows();
        if x.ncols() != self.x_train.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }
        let n_new = x.nrows();
        let k_star = build_gram(x, &self.x_train, &self.kernel);
        let mean = k_star.dot(&self.alpha);
        // Variance: k(x*, x*) - v' v where v = L^{-1} Wˢ k_star.
        let mut var = Array1::<f64>::zeros(n_new);
        for i in 0..n_new {
            let mut ws_k = vec![0.0_f64; n_train];
            for j in 0..n_train {
                ws_k[j] = self.w_sqrt[j] * k_star[[i, j]];
            }
            // Forward solve L v = ws_k.
            let mut v = vec![0.0_f64; n_train];
            for r in 0..n_train {
                let mut s = ws_k[r];
                for c in 0..r {
                    s -= self.l_lower[(r, c)] * v[c];
                }
                v[r] = s / self.l_lower[(r, r)].max(1e-12);
            }
            let v_sq: f64 = v.iter().map(|x| x * x).sum();
            let xi = x.row(i).to_owned();
            let k_xx = self.kernel_compute(xi.as_slice().unwrap(), xi.as_slice().unwrap());
            var[i] = (k_xx - v_sq).max(0.0);
        }
        Ok((mean, var))
    }

    fn kernel_compute(&self, a: &[f64], b: &[f64]) -> f64 {
        // Re-use the kernel's compute method via build_gram on 1×1 arrays.
        let arr_a = Array2::from_shape_vec((1, a.len()), a.to_vec()).unwrap();
        let arr_b = Array2::from_shape_vec((1, b.len()), b.to_vec()).unwrap();
        build_gram(&arr_a, &arr_b, &self.kernel)[[0, 0]]
    }
}

impl Predict<f64> for FittedGaussianProcessClassifier {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let proba = self.predict_proba(x)?;
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            out[i] = if proba[[i, 1]] >= 0.5 {
                self.classes[1]
            } else {
                self.classes[0]
            };
        }
        Ok(out)
    }
}

impl PredictProba<f64> for FittedGaussianProcessClassifier {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (mean, var) = self.latent_predict(x)?;
        // Probit approximation: p(y=1) ≈ σ(f̄ / sqrt(1 + π/8 · v)).
        let n = mean.len();
        let mut out = Array2::<f64>::zeros((n, 2));
        let pi8 = std::f64::consts::PI / 8.0;
        for i in 0..n {
            let denom = (1.0 + pi8 * var[i]).sqrt();
            let p1 = sigmoid(mean[i] / denom);
            out[[i, 0]] = 1.0 - p1;
            out[[i, 1]] = p1;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gpc_separates_two_clusters() {
        // 2-class problem: clusters around (0,0) and (5,5).
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for i in 0..6 {
            let f = i as f64 * 0.1;
            x_data.extend([f, f + 0.1]);
            y_data.push(0.0);
            x_data.extend([5.0 + f, 5.0 - f]);
            y_data.push(1.0);
        }
        let x = Array2::from_shape_vec((12, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);
        let kernel = GpKernel::Rbf {
            length_scale: 2.0,
            signal_var: 1.0,
        };
        let fitted = GaussianProcessClassifier::new(kernel)
            .with_max_iter(50)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        // Should perfectly classify a well-separated problem.
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 0.5)
            .count();
        assert!(correct >= 11, "got {}/{} correct", correct, y.len());

        // predict_proba returns valid probabilities (rows sum to 1).
        let proba = fitted.predict_proba(&x).unwrap();
        for i in 0..12 {
            let s = proba[[i, 0]] + proba[[i, 1]];
            assert!((s - 1.0).abs() < 1e-9, "row {} sum = {}", i, s);
        }
        let _ = array![1.0_f64];
    }
}

impl rustml_core::ClassifierScore<f64> for FittedGaussianProcessClassifier {}

// ---------------------------------------------------------------------------
// Multi-class GPC via one-vs-rest.
// ---------------------------------------------------------------------------

/// Multi-class Gaussian Process Classifier built as a one-vs-rest stack of
/// binary `GaussianProcessClassifier` instances. Mirrors sklearn's
/// `GaussianProcessClassifier(multi_class='one_vs_rest')` for the case of
/// arbitrary discrete class labels.
pub struct MulticlassGaussianProcessClassifier {
    pub kernel: GpKernel,
    pub max_iter: usize,
    pub tol: f64,
}

impl MulticlassGaussianProcessClassifier {
    pub fn new(kernel: GpKernel) -> Self {
        Self {
            kernel,
            max_iter: 100,
            tol: 1e-6,
        }
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }
}

pub struct FittedMulticlassGaussianProcessClassifier {
    pub classes: Vec<f64>,
    pub binary: Vec<FittedGaussianProcessClassifier>,
}

fn clone_kernel_local(k: &GpKernel) -> GpKernel {
    match k {
        GpKernel::Rbf {
            length_scale,
            signal_var,
        } => GpKernel::Rbf {
            length_scale: *length_scale,
            signal_var: *signal_var,
        },
        GpKernel::Matern {
            length_scale,
            signal_var,
            nu,
        } => GpKernel::Matern {
            length_scale: *length_scale,
            signal_var: *signal_var,
            nu: *nu,
        },
        GpKernel::RationalQuadratic {
            length_scale,
            signal_var,
            alpha,
        } => GpKernel::RationalQuadratic {
            length_scale: *length_scale,
            signal_var: *signal_var,
            alpha: *alpha,
        },
        GpKernel::White { noise_level } => GpKernel::White {
            noise_level: *noise_level,
        },
        GpKernel::Constant { value } => GpKernel::Constant { value: *value },
        GpKernel::Sum(a, b) => GpKernel::Sum(
            Box::new(clone_kernel_local(a)),
            Box::new(clone_kernel_local(b)),
        ),
        GpKernel::Product(a, b) => GpKernel::Product(
            Box::new(clone_kernel_local(a)),
            Box::new(clone_kernel_local(b)),
        ),
    }
}

impl Fit<f64> for MulticlassGaussianProcessClassifier {
    type Fitted = FittedMulticlassGaussianProcessClassifier;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "multi-class GPC needs ≥2 classes, found {}",
                classes.len()
            )));
        }
        let mut binary = Vec::with_capacity(classes.len());
        for &c in &classes {
            let y_bin: Array1<f64> = y.mapv(|v| if v == c { 1.0 } else { 0.0 });
            let inner = GaussianProcessClassifier {
                kernel: clone_kernel_local(&self.kernel),
                max_iter: self.max_iter,
                tol: self.tol,
            };
            binary.push(inner.fit(x, &y_bin)?);
        }
        Ok(FittedMulticlassGaussianProcessClassifier { classes, binary })
    }
}

impl Predict<f64> for FittedMulticlassGaussianProcessClassifier {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let proba = self.predict_proba(x)?;
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut best = f64::NEG_INFINITY;
            let mut best_c = 0;
            for c in 0..self.classes.len() {
                if proba[[i, c]] > best {
                    best = proba[[i, c]];
                    best_c = c;
                }
            }
            out[i] = self.classes[best_c];
        }
        Ok(out)
    }
}

impl PredictProba<f64> for FittedMulticlassGaussianProcessClassifier {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let k = self.classes.len();
        let mut p = Array2::<f64>::zeros((n, k));
        for c in 0..k {
            let pc = self.binary[c].predict_proba(x)?;
            // Take the "is-class-c" column (= column 1 since label 1.0 in fit).
            for i in 0..n {
                p[[i, c]] = pc[[i, 1]];
            }
        }
        // Renormalise rows to sum to 1 (sklearn does the same for OvR).
        for i in 0..n {
            let s: f64 = (0..k).map(|c| p[[i, c]]).sum::<f64>().max(1e-12);
            for c in 0..k {
                p[[i, c]] /= s;
            }
        }
        Ok(p)
    }
}

impl rustml_core::ClassifierScore<f64> for FittedMulticlassGaussianProcessClassifier {}

#[cfg(test)]
mod multiclass_tests {
    use super::*;
    use crate::GpKernel;
    use ndarray::Array2;

    #[test]
    fn test_multiclass_gpc_three_classes() {
        // Three clusters at (0,0), (5,0), (0,5).
        let n_per = 6;
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for i in 0..n_per {
            let f = i as f64 * 0.1;
            x_data.extend([f, f]);
            y_data.push(0.0);
            x_data.extend([5.0 + f, f]);
            y_data.push(1.0);
            x_data.extend([f, 5.0 + f]);
            y_data.push(2.0);
        }
        let x = Array2::from_shape_vec((n_per * 3, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);
        let mc = MulticlassGaussianProcessClassifier::new(GpKernel::Rbf {
            length_scale: 2.0,
            signal_var: 1.0,
        })
        .with_max_iter(50);
        let fitted = mc.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 0.5)
            .count();
        assert!(
            correct >= (n_per * 3) * 9 / 10,
            "got {}/{} correct",
            correct,
            n_per * 3
        );
        let p = fitted.predict_proba(&x).unwrap();
        for i in 0..(n_per * 3) {
            let s: f64 = (0..3).map(|c| p[[i, c]]).sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
    }
}
