//! t-distributed Stochastic Neighbor Embedding.
//!
//! Mirrors `sklearn.manifold.TSNE` with the vanilla (no Barnes-Hut) gradient
//! descent. Suitable for n ≲ 1000; the O(n²) cost per iteration becomes the
//! dominant factor past that point.
//!
//! Algorithm (van der Maaten & Hinton 2008):
//! 1. Pairwise squared distances `||x_i - x_j||²`.
//! 2. For each `i`, binary search `σ_i` so that the entropy of the
//!    conditional `p_{j|i} ∝ exp(-d²/2σ_i²)` matches `log(perplexity)`.
//! 3. Joint affinity `p_{ij} = (p_{j|i} + p_{i|j}) / (2n)`.
//! 4. Initialize low-dim `Y` randomly with small variance.
//! 5. Compute student-t low-dim affinities `q_{ij} ∝ (1 + ||y_i - y_j||²)^{-1}`.
//! 6. Gradient: `∂C/∂y_i = 4 Σⱼ (p_{ij} - q_{ij}) (y_i - y_j) / (1 + ||y_i - y_j||²)`.
//! 7. Gradient descent with momentum and "early exaggeration" of `p` for the
//!    first 250 iterations (boosts cluster separation).

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustml_core::{FitUnsupervised, Result, RustMlError};

#[derive(Debug, Clone)]
pub struct TSne {
    pub n_components: usize,
    pub perplexity: f64,
    pub learning_rate: f64,
    pub n_iter: usize,
    pub early_exaggeration: f64,
    pub seed: u64,
}

impl TSne {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 500,
            early_exaggeration: 12.0,
            seed: 0,
        }
    }
    pub fn with_perplexity(mut self, p: f64) -> Self { self.perplexity = p; self }
    pub fn with_learning_rate(mut self, lr: f64) -> Self { self.learning_rate = lr; self }
    pub fn with_n_iter(mut self, n: usize) -> Self { self.n_iter = n; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedTSne {
    pub embedding: Array2<f64>,
    pub kl_divergence: f64,
    pub n_iter: usize,
}

fn squared_distance_matrix(x: &Array2<f64>) -> Vec<Vec<f64>> {
    let n = x.nrows();
    let d = x.ncols();
    let mut m = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sd = 0.0;
            for k in 0..d {
                let dv = x[[i, k]] - x[[j, k]];
                sd += dv * dv;
            }
            m[i][j] = sd;
            m[j][i] = sd;
        }
    }
    m
}

/// Binary-search the bandwidth `β = 1 / (2σ²)` so the conditional has the
/// requested perplexity.
fn conditional_p_row(d_sq_row: &[f64], i: usize, target_log_perp: f64) -> Vec<f64> {
    let n = d_sq_row.len();
    let mut beta_min = f64::NEG_INFINITY;
    let mut beta_max = f64::INFINITY;
    let mut beta = 1.0_f64;
    let max_iter = 50;
    let tol = 1e-5;

    let mut p = vec![0.0_f64; n];
    for _ in 0..max_iter {
        // Compute p_{j|i} ∝ exp(-d² · β); diagonal is zero.
        let mut sum_p = 0.0_f64;
        for j in 0..n {
            if j == i {
                p[j] = 0.0;
                continue;
            }
            p[j] = (-d_sq_row[j] * beta).exp();
            sum_p += p[j];
        }
        let sum_p = sum_p.max(1e-12);
        // Shannon entropy H = log(sum_p) + β · Σ d² p / sum_p.
        let mut hh = 0.0;
        for j in 0..n {
            if j == i { continue; }
            hh += d_sq_row[j] * p[j];
        }
        hh = sum_p.ln() + beta * hh / sum_p;
        // Normalise p in place.
        for j in 0..n {
            p[j] /= sum_p;
        }
        let diff = hh - target_log_perp;
        if diff.abs() < tol {
            break;
        }
        if diff > 0.0 {
            // entropy too high → narrower kernel → larger β
            beta_min = beta;
            beta = if beta_max.is_infinite() { beta * 2.0 } else { (beta + beta_max) / 2.0 };
        } else {
            beta_max = beta;
            beta = if beta_min.is_infinite() { beta / 2.0 } else { (beta + beta_min) / 2.0 };
        }
    }
    p
}

impl FitUnsupervised<f64> for TSne {
    type Fitted = FittedTSne;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let k = self.n_components;
        if n < 2 {
            return Err(RustMlError::EmptyInput("need ≥ 2 samples".into()));
        }
        if k == 0 {
            return Err(RustMlError::InvalidParameter("n_components >= 1".into()));
        }
        // sklearn requires perplexity < n; clamp ours.
        let perp = self.perplexity.min((n - 1) as f64);
        let target_log_perp = perp.ln();

        let d_sq = squared_distance_matrix(x);

        // Build joint P = (P_cond + P_condᵀ) / (2n).
        let mut p_joint = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let row = conditional_p_row(&d_sq[i], i, target_log_perp);
            for j in 0..n {
                p_joint[i][j] += row[j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                p_joint[i][j] = (p_joint[i][j] + p_joint[j][i]) / (2.0 * n as f64);
                if p_joint[i][j] < 1e-12 {
                    p_joint[i][j] = 1e-12;
                }
            }
        }

        // Initialize Y ~ N(0, 1e-4 I).
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut y = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for c in 0..k {
                // Box-Muller-ish: just uniform in a small range, the algorithm
                // is robust to init scale.
                y[[i, c]] = (rng.gen::<f64>() - 0.5) * 0.02;
            }
        }
        let mut y_prev = y.clone();

        // Early-exaggeration phase boosts P for the first stage so clusters
        // separate more cleanly before fine-tuning.
        let exag_iters = (self.n_iter / 4).max(50).min(self.n_iter / 2);
        let exag = self.early_exaggeration;

        let mut kl = 0.0_f64;
        let mut n_iter = 0;
        for iter in 0..self.n_iter {
            n_iter = iter + 1;
            // q_{ij} ∝ (1 + ||y_i - y_j||²)⁻¹
            let mut q_num = vec![vec![0.0_f64; n]; n];
            let mut q_sum = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    if i == j { continue; }
                    let mut sd = 0.0;
                    for c in 0..k {
                        let dv = y[[i, c]] - y[[j, c]];
                        sd += dv * dv;
                    }
                    let v = 1.0 / (1.0 + sd);
                    q_num[i][j] = v;
                    q_sum += v;
                }
            }
            let q_sum = q_sum.max(1e-12);

            // Gradient: ∂C/∂y_i = 4 Σⱼ (p̃ - q) · q_num · (y_i - y_j)
            let momentum = if iter < 250 { 0.5 } else { 0.8 };
            let p_scale = if iter < exag_iters { exag } else { 1.0 };

            let mut grad = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                for j in 0..n {
                    if i == j { continue; }
                    let p_ij = p_joint[i][j] * p_scale;
                    let q_ij = q_num[i][j] / q_sum;
                    let diff = p_ij - q_ij;
                    let f = 4.0 * diff * q_num[i][j];
                    for c in 0..k {
                        grad[[i, c]] += f * (y[[i, c]] - y[[j, c]]);
                    }
                }
            }
            // Update with momentum.
            let lr = self.learning_rate;
            let mut y_new = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                for c in 0..k {
                    y_new[[i, c]] =
                        y[[i, c]] - lr * grad[[i, c]] + momentum * (y[[i, c]] - y_prev[[i, c]]);
                }
            }
            y_prev = y;
            y = y_new;
            // Centre y to keep numerically stable.
            let mut mean = vec![0.0_f64; k];
            for i in 0..n {
                for c in 0..k {
                    mean[c] += y[[i, c]];
                }
            }
            for c in 0..k {
                mean[c] /= n as f64;
            }
            for i in 0..n {
                for c in 0..k {
                    y[[i, c]] -= mean[c];
                }
            }

            // KL divergence (compute occasionally, or at the end).
            if iter + 1 == self.n_iter {
                let mut k_d = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        if i == j { continue; }
                        let p_ij = p_joint[i][j];
                        let q_ij = (q_num[i][j] / q_sum).max(1e-12);
                        k_d += p_ij * (p_ij / q_ij).ln();
                    }
                }
                kl = k_d;
            }
        }

        Ok(FittedTSne {
            embedding: y,
            kl_divergence: kl,
            n_iter,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// On two well-separated clusters, t-SNE should put each cluster's points
    /// closer to its own centroid than to the other's.
    #[test]
    fn test_tsne_separates_two_blobs() {
        let n_per = 15;
        let mut x = Array2::<f64>::zeros((2 * n_per, 5));
        for i in 0..n_per {
            // Cluster A around origin, jitter via deterministic LCG.
            let s = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            x[[i, 0]] = ((s >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.3;
            x[[i, 1]] = (((s.wrapping_mul(13)) >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.3;
            // Cluster B around (10, 10, ...).
            let s2 = ((i + 100) as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            x[[n_per + i, 0]] = 10.0 + ((s2 >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.3;
            x[[n_per + i, 1]] = 10.0 + (((s2.wrapping_mul(13)) >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.3;
        }
        let fitted = TSne::new(2)
            .with_perplexity(5.0)
            .with_learning_rate(10.0)
            .with_n_iter(500)
            .with_seed(1)
            .fit(&x).unwrap();
        let y = fitted.embedding;
        // Compute centroids.
        let mut ca = [0.0; 2];
        let mut cb = [0.0; 2];
        for i in 0..n_per {
            ca[0] += y[[i, 0]]; ca[1] += y[[i, 1]];
            cb[0] += y[[n_per + i, 0]]; cb[1] += y[[n_per + i, 1]];
        }
        for v in ca.iter_mut() { *v /= n_per as f64; }
        for v in cb.iter_mut() { *v /= n_per as f64; }
        // Linear-separability test: project each point onto the centroid-
        // difference direction; the two clusters should be on opposite sides
        // of the midpoint.
        let dir = [cb[0] - ca[0], cb[1] - ca[1]];
        let dir_norm = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-12);
        let mid = [(ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0];
        let mut a_correct = 0;
        let mut b_correct = 0;
        for i in 0..n_per {
            let pa = ((y[[i, 0]] - mid[0]) * dir[0] + (y[[i, 1]] - mid[1]) * dir[1]) / dir_norm;
            let pb = ((y[[n_per + i, 0]] - mid[0]) * dir[0]
                + (y[[n_per + i, 1]] - mid[1]) * dir[1]) / dir_norm;
            if pa < 0.0 { a_correct += 1; }
            if pb > 0.0 { b_correct += 1; }
        }
        // 80% of each cluster must land on the correct side of the
        // centroid-midpoint hyperplane. t-SNE on small datasets is finicky;
        // perfect linear separability isn't guaranteed.
        let thr = (n_per * 4) / 5;
        assert!(
            a_correct >= thr && b_correct >= thr,
            "linear-separability failed: A={}, B={}, threshold={}",
            a_correct, b_correct, thr,
        );
    }
}
