//! Robust regression: TheilSen and RANSAC.
//!
//! Mirrors `sklearn.linear_model.{TheilSenRegressor, RANSACRegressor}`.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rustml_core::{Fit, Predict, Result, RustMlError};

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};

fn solve_psd(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();
    let am = Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let llt = faer::linalg::solvers::Llt::new(am.as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("LLT failed: {e:?}")))?;
    let bm = Mat::from_fn(n, 1, |i, _| b[i]);
    let s = llt.solve(&bm);
    Ok(Array1::from_vec((0..n).map(|i| s[(i, 0)]).collect()))
}

fn ols_on_subset(
    x: &Array2<f64>,
    y: &Array1<f64>,
    idx: &[usize],
    fit_intercept: bool,
) -> Result<(Array1<f64>, f64)> {
    let d = x.ncols();
    let n = idx.len();
    if n < if fit_intercept { d + 1 } else { d } {
        return Err(RustMlError::InvalidParameter(
            "subset too small for OLS".into(),
        ));
    }

    // Optional intercept: append 1-column.
    let ext = if fit_intercept { d + 1 } else { d };
    let mut xs = Array2::<f64>::zeros((n, ext));
    let mut ys = Array1::<f64>::zeros(n);
    for (k, &i) in idx.iter().enumerate() {
        for j in 0..d {
            xs[[k, j]] = x[[i, j]];
        }
        if fit_intercept {
            xs[[k, d]] = 1.0;
        }
        ys[k] = y[i];
    }
    let mut g = Array2::<f64>::zeros((ext, ext));
    let mut z = Array1::<f64>::zeros(ext);
    for i in 0..ext {
        for j in 0..ext {
            let mut s = 0.0;
            for k in 0..n {
                s += xs[[k, i]] * xs[[k, j]];
            }
            g[[i, j]] = s;
        }
        g[[i, i]] += 1e-10;
        let mut s = 0.0;
        for k in 0..n {
            s += xs[[k, i]] * ys[k];
        }
        z[i] = s;
    }
    let beta = solve_psd(&g, &z)?;
    let coef = beta.slice(ndarray::s![..d]).to_owned();
    let intercept = if fit_intercept { beta[d] } else { 0.0 };
    Ok((coef, intercept))
}

// ---------------------------------------------------------------------------
// RANSACRegressor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RansacRegressor {
    pub min_samples: Option<usize>,
    pub residual_threshold: Option<f64>,
    pub max_trials: usize,
    pub stop_n_inliers: usize,
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
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
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
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        let n = x.nrows();
        let d = x.ncols();
        if n < d + 1 {
            return Err(RustMlError::InvalidParameter("too few samples".into()));
        }
        let min_samples = self
            .min_samples
            .unwrap_or(if self.fit_intercept { d + 1 } else { d });

        // sklearn's default residual threshold = MAD(y).
        let threshold = self.residual_threshold.unwrap_or_else(|| {
            let mut ys: Vec<f64> = y.iter().copied().collect();
            ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = ys[ys.len() / 2];
            let mut abs_dev: Vec<f64> = ys.iter().map(|v| (v - median).abs()).collect();
            abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
            abs_dev[abs_dev.len() / 2]
        });

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut best_score = -1isize;
        let mut best_coef = Array1::<f64>::zeros(d);
        let mut best_intercept = 0.0;
        let mut best_inliers = vec![false; n];
        let mut indices: Vec<usize> = (0..n).collect();
        let mut n_trials = 0;

        for trial in 0..self.max_trials {
            n_trials = trial + 1;
            // Random subsample.
            indices.shuffle(&mut rng);
            let subset: Vec<usize> = indices[..min_samples].to_vec();

            let (coef, intercept) = match ols_on_subset(x, y, &subset, self.fit_intercept) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Count inliers.
            let mut inliers = vec![false; n];
            let mut score = 0isize;
            for i in 0..n {
                let mut p = 0.0;
                for j in 0..d {
                    p += x[[i, j]] * coef[j];
                }
                p += intercept;
                if (y[i] - p).abs() < threshold {
                    inliers[i] = true;
                    score += 1;
                }
            }
            if score > best_score {
                best_score = score;
                best_coef = coef;
                best_intercept = intercept;
                best_inliers = inliers;
            }
            if (score as usize) >= self.stop_n_inliers {
                break;
            }
        }

        // Refit on all inliers.
        let inlier_idx: Vec<usize> = best_inliers
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();
        if !inlier_idx.is_empty() {
            if let Ok((c, b)) = ols_on_subset(x, y, &inlier_idx, self.fit_intercept) {
                best_coef = c;
                best_intercept = b;
            }
        }

        Ok(FittedRansacRegressor {
            coef: best_coef,
            intercept: best_intercept,
            inlier_mask: best_inliers,
            n_trials,
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedRansacRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(x.dot(&self.coef).mapv(|v| v + self.intercept))
    }
}

// ---------------------------------------------------------------------------
// TheilSenRegressor (random sub-sampling spatial median)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TheilSenRegressor {
    pub max_subpopulation: usize,
    pub n_subsamples: Option<usize>,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
    pub fit_intercept: bool,
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

/// Spatial median (geometric median) by Weiszfeld iterations.
fn spatial_median(points: &Array2<f64>, max_iter: usize, tol: f64) -> Array1<f64> {
    let d = points.ncols();
    let n = points.nrows();
    // Initial guess: column means.
    let mut m = Array1::<f64>::zeros(d);
    for j in 0..d {
        let s: f64 = points.column(j).sum();
        m[j] = s / n as f64;
    }
    for _ in 0..max_iter {
        let mut num = Array1::<f64>::zeros(d);
        let mut den = 0.0;
        for i in 0..n {
            let mut dist = 0.0;
            for j in 0..d {
                let v = points[[i, j]] - m[j];
                dist += v * v;
            }
            let dist = dist.sqrt().max(1e-12);
            let w = 1.0 / dist;
            for j in 0..d {
                num[j] += w * points[[i, j]];
            }
            den += w;
        }
        let mut m_new = Array1::<f64>::zeros(d);
        for j in 0..d {
            m_new[j] = num[j] / den;
        }
        // Convergence.
        let shift: f64 = (0..d)
            .map(|j| (m_new[j] - m[j]).powi(2))
            .sum::<f64>()
            .sqrt();
        m = m_new;
        if shift < tol {
            break;
        }
    }
    m
}

impl Fit<f64> for TheilSenRegressor {
    type Fitted = FittedTheilSenRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        let n = x.nrows();
        let d = x.ncols();
        let k = self
            .n_subsamples
            .unwrap_or(if self.fit_intercept { d + 1 } else { d });
        if n < k {
            return Err(RustMlError::InvalidParameter(format!(
                "n_subsamples={} > n={}",
                k, n
            )));
        }

        // Enumerate subsamples up to max_subpopulation.
        let mut subsets: Vec<Vec<usize>> = Vec::new();
        let mut rng = StdRng::seed_from_u64(self.seed);
        // Sample with replacement of sets but dedup approximately by hashing.
        use std::collections::HashSet;
        let mut seen = HashSet::<Vec<usize>>::new();
        let cap = self.max_subpopulation.min(1_000_000);
        for _ in 0..cap {
            let mut s: Vec<usize> = (0..k).map(|_| rng.gen_range(0..n)).collect();
            s.sort();
            s.dedup();
            if s.len() < k {
                continue;
            }
            if seen.insert(s.clone()) {
                subsets.push(s);
            }
            if subsets.len() >= cap {
                break;
            }
        }

        // Fit OLS on each subset, collect coefficient vectors (β, b) in ℝ^{d+1}.
        let ext = if self.fit_intercept { d + 1 } else { d };
        let mut coefs = Array2::<f64>::zeros((subsets.len(), ext));
        let mut valid = 0usize;
        for s in &subsets {
            if let Ok((c, b)) = ols_on_subset(x, y, s, self.fit_intercept) {
                for j in 0..d {
                    coefs[[valid, j]] = c[j];
                }
                if self.fit_intercept {
                    coefs[[valid, d]] = b;
                }
                valid += 1;
            }
        }
        if valid == 0 {
            return Err(RustMlError::InvalidParameter(
                "TheilSen could not fit any subset".into(),
            ));
        }
        // Trim to valid rows.
        let mut coefs_trim = Array2::<f64>::zeros((valid, ext));
        for i in 0..valid {
            for j in 0..ext {
                coefs_trim[[i, j]] = coefs[[i, j]];
            }
        }

        let m = spatial_median(&coefs_trim, self.max_iter, self.tol);
        let coef = m.slice(ndarray::s![..d]).to_owned();
        let intercept = if self.fit_intercept { m[d] } else { 0.0 };

        Ok(FittedTheilSenRegressor {
            coef,
            intercept,
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedTheilSenRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(x.dot(&self.coef).mapv(|v| v + self.intercept))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ransac_ignores_outliers() {
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
        let x = Array2::from_shape_vec((x.len(), 1), x).unwrap();
        let y = Array1::from_vec(y);
        let fitted = RansacRegressor::new()
            .with_min_samples(2)
            .with_residual_threshold(0.5)
            .with_max_trials(200)
            .with_seed(1)
            .fit(&x, &y)
            .unwrap();
        assert!((fitted.coef[0] - 2.0).abs() < 0.1);
        assert!((fitted.intercept - 1.0).abs() < 0.5);
        let _ = array![1.0_f64];
    }

    #[test]
    fn test_theil_sen_robust_slope() {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..30 {
            x.push(i as f64);
            y.push(2.0 * (i as f64) + 1.0);
        }
        for k in 0..5 {
            x.push(k as f64);
            y.push(100.0);
        }
        let x = Array2::from_shape_vec((x.len(), 1), x).unwrap();
        let y = Array1::from_vec(y);
        let fitted = TheilSenRegressor::new()
            .with_max_subpopulation(500)
            .with_seed(0)
            .fit(&x, &y)
            .unwrap();
        // With outliers, slope should still be near 2 (TheilSen robust).
        assert!((fitted.coef[0] - 2.0).abs() < 0.5);
    }
}

impl rustml_core::RegressorScore<f64> for FittedRansacRegressor {}
impl rustml_core::RegressorScore<f64> for FittedTheilSenRegressor {}
