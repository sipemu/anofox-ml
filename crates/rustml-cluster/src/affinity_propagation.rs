//! Affinity Propagation.
//!
//! Mirrors `sklearn.cluster.AffinityPropagation`. Message-passing on a
//! similarity matrix `s_{i,k} = -||x_i - x_k||²` until exemplars stabilise.

use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, Result, RustMlError};

#[derive(Debug, Clone)]
pub struct AffinityPropagation {
    pub damping: f64,
    pub max_iter: usize,
    pub convergence_iter: usize,
    pub preference: Option<f64>,
}

impl AffinityPropagation {
    pub fn new() -> Self {
        Self {
            damping: 0.9,
            max_iter: 200,
            convergence_iter: 15,
            preference: None,
        }
    }
    pub fn with_damping(mut self, d: f64) -> Self { self.damping = d; self }
    pub fn with_preference(mut self, p: f64) -> Self { self.preference = Some(p); self }
}

impl Default for AffinityPropagation {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct FittedAffinityPropagation {
    pub labels: Array1<f64>,
    pub cluster_centers_indices: Vec<usize>,
}

impl FitUnsupervised<f64> for AffinityPropagation {
    type Fitted = FittedAffinityPropagation;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }

        // Similarity matrix s_{i,k} = -||x_i - x_k||²
        let mut s = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sd = 0.0;
                for k in 0..x.ncols() {
                    let d = x[[i, k]] - x[[j, k]];
                    sd += d * d;
                }
                s[[i, j]] = -sd;
            }
        }
        // Diagonal: preference = median of off-diagonals (sklearn default).
        let pref = self.preference.unwrap_or_else(|| {
            let mut offdiag = Vec::with_capacity(n * (n - 1));
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        offdiag.push(s[[i, j]]);
                    }
                }
            }
            offdiag.sort_by(|a, b| a.partial_cmp(b).unwrap());
            offdiag[offdiag.len() / 2]
        });
        for i in 0..n {
            s[[i, i]] = pref;
        }
        // Tie-breaking noise (sklearn does this).
        for i in 0..n {
            for j in 0..n {
                s[[i, j]] += 1e-12 * (i as f64 + 1.0) * (j as f64 + 1.0).cos();
            }
        }

        let mut r = Array2::<f64>::zeros((n, n));
        let mut a = Array2::<f64>::zeros((n, n));
        let mut last_exemplars: Option<Vec<bool>> = None;
        let mut converge_count = 0usize;

        for _iter in 0..self.max_iter {
            // Update responsibilities r_{i,k} = s_{i,k} - max_{k' != k}(a_{i,k'} + s_{i,k'})
            let mut new_r = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                // Find top two indices by (a + s) over k.
                let mut top1 = f64::NEG_INFINITY;
                let mut top1_k = 0usize;
                let mut top2 = f64::NEG_INFINITY;
                for k in 0..n {
                    let v = a[[i, k]] + s[[i, k]];
                    if v > top1 {
                        top2 = top1;
                        top1 = v;
                        top1_k = k;
                    } else if v > top2 {
                        top2 = v;
                    }
                }
                for k in 0..n {
                    let other_max = if k == top1_k { top2 } else { top1 };
                    new_r[[i, k]] = s[[i, k]] - other_max;
                }
            }
            // Damping.
            for i in 0..n {
                for k in 0..n {
                    r[[i, k]] = self.damping * r[[i, k]] + (1.0 - self.damping) * new_r[[i, k]];
                }
            }

            // Update availabilities.
            let mut new_a = Array2::<f64>::zeros((n, n));
            for k in 0..n {
                // For i != k: a_{i,k} = min(0, r_{k,k} + sum_{i' != i, k} max(0, r_{i', k}))
                let mut sum_pos = 0.0;
                for ip in 0..n {
                    if ip == k {
                        continue;
                    }
                    sum_pos += r[[ip, k]].max(0.0);
                }
                for i in 0..n {
                    if i == k {
                        // a_{k,k} = sum_{i' != k} max(0, r_{i', k})
                        new_a[[i, k]] = sum_pos;
                    } else {
                        let excl = r[[i, k]].max(0.0);
                        let v = r[[k, k]] + (sum_pos - excl);
                        new_a[[i, k]] = v.min(0.0);
                    }
                }
            }
            for i in 0..n {
                for k in 0..n {
                    a[[i, k]] = self.damping * a[[i, k]] + (1.0 - self.damping) * new_a[[i, k]];
                }
            }

            // Exemplars: indices k where a_{k,k} + r_{k,k} > 0.
            let exemplars: Vec<bool> = (0..n).map(|k| a[[k, k]] + r[[k, k]] > 0.0).collect();
            if let Some(prev) = &last_exemplars {
                if prev == &exemplars {
                    converge_count += 1;
                    if converge_count >= self.convergence_iter {
                        break;
                    }
                } else {
                    converge_count = 0;
                }
            }
            last_exemplars = Some(exemplars);
        }

        // Final cluster centers and labels.
        let exemplars = last_exemplars.unwrap_or_else(|| vec![true; n]);
        let cluster_centers_indices: Vec<usize> = exemplars
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();
        if cluster_centers_indices.is_empty() {
            // Pathological — return single cluster.
            return Ok(FittedAffinityPropagation {
                labels: Array1::<f64>::zeros(n),
                cluster_centers_indices: vec![0],
            });
        }

        let mut labels = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut best = f64::NEG_INFINITY;
            let mut best_c = 0;
            for (c, &k) in cluster_centers_indices.iter().enumerate() {
                let sim = s[[i, k]];
                if sim > best {
                    best = sim;
                    best_c = c;
                }
            }
            labels[i] = best_c as f64;
        }
        Ok(FittedAffinityPropagation {
            labels,
            cluster_centers_indices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ap_two_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.2, 0.1], [-0.1, 0.2],
            [10.0, 10.0], [10.2, 9.9], [9.8, 10.1],
        ];
        let ap = AffinityPropagation::new()
            .with_damping(0.7)
            .with_preference(-1.0);
        let fitted = ap.fit(&x).unwrap();
        assert!(
            fitted.cluster_centers_indices.len() >= 2,
            "expected ≥2 clusters, got {}",
            fitted.cluster_centers_indices.len()
        );
        let l0 = fitted.labels[0];
        for i in 1..3 {
            assert_eq!(fitted.labels[i], l0);
        }
        for i in 3..6 {
            assert_ne!(fitted.labels[i], l0);
        }
    }
}
