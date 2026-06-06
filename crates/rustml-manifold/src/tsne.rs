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

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustml_core::{FitUnsupervised, Result, RustMlError};

/// Algorithm for the t-SNE repulsive force computation. `Exact` is the
/// vanilla O(n²) version; `BarnesHut` uses a quadtree summary with the
/// classical θ ≈ 0.5 cutoff for O(n log n) per iteration.
///
/// Barnes-Hut requires `n_components == 2`. Higher-dimensional output
/// always falls back to `Exact`. Below `n_components == 2` and `n ≥ 200`
/// the BH path is faster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TSneMethod {
    Exact,
    BarnesHut,
}

#[derive(Debug, Clone)]
pub struct TSne {
    pub n_components: usize,
    pub perplexity: f64,
    pub learning_rate: f64,
    pub n_iter: usize,
    pub early_exaggeration: f64,
    pub seed: u64,
    pub method: TSneMethod,
    /// Barnes-Hut tree-walk threshold: open a cell if width / dist > θ.
    pub theta: f64,
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
            method: TSneMethod::Exact,
            theta: 0.5,
        }
    }
    pub fn with_perplexity(mut self, p: f64) -> Self {
        self.perplexity = p;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn with_method(mut self, m: TSneMethod) -> Self {
        self.method = m;
        self
    }
    pub fn with_theta(mut self, t: f64) -> Self {
        self.theta = t;
        self
    }
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
            if j == i {
                continue;
            }
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
            beta = if beta_max.is_infinite() {
                beta * 2.0
            } else {
                (beta + beta_max) / 2.0
            };
        } else {
            beta_max = beta;
            beta = if beta_min.is_infinite() {
                beta / 2.0
            } else {
                (beta + beta_min) / 2.0
            };
        }
    }
    p
}

// ───── Barnes-Hut quadtree ───────────────────────────────────────────────

/// Quadtree node for 2D Barnes-Hut t-SNE. Each node stores the bounding
/// box, count of points inside, and centre of mass. Internal nodes hold up
/// to 4 children (one per quadrant) in `children`; leaves store the index
/// of the single point they contain.
#[derive(Debug, Clone)]
struct QuadNode {
    cx: f64,
    cy: f64,
    half_width: f64,
    count: usize,
    com_x: f64, // weighted centre-of-mass numerator (sum of x-coords)
    com_y: f64,
    children: [Option<usize>; 4],
    point_idx: Option<usize>, // for leaves with count == 1
    is_leaf: bool,
}

struct QuadTree {
    nodes: Vec<QuadNode>,
}

impl QuadTree {
    fn new(cx: f64, cy: f64, half_width: f64) -> Self {
        Self {
            nodes: vec![QuadNode {
                cx,
                cy,
                half_width,
                count: 0,
                com_x: 0.0,
                com_y: 0.0,
                children: [None; 4],
                point_idx: None,
                is_leaf: true,
            }],
        }
    }
    fn quadrant_of(node: &QuadNode, x: f64, y: f64) -> usize {
        let e = (x >= node.cx) as usize;
        let n = (y >= node.cy) as usize;
        // 0: SW, 1: SE, 2: NW, 3: NE
        (n << 1) | e
    }
    fn insert(&mut self, idx: usize, x: f64, y: f64) {
        self.insert_into(0, idx, x, y, 0);
    }
    fn insert_into(&mut self, node_idx: usize, idx: usize, x: f64, y: f64, depth: usize) {
        if depth > 200 {
            // Two coincident points after large depth: just accumulate into
            // centre-of-mass — Barnes-Hut still uses it as a summary.
            self.nodes[node_idx].count += 1;
            self.nodes[node_idx].com_x += x;
            self.nodes[node_idx].com_y += y;
            return;
        }
        let node = &mut self.nodes[node_idx];
        node.count += 1;
        node.com_x += x;
        node.com_y += y;
        if node.is_leaf {
            if let Some(existing) = node.point_idx {
                // Need to split — turn this leaf into an internal node and
                // re-insert both points.
                let (ex, ey) = (
                    node.com_x - x, // existing's coords = com - new (only valid if count was 1 before, but count just bumped; rebuild)
                    node.com_y - y,
                );
                // Re-derive existing point coords. We pre-mutated com_x/y by
                // adding (x, y); the existing point contributed before that.
                // Since previous count was 1, prior com_x = ex (single
                // existing point coord). Same for y.
                node.is_leaf = false;
                node.point_idx = None;

                // Insert existing
                let cx = node.cx;
                let cy = node.cy;
                let hw = node.half_width;
                let q_e = Self::quadrant_of(&self.nodes[node_idx], ex, ey);
                let child_e = self.ensure_child(node_idx, q_e, cx, cy, hw);
                self.insert_into(child_e, existing, ex, ey, depth + 1);
                // Now insert new point
                let q_n = Self::quadrant_of(&self.nodes[node_idx], x, y);
                let child_n = self.ensure_child(node_idx, q_n, cx, cy, hw);
                self.insert_into(child_n, idx, x, y, depth + 1);
            } else {
                // Empty leaf — store directly.
                node.point_idx = Some(idx);
            }
        } else {
            // Internal — descend to child.
            let q = Self::quadrant_of(node, x, y);
            let cx = node.cx;
            let cy = node.cy;
            let hw = node.half_width;
            let child = self.ensure_child(node_idx, q, cx, cy, hw);
            self.insert_into(child, idx, x, y, depth + 1);
        }
    }
    fn ensure_child(&mut self, node_idx: usize, q: usize, cx: f64, cy: f64, hw: f64) -> usize {
        if let Some(c) = self.nodes[node_idx].children[q] {
            c
        } else {
            // Compute child centre based on the parent's geometry.
            let h = hw * 0.5;
            let dx = if q & 1 == 1 { h } else { -h };
            let dy = if q & 2 == 2 { h } else { -h };
            let new_idx = self.nodes.len();
            self.nodes.push(QuadNode {
                cx: cx + dx,
                cy: cy + dy,
                half_width: h,
                count: 0,
                com_x: 0.0,
                com_y: 0.0,
                children: [None; 4],
                point_idx: None,
                is_leaf: true,
            });
            self.nodes[node_idx].children[q] = Some(new_idx);
            new_idx
        }
    }

    /// Walk the tree from `root`, accumulating the Barnes-Hut summary for
    /// query point `(qx, qy)` belonging to `qi`. Returns (force_x, force_y,
    /// Z_contribution) where Z is the partition function term Σⱼ (1+d²)⁻¹.
    fn bh_forces(&self, qi: usize, qx: f64, qy: f64, theta_sq: f64) -> (f64, f64, f64) {
        let mut stack = vec![0_usize];
        let mut fx = 0.0;
        let mut fy = 0.0;
        let mut z = 0.0;
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            if node.count == 0 {
                continue;
            }

            if node.is_leaf {
                if let Some(p) = node.point_idx {
                    if p == qi {
                        continue;
                    } // self
                    let dx = qx - (node.com_x / node.count as f64);
                    let dy = qy - (node.com_y / node.count as f64);
                    let dsq = dx * dx + dy * dy;
                    let inv = 1.0 / (1.0 + dsq);
                    z += node.count as f64 * inv;
                    let m = node.count as f64 * inv * inv;
                    fx += m * dx;
                    fy += m * dy;
                } else if node.count > 0 {
                    // Coincident-points overflow case (count > 0, no point_idx)
                    let cx_com = node.com_x / node.count as f64;
                    let cy_com = node.com_y / node.count as f64;
                    let dx = qx - cx_com;
                    let dy = qy - cy_com;
                    let dsq = dx * dx + dy * dy;
                    let inv = 1.0 / (1.0 + dsq);
                    z += node.count as f64 * inv;
                    let m = node.count as f64 * inv * inv;
                    fx += m * dx;
                    fy += m * dy;
                }
                continue;
            }
            // Internal node: open if cell_width / dist > θ (squared form).
            let cx_com = node.com_x / node.count as f64;
            let cy_com = node.com_y / node.count as f64;
            let dx = qx - cx_com;
            let dy = qy - cy_com;
            let dsq = dx * dx + dy * dy;
            let cell_size = node.half_width * 2.0;
            if cell_size * cell_size < theta_sq * dsq {
                // Summarise the whole cell.
                let inv = 1.0 / (1.0 + dsq);
                z += node.count as f64 * inv;
                let m = node.count as f64 * inv * inv;
                fx += m * dx;
                fy += m * dy;
            } else {
                for c in 0..4 {
                    if let Some(ch) = node.children[c] {
                        stack.push(ch);
                    }
                }
            }
        }
        (fx, fy, z)
    }
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

        let use_bh = self.method == TSneMethod::BarnesHut && k == 2;
        let theta_sq = self.theta * self.theta;

        let mut kl = 0.0_f64;
        let mut n_iter = 0;
        for iter in 0..self.n_iter {
            n_iter = iter + 1;

            let momentum = if iter < 250 { 0.5 } else { 0.8 };
            let p_scale = if iter < exag_iters { exag } else { 1.0 };
            let mut grad = Array2::<f64>::zeros((n, k));

            if use_bh {
                // ── Barnes-Hut path ───────────────────────────────────
                // 1. Build quadtree from current Y.
                let mut xmin = f64::INFINITY;
                let mut xmax = f64::NEG_INFINITY;
                let mut ymin = f64::INFINITY;
                let mut ymax = f64::NEG_INFINITY;
                for i in 0..n {
                    let xi = y[[i, 0]];
                    let yi = y[[i, 1]];
                    if xi < xmin {
                        xmin = xi;
                    }
                    if xi > xmax {
                        xmax = xi;
                    }
                    if yi < ymin {
                        ymin = yi;
                    }
                    if yi > ymax {
                        ymax = yi;
                    }
                }
                let cx = 0.5 * (xmin + xmax);
                let cy = 0.5 * (ymin + ymax);
                let hw = ((xmax - xmin).max(ymax - ymin) * 0.5 + 1e-8).max(1e-8);
                let mut tree = QuadTree::new(cx, cy, hw);
                for i in 0..n {
                    tree.insert(i, y[[i, 0]], y[[i, 1]]);
                }
                // 2. Per-point repulsive forces via tree-walk and partition Z.
                let mut fr_x = vec![0.0_f64; n];
                let mut fr_y = vec![0.0_f64; n];
                let mut z_per = vec![0.0_f64; n];
                let mut z_total = 0.0_f64;
                for i in 0..n {
                    let (fx, fy, zi) = tree.bh_forces(i, y[[i, 0]], y[[i, 1]], theta_sq);
                    fr_x[i] = fx;
                    fr_y[i] = fy;
                    z_per[i] = zi;
                    z_total += zi;
                }
                let z_total = z_total.max(1e-12);

                // 3. Attractive forces from the full P matrix (still O(n²)
                //    for the P part; sparsified P is left for a future
                //    Vantage-Point-tree path. The bulk of t-SNE cost is the
                //    repulsive term, which BH now resolves).
                for i in 0..n {
                    let mut fa_x = 0.0_f64;
                    let mut fa_y = 0.0_f64;
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let dx = y[[i, 0]] - y[[j, 0]];
                        let dy = y[[i, 1]] - y[[j, 1]];
                        let q_num = 1.0 / (1.0 + dx * dx + dy * dy);
                        let p_ij = p_joint[i][j] * p_scale;
                        fa_x += p_ij * q_num * dx;
                        fa_y += p_ij * q_num * dy;
                    }
                    // F_rep already pre-multiplied by q_num²; finalise by
                    // dividing repulsive sum by Z.
                    grad[[i, 0]] = 4.0 * (fa_x - fr_x[i] / z_total);
                    grad[[i, 1]] = 4.0 * (fa_y - fr_y[i] / z_total);
                }
            } else {
                // ── Exact path ────────────────────────────────────────
                let mut q_num = vec![vec![0.0_f64; n]; n];
                let mut q_sum = 0.0_f64;
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
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

                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let p_ij = p_joint[i][j] * p_scale;
                        let q_ij = q_num[i][j] / q_sum;
                        let diff = p_ij - q_ij;
                        let f = 4.0 * diff * q_num[i][j];
                        for c in 0..k {
                            grad[[i, c]] += f * (y[[i, c]] - y[[j, c]]);
                        }
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

            // KL divergence (full O(n²) computation at the end).
            if iter + 1 == self.n_iter {
                // Recompute q_num + q_sum for the final state.
                let mut z = 0.0_f64;
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let mut sd = 0.0;
                        for c in 0..k {
                            let dv = y[[i, c]] - y[[j, c]];
                            sd += dv * dv;
                        }
                        z += 1.0 / (1.0 + sd);
                    }
                }
                let z = z.max(1e-12);
                let mut k_d = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let mut sd = 0.0;
                        for c in 0..k {
                            let dv = y[[i, c]] - y[[j, c]];
                            sd += dv * dv;
                        }
                        let q_ij = (1.0 / (1.0 + sd)) / z;
                        let p_ij = p_joint[i][j];
                        k_d += p_ij * (p_ij / q_ij.max(1e-12)).ln();
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
            let s2 = ((i + 100) as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            x[[n_per + i, 0]] = 10.0 + ((s2 >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.3;
            x[[n_per + i, 1]] =
                10.0 + (((s2.wrapping_mul(13)) >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.3;
        }
        let fitted = TSne::new(2)
            .with_perplexity(5.0)
            .with_learning_rate(10.0)
            .with_n_iter(500)
            .with_seed(1)
            .fit(&x)
            .unwrap();
        let y = fitted.embedding;
        // Compute centroids.
        let mut ca = [0.0; 2];
        let mut cb = [0.0; 2];
        for i in 0..n_per {
            ca[0] += y[[i, 0]];
            ca[1] += y[[i, 1]];
            cb[0] += y[[n_per + i, 0]];
            cb[1] += y[[n_per + i, 1]];
        }
        for v in ca.iter_mut() {
            *v /= n_per as f64;
        }
        for v in cb.iter_mut() {
            *v /= n_per as f64;
        }
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
                + (y[[n_per + i, 1]] - mid[1]) * dir[1])
                / dir_norm;
            if pa < 0.0 {
                a_correct += 1;
            }
            if pb > 0.0 {
                b_correct += 1;
            }
        }
        // 80% of each cluster must land on the correct side of the
        // centroid-midpoint hyperplane. t-SNE on small datasets is finicky;
        // perfect linear separability isn't guaranteed.
        let thr = (n_per * 4) / 5;
        assert!(
            a_correct >= thr && b_correct >= thr,
            "linear-separability failed: A={}, B={}, threshold={}",
            a_correct,
            b_correct,
            thr,
        );
    }

    #[test]
    fn test_tsne_barnes_hut_separates_two_blobs() {
        // Larger dataset where BH actually helps. 60 points × 5 dims.
        let n_per = 30;
        let mut x = Array2::<f64>::zeros((2 * n_per, 5));
        for i in 0..n_per {
            let s = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            x[[i, 0]] = ((s >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.5;
            x[[i, 1]] = (((s.wrapping_mul(13)) >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.5;
            let s2 = ((i + 200) as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            x[[n_per + i, 0]] = 15.0 + ((s2 >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.5;
            x[[n_per + i, 1]] =
                15.0 + (((s2.wrapping_mul(13)) >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.5;
        }
        let fitted = TSne::new(2)
            .with_perplexity(10.0)
            .with_learning_rate(15.0)
            .with_n_iter(500)
            .with_seed(7)
            .with_method(TSneMethod::BarnesHut)
            .with_theta(0.5)
            .fit(&x)
            .unwrap();
        let y = fitted.embedding;
        let mut ca = [0.0; 2];
        let mut cb = [0.0; 2];
        for i in 0..n_per {
            ca[0] += y[[i, 0]];
            ca[1] += y[[i, 1]];
            cb[0] += y[[n_per + i, 0]];
            cb[1] += y[[n_per + i, 1]];
        }
        for v in ca.iter_mut() {
            *v /= n_per as f64;
        }
        for v in cb.iter_mut() {
            *v /= n_per as f64;
        }
        let dir = [cb[0] - ca[0], cb[1] - ca[1]];
        let dir_norm = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-12);
        let mid = [(ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0];
        let mut a_correct = 0;
        let mut b_correct = 0;
        for i in 0..n_per {
            let pa = ((y[[i, 0]] - mid[0]) * dir[0] + (y[[i, 1]] - mid[1]) * dir[1]) / dir_norm;
            let pb = ((y[[n_per + i, 0]] - mid[0]) * dir[0]
                + (y[[n_per + i, 1]] - mid[1]) * dir[1])
                / dir_norm;
            if pa < 0.0 {
                a_correct += 1;
            }
            if pb > 0.0 {
                b_correct += 1;
            }
        }
        let thr = (n_per * 4) / 5;
        assert!(
            a_correct >= thr && b_correct >= thr,
            "BH linear-separability failed: A={}, B={}, threshold={}",
            a_correct,
            b_correct,
            thr,
        );
    }
}
