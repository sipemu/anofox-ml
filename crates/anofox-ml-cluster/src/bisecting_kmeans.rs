//! Bisecting k-means (sklearn's `cluster.BisectingKMeans`): starting from
//! one cluster, repeatedly split the leaf with the largest inertia
//! (`biggest_inertia`) or the most rows (`largest_cluster`) by a 2-means
//! fit, until `n_clusters` leaves remain.
//!
//! Each 2-means fit is sklearn's `_kmeans_single_lloyd` on the mean-centred
//! data: `init` from numpy's `RandomState` (`random`: two rows drawn by
//! weight without replacement; `k-means++`: sklearn's greedy seeding with
//! two local trials), Lloyd steps until the labels repeat or the squared
//! centre shift is at most `tol`. Distances are sklearn's
//! `‖c‖² − 2 x·c` (ties to the first centre), so a seeded fit equals
//! sklearn's up to floating-point summation order.

use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{FitUnsupervisedWeighted, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// sklearn's `init` for the bisections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BisectingInit {
    #[default]
    Random,
    KMeansPlusPlus,
}

/// sklearn's `bisecting_strategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BisectingStrategy {
    #[default]
    BiggestInertia,
    LargestCluster,
}

/// sklearn's `BisectingKMeans` (`algorithm="lloyd"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BisectingKMeans {
    pub n_clusters: usize,
    pub init: BisectingInit,
    /// 2-means fits per bisection (the best inertia is kept).
    pub n_init: usize,
    pub max_iter: usize,
    /// Bound on the summed squared centre shift (sklearn passes `tol` to
    /// each 2-means fit unscaled).
    pub tol: f64,
    pub bisecting_strategy: BisectingStrategy,
    pub seed: u32,
}

impl Default for BisectingKMeans {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            init: BisectingInit::Random,
            n_init: 1,
            max_iter: 300,
            tol: 1e-4,
            bisecting_strategy: BisectingStrategy::BiggestInertia,
            seed: 0,
        }
    }
}

/// One node of the bisection tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    /// Centre in the centred coordinates.
    center: Array1<f64>,
    /// Children (left, right), or the leaf's label.
    children: Option<(usize, usize)>,
    label: usize,
}

/// A fitted [`BisectingKMeans`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedBisectingKMeans {
    /// Leaf centres in leaf order (sklearn's `cluster_centers_`).
    pub cluster_centers: Array2<f64>,
    /// Training labels (sklearn's `labels_`).
    pub labels: Array1<usize>,
    /// Weighted sum of squared distances to the leaf centres
    /// (`inertia_`).
    pub inertia: f64,
    x_mean: Array1<f64>,
    nodes: Vec<Node>,
}

/// Index of the nearest of `centers` by `‖c‖² − 2 x·c` (first on ties).
fn nearest(x: ArrayView1<f64>, centers: &Array2<f64>, sq: &[f64]) -> usize {
    let mut best = (f64::INFINITY, 0);
    for (j, c) in centers.rows().into_iter().enumerate() {
        let d = sq[j] - 2.0 * x.dot(&c);
        if j == 0 || d < best.0 {
            best = (d, j);
        }
    }
    best.1
}

fn sq_norms(c: &Array2<f64>) -> Vec<f64> {
    c.rows().into_iter().map(|r| r.dot(&r)).collect()
}

/// `Σ w_i ‖x_i − c_{l_i}‖²` over the rows labelled `only` (all if `None`).
fn inertia(
    x: ArrayView2<f64>,
    w: &[f64],
    centers: &Array2<f64>,
    labels: &[usize],
    only: Option<usize>,
) -> f64 {
    let mut total = 0.0;
    for (i, row) in x.rows().into_iter().enumerate() {
        if only.is_some_and(|l| l != labels[i]) {
            continue;
        }
        let c = centers.row(labels[i]);
        let d: f64 = row.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
        total += d * w[i];
    }
    total
}

/// sklearn's `_kmeans_plusplus` for two centres (`n_local_trials = 2`).
fn kmeans_plusplus(
    x: ArrayView2<f64>,
    w: &[f64],
    xsq: &[f64],
    rng: &mut RandomState,
) -> Result<Vec<usize>> {
    let n = x.nrows();
    let total: f64 = w.iter().sum();
    let p: Vec<f64> = w.iter().map(|v| v / total).collect();
    let first = rng.choice_weighted(&p, 1, true)?[0];
    // _euclidean_distances(..., squared=True): (−2 x·y + xx) + yy, clipped
    // at 0.
    let dist_to = |c: usize| -> Vec<f64> {
        (0..n)
            .map(|i| ((-2.0 * x.row(c).dot(&x.row(i)) + xsq[c]) + xsq[i]).max(0.0))
            .collect()
    };
    let closest = dist_to(first);
    let pot: f64 = closest.iter().zip(w).map(|(d, v)| d * v).sum();
    let trials = 2;
    let draws: Vec<f64> = (0..trials).map(|_| rng.random_sample() * pot).collect();
    let mut cum = 0.0;
    let cumsum: Vec<f64> = closest
        .iter()
        .zip(w)
        .map(|(d, v)| {
            cum += d * v;
            cum
        })
        .collect();
    let mut best = (f64::INFINITY, first);
    for r in draws {
        let cand = cumsum.partition_point(|&c| c < r).min(n - 1);
        let d = dist_to(cand);
        let p: f64 = d
            .iter()
            .zip(&closest)
            .zip(w)
            .map(|((a, b), v)| a.min(*b) * v)
            .sum();
        if p < best.0 {
            best = (p, cand);
        }
    }
    Ok(vec![first, best.1])
}

/// sklearn's `_kmeans_single_lloyd` with two centres: labels, centres and
/// inertia.
fn lloyd2(
    x: ArrayView2<f64>,
    w: &[f64],
    mut centers: Array2<f64>,
    max_iter: usize,
    tol: f64,
) -> (Vec<usize>, Array2<f64>, f64) {
    let (n, p) = x.dim();
    let k = centers.nrows();
    let mut labels = vec![usize::MAX; n];
    let mut labels_old = labels.clone();
    let mut strict = false;
    for _ in 0..max_iter {
        let sq = sq_norms(&centers);
        let mut new = Array2::<f64>::zeros((k, p));
        let mut weight = vec![0.0; k];
        for (i, row) in x.rows().into_iter().enumerate() {
            let l = nearest(row, &centers, &sq);
            labels[i] = l;
            weight[l] += w[i];
            new.row_mut(l).scaled_add(w[i], &row);
        }
        // sklearn's _relocate_empty_clusters_dense: the farthest rows
        // become the empty clusters' centres.
        let empty: Vec<usize> = (0..k).filter(|&j| weight[j] == 0.0).collect();
        if !empty.is_empty() {
            let mut far: Vec<(f64, usize)> = x
                .rows()
                .into_iter()
                .enumerate()
                .map(|(i, r)| {
                    let c = centers.row(labels[i]);
                    (r.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum(), i)
                })
                .collect();
            far.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (e, &(_, i)) in empty.iter().zip(&far) {
                let old = labels[i];
                new.row_mut(old).scaled_add(-w[i], &x.row(i));
                new.row_mut(*e).assign(&(&x.row(i) * w[i]));
                weight[*e] = w[i];
                weight[old] -= w[i];
            }
        }
        for j in 0..k {
            if weight[j] > 0.0 {
                new.row_mut(j).mapv_inplace(|v| v / weight[j]);
            }
        }
        let shift: f64 = (0..k)
            .map(|j| {
                let d: f64 = centers
                    .row(j)
                    .iter()
                    .zip(new.row(j))
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                d
            })
            .sum();
        centers = new;
        if labels == labels_old {
            strict = true;
            break;
        }
        if shift <= tol {
            break;
        }
        labels_old.clone_from(&labels);
    }
    if !strict {
        let sq = sq_norms(&centers);
        for (i, row) in x.rows().into_iter().enumerate() {
            labels[i] = nearest(row, &centers, &sq);
        }
    }
    let inert = inertia(x, w, &centers, &labels, None);
    (labels, centers, inert)
}

impl FitUnsupervisedWeighted<f64> for BisectingKMeans {
    type Fitted = FittedBisectingKMeans;

    fn fit_unsupervised_weighted_view(
        &self,
        x: ArrayView2<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        let (n, p) = x.dim();
        if n == 0 || p == 0 {
            return Err(RustMlError::EmptyInput("BisectingKMeans needs data".into()));
        }
        if self.n_clusters == 0 || self.n_clusters > n {
            return Err(RustMlError::InvalidParameter(format!(
                "n_clusters must be in 1..={n}, got {}",
                self.n_clusters
            )));
        }
        if self.n_init == 0 || self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_init and max_iter must be at least 1".into(),
            ));
        }
        let w: Vec<f64> = match sample_weight {
            Some(sw) if sw.len() != n => {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight has {} entries for {n} rows",
                    sw.len()
                )))
            }
            Some(sw) => sw.to_vec(),
            None => vec![1.0; n],
        };
        let mut rng = RandomState::new(self.seed);
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let xc = &x - &x_mean;
        let xsq: Vec<f64> = xc.rows().into_iter().map(|r| r.dot(&r)).collect();
        // Leaves: (node, rows, score).
        let mut nodes = vec![Node {
            center: xc.mean_axis(Axis(0)).unwrap(),
            children: None,
            label: 0,
        }];
        let mut leaf_rows: Vec<Option<Vec<usize>>> = vec![Some((0..n).collect())];
        let mut score = vec![0.0];
        for _ in 1..self.n_clusters {
            // The first leaf (in tree order) with the largest score.
            let order = leaf_order(&nodes);
            let target = order
                .iter()
                .copied()
                .fold(None, |best: Option<usize>, l| match best {
                    Some(b) if score[l] <= score[b] => Some(b),
                    _ => Some(l),
                });
            let target = target.expect("at least one leaf");
            let rows = leaf_rows[target].take().expect("leaf rows");
            let xs = xc.select(Axis(0), &rows);
            let ws: Vec<f64> = rows.iter().map(|&i| w[i]).collect();
            let xsq_s: Vec<f64> = rows.iter().map(|&i| xsq[i]).collect();
            let mut best: Option<(Vec<usize>, Array2<f64>, f64)> = None;
            for _ in 0..self.n_init {
                let seeds = match self.init {
                    BisectingInit::Random => {
                        let total: f64 = ws.iter().sum();
                        let pr: Vec<f64> = ws.iter().map(|v| v / total).collect();
                        rng.choice_weighted(&pr, 2, false)?
                    }
                    BisectingInit::KMeansPlusPlus => {
                        kmeans_plusplus(xs.view(), &ws, &xsq_s, &mut rng)?
                    }
                };
                let init = xs.select(Axis(0), &seeds);
                let fit = lloyd2(xs.view(), &ws, init, self.max_iter, self.tol);
                if best.as_ref().is_none_or(|b| fit.2 < b.2 * (1.0 - 1e-6)) {
                    best = Some(fit);
                }
            }
            let (labels, centers, _) = best.expect("n_init >= 1");
            let scores: Vec<f64> = match self.bisecting_strategy {
                BisectingStrategy::BiggestInertia => (0..2)
                    .map(|l| inertia(xs.view(), &ws, &centers, &labels, Some(l)))
                    .collect(),
                BisectingStrategy::LargestCluster => (0..2)
                    .map(|l| labels.iter().filter(|&&v| v == l).count() as f64)
                    .collect(),
            };
            let (left, right) = (nodes.len(), nodes.len() + 1);
            for (side, c) in centers.rows().into_iter().enumerate() {
                nodes.push(Node {
                    center: c.to_owned(),
                    children: None,
                    label: 0,
                });
                leaf_rows.push(Some(
                    rows.iter()
                        .zip(&labels)
                        .filter(|(_, &l)| l == side)
                        .map(|(&i, _)| i)
                        .collect(),
                ));
                score.push(scores[side]);
            }
            nodes[target].children = Some((left, right));
        }
        let mut labels = Array1::<usize>::zeros(n);
        let order = leaf_order(&nodes);
        let mut centers = Array2::<f64>::zeros((order.len(), p));
        for (label, &leaf) in order.iter().enumerate() {
            nodes[leaf].label = label;
            for &i in leaf_rows[leaf].as_ref().expect("leaf rows") {
                labels[i] = label;
            }
            centers
                .row_mut(label)
                .assign(&(&nodes[leaf].center + &x_mean));
        }
        let lab: Vec<usize> = labels.to_vec();
        let inertia_total = inertia(x, &w, &centers, &lab, None);
        Ok(FittedBisectingKMeans {
            cluster_centers: centers,
            labels,
            inertia: inertia_total,
            x_mean,
            nodes,
        })
    }
}

/// Leaves left to right (sklearn's `iter_leaves`).
fn leaf_order(nodes: &[Node]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        match nodes[i].children {
            None => out.push(i),
            Some((l, r)) => {
                stack.push(r);
                stack.push(l);
            }
        }
    }
    out
}

impl FittedBisectingKMeans {
    /// sklearn's `predict`: descend the tree, at each split taking the
    /// nearer child centre.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>> {
        if x.ncols() != self.x_mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_mean.len(),
                x.ncols()
            )));
        }
        Ok(x.rows()
            .into_iter()
            .map(|row| {
                let mut node = 0;
                while let Some((l, r)) = self.nodes[node].children {
                    let pair = ndarray::stack(
                        Axis(0),
                        &[
                            (&self.nodes[l].center + &self.x_mean).view(),
                            (&self.nodes[r].center + &self.x_mean).view(),
                        ],
                    )
                    .expect("two centres");
                    let sq = sq_norms(&pair);
                    node = if nearest(row, &pair, &sq) == 0 { l } else { r };
                }
                self.nodes[node].label
            })
            .collect())
    }
}
