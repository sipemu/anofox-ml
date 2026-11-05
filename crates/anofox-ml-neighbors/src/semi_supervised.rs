//! Graph-based semi-supervised classification (sklearn's `LabelPropagation`
//! and `LabelSpreading`). Unlabelled rows carry the label `−1`.
//!
//! Both build an affinity graph over all rows (an RBF kernel or a
//! k-nearest-neighbour connectivity graph, self included) and iterate the
//! label distributions until their total change is below `tol`.
//! Propagation row-normalises the graph and clamps the labelled rows;
//! spreading uses the symmetric normalised Laplacian and keeps a fraction
//! `1 − alpha` of the initial labels.

use anofox_ml_core::{Fit, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// Affinity of a semi-supervised graph.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GraphKernel {
    /// `exp(−gamma ‖x − y‖²)`.
    Rbf { gamma: f64 },
    /// Connectivity to the `n_neighbors` nearest rows (self included).
    Knn { n_neighbors: usize },
}

impl Default for GraphKernel {
    fn default() -> Self {
        GraphKernel::Rbf { gamma: 20.0 }
    }
}

fn sq_dist(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Indices of the `k` training rows nearest to `q` (ties by index).
fn nearest(train: ArrayView2<f64>, q: ArrayView1<f64>, k: usize) -> Vec<usize> {
    let mut d: Vec<(f64, usize)> = train
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, r)| (sq_dist(r, q), i))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.into_iter().take(k).map(|(_, i)| i).collect()
}

impl GraphKernel {
    /// Affinity between every row of `a` and every row of `b` (`knn`: the
    /// connectivity of each row of `a` to its neighbours in `b`).
    fn matrix(&self, a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
        match *self {
            GraphKernel::Rbf { gamma } => {
                Array2::from_shape_fn((a.nrows(), b.nrows()), |(i, j)| {
                    (-gamma * sq_dist(a.row(i), b.row(j))).exp()
                })
            }
            GraphKernel::Knn { n_neighbors } => {
                let mut m = Array2::zeros((a.nrows(), b.nrows()));
                for (i, r) in a.rows().into_iter().enumerate() {
                    for j in nearest(b, r, n_neighbors) {
                        m[[i, j]] = 1.0;
                    }
                }
                m
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum Variant {
    Propagation,
    Spreading { alpha: f64 },
}

/// Label propagation (Zhu & Ghahramani 2002).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LabelPropagation {
    pub kernel: GraphKernel,
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for LabelPropagation {
    fn default() -> Self {
        Self {
            kernel: GraphKernel::default(),
            max_iter: 1000,
            tol: 1e-3,
        }
    }
}

/// Label spreading (Zhou et al. 2004).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LabelSpreading {
    pub kernel: GraphKernel,
    /// Weight of the neighbours' labels, in (0, 1).
    pub alpha: f64,
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for LabelSpreading {
    fn default() -> Self {
        Self {
            kernel: GraphKernel::default(),
            alpha: 0.2,
            max_iter: 30,
            tol: 1e-3,
        }
    }
}

/// A fitted [`LabelPropagation`] or [`LabelSpreading`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedLabelPropagation {
    kernel: GraphKernel,
    x: Array2<f64>,
    classes: Vec<f64>,
    label_distributions: Array2<f64>,
    transduction: Array1<f64>,
    n_iter: usize,
}

fn fit_graph(
    kernel: GraphKernel,
    variant: Variant,
    max_iter: usize,
    tol: f64,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
) -> Result<FittedLabelPropagation> {
    let n = x.nrows();
    if y.len() != n {
        return Err(RustMlError::ShapeMismatch(format!(
            "x has {n} rows but y has {}",
            y.len()
        )));
    }
    if let GraphKernel::Knn { n_neighbors } = kernel {
        if n_neighbors == 0 || n_neighbors > n {
            return Err(RustMlError::InvalidParameter(format!(
                "n_neighbors must be in [1, {n}]"
            )));
        }
    }
    let mut classes: Vec<f64> = y.iter().copied().filter(|&v| v != -1.0).collect();
    classes.sort_by(f64::total_cmp);
    classes.dedup();
    if classes.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "no labelled rows (all labels are −1)".into(),
        ));
    }
    let k = classes.len();
    let affinity = kernel.matrix(x, x);
    let graph = match variant {
        Variant::Propagation => {
            let mut g = affinity;
            for mut row in g.rows_mut() {
                let s = row.sum();
                row /= s;
            }
            g
        }
        Variant::Spreading { .. } => {
            // −(normalised Laplacian) with a zero diagonal (scipy's
            // `laplacian(normed=True)`: diagonal dropped, column-sum degrees).
            let mut m = affinity;
            for i in 0..n {
                m[[i, i]] = 0.0;
            }
            let w: Vec<f64> = (0..n)
                .map(|j| {
                    let s = m.column(j).sum();
                    if s == 0.0 {
                        1.0
                    } else {
                        s.sqrt()
                    }
                })
                .collect();
            for i in 0..n {
                for j in 0..n {
                    m[[i, j]] /= w[j] * w[i];
                }
            }
            m
        }
    };
    let unlabeled: Vec<bool> = y.iter().map(|&v| v == -1.0).collect();
    let mut dist = Array2::<f64>::zeros((n, k));
    for (i, &v) in y.iter().enumerate() {
        if v != -1.0 {
            dist[[i, classes.partition_point(|&c| c < v)]] = 1.0;
        }
    }
    let mut y_static = dist.clone();
    match variant {
        Variant::Propagation => {
            for (i, &u) in unlabeled.iter().enumerate() {
                if u {
                    y_static.row_mut(i).fill(0.0);
                }
            }
        }
        Variant::Spreading { alpha } => y_static *= 1.0 - alpha,
    }
    let mut previous = Array2::<f64>::zeros((n, k));
    let mut n_iter = 0;
    let mut converged = false;
    for it in 0..max_iter {
        n_iter = it;
        if (&dist - &previous).mapv(f64::abs).sum() < tol {
            converged = true;
            break;
        }
        previous = dist.clone();
        dist = graph.dot(&dist);
        match variant {
            Variant::Propagation => {
                for (i, mut row) in dist.rows_mut().into_iter().enumerate() {
                    if unlabeled[i] {
                        let s = row.sum();
                        if s != 0.0 {
                            row /= s;
                        }
                    } else {
                        row.assign(&y_static.row(i));
                    }
                }
            }
            Variant::Spreading { alpha } => dist = dist * alpha + &y_static,
        }
    }
    if !converged {
        n_iter += 1;
    }
    for mut row in dist.rows_mut() {
        let s = row.sum();
        if s != 0.0 {
            row /= s;
        }
    }
    let transduction = dist
        .rows()
        .into_iter()
        .map(|r| {
            classes[r
                .iter()
                .enumerate()
                .fold(0, |b, (j, &v)| if v > r[b] { j } else { b })]
        })
        .collect();
    Ok(FittedLabelPropagation {
        kernel,
        x: x.to_owned(),
        classes,
        label_distributions: dist,
        transduction,
        n_iter,
    })
}

impl Fit<f64> for LabelPropagation {
    type Fitted = FittedLabelPropagation;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedLabelPropagation> {
        fit_graph(
            self.kernel,
            Variant::Propagation,
            self.max_iter,
            self.tol,
            x,
            y,
        )
    }
}

impl Fit<f64> for LabelSpreading {
    type Fitted = FittedLabelPropagation;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedLabelPropagation> {
        if !(self.alpha > 0.0 && self.alpha < 1.0) {
            return Err(RustMlError::InvalidParameter(format!(
                "alpha must be in (0, 1), got {}",
                self.alpha
            )));
        }
        fit_graph(
            self.kernel,
            Variant::Spreading { alpha: self.alpha },
            self.max_iter,
            self.tol,
            x,
            y,
        )
    }
}

impl FittedLabelPropagation {
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }

    /// Final label distribution of every training row.
    pub fn label_distributions(&self) -> &Array2<f64> {
        &self.label_distributions
    }

    /// Label assigned to every training row (sklearn's `transduction_`).
    pub fn transduction(&self) -> &Array1<f64> {
        &self.transduction
    }

    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl PredictProba<f64> for FittedLabelPropagation {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.x.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x.ncols(),
                x.ncols()
            )));
        }
        let mut p = self
            .kernel
            .matrix(x, self.x.view())
            .dot(&self.label_distributions);
        for mut row in p.rows_mut() {
            let s = row.sum();
            row /= s;
        }
        Ok(p)
    }
}

impl Predict<f64> for FittedLabelPropagation {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let p = self.predict_proba_view(x)?;
        Ok(p.rows()
            .into_iter()
            .map(|r| {
                self.classes[r
                    .iter()
                    .enumerate()
                    .fold(0, |b, (j, &v)| if v > r[b] { j } else { b })]
            })
            .collect())
    }
}
