//! Neighbourhood components analysis (sklearn's
//! `NeighborhoodComponentsAnalysis`): the linear map `A` (`n_components ×
//! n_features`) maximising the expected number of training rows whose
//! stochastic nearest neighbour (softmax over `−‖A xᵢ − A xⱼ‖²`, `j ≠ i`)
//! shares their class.
//!
//! The initial map follows sklearn's `init="auto"`: LDA's scalings when
//! `n_components ≤ min(n_features, n_classes − 1)`, else PCA's components
//! when `n_components < min(n_features, n_samples)`, else the identity. The
//! objective and gradient are sklearn's `_loss_grad_lbfgs`; the optimiser
//! is [`anofox_ml_core::minimize_lbfgsb`] with `maxiter = max_iter`, `ftol =
//! gtol = tol`, as sklearn's `scipy.optimize.minimize(tol=tol)`. Because the
//! L-BFGS iterates are not SciPy's, the result matches sklearn to the
//! optimiser's tolerance, not bit for bit.

use anofox_ml_core::{
    minimize_lbfgsb, Fit, FitUnsupervised, LbfgsbOptions, Result, RustMlError, Transform,
};
use anofox_ml_discriminant::LinearDiscriminantAnalysis;
use anofox_ml_preprocessing::Pca;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// The initial linear map.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum NcaInit {
    /// sklearn's `"auto"` (see the module docs).
    #[default]
    Auto,
    Pca,
    Lda,
    Identity,
    /// An explicit `n_components × n_features` map, row by row.
    Matrix(Vec<Vec<f64>>),
}

/// Neighbourhood components analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NeighborhoodComponentsAnalysis {
    /// Output dimension (`None`: `n_features`).
    pub n_components: Option<usize>,
    pub init: NcaInit,
    /// L-BFGS iterations (sklearn's default 50).
    pub max_iter: usize,
    /// L-BFGS `ftol` and `gtol` (sklearn's default `1e-5`).
    pub tol: f64,
}

impl Default for NeighborhoodComponentsAnalysis {
    fn default() -> Self {
        Self {
            n_components: None,
            init: NcaInit::Auto,
            max_iter: 50,
            tol: 1e-5,
        }
    }
}

/// A fitted [`NeighborhoodComponentsAnalysis`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedNeighborhoodComponentsAnalysis {
    /// The learned map, `n_components × n_features` (sklearn's
    /// `components_`).
    pub components: Array2<f64>,
    /// L-BFGS iterations run (sklearn's `n_iter_`).
    pub n_iter: usize,
    /// Whether the optimiser met its tolerance within `max_iter`.
    pub converged: bool,
}

/// Class codes `0..k` of `y` and `k`.
fn encode(y: ArrayView1<f64>) -> Result<(Vec<usize>, usize)> {
    let mut classes: Vec<f64> = y.to_vec();
    if classes.iter().any(|v| !v.is_finite()) {
        return Err(RustMlError::InvalidParameter(
            "labels must be finite".into(),
        ));
    }
    classes.sort_by(f64::total_cmp);
    classes.dedup();
    let codes = y
        .iter()
        .map(|v| classes.partition_point(|c| c < v))
        .collect();
    Ok((codes, classes.len()))
}

impl NeighborhoodComponentsAnalysis {
    fn initial(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        n_components: usize,
        n_classes: usize,
    ) -> Result<Array2<f64>> {
        let (n, p) = x.dim();
        let init = match &self.init {
            NcaInit::Auto if n_components <= p.min(n_classes.saturating_sub(1)) => NcaInit::Lda,
            NcaInit::Auto if n_components < p.min(n) => NcaInit::Pca,
            NcaInit::Auto => NcaInit::Identity,
            other => other.clone(),
        };
        Ok(match init {
            NcaInit::Identity | NcaInit::Auto => {
                Array2::from_shape_fn((n_components, p), |(i, j)| if i == j { 1.0 } else { 0.0 })
            }
            NcaInit::Pca => {
                let pca = FitUnsupervised::fit(&Pca::new(n_components), &x.to_owned())?;
                pca.components().to_owned()
            }
            NcaInit::Lda => {
                let lda = Fit::fit(
                    &LinearDiscriminantAnalysis::new(),
                    &x.to_owned(),
                    &y.to_owned(),
                )?;
                if lda.scalings.ncols() < n_components {
                    return Err(RustMlError::InvalidParameter(format!(
                        "init=lda gives at most {} components, n_components is {n_components}",
                        lda.scalings.ncols()
                    )));
                }
                lda.scalings
                    .t()
                    .slice(ndarray::s![..n_components, ..])
                    .to_owned()
            }
            NcaInit::Matrix(rows) => {
                if rows.len() != n_components || rows.iter().any(|r| r.len() != p) {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "init must be {n_components} × {p}"
                    )));
                }
                Array2::from_shape_fn((n_components, p), |(i, j)| rows[i][j])
            }
        })
    }
}

/// sklearn's `_loss_grad_lbfgs` (sign +1): the expected number of
/// correctly classified rows and its gradient with respect to `A`.
fn loss_grad(a: &Array2<f64>, x: &Array2<f64>, codes: &[usize]) -> (f64, Array2<f64>) {
    let n = x.nrows();
    let e = x.dot(&a.t()); // n × c
    let sq: Array1<f64> = e.rows().into_iter().map(|r| r.dot(&r)).collect();
    let gram = e.dot(&e.t());
    let mut p = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let mut max = f64::NEG_INFINITY;
        for j in 0..n {
            if i != j {
                let d = (sq[i] + sq[j] - 2.0 * gram[[i, j]]).max(0.0);
                p[[i, j]] = -d;
                max = max.max(-d);
            }
        }
        let mut sum = 0.0;
        for j in 0..n {
            p[[i, j]] = if i == j { 0.0 } else { (p[[i, j]] - max).exp() };
            sum += p[[i, j]];
        }
        p.row_mut(i).mapv_inplace(|v| v / sum);
    }
    let mut loss = 0.0;
    let mut w = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let pi: f64 = (0..n)
            .filter(|&j| codes[j] == codes[i])
            .map(|j| p[[i, j]])
            .sum();
        loss += pi;
        for j in 0..n {
            let masked = if codes[j] == codes[i] { p[[i, j]] } else { 0.0 };
            w[[i, j]] = masked - p[[i, j]] * pi;
        }
    }
    let col_sums = w.sum_axis(Axis(0));
    let mut sym = &w + &w.t();
    for i in 0..n {
        sym[[i, i]] = -col_sums[i];
    }
    let grad = e.t().dot(&sym).dot(x) * 2.0;
    (loss, grad)
}

impl Fit<f64> for NeighborhoodComponentsAnalysis {
    type Fitted = FittedNeighborhoodComponentsAnalysis;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        let (n, p) = x.dim();
        if n < 2 || p == 0 {
            return Err(RustMlError::EmptyInput(
                "NCA needs at least 2 rows and 1 feature".into(),
            ));
        }
        if y.len() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {n} rows but y has {}",
                y.len()
            )));
        }
        let n_components = match (&self.n_components, &self.init) {
            (Some(c), _) => *c,
            (None, NcaInit::Matrix(rows)) => rows.len(),
            (None, _) => p,
        };
        if n_components == 0 || n_components > p {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components must be in 1..={p}, got {n_components}"
            )));
        }
        let (codes, n_classes) = encode(y)?;
        let a0 = self.initial(x, y, n_components, n_classes)?;
        let x = x.to_owned();
        let opts = LbfgsbOptions {
            max_iter: self.max_iter,
            pgtol: self.tol,
            ftol: self.tol,
            ..Default::default()
        };
        let x0: Vec<f64> = a0.iter().copied().collect();
        let res = minimize_lbfgsb(
            |v, g| {
                let a = Array2::from_shape_vec((n_components, p), v.to_vec()).unwrap();
                let (loss, grad) = loss_grad(&a, &x, &codes);
                for (gi, d) in g.iter_mut().zip(grad.iter()) {
                    *gi = -d;
                }
                -loss
            },
            &x0,
            None,
            &opts,
        )?;
        Ok(FittedNeighborhoodComponentsAnalysis {
            converged: res.converged(),
            n_iter: res.n_iter,
            components: Array2::from_shape_vec((n_components, p), res.x)
                .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?,
        })
    }
}

impl FittedNeighborhoodComponentsAnalysis {
    /// NCA's objective at the learned map on `(x, y)`: the expected number
    /// of rows whose stochastic nearest neighbour shares their class.
    pub fn expected_correct(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<f64> {
        if x.ncols() != self.components.ncols() || x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(
                "x must have the fitted features and one row per label".into(),
            ));
        }
        let (codes, _) = encode(y)?;
        Ok(loss_grad(&self.components, &x.to_owned(), &codes).0)
    }
}

impl Transform<f64> for FittedNeighborhoodComponentsAnalysis {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.components.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.components.ncols(),
                x.ncols()
            )));
        }
        Ok(x.dot(&self.components.t()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The analytic gradient matches central differences.
    #[test]
    fn gradient_matches_finite_differences() {
        let x = Array2::from_shape_fn((12, 3), |(i, j)| ((i * 7 + j * 3) % 11) as f64 / 5.0 - 1.0);
        let codes: Vec<usize> = (0..12).map(|i| i % 3).collect();
        let a = Array2::from_shape_fn((2, 3), |(i, j)| 0.3 + 0.2 * i as f64 - 0.1 * j as f64);
        let (_, g) = loss_grad(&a, &x, &codes);
        let h = 1e-6;
        for i in 0..2 {
            for j in 0..3 {
                let (mut up, mut down) = (a.clone(), a.clone());
                up[[i, j]] += h;
                down[[i, j]] -= h;
                let fd =
                    (loss_grad(&up, &x, &codes).0 - loss_grad(&down, &x, &codes).0) / (2.0 * h);
                assert!(
                    (fd - g[[i, j]]).abs() < 1e-6,
                    "{i},{j}: {fd} vs {}",
                    g[[i, j]]
                );
            }
        }
    }
}
