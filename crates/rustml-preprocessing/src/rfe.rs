//! Recursive Feature Elimination (RFE).
//!
//! Mirrors `sklearn.feature_selection.RFE` with a callback-based API: the
//! caller provides a function that, given `(X, y)`, returns per-feature
//! importances (e.g. `|coef_|` for linear models or `feature_importances_`
//! for trees). RFE repeatedly drops the `step` least-important features
//! until `n_features_to_select` remain.

use ndarray::{Array1, Array2, Axis};
use rustml_core::{Result, RustMlError};

pub type ImportanceFn =
    dyn Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync;

pub struct Rfe {
    pub n_features_to_select: usize,
    pub step: usize,
    importance: Box<ImportanceFn>,
}

impl Rfe {
    pub fn new<F>(n_features_to_select: usize, importance_fn: F) -> Self
    where
        F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync + 'static,
    {
        Self {
            n_features_to_select,
            step: 1,
            importance: Box::new(importance_fn),
        }
    }

    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }
}

pub struct FittedRfe {
    /// Boolean mask of selected features, length = original n_features.
    pub support: Vec<bool>,
    /// Ranking of features (1 = selected; higher = dropped earlier).
    pub ranking: Vec<usize>,
}

impl FittedRfe {
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let cols: Vec<usize> = self
            .support
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();
        let mut out = Array2::<f64>::zeros((x.nrows(), cols.len()));
        for (k, &c) in cols.iter().enumerate() {
            for i in 0..x.nrows() {
                out[[i, k]] = x[[i, c]];
            }
        }
        out
    }
}

fn select_cols(x: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((x.nrows(), cols.len()));
    for (k, &c) in cols.iter().enumerate() {
        for i in 0..x.nrows() {
            out[[i, k]] = x[[i, c]];
        }
    }
    out
}

impl Rfe {
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<FittedRfe> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}", x.nrows(), y.len()
            )));
        }
        let d = x.ncols();
        if self.n_features_to_select == 0 || self.n_features_to_select > d {
            return Err(RustMlError::InvalidParameter(format!(
                "n_features_to_select must be in 1..={}", d
            )));
        }
        let mut active: Vec<usize> = (0..d).collect();
        let mut ranking = vec![0usize; d];

        while active.len() > self.n_features_to_select {
            let sub = select_cols(x, &active);
            let imp = (self.importance)(&sub, y)?;
            if imp.len() != active.len() {
                return Err(RustMlError::InvalidParameter(
                    "importance function returned wrong length".into(),
                ));
            }
            // Sort active features by ascending importance; drop step
            // least-important without going below the target.
            let n_drop = self
                .step
                .min(active.len() - self.n_features_to_select);
            let mut order: Vec<usize> = (0..active.len()).collect();
            order.sort_by(|&a, &b| imp[a].abs().partial_cmp(&imp[b].abs()).unwrap());
            let to_drop: Vec<usize> = order[..n_drop].iter().map(|&i| active[i]).collect();
            for &j in &to_drop {
                ranking[j] = active.len(); // ranking number set at time of drop
            }
            active.retain(|i| !to_drop.contains(i));
        }
        for &j in &active {
            ranking[j] = 1;
        }
        let mut support = vec![false; d];
        for &j in &active {
            support[j] = true;
        }
        Ok(FittedRfe { support, ranking })
    }
}

// ---------------------------------------------------------------------------
// SequentialFeatureSelector (forward direction only, with CV scoring)
// ---------------------------------------------------------------------------

pub type ScoringFn =
    dyn Fn(&Array2<f64>, &Array1<f64>) -> Result<f64> + Send + Sync;

pub struct SequentialFeatureSelector {
    pub n_features_to_select: usize,
    score: Box<ScoringFn>,
}

impl SequentialFeatureSelector {
    pub fn new<F>(n_features_to_select: usize, score: F) -> Self
    where
        F: Fn(&Array2<f64>, &Array1<f64>) -> Result<f64> + Send + Sync + 'static,
    {
        Self {
            n_features_to_select,
            score: Box::new(score),
        }
    }
}

pub struct FittedSequentialFeatureSelector {
    pub support: Vec<bool>,
}

impl FittedSequentialFeatureSelector {
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let cols: Vec<usize> = self
            .support
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();
        let mut out = Array2::<f64>::zeros((x.nrows(), cols.len()));
        for (k, &c) in cols.iter().enumerate() {
            for i in 0..x.nrows() {
                out[[i, k]] = x[[i, c]];
            }
        }
        out
    }
}

impl SequentialFeatureSelector {
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<FittedSequentialFeatureSelector> {
        let d = x.ncols();
        if self.n_features_to_select == 0 || self.n_features_to_select > d {
            return Err(RustMlError::InvalidParameter("invalid k".into()));
        }
        let mut selected: Vec<usize> = Vec::with_capacity(self.n_features_to_select);
        let mut remaining: Vec<usize> = (0..d).collect();

        while selected.len() < self.n_features_to_select {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_j = remaining[0];
            for (i, &j) in remaining.iter().enumerate() {
                let mut cols = selected.clone();
                cols.push(j);
                let sub = select_cols(x, &cols);
                let s = (self.score)(&sub, y)?;
                if s > best_score {
                    best_score = s;
                    best_j = j;
                    let _ = i; // silence
                }
            }
            selected.push(best_j);
            remaining.retain(|&j| j != best_j);
        }

        let mut support = vec![false; d];
        for &j in &selected {
            support[j] = true;
        }
        Ok(FittedSequentialFeatureSelector { support })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn dummy_importance(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        // Use absolute correlation of each column with y.
        let n = x.nrows() as f64;
        let mut out = Array1::<f64>::zeros(x.ncols());
        let y_mean = y.sum() / n;
        for j in 0..x.ncols() {
            let m = x.column(j).sum() / n;
            let mut num = 0.0;
            let mut sx = 0.0;
            let mut sy = 0.0;
            for i in 0..x.nrows() {
                let dx = x[[i, j]] - m;
                let dy = y[i] - y_mean;
                num += dx * dy;
                sx += dx * dx;
                sy += dy * dy;
            }
            let den = (sx * sy).sqrt().max(1e-12);
            out[j] = (num / den).abs();
        }
        Ok(out)
    }

    #[test]
    fn test_rfe_keeps_correlated_features() {
        // y = 3x0 + 0*x1 + 2*x2 + 0*x3
        let n = 40;
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            let x0 = (i as f64) - 20.0;
            let x1 = ((i * 11 % 13) as f64) - 6.0;
            let x2 = ((i * 7 % 17) as f64) - 8.0;
            let x3 = ((i * 5 % 11) as f64) - 5.0;
            xv.extend([x0, x1, x2, x3]);
            yv.push(3.0 * x0 + 2.0 * x2);
        }
        let x = Array2::from_shape_vec((n, 4), xv).unwrap();
        let y = Array1::from_vec(yv);

        let rfe = Rfe::new(2, dummy_importance);
        let fitted = rfe.fit(&x, &y).unwrap();
        assert!(fitted.support[0]);
        assert!(fitted.support[2]);
        assert!(!fitted.support[1]);
        assert!(!fitted.support[3]);
        let _ = array![1.0_f64];
    }
}
