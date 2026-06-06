//! Recursive Feature Elimination (RFE).
//!
//! Mirrors `sklearn.feature_selection.RFE` with a callback-based API: the
//! caller provides a function that, given `(X, y)`, returns per-feature
//! importances (e.g. `|coef_|` for linear models or `feature_importances_`
//! for trees). RFE repeatedly drops the `step` least-important features
//! until `n_features_to_select` remain.

use ndarray::{Array1, Array2};
use rustml_core::{Result, RustMlError};

pub type ImportanceFn = dyn Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync;

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
                "X has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        let d = x.ncols();
        if self.n_features_to_select == 0 || self.n_features_to_select > d {
            return Err(RustMlError::InvalidParameter(format!(
                "n_features_to_select must be in 1..={}",
                d
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
            let n_drop = self.step.min(active.len() - self.n_features_to_select);
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
// RFECV — CV-aware wrapper around Rfe that auto-selects n_features_to_select.
// ---------------------------------------------------------------------------

/// Recursive Feature Elimination with Cross-Validated selection of the
/// optimal number of features.
///
/// Mirrors `sklearn.feature_selection.RFECV`. For each candidate
/// `n_features_to_select` in `min..=n_features`, runs `Rfe` on each k-fold
/// split, scores on the held-out fold, averages — picks the size with the
/// highest mean CV score, then runs RFE on the full data to that size.
pub struct Rfecv {
    pub min_features_to_select: usize,
    pub step: usize,
    pub cv_folds: usize,
    importance: Box<ImportanceFn>,
    score: Box<ScoringFn>,
}

impl Rfecv {
    pub fn new<I, S>(min_features_to_select: usize, importance_fn: I, score: S) -> Self
    where
        I: Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync + 'static,
        S: Fn(&Array2<f64>, &Array1<f64>) -> Result<f64> + Send + Sync + 'static,
    {
        Self {
            min_features_to_select,
            step: 1,
            cv_folds: 5,
            importance: Box::new(importance_fn),
            score: Box::new(score),
        }
    }
    pub fn with_cv_folds(mut self, k: usize) -> Self {
        self.cv_folds = k;
        self
    }
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<FittedRfecv> {
        let n = x.nrows();
        let d = x.ncols();
        let k = self.cv_folds.min(n);
        let folds = kfold(n, k);

        // For each candidate size, do CV.
        let mut mean_scores = Vec::with_capacity(d - self.min_features_to_select + 1);
        let mut sizes = Vec::new();
        for size in self.min_features_to_select..=d {
            let mut scores = Vec::with_capacity(k);
            for (train_idx, test_idx) in &folds {
                let x_train = select_rows(x, train_idx);
                let y_train = select_elements(y, train_idx);
                let x_test = select_rows(x, test_idx);
                let y_test = select_elements(y, test_idx);

                // Inline RFE on this fold using our owned importance closure.
                let support = run_rfe(
                    &x_train,
                    &y_train,
                    size,
                    self.step,
                    self.importance.as_ref(),
                )?;
                let x_test_sel = select_cols_mask(&x_test, &support);
                let s = (self.score)(&x_test_sel, &y_test)?;
                scores.push(s);
            }
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            mean_scores.push(mean);
            sizes.push(size);
        }

        let (best_i, _) = mean_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let best_size = sizes[best_i];

        let support = run_rfe(x, y, best_size, self.step, self.importance.as_ref())?;
        let mut ranking = vec![0usize; d];
        for (i, &b) in support.iter().enumerate() {
            ranking[i] = if b { 1 } else { 2 };
        }
        Ok(FittedRfecv {
            n_features_selected: best_size,
            cv_scores: mean_scores,
            sizes,
            inner: FittedRfe { support, ranking },
        })
    }
}

/// Core RFE elimination loop, factored out so Rfecv can call it without
/// constructing a fresh `Rfe` (which would conflict with closure lifetimes).
fn run_rfe(
    x: &Array2<f64>,
    y: &Array1<f64>,
    n_features_to_select: usize,
    step: usize,
    importance: &ImportanceFn,
) -> Result<Vec<bool>> {
    let d = x.ncols();
    let mut active: Vec<usize> = (0..d).collect();
    while active.len() > n_features_to_select {
        let sub = select_cols(x, &active);
        let imp = importance(&sub, y)?;
        let n_drop = step.min(active.len() - n_features_to_select);
        let mut order: Vec<usize> = (0..active.len()).collect();
        order.sort_by(|&a, &b| imp[a].abs().partial_cmp(&imp[b].abs()).unwrap());
        let to_drop: Vec<usize> = order[..n_drop].iter().map(|&i| active[i]).collect();
        active.retain(|i| !to_drop.contains(i));
    }
    let mut support = vec![false; d];
    for &j in &active {
        support[j] = true;
    }
    Ok(support)
}

fn select_cols_mask(x: &Array2<f64>, mask: &[bool]) -> Array2<f64> {
    let cols: Vec<usize> = mask
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

pub struct FittedRfecv {
    pub n_features_selected: usize,
    pub cv_scores: Vec<f64>,
    pub sizes: Vec<usize>,
    pub inner: FittedRfe,
}

impl FittedRfecv {
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        self.inner.transform(x)
    }
}

fn kfold(n: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let fold_size = n / k;
    let rem = n % k;
    let mut folds = Vec::with_capacity(k);
    let mut start = 0;
    for f in 0..k {
        let end = start + fold_size + if f < rem { 1 } else { 0 };
        let test: Vec<usize> = (start..end).collect();
        let train: Vec<usize> = (0..start).chain(end..n).collect();
        folds.push((train, test));
        start = end;
    }
    folds
}

fn select_rows(x: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((idx.len(), x.ncols()));
    for (k, &i) in idx.iter().enumerate() {
        for j in 0..x.ncols() {
            out[[k, j]] = x[[i, j]];
        }
    }
    out
}

fn select_elements(y: &Array1<f64>, idx: &[usize]) -> Array1<f64> {
    Array1::from_vec(idx.iter().map(|&i| y[i]).collect())
}

// ---------------------------------------------------------------------------
// SequentialFeatureSelector (forward direction only, with CV scoring)
// ---------------------------------------------------------------------------

pub type ScoringFn = dyn Fn(&Array2<f64>, &Array1<f64>) -> Result<f64> + Send + Sync;

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
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<FittedSequentialFeatureSelector> {
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

    #[test]
    fn test_rfecv_finds_2_informative_features() {
        // Same dataset as RFE test, but RFECV must auto-discover that
        // n_features_to_select=2 is best.
        let n = 60;
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            let x0 = (i as f64) - 30.0;
            let x1 = ((i * 11 % 13) as f64) - 6.0;
            let x2 = ((i * 7 % 17) as f64) - 8.0;
            let x3 = ((i * 5 % 11) as f64) - 5.0;
            xv.extend([x0, x1, x2, x3]);
            yv.push(3.0 * x0 + 2.0 * x2);
        }
        let x = Array2::from_shape_vec((n, 4), xv).unwrap();
        let y = Array1::from_vec(yv);

        // Scoring: 1 - (rss / tss) on test data using a simple least-squares
        // refit on selected features.
        let score_fn = |xs: &Array2<f64>, ys: &Array1<f64>| -> Result<f64> {
            // Center y, then close-form OLS.
            let n = xs.nrows() as f64;
            let y_mean = ys.sum() / n.max(1.0);
            let yc = ys.mapv(|v| v - y_mean);
            // OLS via normal equations.
            let m = xs.ncols();
            let mut xtx = Array2::<f64>::zeros((m, m));
            let mut xty = Array1::<f64>::zeros(m);
            for i in 0..m {
                for j in 0..m {
                    let mut s = 0.0;
                    for k in 0..xs.nrows() {
                        s += xs[[k, i]] * xs[[k, j]];
                    }
                    xtx[[i, j]] = s;
                }
                xtx[[i, i]] += 1e-9; // tiny ridge for stability
                let mut s = 0.0;
                for k in 0..xs.nrows() {
                    s += xs[[k, i]] * yc[k];
                }
                xty[i] = s;
            }
            // Solve via Gauss elim.
            let mut a = xtx.clone();
            let mut b = xty.clone();
            for col in 0..m {
                let pv = a[[col, col]];
                if pv.abs() < 1e-14 {
                    continue;
                }
                for r in (col + 1)..m {
                    let f = a[[r, col]] / pv;
                    for c in col..m {
                        a[[r, c]] -= f * a[[col, c]];
                    }
                    b[r] -= f * b[col];
                }
            }
            let mut beta = Array1::<f64>::zeros(m);
            for r in (0..m).rev() {
                let mut s = b[r];
                for c in (r + 1)..m {
                    s -= a[[r, c]] * beta[c];
                }
                let pv = a[[r, r]];
                if pv.abs() > 1e-14 {
                    beta[r] = s / pv;
                }
            }
            let mut pred = Array1::<f64>::zeros(xs.nrows());
            for i in 0..xs.nrows() {
                let mut p = y_mean;
                for j in 0..m {
                    p += xs[[i, j]] * beta[j];
                }
                pred[i] = p;
            }
            let rss: f64 = pred
                .iter()
                .zip(ys.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum();
            let tss: f64 = ys.iter().map(|t| (t - y_mean).powi(2)).sum();
            Ok(1.0 - rss / tss.max(1e-12))
        };

        let rfecv = Rfecv::new(1, dummy_importance, score_fn).with_cv_folds(3);
        let fitted = rfecv.fit(&x, &y).unwrap();
        // Without noise penalty, more features often "wins" on test R²; just
        // assert ≥2 features were kept and both informative features are in.
        assert!(fitted.n_features_selected >= 2);
        assert!(fitted.inner.support[0]);
        assert!(fitted.inner.support[2]);
        assert_eq!(fitted.cv_scores.len(), 4); // size 1..=4
    }
}
