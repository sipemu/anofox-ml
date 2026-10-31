//! Recursive Feature Elimination (RFE) and its cross-validated variant.
//!
//! Mirrors `sklearn.feature_selection.RFE` / `RFECV` with a callback-based
//! API: the caller provides a function that, given `(X, y)`, returns
//! per-feature importances (e.g. `|coef_|` for linear models or
//! `feature_importances_` for trees). RFE repeatedly drops the `step`
//! least-important features (by absolute importance) until
//! `n_features_to_select` remain.

use anofox_ml_core::{default_cv_splits, CvSplit, Result, RustMlError};
use ndarray::{Array1, Array2};

pub type ImportanceFn = dyn Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync;

/// Scorer used by [`Rfecv`]: `(x_train, y_train, x_test, y_test) -> score`
/// (higher is better). The closure fits its model on the training fold
/// (restricted to the current feature subset) and scores it on the test
/// fold.
pub type CvScoringFn =
    dyn Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>, &Array1<f64>) -> Result<f64> + Send + Sync;

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
    /// Ranking of features (sklearn `ranking_`): 1 = selected, 2 = dropped
    /// in the last elimination round, 3 = the round before, and so on.
    pub ranking: Vec<usize>,
}

impl FittedRfe {
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        select_cols_mask(x, &self.support)
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

/// One recursive-elimination path (sklearn `RFE._fit`).
///
/// `on_step(active)` is called with the active feature indices at every
/// step *before* features are eliminated, and once more for the final
/// subset (this is where RFECV scores the path). Returns `(support,
/// ranking)` with sklearn's `ranking_` semantics.
fn elimination_path(
    x: &Array2<f64>,
    y: &Array1<f64>,
    n_features_to_select: usize,
    step: usize,
    importance: &ImportanceFn,
    mut on_step: impl FnMut(&[usize]) -> Result<()>,
) -> Result<(Vec<bool>, Vec<usize>)> {
    if step == 0 {
        return Err(RustMlError::InvalidParameter(
            "step must be at least 1".into(),
        ));
    }
    let d = x.ncols();
    let mut support = vec![true; d];
    let mut ranking = vec![1usize; d];
    let mut active: Vec<usize> = (0..d).collect();

    while active.len() > n_features_to_select {
        on_step(&active)?;
        let sub = select_cols(x, &active);
        let imp = importance(&sub, y)?;
        if imp.len() != active.len() {
            return Err(RustMlError::InvalidParameter(format!(
                "importance function returned {} values for {} features",
                imp.len(),
                active.len()
            )));
        }
        // Ascending absolute importance (stable; NaN sorts last so it is
        // never eliminated first, like `np.argsort`).
        let n_drop = step.min(active.len() - n_features_to_select);
        let mut order: Vec<usize> = (0..active.len()).collect();
        order.sort_by(|&a, &b| crate::quantile::total_cmp(&imp[a].abs(), &imp[b].abs()));
        for &i in &order[..n_drop] {
            support[active[i]] = false;
        }
        // `ranking_[~support_] += 1`
        for (r, &s) in ranking.iter_mut().zip(&support) {
            if !s {
                *r += 1;
            }
        }
        active.retain(|&j| support[j]);
    }
    on_step(&active)?;
    Ok((support, ranking))
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
        let (support, ranking) = elimination_path(
            x,
            y,
            self.n_features_to_select,
            self.step,
            self.importance.as_ref(),
            |_| Ok(()),
        )?;
        Ok(FittedRfe { support, ranking })
    }
}

// ---------------------------------------------------------------------------
// RFECV — CV-aware wrapper around Rfe that auto-selects n_features_to_select.
// ---------------------------------------------------------------------------

/// Recursive Feature Elimination with Cross-Validated selection of the
/// optimal number of features (sklearn `RFECV`).
///
/// For every CV split a **single** elimination path is run on the training
/// fold (down to `min_features_to_select`), and the scorer is evaluated on
/// the test fold at every step of that path. The number of features with the
/// highest score summed over folds is chosen (ties → fewer features), then
/// RFE is re-run on the full data to that size.
///
/// Folds come from [`default_cv_splits`] (seeded stratified K-fold for
/// class-label targets, seeded shuffled K-fold otherwise) unless explicit
/// splits are given with [`with_cv_splits`](Self::with_cv_splits).
pub struct Rfecv {
    pub min_features_to_select: usize,
    pub step: usize,
    pub cv_folds: usize,
    /// Seed for the default CV splitter.
    pub random_state: u64,
    cv_splits: Option<Vec<CvSplit>>,
    importance: Box<ImportanceFn>,
    score: Box<CvScoringFn>,
}

impl Rfecv {
    /// `importance_fn(x, y)` returns per-feature importances;
    /// `score(x_train, y_train, x_test, y_test)` fits on the training fold
    /// and returns a test-fold score (higher is better).
    pub fn new<I, S>(min_features_to_select: usize, importance_fn: I, score: S) -> Self
    where
        I: Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync + 'static,
        S: Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>, &Array1<f64>) -> Result<f64>
            + Send
            + Sync
            + 'static,
    {
        Self {
            min_features_to_select,
            step: 1,
            cv_folds: 5,
            random_state: 0,
            cv_splits: None,
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
    /// Seed for the default (shuffled / stratified) K-fold splitter.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
        self
    }
    /// Use explicit `(train, test)` index splits instead of the default
    /// splitter (e.g. from `anofox_ml_core::k_fold`).
    pub fn with_cv_splits(mut self, splits: Vec<CvSplit>) -> Self {
        self.cv_splits = Some(splits);
        self
    }

    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<FittedRfecv> {
        let n = x.nrows();
        let d = x.ncols();
        if n != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}",
                n,
                y.len()
            )));
        }
        if d == 0 {
            return Err(RustMlError::EmptyInput("X has no features".into()));
        }
        if self.min_features_to_select == 0 {
            return Err(RustMlError::InvalidParameter(
                "min_features_to_select must be at least 1".into(),
            ));
        }
        let min_select = self.min_features_to_select.min(d);
        let splits = match &self.cv_splits {
            Some(s) => {
                if s.is_empty() {
                    return Err(RustMlError::InvalidParameter(
                        "need at least one (train, test) split".into(),
                    ));
                }
                if s.iter()
                    .flat_map(|(a, b)| a.iter().chain(b))
                    .any(|&i| i >= n)
                {
                    return Err(RustMlError::InvalidParameter(
                        "CV split index out of range".into(),
                    ));
                }
                s.clone()
            }
            None => default_cv_splits(x, y, self.cv_folds, self.random_state)?,
        };

        // One elimination path per fold, scored at every step.
        let mut split_scores: Vec<Vec<f64>> = Vec::with_capacity(splits.len());
        let mut path_sizes: Vec<usize> = Vec::new();
        for (train_idx, test_idx) in &splits {
            let x_train = select_rows(x, train_idx);
            let y_train = select_elements(y, train_idx);
            let x_test = select_rows(x, test_idx);
            let y_test = select_elements(y, test_idx);
            let mut scores = Vec::new();
            let mut sizes = Vec::new();
            elimination_path(
                &x_train,
                &y_train,
                min_select,
                self.step,
                self.importance.as_ref(),
                |active| {
                    let s = (self.score)(
                        &select_cols(&x_train, active),
                        &y_train,
                        &select_cols(&x_test, active),
                        &y_test,
                    )?;
                    scores.push(s);
                    sizes.push(active.len());
                    Ok(())
                },
            )?;
            // Ascending number of features.
            scores.reverse();
            sizes.reverse();
            path_sizes = sizes;
            split_scores.push(scores);
        }

        let n_steps = path_sizes.len();
        let n_splits = split_scores.len() as f64;
        let sums: Vec<f64> = (0..n_steps)
            .map(|s| split_scores.iter().map(|f| f[s]).sum())
            .collect();
        let mean_scores: Vec<f64> = sums.iter().map(|v| v / n_splits).collect();
        let std_scores: Vec<f64> = (0..n_steps)
            .map(|s| {
                let m = mean_scores[s];
                (split_scores.iter().map(|f| (f[s] - m).powi(2)).sum::<f64>() / n_splits).sqrt()
            })
            .collect();
        // First maximum in ascending size order => fewest features on ties.
        let mut best_i = 0;
        for i in 1..n_steps {
            if crate::quantile::total_cmp_nan_low(&sums[i], &sums[best_i]).is_gt() {
                best_i = i;
            }
        }
        let best_size = path_sizes[best_i];

        let (support, ranking) =
            elimination_path(x, y, best_size, self.step, self.importance.as_ref(), |_| {
                Ok(())
            })?;
        Ok(FittedRfecv {
            n_features_selected: best_size,
            cv_scores: mean_scores,
            cv_std: std_scores,
            split_scores,
            sizes: path_sizes,
            inner: FittedRfe { support, ranking },
        })
    }
}

fn select_cols_mask(x: &Array2<f64>, mask: &[bool]) -> Array2<f64> {
    let cols: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    select_cols(x, &cols)
}

pub struct FittedRfecv {
    pub n_features_selected: usize,
    /// Mean test score over folds per evaluated subset size
    /// (sklearn `cv_results_["mean_test_score"]`), aligned with `sizes`.
    pub cv_scores: Vec<f64>,
    /// Standard deviation of the test score over folds per subset size.
    pub cv_std: Vec<f64>,
    /// Per-fold test scores (`split_scores[fold][step]`), aligned with `sizes`.
    pub split_scores: Vec<Vec<f64>>,
    /// Evaluated numbers of features, ascending
    /// (sklearn `cv_results_["n_features"]`).
    pub sizes: Vec<usize>,
    pub inner: FittedRfe,
}

impl FittedRfecv {
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        self.inner.transform(x)
    }
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

    /// OLS with intercept (centred normal equations, partial pivoting).
    fn ols(x: &Array2<f64>, y: &Array1<f64>) -> (Array1<f64>, f64) {
        let (n, m) = x.dim();
        let xm: Vec<f64> = (0..m).map(|j| x.column(j).sum() / n as f64).collect();
        let ym = y.sum() / n as f64;
        let mut a = Array2::<f64>::zeros((m, m + 1));
        for i in 0..m {
            for j in 0..m {
                a[[i, j]] = (0..n)
                    .map(|k| (x[[k, i]] - xm[i]) * (x[[k, j]] - xm[j]))
                    .sum();
            }
            a[[i, m]] = (0..n).map(|k| (x[[k, i]] - xm[i]) * (y[k] - ym)).sum();
        }
        for c in 0..m {
            let piv = (c..m)
                .max_by(|&p, &q| a[[p, c]].abs().total_cmp(&a[[q, c]].abs()))
                .unwrap();
            for k in 0..=m {
                a.swap([c, k], [piv, k]);
            }
            for r in 0..m {
                if r != c {
                    let f = a[[r, c]] / a[[c, c]];
                    for k in c..=m {
                        a[[r, k]] -= f * a[[c, k]];
                    }
                }
            }
        }
        let beta = Array1::from_iter((0..m).map(|i| a[[i, m]] / a[[i, i]]));
        let intercept = ym - (0..m).map(|j| beta[j] * xm[j]).sum::<f64>();
        (beta, intercept)
    }

    fn ols_importance(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        Ok(ols(x, y).0)
    }

    /// R² on the test fold of an OLS fit on the training fold.
    fn ols_r2(
        xtr: &Array2<f64>,
        ytr: &Array1<f64>,
        xte: &Array2<f64>,
        yte: &Array1<f64>,
    ) -> Result<f64> {
        let (b, c) = ols(xtr, ytr);
        let pred = xte.dot(&b) + c;
        let m = yte.sum() / yte.len() as f64;
        let rss: f64 = pred.iter().zip(yte).map(|(p, t)| (p - t).powi(2)).sum();
        let tss: f64 = yte.iter().map(|t| (t - m).powi(2)).sum();
        Ok(1.0 - rss / tss)
    }

    // Data generated with scikit-learn 1.8.0 / numpy RandomState(3):
    // y = 3 x0 - 2 x2 + 0.6 x4 + noise.
    const X: [[f64; 6]; 30] = [
        [1.789, 0.437, 0.096, -1.863, -0.277, -0.355],
        [-0.083, -0.627, -0.044, -0.477, -1.314, 0.885],
        [0.881, 1.71, 0.05, -0.405, -0.545, -1.546],
        [0.982, -1.101, -1.185, -0.206, 1.486, 0.237],
        [-1.024, -0.713, 0.625, -0.161, -0.769, -0.23],
        [0.745, 1.976, -1.244, -0.626, -0.804, -2.419],
        [-0.924, -1.024, 1.124, -0.132, -1.623, 0.647],
        [-0.356, -1.743, -0.597, -0.589, -0.874, 0.03],
        [-2.248, -0.268, 1.013, 0.853, 1.108, 1.119],
        [1.488, -1.118, 0.846, -1.861, -0.603, -1.914],
        [1.048, 1.334, -0.197, 1.775, -0.675, 0.151],
        [0.153, -1.064, 0.438, 1.939, -1.025, 0.899],
        [-0.155, 1.77, 0.484, 0.676, 0.643, 0.249],
        [-1.396, 1.392, -1.371, 0.239, 0.614, -0.838],
        [0.145, 1.168, -0.024, -0.889, -2.916, -0.972],
        [-0.591, -0.516, -0.96, 0.377, -0.575, -0.109],
        [0.679, -0.855, -0.3, 2.158, 0.874, -1.294],
        [-0.08, 0.564, 1.233, 0.149, -0.531, -0.731],
        [0.645, 0.313, -0.517, -0.189, -0.416, 0.725],
        [-0.69, 0.486, 0.852, 0.486, -0.834, 1.345],
        [-0.678, 0.426, -0.753, -1.744, 0.226, 0.287],
        [-0.077, 0.276, -0.648, -0.737, -0.168, 1.909],
        [0.815, -0.52, 0.559, -0.478, -0.457, 0.859],
        [-0.525, -1.676, -0.906, 0.088, 0.128, 1.242],
        [-0.716, 0.731, 0.426, -0.149, 0.836, 0.492],
        [-0.862, 1.072, -1.221, 0.06, 0.002, 0.425],
        [-0.725, -0.035, -0.141, 0.997, -0.796, 0.073],
        [-0.261, -1.298, 2.676, -0.071, -1.487, 1.409],
        [-1.071, 0.371, 0.863, -0.648, -0.431, -0.54],
        [-0.129, -1.622, -1.236, -0.141, 1.039, 0.632],
    ];
    const Y: [f64; 30] = [
        5.528, -0.741, 2.063, 6.171, -5.393, 3.952, -6.3, -0.317, -7.911, 2.242, 2.956, -1.496,
        -1.086, -1.003, -1.133, -0.433, 3.758, -2.666, 2.691, -4.433, -0.489, 1.01, 1.047, 0.459,
        -2.268, 0.267, -2.026, -7.06, -5.081, 2.592,
    ];

    fn data() -> (Array2<f64>, Array1<f64>) {
        (
            Array2::from_shape_fn((30, 6), |(i, j)| X[i][j]),
            Array1::from_vec(Y.to_vec()),
        )
    }

    #[test]
    fn test_rfe_ranking_matches_sklearn() {
        // Regression for #53: ranking used to be "active.len() at drop
        // time" instead of sklearn's elimination-round rank.
        let (x, y) = data();
        // RFE(LinearRegression(), n_features_to_select=2).ranking_
        let f = Rfe::new(2, ols_importance).fit(&x, &y).unwrap();
        assert_eq!(f.ranking, vec![1, 3, 1, 4, 2, 5]);
        // RFE(LinearRegression(), n_features_to_select=1, step=2).ranking_
        let f = Rfe::new(1, ols_importance)
            .with_step(2)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(f.ranking, vec![1, 3, 2, 4, 3, 4]);
    }

    #[test]
    fn test_rfecv_matches_sklearn_with_explicit_folds() {
        // RFECV(LinearRegression(), cv=KFold(3), scoring="r2"): one path
        // per fold, scorer fitted on the train fold and scored on the test
        // fold.
        let (x, y) = data();
        let folds = anofox_ml_core::k_fold(30, 3).unwrap();
        let f = Rfecv::new(1, ols_importance, ols_r2)
            .with_cv_splits(folds.clone())
            .fit(&x, &y)
            .unwrap();
        assert_eq!(f.sizes, vec![1, 2, 3, 4, 5, 6]);
        let expected = [
            0.4190083449093076,
            0.9523996302626081,
            0.9919079551801736,
            0.9888642366971941,
            0.9873120036260756,
            0.9869803620742612,
        ];
        for (a, b) in f.cv_scores.iter().zip(expected) {
            assert!((a - b).abs() < 1e-10, "{:?}", f.cv_scores);
        }
        assert_eq!(f.n_features_selected, 3);
        assert_eq!(f.inner.ranking, vec![1, 2, 1, 3, 1, 4]);
        assert_eq!(f.split_scores.len(), 3);

        // step=2: path sizes 6 -> 4 -> 2 -> 1.
        let f = Rfecv::new(1, ols_importance, ols_r2)
            .with_step(2)
            .with_cv_splits(folds)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(f.sizes, vec![1, 2, 4, 6]);
        assert_eq!(f.n_features_selected, 4);
        assert_eq!(f.inner.ranking, vec![1, 1, 1, 2, 1, 2]);
        assert!((f.cv_scores[2] - 0.9888642366971941).abs() < 1e-10);
    }

    #[test]
    fn test_rfecv_default_shuffled_folds_are_seeded() {
        let (x, y) = data();
        let a = Rfecv::new(1, ols_importance, ols_r2)
            .with_cv_folds(3)
            .with_random_state(5)
            .fit(&x, &y)
            .unwrap();
        let b = Rfecv::new(1, ols_importance, ols_r2)
            .with_cv_folds(3)
            .with_random_state(5)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(a.cv_scores, b.cv_scores);
        assert!(a.inner.support[0] && a.inner.support[2]);
        assert!(a.n_features_selected >= 2);
        // Scorer must see the training fold: an in-sample-only scorer
        // would make every size look perfect.
        assert!(a.cv_scores[0] < 0.9);
    }
}
