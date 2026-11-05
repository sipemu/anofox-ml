use crate::mutual_information::{mutual_info_classif, mutual_info_regression, DiscreteFeatures};
use crate::special::{chi2_sf, f_sf};
use anofox_ml_core::{Fit, Float, Result, RustMlError, Transform};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Axis, Ix1, Ix2};
use std::collections::HashMap;

/// Pluggable scoring function used by [`SelectKBest`].
///
/// Each variant defines a different univariate statistical test for
/// ranking features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum ScoringFunction {
    /// ANOVA F-value for classification (one-way between-groups F-test).
    ///
    /// For each feature, groups samples by class label and computes
    /// between-class variance / within-class variance. Higher F means the
    /// feature is more discriminative. Requires target labels `y`.
    FClassif,

    /// Univariate linear-regression F-statistic for regression.
    ///
    /// For each feature j, computes the Pearson correlation r with the
    /// target, then F = r^2 * (n-2) / (1 - r^2). Higher F means the
    /// feature has a stronger linear relationship with the target.
    /// Requires target values `y`.
    FRegression,

    /// Chi-squared statistic between non-negative features (counts,
    /// frequencies, one-hot indicators) and class labels (sklearn `chi2`).
    Chi2,

    /// k-NN mutual information with a discrete target (sklearn
    /// `mutual_info_classif`, `n_neighbors = 3`, all features continuous).
    MutualInfoClassif,

    /// k-NN mutual information with a continuous target (sklearn
    /// `mutual_info_regression`, `n_neighbors = 3`, all features continuous).
    MutualInfoRegression,

    /// Feature variance (unsupervised).
    ///
    /// Simply uses the variance of each feature as its score.
    /// Target `y` is ignored when this variant is used.
    Variance,
}

/// Parameters for `SelectKBest` feature selector (unfitted state).
///
/// Selects the top-k features according to a pluggable [`ScoringFunction`].
/// This is more flexible than [`MutualInformationSelector`](crate::MutualInformationSelector),
/// which is hard-coded to mutual information scoring.
///
/// # Example
///
/// ```
/// use anofox_ml_preprocessing::SelectKBest;
/// use anofox_ml_preprocessing::select_k_best::ScoringFunction;
/// use anofox_ml_core::Transform;
/// use ndarray::array;
///
/// // Feature 0 perfectly separates the two classes; feature 1 is noise.
/// let x = array![
///     [0.0, 0.5],
///     [0.0, 0.8],
///     [1.0, 0.3],
///     [1.0, 0.7],
/// ];
/// let y = array![0.0, 0.0, 1.0, 1.0];
///
/// let selector = SelectKBest::new(1, ScoringFunction::FClassif);
/// let fitted = selector.fit(&x, &y).unwrap();
/// let x_selected = fitted.transform(&x).unwrap();
///
/// assert_eq!(x_selected.ncols(), 1);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SelectKBest {
    /// Number of top features to select.
    pub k: usize,
    /// Scoring function to rank features.
    pub scoring_fn: ScoringFunction,
}

impl SelectKBest {
    /// Create a new `SelectKBest` selector that keeps the top `k` features
    /// ranked by the given scoring function.
    pub fn new(k: usize, scoring_fn: ScoringFunction) -> Self {
        Self { k, scoring_fn }
    }

    /// Fit the selector on the given data.
    ///
    /// For [`ScoringFunction::FClassif`] and [`ScoringFunction::FRegression`],
    /// `y` is used as the target variable. For [`ScoringFunction::Variance`],
    /// `y` is ignored.
    pub fn fit<'a, 'b, F: Float>(
        &self,
        x: impl AsArray<'a, F, Ix2>,
        y: impl AsArray<'b, F, Ix1>,
    ) -> Result<FittedSelectKBest<F>> {
        let x: ArrayView2<F> = x.into();
        let y: ArrayView1<F> = y.into();
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        if self.k == 0 {
            return Err(RustMlError::InvalidParameter("k must be at least 1".into()));
        }

        if self.k > n_features {
            return Err(RustMlError::InvalidParameter(format!(
                "k ({}) exceeds number of features ({})",
                self.k, n_features
            )));
        }

        // For supervised modes, validate y length.
        if !matches!(self.scoring_fn, ScoringFunction::Variance) {
            if y.len() != n_samples {
                return Err(RustMlError::ShapeMismatch(format!(
                    "X has {} samples but y has {} elements",
                    n_samples,
                    y.len()
                )));
            }
        }

        let (scores, pvalues) = univariate_scores(&self.scoring_fn, x, y)?;

        // Top-k by score as sklearn: the last k of a stable ascending sort
        // (NaN lowest), so on ties the higher feature indices win.
        let mut feature_scores: Vec<(usize, F)> = scores.iter().copied().enumerate().collect();
        feature_scores
            .sort_by(|a, b| crate::quantile::total_cmp_nan_low(&b.1, &a.1).then(b.0.cmp(&a.0)));

        let mut selected_indices: Vec<usize> = feature_scores
            .iter()
            .take(self.k)
            .map(|&(idx, _)| idx)
            .collect();
        // Sort indices for stable column ordering in transform.
        selected_indices.sort_unstable();

        Ok(FittedSelectKBest {
            scores,
            pvalues,
            selected_indices,
            n_features_in: n_features,
        })
    }
}

/// Per-feature scores and (for the F-tests and χ²) p-values.
pub(crate) fn univariate_scores<F: Float>(
    scoring_fn: &ScoringFunction,
    x: ArrayView2<F>,
    y: ArrayView1<F>,
) -> Result<(Array1<F>, Option<Array1<F>>)> {
    {
        Ok(match scoring_fn {
            ScoringFunction::FClassif => {
                let (s, p) = compute_f_classif(x, y)?;
                (s, Some(p))
            }
            ScoringFunction::FRegression => {
                let (s, p) = compute_f_regression(x, y)?;
                (s, Some(p))
            }
            ScoringFunction::Chi2 => {
                let (s, p) = compute_chi2(x, y)?;
                (s, Some(p))
            }
            ScoringFunction::MutualInfoClassif => (
                mutual_info_classif(x, y, &DiscreteFeatures::Auto, 3, 0)?,
                None,
            ),
            ScoringFunction::MutualInfoRegression => (
                mutual_info_regression(x, y, &DiscreteFeatures::Auto, 3, 0)?,
                None,
            ),
            ScoringFunction::Variance => (compute_variance(x), None),
        })
    }
}

/// Fitted `SelectKBest` -- holds per-feature scores and the indices of the
/// selected top-k features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedSelectKBest<F: Float> {
    /// Per-feature score from the chosen scoring function.
    scores: Array1<F>,
    /// Per-feature p-values (F-tests and chi²; `None` for MI / variance).
    #[serde(default)]
    pvalues: Option<Array1<F>>,
    /// Indices of the top-k features (sorted ascending for stable column ordering).
    selected_indices: Vec<usize>,
    /// Total number of input features (before selection).
    n_features_in: usize,
}

impl<F: Float> FittedSelectKBest<F> {
    pub(crate) fn from_parts(
        scores: Array1<F>,
        pvalues: Option<Array1<F>>,
        selected_indices: Vec<usize>,
        n_features_in: usize,
    ) -> Self {
        Self {
            scores,
            pvalues,
            selected_indices,
            n_features_in,
        }
    }

    /// Per-feature scores computed during fitting.
    pub fn scores(&self) -> &Array1<F> {
        &self.scores
    }

    /// Per-feature p-values (sklearn `pvalues_`) for the F-tests and χ²;
    /// `None` for mutual information and variance scoring.
    pub fn pvalues(&self) -> Option<&Array1<F>> {
        self.pvalues.as_ref()
    }

    /// Indices of the selected features, sorted in ascending order.
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }
}

impl<F: Float> Fit<F> for SelectKBest {
    type Fitted = FittedSelectKBest<F>;
    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<FittedSelectKBest<F>> {
        SelectKBest::fit(self, x, y)
    }
}

impl<F: Float> Transform<F> for FittedSelectKBest<F> {
    fn transform_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features_in {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features_in,
                x.ncols()
            )));
        }

        let n_rows = x.nrows();
        let n_selected = self.selected_indices.len();
        let mut result = Array2::<F>::zeros((n_rows, n_selected));

        for (i, row) in x.rows().into_iter().enumerate() {
            for (out_j, &src_j) in self.selected_indices.iter().enumerate() {
                result[[i, out_j]] = row[src_j];
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Scoring function implementations
// ---------------------------------------------------------------------------

/// Compute ANOVA F-value for each feature (classification).
///
/// For each feature j:
/// - Group samples by class label.
/// - Compute between-class mean square (MSB) and within-class mean square (MSW).
/// - F = MSB / MSW, p = F-distribution survival function with
///   `(n_classes - 1, n_samples - n_classes)` degrees of freedom.
///
/// Like sklearn, a feature with zero within-class variance but non-zero
/// between-class variance gets `F = inf, p = 0`; a constant feature gets
/// `F = 0, p = 1` (sklearn returns NaN there).
fn compute_f_classif<F: Float>(
    x: ArrayView2<F>,
    y: ArrayView1<F>,
) -> Result<(Array1<F>, Array1<F>)> {
    let (n_samples, n_features) = x.dim();
    let n_f = F::from_usize(n_samples).unwrap();

    // Map labels to class indices.
    let mut label_map: HashMap<u64, usize> = HashMap::new();
    let mut class_indices: Vec<usize> = Vec::with_capacity(n_samples);
    for &val in y.iter() {
        let bits = val.to_f64().unwrap().to_bits();
        let next_id = label_map.len();
        let id = *label_map.entry(bits).or_insert(next_id);
        class_indices.push(id);
    }
    let n_classes = label_map.len();

    if n_classes < 2 {
        return Err(RustMlError::InvalidParameter(
            "FClassif requires at least 2 classes".into(),
        ));
    }

    if n_samples <= n_classes {
        return Err(RustMlError::InvalidParameter(
            "not enough samples for FClassif (need more samples than classes)".into(),
        ));
    }

    // Count samples per class.
    let mut class_counts = vec![0usize; n_classes];
    for &c in &class_indices {
        class_counts[c] += 1;
    }

    let mut scores = Array1::<F>::zeros(n_features);
    let mut pvalues = Array1::<F>::ones(n_features);
    let dfb = (n_classes - 1) as f64;
    let dfw = (n_samples - n_classes) as f64;

    for j in 0..n_features {
        let col = x.column(j);

        // Grand mean.
        let grand_mean = col.sum() / n_f;

        // Per-class sum and sum of squares.
        let mut class_sums = vec![F::zero(); n_classes];
        for (i, &val) in col.iter().enumerate() {
            class_sums[class_indices[i]] += val;
        }

        // Between-class sum of squares (SSB).
        let mut ssb = F::zero();
        for c in 0..n_classes {
            let nc = F::from_usize(class_counts[c]).unwrap();
            let class_mean = class_sums[c] / nc;
            let diff = class_mean - grand_mean;
            ssb += nc * diff * diff;
        }

        // Within-class sum of squares (SSW).
        let mut ssw = F::zero();
        for (i, &val) in col.iter().enumerate() {
            let c = class_indices[i];
            let nc = F::from_usize(class_counts[c]).unwrap();
            let class_mean = class_sums[c] / nc;
            let diff = val - class_mean;
            ssw += diff * diff;
        }

        // Degrees of freedom.
        let df_between = F::from_usize(n_classes - 1).unwrap();
        let df_within = F::from_usize(n_samples - n_classes).unwrap();

        let eps = F::from_f64(1e-15).unwrap();
        if ssw < eps {
            // All within-class variance is zero: feature is perfectly
            // discriminative (or constant). Use a large F value.
            scores[j] = if ssb > eps { F::infinity() } else { F::zero() };
        } else {
            let msb = ssb / df_between;
            let msw = ssw / df_within;
            scores[j] = msb / msw;
        }
        pvalues[j] = F::from_f64(f_sf(scores[j].to_f64().unwrap(), dfb, dfw)).unwrap();
    }

    Ok((scores, pvalues))
}

/// Compute univariate linear-regression F-statistic for each feature.
///
/// For each feature j:
///   r = correlation(x[:,j], y)
///   F = r^2 * (n - 2) / (1 - r^2),  p = F.sf(F, 1, n - 2)
fn compute_f_regression<F: Float>(
    x: ArrayView2<F>,
    y: ArrayView1<F>,
) -> Result<(Array1<F>, Array1<F>)> {
    let (n_samples, n_features) = x.dim();

    if n_samples < 3 {
        return Err(RustMlError::InvalidParameter(
            "FRegression requires at least 3 samples".into(),
        ));
    }

    let n_f = F::from_usize(n_samples).unwrap();
    let eps = F::from_f64(1e-15).unwrap();

    // Compute y statistics once.
    let y_mean = y.sum() / n_f;
    let mut y_var = F::zero();
    for &val in y.iter() {
        let diff = val - y_mean;
        y_var += diff * diff;
    }

    let mut scores = Array1::<F>::zeros(n_features);
    let dfw = (n_samples - 2) as f64;

    for j in 0..n_features {
        let col = x.column(j);
        let x_mean = col.sum() / n_f;

        let mut cov_xy = F::zero();
        let mut x_var = F::zero();
        for (&xv, &yv) in col.iter().zip(y.iter()) {
            let dx = xv - x_mean;
            let dy = yv - y_mean;
            cov_xy += dx * dy;
            x_var += dx * dx;
        }

        if x_var < eps || y_var < eps {
            scores[j] = F::zero();
            continue;
        }

        let r = cov_xy / (x_var.sqrt() * y_var.sqrt());
        let r2 = r * r;

        let one = F::one();
        let denom = one - r2;
        if denom < eps {
            // Perfect correlation.
            scores[j] = F::from_f64(1e12).unwrap();
        } else {
            let n_minus_2 = F::from_usize(n_samples - 2).unwrap();
            scores[j] = r2 * n_minus_2 / denom;
        }
    }
    let pvalues = scores.mapv(|f| F::from_f64(f_sf(f.to_f64().unwrap(), 1.0, dfw)).unwrap());

    Ok((scores, pvalues))
}

/// sklearn `chi2`: for non-negative `X` and class labels `y`,
/// `observed = Yᵀ X` (per class feature sums), `expected = class_prob ⊗
/// feature_sum`, `χ² = Σ (obs - exp)² / exp` with `n_classes - 1` dof.
fn compute_chi2<F: Float>(x: ArrayView2<F>, y: ArrayView1<F>) -> Result<(Array1<F>, Array1<F>)> {
    let (n_samples, n_features) = x.dim();
    if x.iter().any(|&v| v < F::zero()) {
        return Err(RustMlError::InvalidParameter(
            "Input X must be non-negative for chi2".into(),
        ));
    }
    let mut label_map: HashMap<u64, usize> = HashMap::new();
    let mut class_idx = Vec::with_capacity(n_samples);
    for &v in y.iter() {
        let v = v.to_f64().unwrap();
        let key = if v == 0.0 { 0.0_f64 } else { v }.to_bits();
        let next = label_map.len();
        class_idx.push(*label_map.entry(key).or_insert(next));
    }
    // sklearn's LabelBinarizer turns a single class into one column; with
    // one class every statistic is 0 (expected == observed).
    let n_classes = label_map.len().max(2);
    let mut observed = vec![vec![0.0_f64; n_features]; n_classes];
    let mut class_count = vec![0.0_f64; n_classes];
    let mut feature_sum = vec![0.0_f64; n_features];
    for (i, &c) in class_idx.iter().enumerate() {
        class_count[c] += 1.0;
        for j in 0..n_features {
            let v = x[[i, j]].to_f64().unwrap();
            observed[c][j] += v;
            feature_sum[j] += v;
        }
    }
    let n = n_samples as f64;
    let dof = (n_classes - 1) as f64;
    let mut scores = Array1::<F>::zeros(n_features);
    let mut pvalues = Array1::<F>::zeros(n_features);
    for j in 0..n_features {
        let mut stat = 0.0;
        for c in 0..n_classes {
            let expected = class_count[c] / n * feature_sum[j];
            let d = observed[c][j] - expected;
            stat += d * d / expected; // 0/0 -> NaN like sklearn
        }
        scores[j] = F::from_f64(stat).unwrap();
        pvalues[j] = F::from_f64(chi2_sf(stat, dof)).unwrap();
    }
    Ok((scores, pvalues))
}

/// Compute per-feature variance (unsupervised scoring).
fn compute_variance<F: Float>(x: ArrayView2<F>) -> Array1<F> {
    let n = F::from_usize(x.nrows()).unwrap();
    let mean = x.sum_axis(Axis(0)) / n;
    let n_features = x.ncols();

    let mut variances = Array1::<F>::zeros(n_features);
    for row in x.rows() {
        for (j, (&val, &m)) in row.iter().zip(mean.iter()).enumerate() {
            let diff = val - m;
            variances[j] += diff * diff;
        }
    }
    variances.mapv_inplace(|v| v / n);
    variances
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sk_data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let x = array![
            [1.0, 2.0, 0.0, 3.0],
            [2.0, 1.0, 1.0, 0.0],
            [3.0, 4.0, 0.0, 1.0],
            [0.0, 2.0, 2.0, 5.0],
            [5.0, 1.0, 0.0, 2.0],
            [2.0, 3.0, 1.0, 1.0],
            [4.0, 0.0, 3.0, 2.0],
            [1.0, 1.0, 0.0, 0.0]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0];
        let yr = array![1.2, 0.3, 2.8, -0.5, 4.1, 1.9, 3.3, 0.2];
        (x, y, yr)
    }

    fn assert_vec(a: &Array1<f64>, b: &[f64], tol: f64) {
        for (x, y) in a.iter().zip(b) {
            assert!(
                (x - y).abs() <= tol * y.abs().max(1.0),
                "got {a:?}, expected {b:?}"
            );
        }
    }

    #[test]
    fn test_scores_and_pvalues_match_sklearn() {
        // sklearn 1.8 f_classif / chi2 / f_regression on the same data.
        let (x, y, yr) = sk_data();
        let f = SelectKBest::new(1, ScoringFunction::FClassif)
            .fit(x.view(), y.view())
            .unwrap();
        assert_vec(
            f.scores(),
            &[
                1.024096385542168,
                0.8173076923076918,
                0.33244680851063807,
                0.04347826086956553,
            ],
            1e-12,
        );
        assert_vec(
            f.pvalues().unwrap(),
            &[
                0.42386794721672477,
                0.4930439479293624,
                0.7318896650420783,
                0.9578112781492958,
            ],
            1e-9,
        );
        let f = SelectKBest::new(2, ScoringFunction::Chi2)
            .fit(x.view(), y.view())
            .unwrap();
        assert_vec(
            f.scores(),
            &[
                2.5185185185185186,
                1.619047619047619,
                1.1904761904761905,
                0.19047619047619047,
            ],
            1e-12,
        );
        assert_vec(
            f.pvalues().unwrap(),
            &[
                0.283864218432727,
                0.4450699538427625,
                0.551431257080004,
                0.909156442876713,
            ],
            1e-9,
        );
        assert_eq!(f.selected_indices(), &[0, 1]);
        let f = SelectKBest::new(1, ScoringFunction::FRegression)
            .fit(x.view(), yr.view())
            .unwrap();
        assert_vec(
            f.scores(),
            &[
                43.422639713293435,
                0.003893002975984549,
                0.01037474002364104,
                0.11906528189910984,
            ],
            1e-10,
        );
        assert_vec(
            f.pvalues().unwrap(),
            &[
                0.0005866287795907776,
                0.9522757151649666,
                0.9221893020094964,
                0.7418250707222199,
            ],
            1e-8,
        );
        // Perfect separation -> F = inf, p = 0 (sklearn).
        let xs: Array2<f64> = array![[0.0, 1.0], [0.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let ys = array![0.0, 0.0, 1.0, 1.0];
        let f = SelectKBest::new(1, ScoringFunction::FClassif)
            .fit(xs.view(), ys.view())
            .unwrap();
        assert!(f.scores()[0].is_infinite());
        assert_eq!(f.pvalues().unwrap()[0], 0.0);
        assert_vec(f.pvalues().unwrap(), &[0.0, 0.10557280900008414], 1e-9);
    }

    #[test]
    fn test_chi2_rejects_negative_input() {
        let x = array![[1.0, -1.0], [0.0, 2.0]];
        let y = array![0.0, 1.0];
        assert!(SelectKBest::new(1, ScoringFunction::Chi2)
            .fit(x.view(), y.view())
            .is_err());
    }

    #[test]
    fn test_mutual_info_scorers() {
        let (x, y, yr) = sk_data();
        let f = SelectKBest::new(2, ScoringFunction::MutualInfoClassif)
            .fit(x.view(), y.view())
            .unwrap();
        assert!(f.pvalues().is_none());
        assert_eq!(f.scores().len(), 4);
        assert!(f.scores().iter().all(|&v| v >= 0.0));
        let f = SelectKBest::new(1, ScoringFunction::MutualInfoRegression)
            .fit(x.view(), yr.view())
            .unwrap();
        assert!(f.scores().iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_f_classif_selects_discriminative_feature() {
        // Feature 0 perfectly separates classes; feature 1 is random noise.
        let x = array![
            [0.0, 0.5],
            [0.0, 0.8],
            [0.0, 0.2],
            [0.0, 0.9],
            [1.0, 0.3],
            [1.0, 0.7],
            [1.0, 0.1],
            [1.0, 0.6],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let selector = SelectKBest::new(1, ScoringFunction::FClassif);
        let fitted = selector.fit(x.view(), y.view()).unwrap();

        assert_eq!(fitted.selected_indices(), &[0]);
        assert!(
            fitted.scores()[0] > fitted.scores()[1],
            "discriminative feature score ({}) should exceed noise ({})",
            fitted.scores()[0],
            fitted.scores()[1]
        );
    }

    #[test]
    fn test_f_regression_selects_correlated_feature() {
        // Feature 0: linearly correlated with y.
        // Feature 1: constant (zero correlation).
        let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [5.0, 5.0],];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let selector = SelectKBest::new(1, ScoringFunction::FRegression);
        let fitted = selector.fit(x.view(), y.view()).unwrap();

        assert_eq!(fitted.selected_indices(), &[0]);
        // Feature 0 has perfect correlation -> very large F.
        assert!(fitted.scores()[0] > 100.0_f64);
        // Feature 1 is constant -> score should be 0.
        assert!(fitted.scores()[1].abs() < 1e-10_f64);
    }

    #[test]
    fn test_variance_scoring_selects_high_variance_feature() {
        // Feature 0: low variance; feature 1: high variance; feature 2: zero variance.
        let x = array![
            [1.0, 10.0, 5.0],
            [1.1, 20.0, 5.0],
            [0.9, 30.0, 5.0],
            [1.0, 40.0, 5.0],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0]; // ignored for Variance

        let selector = SelectKBest::new(1, ScoringFunction::Variance);
        let fitted = selector.fit(x.view(), y.view()).unwrap();

        assert_eq!(fitted.selected_indices(), &[1]);
    }

    #[test]
    fn test_transform_outputs_correct_columns() {
        let x = array![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0],];
        let y = array![1.0, 2.0, 3.0];

        let selector = SelectKBest::new(2, ScoringFunction::FRegression);
        let fitted = selector.fit(x.view(), y.view()).unwrap();
        let result = fitted.transform(&x).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);

        // Verify selected columns are present in the output.
        for &idx in fitted.selected_indices() {
            let original_col: Vec<f64> = x.column(idx).to_vec();
            let out_pos = fitted
                .selected_indices()
                .iter()
                .position(|&i| i == idx)
                .unwrap();
            let result_col: Vec<f64> = result.column(out_pos).to_vec();
            assert_eq!(original_col, result_col);
        }
    }

    #[test]
    fn test_error_k_zero() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let selector = SelectKBest::new(0, ScoringFunction::FClassif);
        let result = selector.fit(x.view(), y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_error_k_exceeds_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let selector = SelectKBest::new(5, ScoringFunction::FClassif);
        let result = selector.fit(x.view(), y.view());
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::InvalidParameter(msg) => {
                assert!(msg.contains("exceeds"), "unexpected message: {}", msg);
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_error_shape_mismatch_x_y() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 2.0]; // 3 labels for 2 samples

        let selector = SelectKBest::new(1, ScoringFunction::FClassif);
        let result = selector.fit(x.view(), y.view());
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::ShapeMismatch(msg) => {
                assert!(msg.contains("samples"), "unexpected message: {}", msg);
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_error_on_empty_input() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<f64>::zeros(0);

        let selector = SelectKBest::new(1, ScoringFunction::FRegression);
        let result = selector.fit(x.view(), y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let selector = SelectKBest::new(1, ScoringFunction::FClassif);
        let fitted = selector.fit(x.view(), y.view()).unwrap();

        let wrong = array![[1.0, 2.0]]; // 2 cols instead of 3
        assert!(fitted.transform(&wrong).is_err());
    }

    #[test]
    fn test_selects_all_when_k_equals_n_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let selector = SelectKBest::new(2, ScoringFunction::FRegression);
        let fitted = selector.fit(x.view(), y.view()).unwrap();

        assert_eq!(fitted.selected_indices().len(), 2);
        assert_eq!(fitted.selected_indices(), &[0, 1]);
    }

    #[test]
    fn test_works_with_f32() {
        let x: Array2<f32> = array![[0.0_f32, 0.5], [0.0, 0.8], [1.0, 0.3], [1.0, 0.7],];
        let y: Array1<f32> = array![0.0_f32, 0.0, 1.0, 1.0];

        let selector = SelectKBest::new(1, ScoringFunction::FClassif);
        let fitted = selector.fit(x.view(), y.view()).unwrap();

        assert_eq!(fitted.selected_indices().len(), 1);
        let result = fitted.transform(&x).unwrap();
        assert_eq!(result.ncols(), 1);
    }
}
