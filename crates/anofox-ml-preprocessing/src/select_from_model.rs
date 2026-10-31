use crate::quantile::{percentile_sorted, sorted_non_nan};
use anofox_ml_core::{Float, Result, RustMlError, Transform};
use ndarray::{Array1, Array2};

/// Importance threshold rule for [`SelectFromModel`] (sklearn's
/// `threshold` parameter). Features whose (absolute) importance is
/// `>= threshold` are kept.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Threshold {
    /// A fixed value.
    Value(f64),
    /// The mean of the importances (`"mean"`, sklearn's default).
    Mean,
    /// The median of the importances (`"median"`).
    Median,
    /// `factor * mean` (e.g. `"1.25*mean"`).
    ScaledMean(f64),
    /// `factor * median` (e.g. `"0.5*median"`).
    ScaledMedian(f64),
}

impl Threshold {
    fn resolve(&self, importances: &[f64]) -> f64 {
        let mean = || importances.iter().sum::<f64>() / importances.len() as f64;
        let median = || percentile_sorted(&sorted_non_nan(importances.iter().copied()), 0.5);
        match *self {
            Threshold::Value(v) => v,
            Threshold::Mean => mean(),
            Threshold::Median => median(),
            Threshold::ScaledMean(f) => f * mean(),
            Threshold::ScaledMedian(f) => f * median(),
        }
    }
}

/// Parameters for `SelectFromModel` feature selector (unfitted state).
///
/// Selects features based on a pre-computed importance vector (e.g. a
/// fitted random forest's `feature_importances()`) or a coefficient matrix
/// of a linear model, mirroring `sklearn.feature_selection.SelectFromModel`:
///
/// * importances are taken in **absolute value** (a coefficient of `-3` is
///   as important as `+3`); for a 2-D coefficient matrix
///   (`n_targets x n_features`, see [`fit_coef`](Self::fit_coef)) the
///   column-wise `norm_order`-norm is used (default L1);
/// * the threshold defaults to the **mean** importance, like sklearn —
///   also when `max_features` is set (use
///   `with_threshold(f64::NEG_INFINITY)` to select purely by
///   `max_features`);
/// * with `max_features`, the top-`max_features` features (ties broken by
///   index) are candidates and the threshold is then applied on top.
///
/// # Example
///
/// ```
/// use anofox_ml_preprocessing::SelectFromModel;
/// use anofox_ml_core::Transform;
/// use ndarray::array;
///
/// let importances = array![0.05, 0.40, 0.10, 0.45];
///
/// // Select features with importance >= 0.20
/// let selector = SelectFromModel::new().with_threshold(0.20);
/// let fitted = selector.fit(&importances).unwrap();
///
/// assert_eq!(fitted.selected_indices(), &[1, 3]);
///
/// let x = array![
///     [1.0, 2.0, 3.0, 4.0],
///     [5.0, 6.0, 7.0, 8.0],
/// ];
/// let x_selected = fitted.transform(&x).unwrap();
/// assert_eq!(x_selected.ncols(), 2);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SelectFromModel {
    /// Threshold rule; `None` means the sklearn default ([`Threshold::Mean`]).
    pub threshold: Option<Threshold>,
    /// Maximum number of features to select (top by importance).
    /// If `None`, no cap is applied.
    pub max_features: Option<usize>,
    /// Norm used to collapse a 2-D coefficient matrix to one importance per
    /// feature (sklearn `norm_order`, default 1). `f64::INFINITY` = max-abs.
    #[serde(default = "default_norm_order")]
    pub norm_order: f64,
}

fn default_norm_order() -> f64 {
    1.0
}

impl SelectFromModel {
    /// Create a new `SelectFromModel` (mean threshold, no feature cap).
    pub fn new() -> Self {
        Self {
            threshold: None,
            max_features: None,
            norm_order: 1.0,
        }
    }

    /// Set a fixed minimum importance threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(Threshold::Value(threshold));
        self
    }

    /// Set the threshold rule (mean, median, scaled, fixed value).
    pub fn with_threshold_rule(mut self, threshold: Threshold) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set the maximum number of features to select.
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Set the norm used by [`fit_coef`](Self::fit_coef).
    pub fn with_norm_order(mut self, norm_order: f64) -> Self {
        self.norm_order = norm_order;
        self
    }

    /// Fit on a 2-D coefficient matrix of shape `(n_targets, n_features)`
    /// (e.g. a multi-class linear model's `coef_`). Each feature's
    /// importance is the `norm_order`-norm of its column.
    pub fn fit_coef(&self, coef: &Array2<f64>) -> Result<FittedSelectFromModel> {
        let p = self.norm_order;
        if p.is_nan() || p < 1.0 {
            return Err(RustMlError::InvalidParameter(
                "norm_order must be >= 1 (or infinity)".into(),
            ));
        }
        let importances = Array1::from_iter(coef.columns().into_iter().map(|c| {
            if p.is_infinite() {
                c.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
            } else if p == 1.0 {
                c.iter().map(|v| v.abs()).sum()
            } else {
                c.iter().map(|v| v.abs().powf(p)).sum::<f64>().powf(1.0 / p)
            }
        }));
        self.fit(&importances)
    }

    /// Fit the selector on a pre-computed feature importance vector
    /// (absolute values are used).
    ///
    /// Returns an error if the importance vector is empty or no feature
    /// passes the selection.
    pub fn fit(&self, importances: &Array1<f64>) -> Result<FittedSelectFromModel> {
        let n_features = importances.len();

        if n_features == 0 {
            return Err(RustMlError::EmptyInput(
                "importances vector is empty".into(),
            ));
        }

        if let Some(max_f) = self.max_features {
            if max_f == 0 {
                return Err(RustMlError::InvalidParameter(
                    "max_features must be at least 1".into(),
                ));
            }
        }

        let scores: Vec<f64> = importances.iter().map(|v| v.abs()).collect();
        let threshold = self.threshold.unwrap_or(Threshold::Mean).resolve(&scores);

        // Candidates: top-`max_features` (stable, ties by index), or all.
        let mut mask = vec![true; n_features];
        if let Some(max_f) = self.max_features {
            let mut order: Vec<usize> = (0..n_features).collect();
            order.sort_by(|&a, &b| crate::quantile::total_cmp_nan_low(&scores[b], &scores[a]));
            mask = vec![false; n_features];
            for &i in order.iter().take(max_f) {
                mask[i] = true;
            }
        }
        // `mask[scores < threshold] = False` (NaN scores are never kept).
        for (m, &sc) in mask.iter_mut().zip(&scores) {
            if sc.is_nan() || sc < threshold {
                *m = false;
            }
        }

        let selected_indices: Vec<usize> = (0..n_features).filter(|&i| mask[i]).collect();
        if selected_indices.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "no features meet the selection criteria".into(),
            ));
        }

        Ok(FittedSelectFromModel {
            importances: Array1::from_vec(scores),
            selected_indices,
            n_features_in: n_features,
            threshold,
        })
    }
}

impl Default for SelectFromModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted `SelectFromModel` -- holds the (absolute) importances and the
/// indices of the selected features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedSelectFromModel {
    /// Per-feature (absolute) importance vector.
    importances: Array1<f64>,
    /// Indices of the selected features, sorted ascending.
    selected_indices: Vec<usize>,
    /// Total number of input features (before selection).
    n_features_in: usize,
    /// Resolved threshold value (sklearn `threshold_`).
    #[serde(default)]
    threshold: f64,
}

impl FittedSelectFromModel {
    /// Per-feature absolute importances used for selection.
    pub fn importances(&self) -> &Array1<f64> {
        &self.importances
    }

    /// Indices of the selected features, sorted in ascending order.
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }

    /// Number of features that survived selection.
    pub fn n_features_selected(&self) -> usize {
        self.selected_indices.len()
    }

    /// The resolved numeric threshold (sklearn `threshold_`).
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl<F: Float> Transform<F> for FittedSelectFromModel {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_threshold_selects_important_features() {
        let importances = array![0.05, 0.40, 0.10, 0.45];

        let selector = SelectFromModel::new().with_threshold(0.20);
        let fitted = selector.fit(&importances).unwrap();

        assert_eq!(fitted.selected_indices(), &[1, 3]);
    }

    #[test]
    fn test_max_features_selects_top_n() {
        let importances = array![0.1, 0.5, 0.3, 0.8, 0.2];

        let selector = SelectFromModel::new()
            .with_threshold(f64::NEG_INFINITY)
            .with_max_features(2);
        let fitted = selector.fit(&importances).unwrap();

        // Top 2 by importance: index 3 (0.8) and index 1 (0.5), sorted -> [1, 3]
        assert_eq!(fitted.selected_indices(), &[1, 3]);
    }

    #[test]
    fn test_threshold_and_max_features_combined() {
        let importances = array![0.05, 0.40, 0.30, 0.45, 0.35];

        // Threshold 0.20 keeps indices [1, 2, 3, 4], then max_features=2 keeps top 2.
        let selector = SelectFromModel::new()
            .with_threshold(0.20)
            .with_max_features(2);
        let fitted = selector.fit(&importances).unwrap();

        // Top 2 after threshold: index 3 (0.45) and index 1 (0.40), sorted -> [1, 3]
        assert_eq!(fitted.selected_indices(), &[1, 3]);
    }

    #[test]
    fn test_transform_selects_correct_columns() {
        let importances = array![0.1, 0.9, 0.5];

        let selector = SelectFromModel::new()
            .with_threshold(f64::NEG_INFINITY)
            .with_max_features(2);
        let fitted = selector.fit(&importances).unwrap();

        // Should select indices 1 (0.9) and 2 (0.5), sorted -> [1, 2]
        assert_eq!(fitted.selected_indices(), &[1, 2]);

        let x = array![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0],];
        let result = fitted.transform(&x).unwrap();

        assert_eq!(result.dim(), (2, 2));
        assert_eq!(result[[0, 0]], 20.0);
        assert_eq!(result[[0, 1]], 30.0);
        assert_eq!(result[[1, 0]], 50.0);
        assert_eq!(result[[1, 1]], 60.0);
    }

    #[test]
    fn test_default_threshold_is_mean_of_abs_like_sklearn() {
        // Regression for #53: raw signed values were compared, so a large
        // negative coefficient was dropped. sklearn 1.8 with coef_ =
        // [0.5, -3.0, 0.1, 1.0]:
        //   threshold=None      -> threshold_ 1.15,  support [1]
        //   threshold='median'  -> threshold_ 0.75,  support [1, 3]
        //   threshold='1.5*mean'-> threshold_ 1.725, support [1]
        //   max_features=2      -> threshold_ 1.15,  support [1]
        let coef = array![0.5, -3.0, 0.1, 1.0];
        let f = SelectFromModel::new().fit(&coef).unwrap();
        assert!((f.threshold() - 1.15).abs() < 1e-12);
        assert_eq!(f.selected_indices(), &[1]);
        let f = SelectFromModel::new()
            .with_threshold_rule(Threshold::Median)
            .fit(&coef)
            .unwrap();
        assert!((f.threshold() - 0.75).abs() < 1e-12);
        assert_eq!(f.selected_indices(), &[1, 3]);
        let f = SelectFromModel::new()
            .with_threshold_rule(Threshold::ScaledMean(1.5))
            .fit(&coef)
            .unwrap();
        assert!((f.threshold() - 1.725).abs() < 1e-12);
        assert_eq!(f.selected_indices(), &[1]);
        let f = SelectFromModel::new()
            .with_max_features(2)
            .fit(&coef)
            .unwrap();
        assert_eq!(f.selected_indices(), &[1]);
        assert_eq!(f.importances()[1], 3.0);
    }

    #[test]
    fn test_fit_coef_uses_l1_norm_over_targets() {
        // sklearn: coef_ = [[0.5,-3,0.1,1],[1,0.2,-0.2,-1]] -> importances
        //   [1.5, 3.2, 0.3, 2.0], threshold_ 1.75, support [1, 3]
        let coef = array![[0.5, -3.0, 0.1, 1.0], [1.0, 0.2, -0.2, -1.0]];
        let f = SelectFromModel::new().fit_coef(&coef).unwrap();
        assert!((f.threshold() - 1.75).abs() < 1e-12);
        assert_eq!(f.selected_indices(), &[1, 3]);
        let f2 = SelectFromModel::new()
            .with_norm_order(f64::INFINITY)
            .fit_coef(&coef)
            .unwrap();
        assert_eq!(f2.importances().to_vec(), vec![1.0, 3.0, 0.2, 1.0]);
    }

    #[test]
    fn test_error_no_features_survive_threshold() {
        let importances = array![0.01, 0.02, 0.03];

        let selector = SelectFromModel::new().with_threshold(0.50);
        let result = selector.fit(&importances);
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::InvalidParameter(msg) => {
                assert!(msg.contains("no features"), "unexpected message: {}", msg);
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_error_empty_importances() {
        let importances = Array1::<f64>::zeros(0);

        let selector = SelectFromModel::new().with_threshold(0.0);
        let result = selector.fit(&importances);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let importances = array![0.5, 0.5, 0.5];

        let selector = SelectFromModel::new().with_threshold(0.0);
        let fitted = selector.fit(&importances).unwrap();

        let wrong = array![[1.0, 2.0]]; // 2 cols instead of 3
        assert!(Transform::<f64>::transform(&fitted, &wrong).is_err());
    }

    #[test]
    fn test_works_with_f32_transform() {
        let importances = array![0.1, 0.9];

        let selector = SelectFromModel::new().with_max_features(1);
        let fitted = selector.fit(&importances).unwrap();

        assert_eq!(fitted.selected_indices(), &[1]);

        let x: Array2<f32> = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let result = Transform::<f32>::transform(&fitted, &x).unwrap();
        assert_eq!(result.dim(), (2, 1));
        assert_eq!(result[[0, 0]], 2.0_f32);
    }

    #[test]
    fn test_max_features_zero_is_error() {
        let importances = array![0.1, 0.2];

        let selector = SelectFromModel::new().with_max_features(0);
        let result = selector.fit(&importances);
        assert!(result.is_err());
    }

    #[test]
    fn test_n_features_selected() {
        let importances = array![0.1, 0.5, 0.3, 0.8];

        let selector = SelectFromModel::new().with_threshold(0.25);
        let fitted = selector.fit(&importances).unwrap();

        assert_eq!(fitted.n_features_selected(), 3); // indices 1, 2, 3
        assert_eq!(fitted.selected_indices(), &[1, 2, 3]);
    }
}
