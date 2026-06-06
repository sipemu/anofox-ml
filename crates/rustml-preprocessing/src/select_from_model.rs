use ndarray::{Array1, Array2};
use rustml_core::{Float, Result, RustMlError, Transform};

/// Parameters for `SelectFromModel` feature selector (unfitted state).
///
/// Selects features based on a pre-computed importance vector (e.g., from a
/// fitted `RandomForestClassifier`'s
/// `feature_importances()`).
///
/// Features can be selected in two ways:
/// - **Threshold**: keep all features with importance >= threshold.
/// - **Max features**: keep the top `max_features` features by importance.
///
/// If both are specified, threshold is applied first and then the result is
/// capped to `max_features`.
///
/// Since this selector does not learn from raw data (it uses pre-computed
/// importances), it exposes a custom `fit` method instead of implementing the
/// standard [`FitUnsupervised`](rustml_core::FitUnsupervised) trait.
///
/// # Example
///
/// ```
/// use rustml_preprocessing::SelectFromModel;
/// use rustml_core::Transform;
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
    /// Features with importance >= threshold are selected.
    /// If `None`, no threshold filtering is applied.
    pub threshold: Option<f64>,
    /// Maximum number of features to select (top by importance).
    /// If `None`, no cap is applied.
    pub max_features: Option<usize>,
}

impl SelectFromModel {
    /// Create a new `SelectFromModel` with no threshold and no feature cap.
    ///
    /// At least one of `threshold` or `max_features` must be set before calling
    /// [`fit`](Self::fit).
    pub fn new() -> Self {
        Self {
            threshold: None,
            max_features: None,
        }
    }

    /// Set the minimum importance threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set the maximum number of features to select.
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Fit the selector on a pre-computed feature importance vector.
    ///
    /// Returns an error if the importance vector is empty or if neither
    /// `threshold` nor `max_features` is set.
    pub fn fit(&self, importances: &Array1<f64>) -> Result<FittedSelectFromModel> {
        let n_features = importances.len();

        if n_features == 0 {
            return Err(RustMlError::EmptyInput(
                "importances vector is empty".into(),
            ));
        }

        if self.threshold.is_none() && self.max_features.is_none() {
            return Err(RustMlError::InvalidParameter(
                "at least one of threshold or max_features must be set".into(),
            ));
        }

        if let Some(max_f) = self.max_features {
            if max_f == 0 {
                return Err(RustMlError::InvalidParameter(
                    "max_features must be at least 1".into(),
                ));
            }
        }

        // Step 1: Apply threshold filter.
        let mut candidates: Vec<(usize, f64)> = if let Some(thresh) = self.threshold {
            importances
                .iter()
                .copied()
                .enumerate()
                .filter(|&(_, imp)| imp >= thresh)
                .collect()
        } else {
            importances.iter().copied().enumerate().collect()
        };

        // Step 2: If max_features is set, keep only the top-N by importance.
        if let Some(max_f) = self.max_features {
            if candidates.len() > max_f {
                // Sort descending by importance; break ties by index (ascending).
                candidates.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                candidates.truncate(max_f);
            }
        }

        if candidates.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "no features meet the selection criteria".into(),
            ));
        }

        // Sort by index for stable column ordering.
        let mut selected_indices: Vec<usize> = candidates.iter().map(|&(idx, _)| idx).collect();
        selected_indices.sort_unstable();

        Ok(FittedSelectFromModel {
            importances: importances.clone(),
            selected_indices,
            n_features_in: n_features,
        })
    }
}

impl Default for SelectFromModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted `SelectFromModel` -- holds the original importances and the indices
/// of the selected features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedSelectFromModel {
    /// Original per-feature importance vector.
    importances: Array1<f64>,
    /// Indices of the selected features, sorted ascending.
    selected_indices: Vec<usize>,
    /// Total number of input features (before selection).
    n_features_in: usize,
}

impl FittedSelectFromModel {
    /// Per-feature importances supplied during fitting.
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

        let selector = SelectFromModel::new().with_max_features(2);
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

        let selector = SelectFromModel::new().with_max_features(2);
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
    fn test_error_no_criteria_set() {
        let importances = array![0.1, 0.2, 0.3];

        let selector = SelectFromModel::new(); // neither threshold nor max_features
        let result = selector.fit(&importances);
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("threshold") || msg.contains("max_features"),
                    "unexpected message: {}",
                    msg
                );
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
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
