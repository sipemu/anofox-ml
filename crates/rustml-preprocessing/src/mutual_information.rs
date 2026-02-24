use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Result, RustMlError, Transform};
use std::collections::HashMap;

/// Parameters for `MutualInformationSelector` (unfitted state).
///
/// Selects the top-k features by mutual information with the target variable.
/// Mutual information measures the dependency between two variables: a higher
/// score means the feature is more informative about the target.
///
/// For continuous features, values are discretized into equal-width bins before
/// computing mutual information. The number of bins can be tuned via `n_bins`.
///
/// This is a **supervised** feature selector and requires target labels `y`.
///
/// # Example
///
/// ```
/// use rustml_preprocessing::MutualInformationSelector;
/// use rustml_core::{Fit, Transform};
/// use ndarray::array;
///
/// let x = array![
///     [1.0, 100.0],
///     [2.0, 200.0],
///     [1.0, 300.0],
///     [2.0, 400.0],
/// ];
/// let y = array![0.0, 1.0, 0.0, 1.0]; // perfectly correlated with col 0
///
/// let selector = MutualInformationSelector::new(1);
/// let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();
/// let x_selected = fitted.transform(&x).unwrap();
///
/// assert_eq!(x_selected.ncols(), 1);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MutualInformationSelector {
    /// Number of top features to select.
    pub n_features_to_select: usize,
    /// Number of equal-width bins for discretizing continuous features.
    pub n_bins: usize,
}

impl MutualInformationSelector {
    /// Create a new selector that keeps the top `n_features_to_select` features.
    pub fn new(n_features_to_select: usize) -> Self {
        Self {
            n_features_to_select,
            n_bins: 10,
        }
    }

    /// Set the number of bins for discretizing continuous features.
    pub fn with_n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }
}

/// Fitted `MutualInformationSelector` — holds per-feature MI scores and
/// the indices of the selected top-k features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMutualInformationSelector<F: Float> {
    /// Mutual information score for each feature.
    mi_scores: Array1<F>,
    /// Indices of the top-k features (sorted by index for stable column ordering).
    selected_indices: Vec<usize>,
    /// Total number of input features (before selection).
    n_features_in: usize,
}

impl<F: Float> FittedMutualInformationSelector<F> {
    /// Per-feature mutual information scores.
    pub fn mi_scores(&self) -> &Array1<F> {
        &self.mi_scores
    }

    /// Indices of the selected features, sorted in ascending order.
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }
}

/// Discretize a 1-D array of continuous values into `n_bins` equal-width bins.
///
/// Returns a `Vec<usize>` where each element is the bin index (0..n_bins-1).
/// If all values are identical (zero range), every sample is placed in bin 0.
fn discretize<F: Float>(values: &[F], n_bins: usize) -> Vec<usize> {
    let mut min_val = values[0];
    let mut max_val = values[0];
    for &v in values.iter().skip(1) {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    let range = max_val - min_val;
    let eps = F::from_f64(1e-15).unwrap();

    if range < eps {
        // All values identical -> single bin.
        return vec![0; values.len()];
    }

    let n_bins_f = F::from_usize(n_bins).unwrap();
    let max_bin = n_bins - 1;

    values
        .iter()
        .map(|&v| {
            let normalized = (v - min_val) / range; // [0, 1]
            let bin = (normalized * n_bins_f).to_usize().unwrap_or(max_bin);
            bin.min(max_bin)
        })
        .collect()
}

/// Compute mutual information MI(X, Y) between two discrete integer-valued
/// random variables represented as parallel slices.
///
/// MI(X, Y) = sum_{x,y} p(x,y) * log2(p(x,y) / (p(x) * p(y)))
///
/// Convention: 0 * log(0) = 0.
fn mutual_information_discrete<F: Float>(x_bins: &[usize], y_labels: &[usize]) -> F {
    let n = x_bins.len();
    let n_f = F::from_usize(n).unwrap();

    // Count joint and marginal frequencies.
    let mut joint: HashMap<(usize, usize), usize> = HashMap::new();
    let mut x_counts: HashMap<usize, usize> = HashMap::new();
    let mut y_counts: HashMap<usize, usize> = HashMap::new();

    for (&xb, &yb) in x_bins.iter().zip(y_labels.iter()) {
        *joint.entry((xb, yb)).or_insert(0) += 1;
        *x_counts.entry(xb).or_insert(0) += 1;
        *y_counts.entry(yb).or_insert(0) += 1;
    }

    let mut mi = F::zero();
    for (&(xb, yb), &count) in &joint {
        if count == 0 {
            continue;
        }
        let p_xy = F::from_usize(count).unwrap() / n_f;
        let p_x = F::from_usize(x_counts[&xb]).unwrap() / n_f;
        let p_y = F::from_usize(y_counts[&yb]).unwrap() / n_f;

        let ratio = p_xy / (p_x * p_y);
        mi += p_xy * ratio.ln();
    }

    // Clamp to zero in case of floating-point noise producing a tiny negative.
    if mi < F::zero() {
        F::zero()
    } else {
        mi
    }
}

/// Convert target labels to discrete integer class indices.
///
/// Returns `(label_indices, n_classes)`. Unique labels are discovered in
/// order of first appearance.
fn labels_to_indices<F: Float>(y: &Array1<F>) -> Vec<usize> {
    let mut label_map: HashMap<u64, usize> = HashMap::new();
    let mut indices = Vec::with_capacity(y.len());

    for &val in y.iter() {
        // Use bit representation as hash key for exact equality.
        let bits = val.to_f64().unwrap().to_bits();
        let next_id = label_map.len();
        let id = *label_map.entry(bits).or_insert(next_id);
        indices.push(id);
    }

    indices
}

impl<F: Float> Fit<F> for MutualInformationSelector {
    type Fitted = FittedMutualInformationSelector<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        if y.len() != n_samples {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} samples but y has {} elements",
                n_samples,
                y.len()
            )));
        }

        if self.n_features_to_select == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_features_to_select must be at least 1".into(),
            ));
        }

        if self.n_features_to_select > n_features {
            return Err(RustMlError::InvalidParameter(format!(
                "n_features_to_select ({}) exceeds number of features ({})",
                self.n_features_to_select, n_features
            )));
        }

        if self.n_bins == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_bins must be at least 1".into(),
            ));
        }

        // Convert target labels to integer indices.
        let y_indices = labels_to_indices(y);

        // Compute MI for each feature column.
        let mut mi_scores = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let col: Vec<F> = x.column(j).to_vec();
            let x_bins = discretize(&col, self.n_bins);
            mi_scores[j] = mutual_information_discrete::<F>(&x_bins, &y_indices);
        }

        // Select top-k features by MI score.
        let mut feature_scores: Vec<(usize, F)> =
            mi_scores.iter().copied().enumerate().collect();
        // Sort descending by score; break ties by feature index (ascending).
        feature_scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let mut selected_indices: Vec<usize> = feature_scores
            .iter()
            .take(self.n_features_to_select)
            .map(|&(idx, _)| idx)
            .collect();
        // Sort indices for stable column ordering in transform.
        selected_indices.sort_unstable();

        Ok(FittedMutualInformationSelector {
            mi_scores,
            selected_indices,
            n_features_in: n_features,
        })
    }
}

impl<F: Float> Transform<F> for FittedMutualInformationSelector<F> {
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
    fn test_selects_informative_feature_over_noise() {
        // Feature 0: perfectly predicts the class (0->class0, 1->class1).
        // Feature 1: random noise, uncorrelated with class.
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

        let selector = MutualInformationSelector::new(1).with_n_bins(2);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();

        // Feature 0 should be selected (it perfectly separates the classes).
        assert_eq!(fitted.selected_indices(), &[0]);

        // MI of feature 0 should be substantially larger than feature 1.
        assert!(
            fitted.mi_scores()[0] > fitted.mi_scores()[1],
            "informative feature MI ({}) should be > noise MI ({})",
            fitted.mi_scores()[0],
            fitted.mi_scores()[1]
        );
    }

    #[test]
    fn test_scores_are_non_negative() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = array![0.0, 1.0, 0.0, 1.0];

        let selector = MutualInformationSelector::new(2);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();

        for (i, &score) in fitted.mi_scores().iter().enumerate() {
            assert!(
                score >= 0.0,
                "MI score for feature {} is negative: {}",
                i,
                score
            );
        }
    }

    #[test]
    fn test_transform_outputs_correct_shape() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ];
        let y = array![0.0, 1.0, 0.0];

        let selector = MutualInformationSelector::new(2);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();
        let result = fitted.transform(&x).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_selects_all_when_k_equals_n_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0.0, 1.0, 0.0];

        let selector = MutualInformationSelector::new(2);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();

        assert_eq!(fitted.selected_indices(), &[0, 1]);
    }

    #[test]
    fn test_shape_mismatch_x_y() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 2.0]; // 3 labels for 2 samples

        let selector = MutualInformationSelector::new(1);
        let result = Fit::<f64>::fit(&selector, &x, &y);

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

        let selector = MutualInformationSelector::new(1);
        let result = Fit::<f64>::fit(&selector, &x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_error_n_features_to_select_exceeds_n_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let selector = MutualInformationSelector::new(5); // only 2 features
        let result = Fit::<f64>::fit(&selector, &x, &y);

        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("n_features_to_select"),
                    "unexpected message: {}",
                    msg
                );
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = array![0.0, 1.0];

        let selector = MutualInformationSelector::new(1);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();

        let wrong = array![[1.0, 2.0]]; // 2 cols instead of 3
        assert!(fitted.transform(&wrong).is_err());
    }

    #[test]
    fn test_works_with_f32() {
        let x: Array2<f32> = array![
            [0.0_f32, 0.5],
            [0.0, 0.8],
            [1.0, 0.3],
            [1.0, 0.7],
        ];
        let y: Array1<f32> = array![0.0_f32, 0.0, 1.0, 1.0];

        let selector = MutualInformationSelector::new(1).with_n_bins(2);
        let fitted = Fit::<f32>::fit(&selector, &x, &y).unwrap();

        assert_eq!(fitted.selected_indices().len(), 1);
        let result = fitted.transform(&x).unwrap();
        assert_eq!(result.ncols(), 1);
    }

    #[test]
    fn test_multiclass_labels() {
        // Feature 0 has 3 bins matching 3 classes; feature 1 is constant.
        let x = array![
            [0.0, 5.0],
            [0.0, 5.0],
            [0.5, 5.0],
            [0.5, 5.0],
            [1.0, 5.0],
            [1.0, 5.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let selector = MutualInformationSelector::new(1).with_n_bins(3);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();

        // Feature 0 should be selected (feature 1 has zero MI since it's constant).
        assert_eq!(fitted.selected_indices(), &[0]);
    }
}
