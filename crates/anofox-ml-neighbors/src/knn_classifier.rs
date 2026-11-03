use anofox_ml_core::{Fit, Float, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix2};

use crate::distance::DistanceMetric;
use crate::index::{NeighborAlgorithm, NeighborIndex};

/// Weighting strategy for KNN.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum WeightFunction {
    /// All neighbors contribute equally.
    #[default]
    Uniform,
    /// Closer neighbors have more influence (weight = 1/distance). A query
    /// with neighbours at distance 0 uses only those, with equal weights
    /// (sklearn).
    Distance,
}

/// Neighbour weights of one query (sklearn's `_get_weights`).
pub(crate) fn neighbor_weights<F: Float>(dists: &[F], weights: WeightFunction) -> Vec<F> {
    match weights {
        WeightFunction::Uniform => vec![F::one(); dists.len()],
        WeightFunction::Distance => {
            if dists.iter().any(|&d| d == F::zero()) {
                dists
                    .iter()
                    .map(|&d| if d == F::zero() { F::one() } else { F::zero() })
                    .collect()
            } else {
                dists.iter().map(|&d| F::one() / d).collect()
            }
        }
    }
}

/// KNN classifier parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnnClassifier {
    pub n_neighbors: usize,
    pub metric: DistanceMetric,
    pub weights: WeightFunction,
    /// Neighbour search algorithm (sklearn's `algorithm`).
    #[serde(default)]
    pub algorithm: NeighborAlgorithm,
}

impl KnnClassifier {
    /// Create a new `KnnClassifier` with the given number of neighbors and default parameters.
    pub fn new(k: usize) -> Self {
        Self {
            n_neighbors: k,
            metric: DistanceMetric::default(),
            weights: WeightFunction::default(),
            algorithm: NeighborAlgorithm::default(),
        }
    }

    /// Set the weighting strategy for neighbor contributions.
    pub fn with_weights(mut self, weights: WeightFunction) -> Self {
        self.weights = weights;
        self
    }

    /// Set the distance metric used to find nearest neighbors.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the neighbour search algorithm.
    pub fn with_algorithm(mut self, algorithm: NeighborAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

impl Default for KnnClassifier {
    fn default() -> Self {
        Self::new(5)
    }
}

/// Fitted KNN classifier — stores training data (lazy learner) in a
/// [`NeighborIndex`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedKnnClassifier<F: Float> {
    index: NeighborIndex<F>,
    /// Sorted class labels.
    classes: Vec<F>,
    /// Class index of every training sample.
    y_class: Vec<usize>,
    n_neighbors: usize,
    weights: WeightFunction,
}

/// Check `X`/`y` and `n_neighbors` for the k-nearest-neighbour estimators.
pub(crate) fn check_knn_fit<F: Float>(x: ArrayView2<F>, y: ArrayView1<F>, k: usize) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(RustMlError::EmptyInput("training data is empty".into()));
    }
    if k == 0 {
        return Err(RustMlError::InvalidParameter(
            "n_neighbors must be > 0".into(),
        ));
    }
    if k > x.nrows() {
        return Err(RustMlError::InvalidParameter(format!(
            "n_neighbors ({}) > number of training samples ({})",
            k,
            x.nrows()
        )));
    }
    Ok(())
}

/// Sorted distinct labels and the class index of every label.
pub(crate) fn encode_classes<F: Float>(y: ArrayView1<F>) -> (Vec<F>, Vec<usize>) {
    let mut classes: Vec<F> = y.to_vec();
    classes.sort_by(crate::total_cmp);
    classes.dedup_by(|a, b| crate::total_cmp(a, b).is_eq());
    let idx = y
        .iter()
        .map(|v| {
            classes
                .binary_search_by(|c| crate::total_cmp(c, v))
                .expect("label in classes")
        })
        .collect();
    (classes, idx)
}

impl<F: Float> Fit<F> for KnnClassifier {
    type Fitted = FittedKnnClassifier<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        check_knn_fit(x, y, self.n_neighbors)?;
        let index = NeighborIndex::new(x, self.metric, self.algorithm, Some(self.n_neighbors))?;
        let (classes, y_class) = encode_classes(y);
        Ok(FittedKnnClassifier {
            index,
            classes,
            y_class,
            n_neighbors: self.n_neighbors,
            weights: self.weights,
        })
    }
}

impl<F: Float + Send + Sync> Predict<F> for FittedKnnClassifier<F> {
    /// The class with the largest (weighted) vote; ties go to the smallest
    /// label, as in sklearn.
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        let proba = self.predict_proba(x)?;
        Ok(proba
            .rows()
            .into_iter()
            .map(|row| self.classes[argmax_first(row.iter().copied())])
            .collect())
    }
}

/// Index of the first maximum.
pub(crate) fn argmax_first<F: Float>(values: impl Iterator<Item = F>) -> usize {
    let mut best = (0, F::neg_infinity());
    for (i, v) in values.enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0
}

impl<F: Float> FittedKnnClassifier<F> {
    /// Sorted class labels (columns of [`predict_proba`](Self::predict_proba)).
    pub fn classes(&self) -> &[F] {
        &self.classes
    }

    /// The neighbour index.
    pub fn index(&self) -> &NeighborIndex<F> {
        &self.index
    }

    /// Distances and training indices of the `k` (default `n_neighbors`)
    /// nearest training samples of every row of `x`.
    pub fn kneighbors(
        &self,
        x: &Array2<F>,
        k: Option<usize>,
    ) -> Result<(Array2<F>, Array2<usize>)> {
        self.index
            .kneighbors(x.view(), k.unwrap_or(self.n_neighbors))
    }

    /// Class probabilities: the (weighted) share of each class among the
    /// neighbours.
    pub fn predict_proba<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array2<F>> {
        let x: ArrayView2<F> = x.into();
        let (dist, idx) = self.index.kneighbors(x, self.n_neighbors)?;
        let mut proba = Array2::zeros((x.nrows(), self.classes.len()));
        for (i, mut row) in proba.rows_mut().into_iter().enumerate() {
            let d = dist.row(i).to_vec();
            let w = neighbor_weights(&d, self.weights);
            for (&j, &wj) in idx.row(i).iter().zip(&w) {
                row[self.y_class[j]] += wj;
            }
            let total = row.sum();
            if total > F::zero() {
                row /= total;
            }
        }
        Ok(proba)
    }
}

impl<F: Float> PredictProba<F> for FittedKnnClassifier<F> {
    fn predict_proba_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        FittedKnnClassifier::predict_proba(self, x)
    }
}

/// Weighted majority vote among (distance, label) neighbours: ties go to
/// the smallest label (sklearn's `mode` / `weighted_mode`).
#[cfg(test)]
fn weighted_majority_vote<F: Float>(neighbors: &[(F, F)], weights: WeightFunction) -> F {
    let d: Vec<F> = neighbors.iter().map(|n| n.0).collect();
    let w = neighbor_weights(&d, weights);
    let mut votes: Vec<(F, F)> = neighbors.iter().map(|n| n.1).zip(w).collect();
    votes.sort_by(|a, b| crate::total_cmp(&a.0, &b.0));
    let mut best = (votes[0].0, F::neg_infinity());
    let mut i = 0;
    while i < votes.len() {
        let class = votes[i].0;
        let mut total = F::zero();
        while i < votes.len() && crate::total_cmp(&votes[i].0, &class).is_eq() {
            total += votes[i].1;
            i += 1;
        }
        if total > best.1 {
            best = (class, total);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_knn_simple() {
        let x_train = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let knn = KnnClassifier {
            n_neighbors: 3,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test = array![[0.05, 0.05], [10.05, 10.05]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_distance_weights() {
        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 0.0, 1.0];

        let knn = KnnClassifier {
            n_neighbors: 3,
            weights: WeightFunction::Distance,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test = array![[1.9]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_manhattan() {
        let x_train = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let knn = KnnClassifier {
            n_neighbors: 3,
            metric: DistanceMetric::Manhattan,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test = array![[0.05, 0.05]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0]];
        let y = array![0.0, 1.0];
        let knn = KnnClassifier::default();
        assert!(Fit::fit(&knn, &x, &y).is_err());
    }

    #[test]
    fn test_k1_nearest_single() {
        // k=1 should return the label of the single nearest neighbor.
        let x_train = array![[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]];
        let y_train = array![0.0, 1.0, 2.0];

        let knn = KnnClassifier::new(1);
        let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // Point nearest to [5.0, 5.0] (class 1).
        let x_test = array![[4.9, 4.9]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 1.0, epsilon = 1e-10);

        // Point nearest to [0.0, 0.0] (class 0).
        let x_test2 = array![[0.1, 0.1]];
        let preds2 = fitted.predict(&x_test2).unwrap();
        assert_abs_diff_eq!(preds2[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_k_equals_n_samples() {
        // k = n_samples, all neighbors participate in the vote.
        let x_train = array![[0.0], [1.0], [2.0], [3.0]];
        let y_train = array![0.0, 0.0, 0.0, 1.0];

        let knn = KnnClassifier::new(4); // k = n_samples = 4
        let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // Majority is class 0 (3 vs 1), should predict 0 everywhere with uniform weights.
        let x_test = array![[1.5]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_distance() {
        // Cosine distance cares about angle, not magnitude.
        // Two directions: roughly (1,0) and (0,1).
        let x_train = array![
            [10.0, 0.1], // class 0 — points along x-axis
            [20.0, 0.2], // class 0
            [0.1, 10.0], // class 1 — points along y-axis
            [0.2, 20.0]  // class 1
        ];
        let y_train = array![0.0, 0.0, 1.0, 1.0];

        let knn = KnnClassifier::new(2).with_metric(DistanceMetric::Cosine);
        let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // A point along x-axis should be class 0.
        let x_test = array![[5.0, 0.05]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);

        // A point along y-axis should be class 1.
        let x_test2 = array![[0.05, 5.0]];
        let preds2 = fitted.predict(&x_test2).unwrap();
        assert_abs_diff_eq!(preds2[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_duplicate_training_points() {
        // Duplicate points should not cause issues.
        let x_train = array![
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [10.0, 10.0]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0];

        let knn = KnnClassifier::new(3);
        let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // Predicting at a duplicate point should work.
        let x_test = array![[1.0, 1.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_support() {
        let x_train: Array2<f32> = array![[0.0f32, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]];
        let y_train: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];

        let knn = KnnClassifier::new(2);
        let fitted: FittedKnnClassifier<f32> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test: Array2<f32> = array![[0.05f32, 0.05], [10.05, 10.05]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(preds[1], 1.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let knn = KnnClassifier::new(3);
        let result: Result<FittedKnnClassifier<f64>> = Fit::fit(&knn, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_k_zero_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let knn = KnnClassifier::new(0);
        let result: Result<FittedKnnClassifier<f64>> = Fit::fit(&knn, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(msg)) => {
                assert!(msg.contains("0") || msg.to_lowercase().contains("neighbor"));
            }
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_k_too_large_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let knn = KnnClassifier::new(5); // k=5 but only 2 samples
        let result: Result<FittedKnnClassifier<f64>> = Fit::fit(&knn, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(msg)) => {
                assert!(msg.contains("5"), "error should mention k value");
                assert!(msg.contains("2"), "error should mention sample count");
            }
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    /// Regression for #42: a tied vote used to depend on `HashMap`
    /// iteration order (0.0 in ~half the runs, 1.0 in the rest). Ties must
    /// resolve to the smallest class label, as sklearn does.
    #[test]
    fn test_tied_vote_is_deterministic_smallest_label() {
        let x_train = array![[-1.0, 0.0], [1.0, 0.0], [0.0, -2.0], [0.0, 2.0]];
        let y_train = array![1.0, 0.0, 3.0, 2.0];
        let query = array![[0.0, 0.0], [0.0, 0.1]];
        for _ in 0..20 {
            for weights in [WeightFunction::Uniform, WeightFunction::Distance] {
                let knn = KnnClassifier::new(2).with_weights(weights);
                let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();
                let p = fitted.predict(&query).unwrap();
                // Neighbours of (0,0)/(0,0.1): the two points at distance 1
                // (classes 1 and 0) -> tie -> 0.
                assert_eq!(p[0], 0.0, "weights={weights:?}");
                assert_eq!(p[1], 0.0, "weights={weights:?}");
            }
            // Four-way tie with k = 4 -> smallest label 0; brute-force path too.
            for metric in [DistanceMetric::Euclidean, DistanceMetric::Manhattan] {
                let knn = KnnClassifier::new(4).with_metric(metric);
                let fitted = Fit::fit(&knn, &x_train, &array![2.0, 1.0, 2.0, 1.0]).unwrap();
                assert_eq!(fitted.predict(&array![[0.0, 0.0]]).unwrap()[0], 1.0);
            }
        }
    }

    #[test]
    fn test_weighted_majority_vote_direct() {
        let n = [(1.0, 5.0), (1.0, 3.0), (1.0, 5.0), (1.0, 3.0), (2.0, 7.0)];
        assert_eq!(weighted_majority_vote(&n, WeightFunction::Uniform), 3.0);
        assert_eq!(
            weighted_majority_vote(&n[..3], WeightFunction::Uniform),
            5.0
        );
    }

    /// Regression for #42: Fortran-order training / query data used to
    /// panic in `row(i).as_slice().unwrap()`.
    #[test]
    fn test_fortran_order_input() {
        use ndarray::{Array2, ShapeBuilder};
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v: Vec<f64> = x.t().iter().copied().collect();
        let xf = Array2::from_shape_vec((6, 2).f(), v).unwrap();
        assert!(!xf.is_standard_layout());
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Cosine,
        ] {
            let knn = KnnClassifier::new(3).with_metric(metric);
            let a = Fit::fit(&knn, &x, &y).unwrap().predict(&x).unwrap();
            let b = Fit::fit(&knn, &xf, &y).unwrap().predict(&xf).unwrap();
            assert_eq!(a, b, "metric={metric:?}");
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;
        use std::collections::HashSet;

        /// Generate deterministic training data for classification.
        fn make_classification_data(
            n_samples: usize,
            n_features: usize,
            n_classes: usize,
            seed: u64,
        ) -> (Array2<f64>, Array1<f64>) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut x_data = Vec::with_capacity(n_samples * n_features);
            let mut y_data = Vec::with_capacity(n_samples);

            for i in 0..n_samples {
                for j in 0..n_features {
                    let mut h = DefaultHasher::new();
                    seed.hash(&mut h);
                    (i as u64).hash(&mut h);
                    (j as u64).hash(&mut h);
                    let bits = h.finish();
                    let v = (bits as f64 / u64::MAX as f64) * 20.0 - 10.0;
                    x_data.push(v);
                }
                let mut h = DefaultHasher::new();
                seed.hash(&mut h);
                (i as u64).hash(&mut h);
                0xDEAD_BEEFu64.hash(&mut h);
                let label = (h.finish() % n_classes as u64) as f64;
                y_data.push(label);
            }

            let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
            let y = Array1::from_vec(y_data);
            (x, y)
        }

        proptest! {
            #[test]
            fn predictions_are_valid_training_labels(
                n_samples in 4..30usize,
                n_features in 1..5usize,
                k in 1..4usize,
                seed in 0u64..1000,
            ) {
                let n_classes = 3;
                let (x, y) = make_classification_data(n_samples, n_features, n_classes, seed);

                // k must not exceed n_samples
                let k = k.min(n_samples);

                // Collect unique training labels
                let train_labels: HashSet<u64> = y.iter()
                    .map(|&v| v.to_bits())
                    .collect();

                let knn = KnnClassifier::new(k);
                let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x, &y).unwrap();
                let preds = fitted.predict(&x).unwrap();

                for (i, &p) in preds.iter().enumerate() {
                    prop_assert!(
                        train_labels.contains(&p.to_bits()),
                        "prediction {} at index {} is not a valid training label",
                        p, i
                    );
                }
            }

            #[test]
            fn k1_returns_exact_nearest_neighbor(
                n_samples in 2..30usize,
                n_features in 1..5usize,
                seed in 0u64..1000,
            ) {
                let n_classes = 3;
                let (x, y) = make_classification_data(n_samples, n_features, n_classes, seed);

                let knn = KnnClassifier::new(1);
                let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x, &y).unwrap();

                // Predicting on the training data with k=1 should return each point's own label,
                // because the nearest neighbor of a training point is itself (distance = 0).
                let preds = fitted.predict(&x).unwrap();

                for (i, (&pred, &expected)) in preds.iter().zip(y.iter()).enumerate() {
                    prop_assert!(
                        (pred - expected).abs() < 1e-10,
                        "k=1 prediction at index {} was {} but expected {}",
                        i, pred, expected
                    );
                }
            }
        }
    }
}
