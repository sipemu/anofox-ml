use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rustml_core::Float;

/// Convert a Float value to a u64 key suitable for HashMap use.
/// Uses f64 bit representation for exact equality matching.
#[inline]
fn float_key<F: Float>(v: F) -> u64 {
    v.to_f64().unwrap().to_bits()
}

/// Criterion for evaluating splits.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SplitCriterion {
    /// Gini impurity (for classification).
    Gini,
    /// Entropy / information gain (for classification).
    Entropy,
    /// Mean squared error (for regression).
    Mse,
}

/// Result of finding the best split at a node.
#[derive(Debug, Clone)]
pub struct BestSplit<F: Float> {
    pub feature_index: usize,
    pub threshold: F,
    pub left_indices: Vec<usize>,
    pub right_indices: Vec<usize>,
    pub improvement: F,
}

/// Find the best split over all features and thresholds.
///
/// Uses an incremental class-count / running-sum approach so that each
/// candidate threshold is evaluated in O(k) (classification, k = n_classes)
/// or O(1) (regression) instead of O(n).
pub fn find_best_split<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    criterion: SplitCriterion,
    min_samples_leaf: usize,
) -> Option<BestSplit<F>> {
    let n_features = x.ncols();
    let n = indices.len();
    if n < 2 * min_samples_leaf {
        return None;
    }

    let parent_impurity = compute_impurity(y, indices, criterion);

    match criterion {
        SplitCriterion::Gini | SplitCriterion::Entropy => {
            find_best_split_classification(
                x, y, indices, criterion, min_samples_leaf, n_features, n, parent_impurity,
            )
        }
        SplitCriterion::Mse => {
            find_best_split_regression(
                x, y, indices, min_samples_leaf, n_features, n, parent_impurity,
            )
        }
    }
}

/// Sort indices by feature value, filling the provided buffer.
///
/// Clears `sorted_pairs` and fills it with `(feature_value, original_index)`
/// pairs sorted by feature value.
#[inline]
fn sort_feature_pairs<F: Float>(
    x: &Array2<F>,
    indices: &[usize],
    feature: usize,
    sorted_pairs: &mut Vec<(F, usize)>,
) {
    sorted_pairs.clear();
    sorted_pairs.extend(indices.iter().map(|&i| (x[[i, feature]], i)));
    sorted_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
}

/// If `improvement` beats `best_improvement`, update the best split.
///
/// Reconstructs left/right index vectors from the sorted pairs at
/// the given split position.
#[inline]
fn try_update_best_split<F: Float>(
    improvement: F,
    best_improvement: &mut F,
    best: &mut Option<BestSplit<F>>,
    feature: usize,
    threshold: F,
    sorted_pairs: &[(F, usize)],
    pos: usize,
) {
    if improvement > *best_improvement {
        *best_improvement = improvement;
        let left_indices: Vec<usize> =
            sorted_pairs[..=pos].iter().map(|&(_, idx)| idx).collect();
        let right_indices: Vec<usize> =
            sorted_pairs[pos + 1..].iter().map(|&(_, idx)| idx).collect();
        *best = Some(BestSplit {
            feature_index: feature,
            threshold,
            left_indices,
            right_indices,
            improvement,
        });
    }
}

/// Accumulator that tracks split impurity incrementally as samples move
/// from the "right" partition to the "left" partition.
trait SplitAccumulator<F: Float> {
    /// Create a new accumulator with all samples in "right".
    fn new(y: &Array1<F>, indices: &[usize]) -> Self;
    /// Move sample `idx` from right to left.
    fn move_to_left(&mut self, y: &Array1<F>, idx: usize);
    /// Compute weighted impurity: (n_left/n)*left + (n_right/n)*right.
    fn weighted_impurity(&self, n: usize) -> F;
}

/// Classification accumulator using incremental class counts.
struct ClassificationAccumulator<F: Float> {
    left_counts: Vec<usize>,
    right_counts: Vec<usize>,
    n_left: usize,
    n_right: usize,
    criterion: SplitCriterion,
    class_map: HashMap<u64, usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> SplitAccumulator<F> for ClassificationAccumulator<F> {
    fn new(y: &Array1<F>, indices: &[usize]) -> Self {
        let class_map = build_class_map(y, indices);
        let n_classes = class_map.len();

        let mut total_counts = vec![0usize; n_classes];
        for &i in indices {
            let cls = class_map[&float_key(y[i])];
            total_counts[cls] += 1;
        }

        Self {
            left_counts: vec![0usize; n_classes],
            right_counts: total_counts,
            n_left: 0,
            n_right: indices.len(),
            criterion: SplitCriterion::Gini, // overwritten by with_criterion
            class_map,
            _marker: std::marker::PhantomData,
        }
    }

    fn move_to_left(&mut self, y: &Array1<F>, idx: usize) {
        let cls = self.class_map[&float_key(y[idx])];
        self.left_counts[cls] += 1;
        self.right_counts[cls] -= 1;
        self.n_left += 1;
        self.n_right -= 1;
    }

    fn weighted_impurity(&self, n: usize) -> F {
        let n_f = F::from_usize(n).unwrap();
        let nl = F::from_usize(self.n_left).unwrap();
        let nr = F::from_usize(self.n_right).unwrap();
        let left_imp = impurity_from_counts(&self.left_counts, self.n_left, self.criterion);
        let right_imp = impurity_from_counts(&self.right_counts, self.n_right, self.criterion);
        (nl / n_f) * left_imp + (nr / n_f) * right_imp
    }
}

impl<F: Float> ClassificationAccumulator<F> {
    fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    fn n_left(&self) -> usize {
        self.n_left
    }

    fn n_right(&self) -> usize {
        self.n_right
    }
}

/// Regression accumulator using running sum and sum-of-squares.
struct RegressionAccumulator<F: Float> {
    left_sum: F,
    left_sum_sq: F,
    right_sum: F,
    right_sum_sq: F,
    n_left: usize,
    n_right: usize,
}

impl<F: Float> SplitAccumulator<F> for RegressionAccumulator<F> {
    fn new(y: &Array1<F>, indices: &[usize]) -> Self {
        let mut total_sum = F::zero();
        let mut total_sum_sq = F::zero();
        for &i in indices {
            let v = y[i];
            total_sum += v;
            total_sum_sq += v * v;
        }

        Self {
            left_sum: F::zero(),
            left_sum_sq: F::zero(),
            right_sum: total_sum,
            right_sum_sq: total_sum_sq,
            n_left: 0,
            n_right: indices.len(),
        }
    }

    fn move_to_left(&mut self, y: &Array1<F>, idx: usize) {
        let v = y[idx];
        self.left_sum += v;
        self.left_sum_sq += v * v;
        self.right_sum -= v;
        self.right_sum_sq -= v * v;
        self.n_left += 1;
        self.n_right -= 1;
    }

    fn weighted_impurity(&self, n: usize) -> F {
        let n_f = F::from_usize(n).unwrap();
        let nl = F::from_usize(self.n_left).unwrap();
        let nr = F::from_usize(self.n_right).unwrap();
        // MSE = sum_sq/n - (sum/n)^2
        let left_mse = self.left_sum_sq / nl - (self.left_sum / nl) * (self.left_sum / nl);
        let right_mse = self.right_sum_sq / nr - (self.right_sum / nr) * (self.right_sum / nr);
        (nl / n_f) * left_mse + (nr / n_f) * right_mse
    }
}

impl<F: Float> RegressionAccumulator<F> {
    fn n_left(&self) -> usize {
        self.n_left
    }

    fn n_right(&self) -> usize {
        self.n_right
    }
}

/// Unified split-finding loop parameterised by accumulator type.
///
/// Scans each feature's sorted values, moving samples from right to left
/// and evaluating candidate splits via the accumulator's impurity method.
#[allow(clippy::too_many_arguments)]
fn find_best_split_inner<F, A>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    min_samples_leaf: usize,
    n_features: usize,
    n: usize,
    parent_impurity: F,
    acc_init: impl Fn() -> A,
    n_left_fn: impl Fn(&A) -> usize,
    n_right_fn: impl Fn(&A) -> usize,
) -> Option<BestSplit<F>>
where
    F: Float,
    A: SplitAccumulator<F>,
{
    let mut best: Option<BestSplit<F>> = None;
    let mut best_improvement = F::neg_infinity();

    let mut sorted_pairs: Vec<(F, usize)> = Vec::with_capacity(n);

    for feature in 0..n_features {
        sort_feature_pairs(x, indices, feature, &mut sorted_pairs);

        let mut acc = acc_init();

        for pos in 0..n - 1 {
            let (cur_val, cur_idx) = sorted_pairs[pos];
            acc.move_to_left(y, cur_idx);

            // Only consider a split between distinct values
            let next_val = sorted_pairs[pos + 1].0;
            if (next_val - cur_val).abs() < F::from_f64(1e-15).unwrap() {
                continue;
            }

            // Check min_samples_leaf constraint
            if n_left_fn(&acc) < min_samples_leaf || n_right_fn(&acc) < min_samples_leaf {
                continue;
            }

            let threshold = (cur_val + next_val) / (F::one() + F::one());
            let improvement = parent_impurity - acc.weighted_impurity(n);

            try_update_best_split(
                improvement,
                &mut best_improvement,
                &mut best,
                feature,
                threshold,
                &sorted_pairs,
                pos,
            );
        }
    }

    best
}

/// Classification split finding with incremental class counts.
#[allow(clippy::too_many_arguments)]
fn find_best_split_classification<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    criterion: SplitCriterion,
    min_samples_leaf: usize,
    n_features: usize,
    n: usize,
    parent_impurity: F,
) -> Option<BestSplit<F>> {
    find_best_split_inner(
        x,
        y,
        indices,
        min_samples_leaf,
        n_features,
        n,
        parent_impurity,
        || ClassificationAccumulator::<F>::new(y, indices).with_criterion(criterion),
        |acc| acc.n_left(),
        |acc| acc.n_right(),
    )
}

/// Regression split finding with running sum and sum-of-squares.
fn find_best_split_regression<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    min_samples_leaf: usize,
    n_features: usize,
    n: usize,
    parent_impurity: F,
) -> Option<BestSplit<F>> {
    find_best_split_inner(
        x,
        y,
        indices,
        min_samples_leaf,
        n_features,
        n,
        parent_impurity,
        || RegressionAccumulator::<F>::new(y, indices),
        |acc| acc.n_left(),
        |acc| acc.n_right(),
    )
}

/// Build a mapping from class label (as f64 bits) to contiguous index.
fn build_class_map<F: Float>(y: &Array1<F>, indices: &[usize]) -> HashMap<u64, usize> {
    let mut map = HashMap::new();
    let mut next_idx = 0;
    for &i in indices {
        let bits = float_key(y[i]);
        if let std::collections::hash_map::Entry::Vacant(e) = map.entry(bits) {
            e.insert(next_idx);
            next_idx += 1;
        }
    }
    map
}

/// Compute Gini or Entropy impurity from class counts (O(k) where k = n_classes).
#[inline]
fn impurity_from_counts<F: Float>(counts: &[usize], total: usize, criterion: SplitCriterion) -> F {
    let n = F::from_usize(total).unwrap();
    match criterion {
        SplitCriterion::Gini => {
            let sum_sq: F = counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = F::from_usize(c).unwrap() / n;
                    p * p
                })
                .fold(F::zero(), |a, b| a + b);
            F::one() - sum_sq
        }
        SplitCriterion::Entropy => {
            let sum: F = counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = F::from_usize(c).unwrap() / n;
                    p * p.ln()
                })
                .fold(F::zero(), |a, b| a + b);
            -sum
        }
        SplitCriterion::Mse => unreachable!("MSE does not use class counts"),
    }
}

/// Compute impurity for a subset of samples.
#[inline]
pub fn compute_impurity<F: Float>(y: &Array1<F>, indices: &[usize], criterion: SplitCriterion) -> F {
    match criterion {
        SplitCriterion::Gini => gini(y, indices),
        SplitCriterion::Entropy => entropy(y, indices),
        SplitCriterion::Mse => mse_impurity(y, indices),
    }
}

#[inline]
fn gini<F: Float>(y: &Array1<F>, indices: &[usize]) -> F {
    let n = F::from_usize(indices.len()).unwrap();
    let class_counts = count_classes(y, indices);

    let sum_sq: F = class_counts
        .iter()
        .map(|&(_, count)| {
            let p = F::from_usize(count).unwrap() / n;
            p * p
        })
        .fold(F::zero(), |a, b| a + b);

    F::one() - sum_sq
}

#[inline]
fn entropy<F: Float>(y: &Array1<F>, indices: &[usize]) -> F {
    let n = F::from_usize(indices.len()).unwrap();
    let class_counts = count_classes(y, indices);

    let sum: F = class_counts
        .iter()
        .map(|&(_, count)| {
            let p = F::from_usize(count).unwrap() / n;
            if p > F::zero() {
                p * p.ln()
            } else {
                F::zero()
            }
        })
        .fold(F::zero(), |a, b| a + b);

    -sum
}

#[inline]
fn mse_impurity<F: Float>(y: &Array1<F>, indices: &[usize]) -> F {
    let n = F::from_usize(indices.len()).unwrap();
    let mean: F = indices.iter().map(|&i| y[i]).fold(F::zero(), |a, b| a + b) / n;

    indices
        .iter()
        .map(|&i| (y[i] - mean) * (y[i] - mean))
        .fold(F::zero(), |a, b| a + b)
        / n
}

/// Count occurrences of each class in a subset.
pub fn count_classes<F: Float>(y: &Array1<F>, indices: &[usize]) -> Vec<(F, usize)> {
    let mut map: HashMap<u64, (F, usize)> = HashMap::new();
    for &i in indices {
        let val = y[i];
        let bits = float_key(val);
        map.entry(bits)
            .and_modify(|e| e.1 += 1)
            .or_insert((val, 1));
    }
    let mut counts: Vec<(F, usize)> = map.into_values().collect();
    counts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    counts
}

/// Compute the majority class (for classification) or mean (for regression).
#[inline]
pub fn leaf_value<F: Float>(y: &Array1<F>, indices: &[usize], criterion: SplitCriterion) -> F {
    match criterion {
        SplitCriterion::Mse => {
            let n = F::from_usize(indices.len()).unwrap();
            indices.iter().map(|&i| y[i]).fold(F::zero(), |a, b| a + b) / n
        }
        SplitCriterion::Gini | SplitCriterion::Entropy => {
            let counts = count_classes(y, indices);
            counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .unwrap()
                .0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_gini_pure() {
        let y = array![1.0, 1.0, 1.0];
        let indices = vec![0, 1, 2];
        assert_abs_diff_eq!(gini(&y, &indices), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gini_balanced() {
        let y = array![0.0, 1.0];
        let indices = vec![0, 1];
        assert_abs_diff_eq!(gini(&y, &indices), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_pure() {
        let y = array![5.0, 5.0, 5.0];
        let indices = vec![0, 1, 2];
        assert_abs_diff_eq!(mse_impurity(&y, &indices), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_find_best_split() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let indices = vec![0, 1, 2, 3];

        let split = find_best_split(&x, &y, &indices, SplitCriterion::Gini, 1).unwrap();
        // Should split between 2.0 and 3.0
        assert!(split.threshold > 2.0 && split.threshold < 3.0);
    }

    #[test]
    fn test_find_best_split_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 1.5, 10.0, 10.5];
        let indices = vec![0, 1, 2, 3];

        let split = find_best_split(&x, &y, &indices, SplitCriterion::Mse, 1).unwrap();
        // Should split between 2.0 and 3.0
        assert!(split.threshold > 2.0 && split.threshold < 3.0);
        assert_eq!(split.left_indices.len(), 2);
        assert_eq!(split.right_indices.len(), 2);
    }

    #[test]
    fn test_count_classes_uses_exact_bits() {
        let y = array![0.0, 1.0, 0.0, 2.0, 1.0];
        let indices = vec![0, 1, 2, 3, 4];
        let counts = count_classes(&y, &indices);
        assert_eq!(counts.len(), 3);
        // Sorted by value: (0.0, 2), (1.0, 2), (2.0, 1)
        assert_eq!(counts[0].1, 2); // class 0.0
        assert_eq!(counts[1].1, 2); // class 1.0
        assert_eq!(counts[2].1, 1); // class 2.0
    }

    #[test]
    fn test_find_best_split_entropy() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let indices = vec![0, 1, 2, 3];

        let split = find_best_split(&x, &y, &indices, SplitCriterion::Entropy, 1).unwrap();
        assert!(split.threshold > 2.0 && split.threshold < 3.0);
    }

    #[test]
    fn test_min_samples_leaf_respected() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let indices = vec![0, 1, 2, 3];

        // min_samples_leaf=3 means no valid split with 4 samples
        let split = find_best_split(&x, &y, &indices, SplitCriterion::Gini, 3);
        assert!(split.is_none());
    }
}
