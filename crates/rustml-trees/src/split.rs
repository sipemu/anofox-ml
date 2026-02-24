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

/// Classification split finding with incremental class counts.
///
/// For each feature:
/// 1. Sort indices by feature value.
/// 2. Start with all samples in "right"; maintain left_counts/right_counts
///    as arrays indexed by class.
/// 3. Scan sorted values left→right: move each sample from right→left,
///    update counts in O(1), compute impurity from counts in O(k).
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
    // Build a class-index map: map each distinct class label to a contiguous index.
    let class_map = build_class_map(y, indices);
    let n_classes = class_map.len();

    // Count total classes for "right" initialization.
    let mut total_counts = vec![0usize; n_classes];
    for &i in indices {
        let cls = class_map[&float_key(y[i])];
        total_counts[cls] += 1;
    }

    let mut best: Option<BestSplit<F>> = None;
    let mut best_improvement = F::neg_infinity();
    let n_f = F::from_usize(n).unwrap();

    // Reusable buffer for sorted (feature_value, original_index) pairs
    let mut sorted_pairs: Vec<(F, usize)> = Vec::with_capacity(n);

    for feature in 0..n_features {
        // Sort indices by feature value
        sorted_pairs.clear();
        sorted_pairs.extend(indices.iter().map(|&i| (x[[i, feature]], i)));
        sorted_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Initialize: all samples in "right", none in "left"
        let mut left_counts = vec![0usize; n_classes];
        let mut right_counts = total_counts.clone();
        let mut n_left = 0usize;
        let mut n_right = n;

        // Scan left→right through sorted values
        for pos in 0..n - 1 {
            let (cur_val, cur_idx) = sorted_pairs[pos];
            let cls = class_map[&float_key(y[cur_idx])];

            // Move this sample from right to left
            left_counts[cls] += 1;
            right_counts[cls] -= 1;
            n_left += 1;
            n_right -= 1;

            // Only consider a split between distinct values
            let next_val = sorted_pairs[pos + 1].0;
            if (next_val - cur_val).abs() < F::from_f64(1e-15).unwrap() {
                continue;
            }

            // Check min_samples_leaf constraint
            if n_left < min_samples_leaf || n_right < min_samples_leaf {
                continue;
            }

            let threshold = (cur_val + next_val) / (F::one() + F::one());

            let nl = F::from_usize(n_left).unwrap();
            let nr = F::from_usize(n_right).unwrap();
            let left_impurity = impurity_from_counts(&left_counts, n_left, criterion);
            let right_impurity = impurity_from_counts(&right_counts, n_right, criterion);

            let improvement =
                parent_impurity - (nl / n_f) * left_impurity - (nr / n_f) * right_impurity;

            if improvement > best_improvement {
                best_improvement = improvement;
                // Reconstruct left/right indices from sorted order
                let left_indices: Vec<usize> =
                    sorted_pairs[..=pos].iter().map(|&(_, idx)| idx).collect();
                let right_indices: Vec<usize> =
                    sorted_pairs[pos + 1..].iter().map(|&(_, idx)| idx).collect();
                best = Some(BestSplit {
                    feature_index: feature,
                    threshold,
                    left_indices,
                    right_indices,
                    improvement,
                });
            }
        }
    }

    best
}

/// Regression split finding with running sum and sum-of-squares.
///
/// For each feature, maintain running left_sum, left_sum_sq, right_sum,
/// right_sum_sq so MSE can be computed in O(1) per threshold:
///   MSE = sum_sq/n - (sum/n)²
fn find_best_split_regression<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    min_samples_leaf: usize,
    n_features: usize,
    n: usize,
    parent_impurity: F,
) -> Option<BestSplit<F>> {
    // Precompute total sum and sum of squares
    let mut total_sum = F::zero();
    let mut total_sum_sq = F::zero();
    for &i in indices {
        let v = y[i];
        total_sum += v;
        total_sum_sq += v * v;
    }

    let mut best: Option<BestSplit<F>> = None;
    let mut best_improvement = F::neg_infinity();
    let n_f = F::from_usize(n).unwrap();

    let mut sorted_pairs: Vec<(F, usize)> = Vec::with_capacity(n);

    for feature in 0..n_features {
        sorted_pairs.clear();
        sorted_pairs.extend(indices.iter().map(|&i| (x[[i, feature]], i)));
        sorted_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut left_sum = F::zero();
        let mut left_sum_sq = F::zero();
        let mut n_left = 0usize;

        for pos in 0..n - 1 {
            let (cur_val, cur_idx) = sorted_pairs[pos];
            let v = y[cur_idx];

            left_sum += v;
            left_sum_sq += v * v;
            n_left += 1;
            let n_right = n - n_left;

            let next_val = sorted_pairs[pos + 1].0;
            if (next_val - cur_val).abs() < F::from_f64(1e-15).unwrap() {
                continue;
            }

            if n_left < min_samples_leaf || n_right < min_samples_leaf {
                continue;
            }

            let threshold = (cur_val + next_val) / (F::one() + F::one());

            let nl = F::from_usize(n_left).unwrap();
            let nr = F::from_usize(n_right).unwrap();

            let right_sum = total_sum - left_sum;
            let right_sum_sq = total_sum_sq - left_sum_sq;

            // MSE = sum_sq/n - (sum/n)^2
            let left_mse = left_sum_sq / nl - (left_sum / nl) * (left_sum / nl);
            let right_mse = right_sum_sq / nr - (right_sum / nr) * (right_sum / nr);

            let improvement =
                parent_impurity - (nl / n_f) * left_mse - (nr / n_f) * right_mse;

            if improvement > best_improvement {
                best_improvement = improvement;
                let left_indices: Vec<usize> =
                    sorted_pairs[..=pos].iter().map(|&(_, idx)| idx).collect();
                let right_indices: Vec<usize> =
                    sorted_pairs[pos + 1..].iter().map(|&(_, idx)| idx).collect();
                best = Some(BestSplit {
                    feature_index: feature,
                    threshold,
                    left_indices,
                    right_indices,
                    improvement,
                });
            }
        }
    }

    best
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
