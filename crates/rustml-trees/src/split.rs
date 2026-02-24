use ndarray::{Array1, Array2};
use rustml_core::Float;

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
pub fn find_best_split<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    criterion: SplitCriterion,
    min_samples_leaf: usize,
) -> Option<BestSplit<F>> {
    let n_features = x.ncols();
    let mut best: Option<BestSplit<F>> = None;
    let mut best_improvement = F::neg_infinity();

    let parent_impurity = compute_impurity(y, indices, criterion);

    for feature in 0..n_features {
        // Get sorted unique thresholds (midpoints between consecutive values)
        let mut feature_vals: Vec<(F, usize)> = indices
            .iter()
            .map(|&i| (x[[i, feature]], i))
            .collect();
        feature_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Try midpoints between consecutive distinct values
        let mut prev_val = feature_vals[0].0;
        for &(cur_val, _) in &feature_vals[1..] {

            if (cur_val - prev_val).abs() < F::from_f64(1e-15).unwrap() {
                continue;
            }

            let threshold = (prev_val + cur_val) / (F::one() + F::one());
            prev_val = cur_val;

            let (left, right): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .partition(|&&i| x[[i, feature]] <= threshold);

            if left.len() < min_samples_leaf || right.len() < min_samples_leaf {
                continue;
            }

            let n = F::from_usize(indices.len()).unwrap();
            let nl = F::from_usize(left.len()).unwrap();
            let nr = F::from_usize(right.len()).unwrap();

            let left_impurity = compute_impurity(y, &left, criterion);
            let right_impurity = compute_impurity(y, &right, criterion);

            let improvement =
                parent_impurity - (nl / n) * left_impurity - (nr / n) * right_impurity;

            if improvement > best_improvement {
                best_improvement = improvement;
                best = Some(BestSplit {
                    feature_index: feature,
                    threshold,
                    left_indices: left,
                    right_indices: right,
                    improvement,
                });
            }
        }
    }

    best
}

/// Compute impurity for a subset of samples.
pub fn compute_impurity<F: Float>(y: &Array1<F>, indices: &[usize], criterion: SplitCriterion) -> F {
    match criterion {
        SplitCriterion::Gini => gini(y, indices),
        SplitCriterion::Entropy => entropy(y, indices),
        SplitCriterion::Mse => mse_impurity(y, indices),
    }
}

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
    let mut counts: Vec<(F, usize)> = Vec::new();
    for &i in indices {
        let val = y[i];
        if let Some(entry) = counts
            .iter_mut()
            .find(|(c, _)| (*c - val).abs() < F::from_f64(1e-9).unwrap())
        {
            entry.1 += 1;
        } else {
            counts.push((val, 1));
        }
    }
    counts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    counts
}

/// Compute the majority class (for classification) or mean (for regression).
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
}
