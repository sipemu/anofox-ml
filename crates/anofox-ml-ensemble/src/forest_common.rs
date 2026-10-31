//! Helpers shared by the random-forest / extra-trees / bagging ensembles.

use std::borrow::Cow;

use anofox_ml_core::{Float, Result, RustMlError};
use anofox_ml_trees::node::TreeNode;
use anofox_ml_trees::split::{
    child_seed, complement_features, compute_impurity, compute_weighted_impurity,
    find_random_split_weighted, make_leaf, select_feature_subset, splitmix64, SplitCriterion,
};
use anofox_ml_trees::MaxFeatures;
use ndarray::{Array1, Array2};

/// Validate a `max_features` setting against the data and resolve it to a
/// concrete per-split feature count (`None` = all features).
pub(crate) fn resolve_max_features(
    max_features: Option<MaxFeatures>,
    n_features: usize,
) -> Result<Option<usize>> {
    match max_features {
        None => Ok(None),
        Some(MaxFeatures::Fixed(k)) if k == 0 || k > n_features => {
            Err(RustMlError::InvalidParameter(format!(
                "max_features={k} is invalid for data with {n_features} features"
            )))
        }
        Some(MaxFeatures::Fraction(f)) if !(f > 0.0 && f <= 1.0) => Err(
            RustMlError::InvalidParameter(format!("max_features fraction {f} must be in (0, 1]")),
        ),
        Some(mf) => Ok(Some(mf.resolve(n_features))),
    }
}

/// `true` when `feature_indices` is exactly `0..n_features`.
pub(crate) fn is_identity(feature_indices: &[usize], n_features: usize) -> bool {
    feature_indices.len() == n_features && feature_indices.iter().enumerate().all(|(i, &f)| i == f)
}

/// Input matrix for a tree trained on `feature_indices`. Trees trained by the
/// current code use all features (per-split subsampling happens inside the
/// tree), so this borrows `x`; models deserialized from older versions that
/// stored a per-tree column subset get the matching sub-matrix.
pub(crate) fn tree_input<'a, F: Float>(
    x: &'a Array2<F>,
    feature_indices: &[usize],
) -> Cow<'a, Array2<F>> {
    if is_identity(feature_indices, x.ncols()) {
        Cow::Borrowed(x)
    } else {
        let n_rows = x.nrows();
        let mut out = Array2::<F>::zeros((n_rows, feature_indices.len()));
        for (j_new, &j) in feature_indices.iter().enumerate() {
            out.column_mut(j_new).assign(&x.column(j));
        }
        Cow::Owned(out)
    }
}

/// Copy the given rows (all columns) of `x` into a new standard-layout matrix.
/// Works for any memory layout of `x`.
pub(crate) fn select_rows<F: Float>(x: &Array2<F>, row_indices: &[usize]) -> Array2<F> {
    let mut out = Array2::<F>::zeros((row_indices.len(), x.ncols()));
    for (r_new, &r) in row_indices.iter().enumerate() {
        out.row_mut(r_new).assign(&x.row(r));
    }
    out
}

/// Index of the first maximum (ties -> lowest index, like `numpy.argmax`).
/// Fit one forest tree on the bootstrap rows, passing the matching slice of
/// the forest-level sample weights. A bootstrap draw whose weights are all
/// zero carries no information about the weighting and is fitted unweighted.
pub(crate) fn fit_tree_on_rows<F, T, FT>(
    tree: &T,
    x: &Array2<F>,
    y: &Array1<F>,
    rows: &[usize],
    sample_weight: Option<&Array1<F>>,
) -> Result<FT>
where
    F: Float,
    T: anofox_ml_core::Fit<F, Fitted = FT> + anofox_ml_core::FitWeighted<F, Fitted = FT>,
{
    let x_sub = select_rows(x, rows);
    let y_sub = y.select(ndarray::Axis(0), rows);
    match sample_weight {
        Some(w) => {
            let w_sub = w.select(ndarray::Axis(0), rows);
            if w_sub.iter().fold(F::zero(), |a, &b| a + b) > F::zero() {
                tree.fit_weighted(&x_sub, &y_sub, Some(&w_sub))
            } else {
                tree.fit(&x_sub, &y_sub)
            }
        }
        None => tree.fit(&x_sub, &y_sub),
    }
}

pub(crate) fn argmax_first<F: Float>(values: impl IntoIterator<Item = F>) -> usize {
    let mut best_i = 0;
    let mut best_v = F::neg_infinity();
    for (i, v) in values.into_iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

/// Sorted, de-duplicated class labels of `y`.
pub(crate) fn sorted_classes<F: Float>(y: &Array1<F>) -> Vec<F> {
    let all: Vec<usize> = (0..y.len()).collect();
    anofox_ml_trees::split::count_classes(y, &all)
        .into_iter()
        .map(|(c, _)| c)
        .collect()
}

/// Position of `label` in the sorted class list (exact match).
#[inline]
pub(crate) fn class_index<F: Float>(classes: &[F], label: F) -> Option<usize> {
    classes.iter().position(|&c| c == label)
}

/// Parameters for building one extremely-randomized tree.
pub(crate) struct ExtraTreeParams {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub criterion: SplitCriterion,
    /// Per-split feature count (`None` = all features).
    pub max_features_k: Option<usize>,
    pub n_features: usize,
}

/// Build an extremely-randomized tree: at each node draw a fresh random
/// subset of `max_features_k` features and a random threshold per feature.
///
/// `node_seed` drives both the feature subset and the thresholds (via
/// independent derived streams); children get SplitMix64-derived seeds so
/// every node position in every tree sees independent randomness.
pub(crate) fn build_extra_tree<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    depth: usize,
    params: &ExtraTreeParams,
    node_seed: u64,
    weights: Option<&Array1<F>>,
) -> TreeNode<F> {
    let n_samples = indices.len();
    let impurity = match weights {
        Some(w) => compute_weighted_impurity(y, indices, w, params.criterion),
        None => compute_impurity(y, indices, params.criterion),
    };

    let should_stop = n_samples < params.min_samples_split
        || params.max_depth.is_some_and(|d| depth >= d)
        || impurity < F::from_f64(1e-15).unwrap();

    if should_stop {
        return make_leaf(y, indices, params.criterion, weights);
    }

    let threshold_seed = splitmix64(node_seed ^ 0x5EED_7123_A5A5_0001);
    let find = |features: &[usize]| {
        find_random_split_weighted(
            x,
            y,
            indices,
            params.criterion,
            params.min_samples_leaf,
            features,
            threshold_seed,
            weights,
        )
    };
    let split = match params.max_features_k {
        Some(k) if k < params.n_features => {
            let subset = select_feature_subset(params.n_features, k, node_seed);
            find(&subset).or_else(|| find(&complement_features(params.n_features, &subset)))
        }
        _ => {
            let all: Vec<usize> = (0..params.n_features).collect();
            find(&all)
        }
    };

    match split {
        Some(split) => {
            let left = build_extra_tree(
                x,
                y,
                &split.left_indices,
                depth + 1,
                params,
                child_seed(node_seed, 0),
                weights,
            );
            let right = build_extra_tree(
                x,
                y,
                &split.right_indices,
                depth + 1,
                params,
                child_seed(node_seed, 1),
                weights,
            );
            TreeNode::Split {
                feature_index: split.feature_index,
                threshold: split.threshold,
                left: Box::new(left),
                right: Box::new(right),
                n_samples,
                impurity,
            }
        }
        None => make_leaf(y, indices, params.criterion, weights),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_max_features_validates() {
        assert_eq!(resolve_max_features(None, 10).unwrap(), None);
        assert_eq!(
            resolve_max_features(Some(MaxFeatures::Sqrt), 16).unwrap(),
            Some(4)
        );
        assert!(resolve_max_features(Some(MaxFeatures::Fixed(0)), 4).is_err());
        assert!(resolve_max_features(Some(MaxFeatures::Fixed(5)), 4).is_err());
        assert!(resolve_max_features(Some(MaxFeatures::Fraction(0.0)), 4).is_err());
        assert_eq!(
            resolve_max_features(Some(MaxFeatures::Fraction(1.0)), 4).unwrap(),
            Some(4)
        );
    }

    #[test]
    fn argmax_first_breaks_ties_low() {
        assert_eq!(argmax_first([0.5_f64, 0.5]), 0);
        assert_eq!(argmax_first([0.2_f64, 0.4, 0.4]), 1);
    }
}
