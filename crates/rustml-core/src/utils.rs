use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::error::{Result, RustMlError};
use crate::float::Float;

/// The return type for [`train_test_split`]: `(X_train, X_test, y_train, y_test)`.
pub type TrainTestSplit<F> = (Array2<F>, Array2<F>, Array1<F>, Array1<F>);

/// Split arrays into random train and test subsets.
///
/// Returns `(X_train, X_test, y_train, y_test)`.
pub fn train_test_split<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    test_size: f64,
    seed: u64,
) -> Result<TrainTestSplit<F>> {
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(RustMlError::EmptyInput("input is empty".into()));
    }
    if !(0.0..=1.0).contains(&test_size) {
        return Err(RustMlError::InvalidParameter(
            "test_size must be between 0.0 and 1.0".into(),
        ));
    }

    let n = x.nrows();
    let n_test = (n as f64 * test_size).round() as usize;
    let n_test = n_test.max(1).min(n - 1); // at least 1 in each split

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let test_indices = &indices[..n_test];
    let train_indices = &indices[n_test..];

    Ok((
        select_rows(x, train_indices),
        select_rows(x, test_indices),
        select_elements(y, train_indices),
        select_elements(y, test_indices),
    ))
}

/// Select rows from a 2D array by indices.
fn select_rows<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let ncols = x.ncols();
    let mut result = Array2::<F>::zeros((indices.len(), ncols));
    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&x.row(idx));
    }
    result
}

/// Select elements from a 1D array by indices.
fn select_elements<F: Float>(y: &Array1<F>, indices: &[usize]) -> Array1<F> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

/// K-fold cross-validation score.
///
/// Splits data into `k` folds, trains on k-1 folds, evaluates on the held-out fold.
/// Returns a vector of k scores (one per fold).
///
/// `scorer` is a function that takes (y_true, y_pred) and returns a score (higher is better).
/// `fit_predict` is a function that takes (X_train, y_train, X_test) and returns predictions.
pub fn cross_val_score<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<Vec<F>> {
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        )));
    }
    if k < 2 {
        return Err(RustMlError::InvalidParameter(
            "k must be >= 2 for cross-validation".into(),
        ));
    }
    if k > x.nrows() {
        return Err(RustMlError::InvalidParameter(format!(
            "k ({}) > number of samples ({})",
            k,
            x.nrows()
        )));
    }

    let n = x.nrows();
    let fold_size = n / k;
    let mut scores = Vec::with_capacity(k);

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { n } else { test_start + fold_size };

        let test_indices: Vec<usize> = (test_start..test_end).collect();
        let train_indices: Vec<usize> = (0..test_start).chain(test_end..n).collect();

        let x_train = select_rows(x, &train_indices);
        let y_train = select_elements(y, &train_indices);
        let x_test = select_rows(x, &test_indices);
        let y_test = select_elements(y, &test_indices);

        let y_pred = fit_predict(&x_train, &y_train, &x_test)?;
        let score = scorer(&y_test, &y_pred)?;
        scores.push(score);
    }

    Ok(scores)
}

/// Generate stratified k-fold cross-validation splits.
///
/// Returns a `Vec` of `(train_indices, test_indices)` tuples, one per fold.
/// Each fold has approximately the same proportion of each class as the full dataset.
///
/// Samples are grouped by class label, shuffled within each group using the
/// provided `seed`, then distributed across folds in a round-robin fashion.
pub fn stratified_k_fold<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(RustMlError::EmptyInput("input is empty".into()));
    }
    if k < 2 {
        return Err(RustMlError::InvalidParameter(
            "k must be >= 2 for cross-validation".into(),
        ));
    }
    if k > x.nrows() {
        return Err(RustMlError::InvalidParameter(format!(
            "k ({}) > number of samples ({})",
            k,
            x.nrows()
        )));
    }

    let n = x.nrows();

    // Group sample indices by class label.
    // We use the string representation of the float as a key to group by class.
    let mut class_indices: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        let label = format!("{}", y[i]);
        class_indices.entry(label).or_default().push(i);
    }

    // Shuffle within each class group.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for indices in class_indices.values_mut() {
        indices.shuffle(&mut rng);
    }

    // Distribute indices round-robin across folds.
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k];
    for indices in class_indices.values() {
        for (i, &idx) in indices.iter().enumerate() {
            folds[i % k].push(idx);
        }
    }

    // Build (train, test) pairs.
    let mut result = Vec::with_capacity(k);
    for fold in 0..k {
        let test_indices = folds[fold].clone();
        let train_indices: Vec<usize> = folds
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != fold)
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        result.push((train_indices, test_indices));
    }

    Ok(result)
}

/// Stratified k-fold cross-validation score.
///
/// Like [`cross_val_score`], but uses stratified folds to preserve class
/// proportions in each fold. Useful for classification tasks with imbalanced classes.
///
/// `scorer` takes `(y_true, y_pred)` and returns a score (higher is better).
/// `fit_predict` takes `(X_train, y_train, X_test)` and returns predictions.
pub fn cross_val_score_stratified<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<Vec<F>> {
    let folds = stratified_k_fold(x, y, k, seed)?;
    let mut scores = Vec::with_capacity(k);

    for (train_indices, test_indices) in &folds {
        let x_train = select_rows(x, train_indices);
        let y_train = select_elements(y, train_indices);
        let x_test = select_rows(x, test_indices);
        let y_test = select_elements(y, test_indices);

        let y_pred = fit_predict(&x_train, &y_train, &x_test)?;
        let score = scorer(&y_test, &y_pred)?;
        scores.push(score);
    }

    Ok(scores)
}

/// Results from a grid search with cross-validation.
#[derive(Debug, Clone)]
pub struct GridSearchResult<F: Float> {
    /// Best mean cross-validation score across all parameter configurations.
    pub best_score: F,
    /// Index into the `param_configs` slice that achieved the best score.
    pub best_params_index: usize,
    /// Cross-validation scores for every configuration and fold: `cv_scores[config][fold]`.
    pub cv_scores: Vec<Vec<F>>,
    /// Mean CV score for each parameter configuration.
    pub mean_scores: Vec<F>,
}

/// Grid search with stratified cross-validation.
///
/// Evaluates each parameter configuration in `param_configs` using stratified k-fold
/// cross-validation and returns the configuration that achieved the highest mean score.
///
/// Each element of `param_configs` is a `fit_predict` closure that encapsulates a
/// particular hyperparameter setting. This design keeps the grid search generic:
/// callers construct the closures with whatever parameters they want to tune.
///
/// `scorer` takes `(y_true, y_pred)` and returns a score (higher is better).
///
/// # Errors
///
/// Returns `InvalidParameter` if `param_configs` is empty, or propagates errors from
/// `stratified_k_fold` and the individual `fit_predict` / `scorer` calls.
pub fn grid_search_cv<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
    param_configs: &[impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>],
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<GridSearchResult<F>> {
    if param_configs.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "param_configs must not be empty".into(),
        ));
    }

    let folds = stratified_k_fold(x, y, k, seed)?;

    let mut cv_scores: Vec<Vec<F>> = Vec::with_capacity(param_configs.len());

    for fit_predict in param_configs {
        let mut fold_scores = Vec::with_capacity(k);
        for (train_indices, test_indices) in &folds {
            let x_train = select_rows(x, train_indices);
            let y_train = select_elements(y, train_indices);
            let x_test = select_rows(x, test_indices);
            let y_test = select_elements(y, test_indices);

            let y_pred = fit_predict(&x_train, &y_train, &x_test)?;
            let score = scorer(&y_test, &y_pred)?;
            fold_scores.push(score);
        }
        cv_scores.push(fold_scores);
    }

    let mean_scores: Vec<F> = cv_scores
        .iter()
        .map(|scores| {
            let sum: F = scores.iter().copied().sum();
            sum / F::from_usize(scores.len()).unwrap()
        })
        .collect();

    let (best_params_index, &best_score) = mean_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap(); // safe: param_configs is non-empty

    Ok(GridSearchResult {
        best_score,
        best_params_index,
        cv_scores,
        mean_scores,
    })
}

#[cfg(test)]
#[allow(clippy::type_complexity)]
mod tests {
    use super::*;
    use ndarray::array;

    // ---------------------------------------------------------------
    // Helpers used across multiple tests
    // ---------------------------------------------------------------

    /// Accuracy scorer: fraction of correct predictions.
    fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| (**t - **p).abs() < 1e-9)
            .count();
        Ok(correct as f64 / y_true.len() as f64)
    }

    /// Trivial predictor that always returns 0.0.
    fn predict_zero(
        _x_train: &Array2<f64>,
        _y_train: &Array1<f64>,
        x_test: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        Ok(Array1::zeros(x_test.nrows()))
    }

    /// Majority-class predictor: returns the most frequent label in y_train.
    fn predict_majority(
        _x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_test: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let mut counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for &v in y_train.iter() {
            *counts.entry(format!("{v}")).or_default() += 1;
        }
        let majority_label: f64 = counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .unwrap()
            .0
            .parse()
            .unwrap();
        Ok(Array1::from_elem(x_test.nrows(), majority_label))
    }

    // ---------------------------------------------------------------
    // train_test_split tests (existing)
    // ---------------------------------------------------------------

    #[test]
    fn test_train_test_split_sizes() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let (x_train, x_test, y_train, y_test) =
            train_test_split(&x, &y, 0.3, 42).unwrap();

        assert_eq!(x_train.nrows(), 7);
        assert_eq!(x_test.nrows(), 3);
        assert_eq!(y_train.len(), 7);
        assert_eq!(y_test.len(), 3);
    }

    #[test]
    fn test_train_test_split_deterministic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let (_, x_test1, _, _) = train_test_split(&x, &y, 0.4, 42).unwrap();
        let (_, x_test2, _, _) = train_test_split(&x, &y, 0.4, 42).unwrap();

        assert_eq!(x_test1, x_test2);
    }

    #[test]
    fn test_train_test_split_no_overlap() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let (_, _, y_train, y_test) = train_test_split(&x, &y, 0.4, 42).unwrap();

        // All original values should appear exactly once
        let mut all: Vec<f64> = y_train.iter().chain(y_test.iter()).copied().collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(all, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    // ---------------------------------------------------------------
    // cross_val_score tests (existing)
    // ---------------------------------------------------------------

    #[test]
    fn test_cross_val_score_basic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let scores = cross_val_score::<f64>(&x, &y, 3, predict_zero, accuracy).unwrap();

        assert_eq!(scores.len(), 3);
    }

    // ---------------------------------------------------------------
    // stratified_k_fold tests
    // ---------------------------------------------------------------

    #[test]
    fn test_stratified_k_fold_returns_correct_number_of_folds() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let folds = stratified_k_fold(&x, &y, 3, 42).unwrap();
        assert_eq!(folds.len(), 3);
    }

    #[test]
    fn test_stratified_k_fold_covers_all_indices() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let folds = stratified_k_fold(&x, &y, 3, 42).unwrap();

        // Collect all test indices; each sample should appear in exactly one test fold.
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.clone()).collect();
        all_test.sort();
        assert_eq!(all_test, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_stratified_k_fold_no_overlap_between_train_and_test() {
        let x = array![
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let folds = stratified_k_fold(&x, &y, 2, 99).unwrap();

        for (train, test) in &folds {
            // No overlap
            for t in test {
                assert!(!train.contains(t), "test index {} found in train set", t);
            }
            // Together they cover all samples
            let mut combined: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
            combined.sort();
            assert_eq!(combined, (0..8).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_stratified_k_fold_preserves_class_proportions() {
        // 6 samples of class 0, 3 samples of class 1 => ratio 2:1
        let x = array![
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let folds = stratified_k_fold(&x, &y, 3, 42).unwrap();

        for (_, test_indices) in &folds {
            let class_0 = test_indices.iter().filter(|&&i| y[i] == 0.0).count();
            let class_1 = test_indices.iter().filter(|&&i| y[i] == 1.0).count();
            // With 6 class-0 and 3 class-1 across 3 folds, each fold should get 2 and 1.
            assert_eq!(class_0, 2, "expected 2 class-0 samples per fold");
            assert_eq!(class_1, 1, "expected 1 class-1 sample per fold");
        }
    }

    #[test]
    fn test_stratified_k_fold_deterministic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let folds1 = stratified_k_fold(&x, &y, 3, 42).unwrap();
        let folds2 = stratified_k_fold(&x, &y, 3, 42).unwrap();

        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_stratified_k_fold_errors_on_k_less_than_2() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let err = stratified_k_fold(&x, &y, 1, 42).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    #[test]
    fn test_stratified_k_fold_errors_on_k_greater_than_n() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let err = stratified_k_fold(&x, &y, 3, 42).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    #[test]
    fn test_stratified_k_fold_errors_on_shape_mismatch() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![0.0, 1.0];

        let err = stratified_k_fold(&x, &y, 2, 42).unwrap_err();
        assert!(matches!(err, RustMlError::ShapeMismatch(_)));
    }

    #[test]
    fn test_stratified_k_fold_errors_on_empty_input() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let err = stratified_k_fold(&x, &y, 2, 42).unwrap_err();
        assert!(matches!(err, RustMlError::EmptyInput(_)));
    }

    // ---------------------------------------------------------------
    // cross_val_score_stratified tests
    // ---------------------------------------------------------------

    #[test]
    fn test_cross_val_score_stratified_returns_k_scores() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let scores =
            cross_val_score_stratified(&x, &y, 3, 42, predict_zero, accuracy).unwrap();

        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_cross_val_score_stratified_perfect_predictor() {
        // A "cheating" predictor that returns y_train labels (only works because
        // with stratified folds each test fold's labels match a known pattern).
        // Instead, just use a constant predictor on a homogeneous class.
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // all same class

        let scores =
            cross_val_score_stratified(&x, &y, 3, 42, predict_zero, accuracy).unwrap();

        // All predictions are 0.0, all true values are 0.0 => perfect accuracy
        for &s in &scores {
            assert!((s - 1.0).abs() < 1e-9, "expected accuracy 1.0, got {s}");
        }
    }

    #[test]
    fn test_cross_val_score_stratified_deterministic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let s1 = cross_val_score_stratified(&x, &y, 3, 42, predict_zero, accuracy).unwrap();
        let s2 = cross_val_score_stratified(&x, &y, 3, 42, predict_zero, accuracy).unwrap();

        assert_eq!(s1, s2);
    }

    #[test]
    fn test_cross_val_score_stratified_propagates_errors() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0]; // shape mismatch

        let err = cross_val_score_stratified(&x, &y, 2, 42, predict_zero, accuracy).unwrap_err();
        assert!(matches!(err, RustMlError::ShapeMismatch(_)));
    }

    #[test]
    fn test_cross_val_score_stratified_majority_class_baseline() {
        // With balanced 50/50 classes a majority predictor should get ~50% accuracy.
        let x = array![
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let scores =
            cross_val_score_stratified(&x, &y, 5, 42, predict_majority, accuracy).unwrap();

        let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        // Majority predictor on a 50/50 split should hover around 0.5
        assert!(
            (0.3..=0.7).contains(&mean),
            "expected mean accuracy around 0.5, got {mean}"
        );
    }

    // ---------------------------------------------------------------
    // grid_search_cv tests
    // ---------------------------------------------------------------

    #[test]
    fn test_grid_search_cv_selects_best_config() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        // Config 0: always predict 0 (50% accuracy on balanced data)
        // Config 1: always predict 1 (50% accuracy on balanced data)
        // Config 2: always predict the majority class from training set
        let configs: Vec<Box<dyn Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>>> = vec![
            Box::new(|_xt, _yt, x_te| Ok(Array1::from_elem(x_te.nrows(), 0.0))),
            Box::new(|_xt, _yt, x_te| Ok(Array1::from_elem(x_te.nrows(), 1.0))),
            Box::new(predict_majority),
        ];

        let result = grid_search_cv(&x, &y, 3, 42, &configs, accuracy).unwrap();

        assert_eq!(result.cv_scores.len(), 3);
        assert_eq!(result.mean_scores.len(), 3);
        assert!(result.best_params_index < 3);
        assert!(result.best_score >= result.mean_scores[0]);
        assert!(result.best_score >= result.mean_scores[1]);
        assert!(result.best_score >= result.mean_scores[2]);
    }

    #[test]
    fn test_grid_search_cv_single_config() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let configs: Vec<fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>> =
            vec![predict_zero];

        let result = grid_search_cv(&x, &y, 2, 42, &configs, accuracy).unwrap();

        assert_eq!(result.best_params_index, 0);
        assert_eq!(result.cv_scores.len(), 1);
        assert_eq!(result.cv_scores[0].len(), 2); // 2 folds
    }

    #[test]
    fn test_grid_search_cv_empty_configs_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let configs: Vec<fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>> =
            vec![];

        let err = grid_search_cv(&x, &y, 2, 42, &configs, accuracy).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    #[test]
    fn test_grid_search_cv_best_score_matches_mean() {
        let x = array![
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let configs: Vec<Box<dyn Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>>> = vec![
            Box::new(|_xt, _yt, x_te| Ok(Array1::from_elem(x_te.nrows(), 0.0))),
            Box::new(|_xt, _yt, x_te| Ok(Array1::from_elem(x_te.nrows(), 1.0))),
        ];

        let result = grid_search_cv(&x, &y, 2, 42, &configs, accuracy).unwrap();

        let best_mean = result.mean_scores[result.best_params_index];
        assert!(
            (result.best_score - best_mean).abs() < 1e-12,
            "best_score should equal mean_scores[best_params_index]"
        );
    }

    #[test]
    fn test_grid_search_cv_deterministic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let configs: Vec<fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>> =
            vec![predict_zero, predict_majority];

        let r1 = grid_search_cv(&x, &y, 3, 42, &configs, accuracy).unwrap();
        let r2 = grid_search_cv(&x, &y, 3, 42, &configs, accuracy).unwrap();

        assert_eq!(r1.best_params_index, r2.best_params_index);
        assert_eq!(r1.cv_scores, r2.cv_scores);
    }

    #[test]
    fn test_grid_search_cv_correct_fold_count() {
        let x = array![
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let configs: Vec<fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>> =
            vec![predict_zero, predict_majority];

        let result = grid_search_cv(&x, &y, 5, 42, &configs, accuracy).unwrap();

        for scores in &result.cv_scores {
            assert_eq!(scores.len(), 5, "each config should have k=5 fold scores");
        }
    }
}
