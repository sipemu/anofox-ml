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

/// Evaluate a single fold: split data by the given indices, fit, predict, and score.
fn evaluate_fold<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    train_indices: &[usize],
    test_indices: &[usize],
    fit_predict: &impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    scorer: &impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<F> {
    let x_train = select_rows(x, train_indices);
    let y_train = select_elements(y, train_indices);
    let x_test = select_rows(x, test_indices);
    let y_test = select_elements(y, test_indices);

    let y_pred = fit_predict(&x_train, &y_train, &x_test)?;
    scorer(&y_test, &y_pred)
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
        let test_end = if fold == k - 1 {
            n
        } else {
            test_start + fold_size
        };

        let test_indices: Vec<usize> = (test_start..test_end).collect();
        let train_indices: Vec<usize> = (0..test_start).chain(test_end..n).collect();

        let score = evaluate_fold(x, y, &train_indices, &test_indices, &fit_predict, &scorer)?;
        scores.push(score);
    }

    Ok(scores)
}

/// Group sample indices by their class label (string representation of the float value).
fn group_indices_by_class<F: Float>(
    y: &Array1<F>,
    n: usize,
) -> std::collections::BTreeMap<String, Vec<usize>> {
    let mut class_indices: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        let label = format!("{}", y[i]);
        class_indices.entry(label).or_default().push(i);
    }
    class_indices
}

/// Distribute class indices across `k` folds in round-robin order.
fn distribute_round_robin(
    class_indices: &std::collections::BTreeMap<String, Vec<usize>>,
    k: usize,
) -> Vec<Vec<usize>> {
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k];
    for indices in class_indices.values() {
        for (i, &idx) in indices.iter().enumerate() {
            folds[i % k].push(idx);
        }
    }
    folds
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

    let mut class_indices = group_indices_by_class(y, n);

    // Shuffle within each class group.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for indices in class_indices.values_mut() {
        indices.shuffle(&mut rng);
    }

    let folds = distribute_round_robin(&class_indices, k);

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
        let score = evaluate_fold(x, y, train_indices, test_indices, &fit_predict, &scorer)?;
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
            let score = evaluate_fold(x, y, train_indices, test_indices, fit_predict, &scorer)?;
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

/// Non-stratified K-fold splitting.
///
/// Splits indices `0..n_samples` into `k` consecutive folds. Returns a vector
/// of `(train_indices, test_indices)` pairs — one per fold.
///
/// # Errors
///
/// Returns `InvalidParameter` if `k < 2` or `k > n_samples`.
pub fn k_fold(n_samples: usize, k: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if k < 2 {
        return Err(RustMlError::InvalidParameter(
            "k must be >= 2 for cross-validation".into(),
        ));
    }
    if k > n_samples {
        return Err(RustMlError::InvalidParameter(format!(
            "k ({}) > number of samples ({})",
            k, n_samples
        )));
    }

    let fold_size = n_samples / k;
    let remainder = n_samples % k;
    let mut folds = Vec::with_capacity(k);
    let mut start = 0;

    for i in 0..k {
        // Distribute the remainder across the first `remainder` folds.
        let size = fold_size + if i < remainder { 1 } else { 0 };
        let end = start + size;
        let test_indices: Vec<usize> = (start..end).collect();
        let train_indices: Vec<usize> = (0..start).chain(end..n_samples).collect();
        folds.push((train_indices, test_indices));
        start = end;
    }

    Ok(folds)
}

/// Random train/test split indices (shuffle split).
///
/// Returns `n_splits` independent random `(train_indices, test_indices)` pairs.
/// `test_size` is the fraction of samples in the test set (must be in `(0, 1)`).
/// Each split is produced by an independent shuffle seeded from `seed + split_index`.
///
/// # Errors
///
/// Returns `InvalidParameter` if `test_size` is not in `(0, 1)`, if `n_splits < 1`,
/// or if `n_samples < 2`.
pub fn shuffle_split(
    n_samples: usize,
    n_splits: usize,
    test_size: f64,
    seed: u64,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(RustMlError::InvalidParameter(
            "test_size must be in (0, 1)".into(),
        ));
    }
    if n_splits < 1 {
        return Err(RustMlError::InvalidParameter(
            "n_splits must be >= 1".into(),
        ));
    }
    if n_samples < 2 {
        return Err(RustMlError::InvalidParameter(
            "n_samples must be >= 2 for shuffle split".into(),
        ));
    }

    let n_test = ((n_samples as f64 * test_size).round() as usize)
        .max(1)
        .min(n_samples - 1);
    let mut result = Vec::with_capacity(n_splits);

    for split in 0..n_splits {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(split as u64));
        indices.shuffle(&mut rng);

        let test_indices = indices[..n_test].to_vec();
        let train_indices = indices[n_test..].to_vec();
        result.push((train_indices, test_indices));
    }

    Ok(result)
}

/// Leave-one-out cross-validation splits.
///
/// Returns `n_samples` folds. In fold *i* the test set contains only sample *i*
/// and the training set contains all other samples.
pub fn leave_one_out(n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut folds = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let test_indices = vec![i];
        let train_indices: Vec<usize> = (0..i).chain(i + 1..n_samples).collect();
        folds.push((train_indices, test_indices));
    }
    folds
}

/// Time-series aware forward-chaining splits.
///
/// In split *k* (0-indexed) the training set is `[0..train_end]` and the test set
/// is `[train_end..test_end]`. The training set grows with each fold, ensuring
/// that future data never leaks into the training set.
///
/// The minimum training size is `n_samples / (n_splits + 1)` (rounded), and each
/// subsequent fold adds one chunk of that size to the training set.
///
/// # Errors
///
/// Returns `InvalidParameter` if `n_splits < 1` or `n_splits >= n_samples`.
pub fn time_series_split(
    n_samples: usize,
    n_splits: usize,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if n_splits < 1 {
        return Err(RustMlError::InvalidParameter(
            "n_splits must be >= 1".into(),
        ));
    }
    if n_splits >= n_samples {
        return Err(RustMlError::InvalidParameter(format!(
            "n_splits ({}) must be < n_samples ({})",
            n_splits, n_samples
        )));
    }

    // Match sklearn's TimeSeriesSplit: test_size = n / (n_splits + 1),
    // initial train_size = n - n_splits * test_size.
    let test_size = n_samples / (n_splits + 1);
    let test_size = test_size.max(1);
    let mut result = Vec::with_capacity(n_splits);

    for i in 0..n_splits {
        let test_start = n_samples - (n_splits - i) * test_size;
        let test_end = test_start + test_size;

        let train_indices: Vec<usize> = (0..test_start).collect();
        let test_indices: Vec<usize> = (test_start..test_end).collect();
        result.push((train_indices, test_indices));
    }

    Ok(result)
}

/// Repeated K-fold cross-validation splits.
///
/// Repeats K-fold `n_repeats` times, each time with a different random shuffle
/// seeded from `seed + repeat_index`. Returns `k * n_repeats` folds total.
///
/// # Errors
///
/// Returns `InvalidParameter` if `k < 2`, `k > n_samples`, or `n_repeats < 1`.
pub fn repeated_k_fold(
    n_samples: usize,
    k: usize,
    n_repeats: usize,
    seed: u64,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if k < 2 {
        return Err(RustMlError::InvalidParameter(
            "k must be >= 2 for cross-validation".into(),
        ));
    }
    if k > n_samples {
        return Err(RustMlError::InvalidParameter(format!(
            "k ({}) > number of samples ({})",
            k, n_samples
        )));
    }
    if n_repeats < 1 {
        return Err(RustMlError::InvalidParameter(
            "n_repeats must be >= 1".into(),
        ));
    }

    let mut result = Vec::with_capacity(k * n_repeats);

    for repeat in 0..n_repeats {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(repeat as u64));
        indices.shuffle(&mut rng);

        let fold_size = n_samples / k;
        let remainder = n_samples % k;
        let mut start = 0;

        for i in 0..k {
            let size = fold_size + if i < remainder { 1 } else { 0 };
            let end = start + size;

            let test_indices: Vec<usize> = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();
            result.push((train_indices, test_indices));
            start = end;
        }
    }

    Ok(result)
}

/// Randomized search with stratified cross-validation.
///
/// Randomly samples `n_iter` parameter configurations from `param_sampler` and
/// evaluates each using stratified k-fold cross-validation. Returns the
/// configuration that achieved the highest mean score.
///
/// This is more efficient than grid search when the parameter space is large,
/// as it explores the space randomly rather than exhaustively.
///
/// `param_sampler` is a closure that takes a random seed and returns a
/// `fit_predict` closure for that configuration.
///
/// # Errors
///
/// Returns `InvalidParameter` if `n_iter == 0`, or propagates errors from
/// `stratified_k_fold` and the individual evaluations.
pub fn randomized_search_cv<F, S, P>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
    n_iter: usize,
    param_sampler: S,
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<GridSearchResult<F>>
where
    F: Float,
    S: Fn(u64) -> P,
    P: Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
{
    if n_iter == 0 {
        return Err(RustMlError::InvalidParameter("n_iter must be > 0".into()));
    }

    let folds = stratified_k_fold(x, y, k, seed)?;

    let mut cv_scores: Vec<Vec<F>> = Vec::with_capacity(n_iter);
    let mut mean_scores: Vec<F> = Vec::with_capacity(n_iter);

    for i in 0..n_iter {
        let config_seed = seed
            .wrapping_add(i as u64)
            .wrapping_mul(6364136223846793005);
        let fit_predict = param_sampler(config_seed);

        let mut fold_scores = Vec::with_capacity(k);
        for (train_indices, test_indices) in &folds {
            let score = evaluate_fold(x, y, train_indices, test_indices, &fit_predict, &scorer)?;
            fold_scores.push(score);
        }

        let sum: F = fold_scores.iter().copied().sum();
        let mean = sum / F::from_usize(fold_scores.len()).unwrap();
        mean_scores.push(mean);
        cv_scores.push(fold_scores);
    }

    let (best_params_index, &best_score) = mean_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    Ok(GridSearchResult {
        best_score,
        best_params_index,
        cv_scores,
        mean_scores,
    })
}

/// Cross-validated predictions.
///
/// Like [`cross_val_score`], but returns out-of-fold predictions for every
/// sample instead of per-fold scores. Each sample gets a prediction from
/// the model trained without it. Uses stratified K-fold splitting.
///
/// Returns an `Array1<F>` of length `n_samples` with the out-of-fold
/// predictions arranged in the original sample order.
pub fn cross_val_predict<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
) -> Result<Array1<F>> {
    let folds = stratified_k_fold(x, y, k, seed)?;
    let n = x.nrows();
    let mut predictions = Array1::zeros(n);

    for (train_indices, test_indices) in &folds {
        let x_train = select_rows(x, train_indices);
        let y_train = select_elements(y, train_indices);
        let x_test = select_rows(x, test_indices);

        let y_pred = fit_predict(&x_train, &y_train, &x_test)?;

        for (local_idx, &global_idx) in test_indices.iter().enumerate() {
            predictions[global_idx] = y_pred[local_idx];
        }
    }

    Ok(predictions)
}

/// Repeated stratified K-fold splitting.
///
/// Repeats stratified K-fold `n_repeats` times with different random shuffles.
/// Returns `k * n_repeats` fold pairs total.
pub fn repeated_stratified_k_fold<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    n_repeats: usize,
    seed: u64,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if n_repeats == 0 {
        return Err(RustMlError::InvalidParameter(
            "n_repeats must be > 0".into(),
        ));
    }

    let mut all_folds = Vec::with_capacity(k * n_repeats);
    for r in 0..n_repeats {
        let repeat_seed = seed
            .wrapping_add(r as u64)
            .wrapping_mul(6364136223846793005);
        let folds = stratified_k_fold(x, y, k, repeat_seed)?;
        all_folds.extend(folds);
    }

    Ok(all_folds)
}

/// Stratified shuffle split.
///
/// Like [`shuffle_split`] but preserves the class distribution in each split.
/// Returns `n_splits` random (train, test) index pairs where each split
/// maintains approximately the same class proportions as the full dataset.
pub fn stratified_shuffle_split<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    n_splits: usize,
    test_size: f64,
    seed: u64,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(RustMlError::InvalidParameter(
            "test_size must be in (0, 1)".into(),
        ));
    }
    if n_splits == 0 {
        return Err(RustMlError::InvalidParameter("n_splits must be > 0".into()));
    }
    let n = x.nrows();
    if n < 2 {
        return Err(RustMlError::InvalidParameter(
            "need at least 2 samples".into(),
        ));
    }

    // Group indices by class
    let mut class_indices: std::collections::HashMap<u64, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &val) in y.iter().enumerate() {
        let key = val.to_f64().unwrap().to_bits();
        class_indices.entry(key).or_default().push(i);
    }

    let mut result = Vec::with_capacity(n_splits);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for _ in 0..n_splits {
        let mut test_indices = Vec::new();
        let mut train_indices = Vec::new();

        for indices in class_indices.values_mut() {
            indices.shuffle(&mut rng);
            let n_test = (indices.len() as f64 * test_size).max(1.0).ceil() as usize;
            let n_test = n_test.min(indices.len().saturating_sub(1)); // keep at least 1 for train

            test_indices.extend_from_slice(&indices[..n_test]);
            train_indices.extend_from_slice(&indices[n_test..]);
        }

        train_indices.sort_unstable();
        test_indices.sort_unstable();
        result.push((train_indices, test_indices));
    }

    Ok(result)
}

/// Leave-P-Out cross-validation.
///
/// Generates all possible splits where exactly `p` samples are held out
/// as the test set and the remaining `n - p` samples form the training set.
///
/// Warning: produces C(n, p) folds which grows combinatorially. Use small
/// values of p.
pub fn leave_p_out(n_samples: usize, p: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if p == 0 || p >= n_samples {
        return Err(RustMlError::InvalidParameter(format!(
            "p must be in [1, n_samples), got p={} n_samples={}",
            p, n_samples
        )));
    }

    let mut result = Vec::new();
    let mut combo = Vec::with_capacity(p);

    fn generate(
        start: usize,
        n: usize,
        p: usize,
        combo: &mut Vec<usize>,
        result: &mut Vec<(Vec<usize>, Vec<usize>)>,
    ) {
        if combo.len() == p {
            let test: Vec<usize> = combo.clone();
            let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
            let train: Vec<usize> = (0..n).filter(|i| !test_set.contains(i)).collect();
            result.push((train, test));
            return;
        }
        let remaining = p - combo.len();
        for i in start..=(n - remaining) {
            combo.push(i);
            generate(i + 1, n, p, combo, result);
            combo.pop();
        }
    }

    generate(0, n_samples, p, &mut combo, &mut result);
    Ok(result)
}

/// Group K-fold cross-validation.
///
/// Splits data so that samples from the same group are always in the same
/// fold. This prevents data leakage when samples within a group are
/// correlated (e.g., multiple measurements from the same subject).
///
/// `groups` is an array of group labels (integers) with one entry per sample.
pub fn group_k_fold(groups: &Array1<usize>, k: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if k < 2 {
        return Err(RustMlError::InvalidParameter("k must be >= 2".into()));
    }

    // Collect unique groups
    let mut unique_groups: Vec<usize> = groups.iter().copied().collect();
    unique_groups.sort_unstable();
    unique_groups.dedup();

    let n_groups = unique_groups.len();
    if k > n_groups {
        return Err(RustMlError::InvalidParameter(format!(
            "k={} exceeds number of groups={}",
            k, n_groups
        )));
    }

    // Assign each group to a fold (round-robin)
    let mut group_to_fold: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for (i, &g) in unique_groups.iter().enumerate() {
        group_to_fold.insert(g, i % k);
    }

    // Build fold index lists
    let mut fold_indices: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &g) in groups.iter().enumerate() {
        let fold = group_to_fold[&g];
        fold_indices[fold].push(i);
    }

    // Generate train/test pairs
    let mut result = Vec::with_capacity(k);
    for test_fold in 0..k {
        let test_indices = fold_indices[test_fold].clone();
        let train_indices: Vec<usize> = (0..k)
            .filter(|&f| f != test_fold)
            .flat_map(|f| fold_indices[f].iter().copied())
            .collect();
        result.push((train_indices, test_indices));
    }

    Ok(result)
}

/// Result from [`cross_validate`].
pub struct CrossValidateResult<F: Float> {
    /// Per-fold scores for each metric.
    pub scores: Vec<Vec<F>>,
    /// Mean score for each metric.
    pub mean_scores: Vec<F>,
    /// Fit time per fold in seconds.
    pub fit_times: Vec<f64>,
    /// Score time per fold in seconds.
    pub score_times: Vec<f64>,
}

/// Cross-validate with multiple metrics and timing.
///
/// Like [`cross_val_score_stratified`] but returns per-fold fit/score times
/// and supports multiple scoring functions.
pub fn cross_validate<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    scorers: &[&dyn Fn(&Array1<F>, &Array1<F>) -> Result<F>],
) -> Result<CrossValidateResult<F>> {
    if scorers.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "need at least one scorer".into(),
        ));
    }

    let folds = stratified_k_fold(x, y, k, seed)?;
    let n_metrics = scorers.len();

    let mut all_scores: Vec<Vec<F>> = vec![Vec::with_capacity(k); n_metrics];
    let mut fit_times = Vec::with_capacity(k);
    let mut score_times = Vec::with_capacity(k);

    for (train_indices, test_indices) in &folds {
        let x_train = select_rows(x, train_indices);
        let y_train = select_elements(y, train_indices);
        let x_test = select_rows(x, test_indices);
        let y_test = select_elements(y, test_indices);

        let fit_start = std::time::Instant::now();
        let y_pred = fit_predict(&x_train, &y_train, &x_test)?;
        let fit_elapsed = fit_start.elapsed().as_secs_f64();
        fit_times.push(fit_elapsed);

        let score_start = std::time::Instant::now();
        for (m, scorer) in scorers.iter().enumerate() {
            let s = scorer(&y_test, &y_pred)?;
            all_scores[m].push(s);
        }
        let score_elapsed = score_start.elapsed().as_secs_f64();
        score_times.push(score_elapsed);
    }

    let mean_scores: Vec<F> = all_scores
        .iter()
        .map(|scores| {
            let sum: F = scores.iter().copied().sum();
            sum / F::from_usize(scores.len()).unwrap()
        })
        .collect();

    Ok(CrossValidateResult {
        scores: all_scores,
        mean_scores,
        fit_times,
        score_times,
    })
}

/// Generate a learning curve: scores as a function of training set size.
///
/// Trains the model on increasing subsets of the training data and evaluates
/// on a held-out test set (the last fold of a stratified K-fold split).
///
/// `train_sizes` is a slice of fractions in (0, 1], e.g. `[0.1, 0.3, 0.5, 0.7, 1.0]`.
///
/// Returns `(train_sizes_abs, train_scores, test_scores)` where each inner
/// Vec has length `train_sizes.len()`.
pub fn learning_curve<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
    train_sizes: &[f64],
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<(Vec<usize>, Vec<Vec<F>>, Vec<Vec<F>>)> {
    if train_sizes.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "train_sizes must not be empty".into(),
        ));
    }
    for &ts in train_sizes {
        if ts <= 0.0 || ts > 1.0 {
            return Err(RustMlError::InvalidParameter(format!(
                "train_size must be in (0, 1], got {}",
                ts
            )));
        }
    }

    let folds = stratified_k_fold(x, y, k, seed)?;
    let mut train_sizes_abs = Vec::with_capacity(train_sizes.len());
    let mut all_train_scores: Vec<Vec<F>> = vec![Vec::with_capacity(k); train_sizes.len()];
    let mut all_test_scores: Vec<Vec<F>> = vec![Vec::with_capacity(k); train_sizes.len()];

    for (train_indices, test_indices) in &folds {
        let x_test = select_rows(x, test_indices);
        let y_test = select_elements(y, test_indices);

        for (si, &frac) in train_sizes.iter().enumerate() {
            let n_train = ((train_indices.len() as f64 * frac).ceil() as usize)
                .max(1)
                .min(train_indices.len());
            let sub_train = &train_indices[..n_train];

            if si == 0 || train_sizes_abs.len() <= si {
                if train_sizes_abs.len() <= si {
                    train_sizes_abs.push(n_train);
                }
            }

            let x_train = select_rows(x, sub_train);
            let y_train = select_elements(y, sub_train);

            let y_pred_train = fit_predict(&x_train, &y_train, &x_train)?;
            let train_score = scorer(&y_train, &y_pred_train)?;
            all_train_scores[si].push(train_score);

            let y_pred_test = fit_predict(&x_train, &y_train, &x_test)?;
            let test_score = scorer(&y_test, &y_pred_test)?;
            all_test_scores[si].push(test_score);
        }
    }

    Ok((train_sizes_abs, all_train_scores, all_test_scores))
}

/// Generate a validation curve: scores as a function of a hyperparameter.
///
/// Evaluates performance across different hyperparameter values using
/// stratified K-fold cross-validation.
///
/// `param_configs` is a slice of `fit_predict` closures, one per hyperparameter value.
///
/// Returns `(train_scores, test_scores)` where each inner Vec has k fold scores.
pub fn validation_curve<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
    param_configs: &[impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>],
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<(Vec<Vec<F>>, Vec<Vec<F>>)> {
    if param_configs.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "param_configs must not be empty".into(),
        ));
    }

    let folds = stratified_k_fold(x, y, k, seed)?;
    let n_configs = param_configs.len();
    let mut train_scores: Vec<Vec<F>> = vec![Vec::with_capacity(k); n_configs];
    let mut test_scores: Vec<Vec<F>> = vec![Vec::with_capacity(k); n_configs];

    for (train_indices, test_indices) in &folds {
        let x_train = select_rows(x, train_indices);
        let y_train = select_elements(y, train_indices);
        let x_test = select_rows(x, test_indices);
        let y_test = select_elements(y, test_indices);

        for (ci, fit_predict) in param_configs.iter().enumerate() {
            // Train score
            let y_pred_train = fit_predict(&x_train, &y_train, &x_train)?;
            let ts = scorer(&y_train, &y_pred_train)?;
            train_scores[ci].push(ts);

            // Test score
            let y_pred_test = fit_predict(&x_train, &y_train, &x_test)?;
            let vs = scorer(&y_test, &y_pred_test)?;
            test_scores[ci].push(vs);
        }
    }

    Ok((train_scores, test_scores))
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
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
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
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3, 42).unwrap();

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
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
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
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
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

        let scores = cross_val_score_stratified(&x, &y, 3, 42, predict_zero, accuracy).unwrap();

        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_cross_val_score_stratified_perfect_predictor() {
        // A "cheating" predictor that returns y_train labels (only works because
        // with stratified folds each test fold's labels match a known pattern).
        // Instead, just use a constant predictor on a homogeneous class.
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // all same class

        let scores = cross_val_score_stratified(&x, &y, 3, 42, predict_zero, accuracy).unwrap();

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
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let scores = cross_val_score_stratified(&x, &y, 5, 42, predict_majority, accuracy).unwrap();

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
        let configs: Vec<
            Box<dyn Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>>,
        > = vec![
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
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let configs: Vec<
            Box<dyn Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>>,
        > = vec![
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
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let configs: Vec<fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>> =
            vec![predict_zero, predict_majority];

        let result = grid_search_cv(&x, &y, 5, 42, &configs, accuracy).unwrap();

        for scores in &result.cv_scores {
            assert_eq!(scores.len(), 5, "each config should have k=5 fold scores");
        }
    }

    // ---------------------------------------------------------------
    // k_fold tests
    // ---------------------------------------------------------------

    #[test]
    fn test_k_fold_correct_number_of_folds() {
        let folds = k_fold(10, 5).unwrap();
        assert_eq!(folds.len(), 5);
    }

    #[test]
    fn test_k_fold_disjoint_and_full_coverage() {
        let folds = k_fold(10, 3).unwrap();

        // Every index appears in exactly one test fold.
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.clone()).collect();
        all_test.sort();
        assert_eq!(all_test, (0..10).collect::<Vec<_>>());

        // Train and test within each fold are disjoint and cover all samples.
        for (train, test) in &folds {
            for t in test {
                assert!(!train.contains(t), "test index {} found in train set", t);
            }
            let mut combined: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
            combined.sort();
            assert_eq!(combined, (0..10).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_k_fold_minimum_k() {
        let folds = k_fold(4, 2).unwrap();
        assert_eq!(folds.len(), 2);
        assert_eq!(folds[0].1.len(), 2);
        assert_eq!(folds[1].1.len(), 2);
    }

    #[test]
    fn test_k_fold_k_equals_n_samples() {
        // Leave-one-out equivalent
        let folds = k_fold(5, 5).unwrap();
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(test.len(), 1);
            assert_eq!(train.len(), 4);
        }
    }

    #[test]
    fn test_k_fold_uneven_split() {
        // 7 samples, 3 folds: first fold gets 3, next two get 2 each
        let folds = k_fold(7, 3).unwrap();
        let test_sizes: Vec<usize> = folds.iter().map(|(_, t)| t.len()).collect();
        assert_eq!(test_sizes.iter().sum::<usize>(), 7);
        // The first fold(s) get the extra sample
        assert_eq!(test_sizes[0], 3); // 7/3 = 2 remainder 1 => first fold gets 2+1=3
        assert_eq!(test_sizes[1], 2);
        assert_eq!(test_sizes[2], 2);
    }

    #[test]
    fn test_k_fold_error_k_less_than_2() {
        let err = k_fold(10, 1).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    #[test]
    fn test_k_fold_error_k_greater_than_n() {
        let err = k_fold(3, 5).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    // ---------------------------------------------------------------
    // shuffle_split tests
    // ---------------------------------------------------------------

    #[test]
    fn test_shuffle_split_correct_number_of_splits() {
        let splits = shuffle_split(10, 3, 0.2, 42).unwrap();
        assert_eq!(splits.len(), 3);
    }

    #[test]
    fn test_shuffle_split_correct_sizes() {
        let splits = shuffle_split(10, 1, 0.3, 42).unwrap();
        let (train, test) = &splits[0];
        assert_eq!(test.len(), 3); // 10 * 0.3 = 3
        assert_eq!(train.len(), 7);
    }

    #[test]
    fn test_shuffle_split_disjoint_and_full_coverage() {
        let splits = shuffle_split(10, 5, 0.2, 42).unwrap();
        for (train, test) in &splits {
            // Train and test must be disjoint
            for t in test {
                assert!(!train.contains(t), "test index {} found in train set", t);
            }
            // Together they cover all samples
            let mut combined: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
            combined.sort();
            assert_eq!(combined, (0..10).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_shuffle_split_deterministic() {
        let s1 = shuffle_split(10, 3, 0.3, 42).unwrap();
        let s2 = shuffle_split(10, 3, 0.3, 42).unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_shuffle_split_different_seeds_differ() {
        let s1 = shuffle_split(20, 1, 0.3, 42).unwrap();
        let s2 = shuffle_split(20, 1, 0.3, 99).unwrap();
        // Very unlikely to be identical with different seeds
        assert_ne!(s1[0].1, s2[0].1);
    }

    #[test]
    fn test_shuffle_split_error_test_size_zero() {
        let err = shuffle_split(10, 1, 0.0, 42).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    #[test]
    fn test_shuffle_split_error_test_size_one() {
        let err = shuffle_split(10, 1, 1.0, 42).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    #[test]
    fn test_shuffle_split_error_n_samples_too_small() {
        let err = shuffle_split(1, 1, 0.5, 42).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    // ---------------------------------------------------------------
    // leave_one_out tests
    // ---------------------------------------------------------------

    #[test]
    fn test_leave_one_out_correct_number_of_folds() {
        let folds = leave_one_out(5);
        assert_eq!(folds.len(), 5);
    }

    #[test]
    fn test_leave_one_out_each_fold_has_one_test() {
        let folds = leave_one_out(4);
        for (i, (train, test)) in folds.iter().enumerate() {
            assert_eq!(test.len(), 1);
            assert_eq!(test[0], i);
            assert_eq!(train.len(), 3);
        }
    }

    #[test]
    fn test_leave_one_out_disjoint_and_full_coverage() {
        let folds = leave_one_out(5);
        for (train, test) in &folds {
            assert!(!train.contains(&test[0]));
            let mut combined: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
            combined.sort();
            assert_eq!(combined, (0..5).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_leave_one_out_all_indices_tested() {
        let folds = leave_one_out(6);
        let all_test: Vec<usize> = folds.iter().map(|(_, t)| t[0]).collect();
        assert_eq!(all_test, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_leave_one_out_single_sample() {
        let folds = leave_one_out(1);
        assert_eq!(folds.len(), 1);
        assert_eq!(folds[0].0.len(), 0); // empty train
        assert_eq!(folds[0].1, vec![0]);
    }

    #[test]
    fn test_leave_one_out_empty() {
        let folds = leave_one_out(0);
        assert!(folds.is_empty());
    }

    // ---------------------------------------------------------------
    // time_series_split tests
    // ---------------------------------------------------------------

    #[test]
    fn test_time_series_split_correct_number_of_splits() {
        let splits = time_series_split(10, 3).unwrap();
        assert_eq!(splits.len(), 3);
    }

    #[test]
    fn test_time_series_split_training_set_grows() {
        let splits = time_series_split(12, 3).unwrap();
        for i in 1..splits.len() {
            assert!(
                splits[i].0.len() > splits[i - 1].0.len(),
                "training set should grow: fold {} has {} but fold {} has {}",
                i,
                splits[i].0.len(),
                i - 1,
                splits[i - 1].0.len()
            );
        }
    }

    #[test]
    fn test_time_series_split_no_future_leak() {
        let splits = time_series_split(10, 3).unwrap();
        for (train, test) in &splits {
            let max_train = *train.iter().max().unwrap();
            let min_test = *test.iter().min().unwrap();
            assert!(
                max_train < min_test,
                "training set max ({}) must be less than test set min ({})",
                max_train,
                min_test
            );
        }
    }

    #[test]
    fn test_time_series_split_last_fold_extends_to_end() {
        let splits = time_series_split(10, 3).unwrap();
        let (_, last_test) = splits.last().unwrap();
        assert_eq!(
            *last_test.last().unwrap(),
            9,
            "last fold test set should reach the final sample"
        );
    }

    #[test]
    fn test_time_series_split_minimum_splits() {
        let splits = time_series_split(4, 1).unwrap();
        assert_eq!(splits.len(), 1);
        let (train, test) = &splits[0];
        assert!(!train.is_empty());
        assert!(!test.is_empty());
    }

    #[test]
    fn test_time_series_split_error_n_splits_zero() {
        let err = time_series_split(10, 0).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    #[test]
    fn test_time_series_split_error_n_splits_ge_n_samples() {
        let err = time_series_split(5, 5).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    // ---------------------------------------------------------------
    // repeated_k_fold tests
    // ---------------------------------------------------------------

    #[test]
    fn test_repeated_k_fold_correct_total_folds() {
        let folds = repeated_k_fold(10, 5, 3, 42).unwrap();
        assert_eq!(folds.len(), 15); // 5 * 3
    }

    #[test]
    fn test_repeated_k_fold_each_fold_disjoint_and_covers_all() {
        let folds = repeated_k_fold(8, 4, 2, 42).unwrap();
        for (train, test) in &folds {
            for t in test {
                assert!(!train.contains(t), "test index {} found in train set", t);
            }
            let mut combined: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
            combined.sort();
            assert_eq!(combined, (0..8).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_repeated_k_fold_each_repeat_covers_all_samples() {
        let folds = repeated_k_fold(10, 5, 3, 42).unwrap();
        // Each group of 5 folds (one repeat) should have all 10 samples in their test sets.
        for repeat in 0..3 {
            let start = repeat * 5;
            let mut all_test: Vec<usize> = folds[start..start + 5]
                .iter()
                .flat_map(|(_, t)| t.clone())
                .collect();
            all_test.sort();
            assert_eq!(all_test, (0..10).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_repeated_k_fold_different_repeats_differ() {
        let folds = repeated_k_fold(10, 5, 2, 42).unwrap();
        // First fold of repeat 0 vs first fold of repeat 1 should differ (different shuffles).
        let test_r0 = &folds[0].1;
        let test_r1 = &folds[5].1;
        // They can theoretically be the same but with 10 samples it's astronomically unlikely.
        assert_ne!(test_r0, test_r1);
    }

    #[test]
    fn test_repeated_k_fold_deterministic() {
        let f1 = repeated_k_fold(10, 5, 2, 42).unwrap();
        let f2 = repeated_k_fold(10, 5, 2, 42).unwrap();
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_repeated_k_fold_error_k_less_than_2() {
        let err = repeated_k_fold(10, 1, 3, 42).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    #[test]
    fn test_repeated_k_fold_error_k_greater_than_n() {
        let err = repeated_k_fold(3, 5, 2, 42).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    #[test]
    fn test_repeated_k_fold_error_n_repeats_zero() {
        let err = repeated_k_fold(10, 5, 0, 42).unwrap_err();
        assert!(matches!(err, RustMlError::InvalidParameter(_)));
    }

    // --- RandomizedSearchCV ---

    #[test]
    fn test_randomized_search_cv_basic() {
        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
            [13.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        // Sampler that always returns the same trivial model
        let sampler = |_seed: u64| {
            move |_x_train: &Array2<f64>, y_train: &Array1<f64>, x_test: &Array2<f64>| {
                // Predict majority class
                let ones: usize = y_train.iter().filter(|&&v| v == 1.0).count();
                let majority = if ones > y_train.len() / 2 { 1.0 } else { 0.0 };
                Ok(Array1::from_elem(x_test.nrows(), majority))
            }
        };

        let result = randomized_search_cv(&x, &y, 2, 42, 5, sampler, |y_true, y_pred| {
            let correct: usize = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(&a, &b)| a == b)
                .count();
            Ok(correct as f64 / y_true.len() as f64)
        })
        .unwrap();

        assert_eq!(result.cv_scores.len(), 5);
        assert_eq!(result.mean_scores.len(), 5);
        assert!(result.best_params_index < 5);
    }

    #[test]
    fn test_randomized_search_cv_n_iter_zero() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let result = randomized_search_cv(
            &x,
            &y,
            2,
            0,
            0,
            |_seed: u64| {
                move |_: &Array2<f64>, _: &Array1<f64>, x_t: &Array2<f64>| {
                    Ok(Array1::zeros(x_t.nrows()))
                }
            },
            |_, _| Ok(0.5),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_randomized_search_cv_selects_best() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        // Even seeds predict all 0s, odd seeds predict based on threshold
        let sampler = |seed: u64| {
            move |_x_train: &Array2<f64>, _y_train: &Array1<f64>, x_test: &Array2<f64>| {
                if seed % 2 == 0 {
                    Ok(Array1::from_elem(x_test.nrows(), 0.0))
                } else {
                    Ok(x_test.column(0).mapv(|v| if v > 4.0 { 1.0 } else { 0.0 }))
                }
            }
        };

        let result = randomized_search_cv(&x, &y, 2, 42, 4, sampler, |y_true, y_pred| {
            let correct: usize = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(&a, &b)| a == b)
                .count();
            Ok(correct as f64 / y_true.len() as f64)
        })
        .unwrap();

        // The best score should be achievable
        assert!(result.best_score >= 0.0 && result.best_score <= 1.0);
    }

    // --- cross_val_predict ---

    #[test]
    fn test_cross_val_predict_length() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let preds = cross_val_predict(&x, &y, 2, 42, |_xt, _yt, x_te| {
            Ok(Array1::zeros(x_te.nrows()))
        })
        .unwrap();

        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_cross_val_predict_covers_all_samples() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let preds = cross_val_predict(&x, &y, 3, 42, |_xt, yt, x_te| {
            let majority = if yt.iter().filter(|&&v| v == 1.0).count() > yt.len() / 2 {
                1.0
            } else {
                0.0
            };
            Ok(Array1::from_elem(x_te.nrows(), majority))
        })
        .unwrap();

        // Every prediction should be 0 or 1
        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0);
        }
    }

    // --- repeated_stratified_k_fold ---

    #[test]
    fn test_repeated_stratified_k_fold_count() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let folds = repeated_stratified_k_fold(&x, &y, 3, 2, 42).unwrap();
        assert_eq!(folds.len(), 6); // 3 folds * 2 repeats
    }

    #[test]
    fn test_repeated_stratified_k_fold_error_zero_repeats() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        assert!(repeated_stratified_k_fold(&x, &y, 2, 0, 42).is_err());
    }

    // --- stratified_shuffle_split ---

    #[test]
    fn test_stratified_shuffle_split_count() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let splits = stratified_shuffle_split(&x, &y, 5, 0.25, 42).unwrap();
        assert_eq!(splits.len(), 5);

        for (train, test) in &splits {
            assert!(!test.is_empty());
            assert!(!train.is_empty());
            // No overlap
            let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
            for &t in train {
                assert!(!test_set.contains(&t));
            }
        }
    }

    #[test]
    fn test_stratified_shuffle_split_invalid_test_size() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        assert!(stratified_shuffle_split(&x, &y, 3, 0.0, 42).is_err());
        assert!(stratified_shuffle_split(&x, &y, 3, 1.0, 42).is_err());
    }

    // --- leave_p_out ---

    #[test]
    fn test_leave_p_out_basic() {
        let folds = leave_p_out(4, 2).unwrap();
        assert_eq!(folds.len(), 6); // C(4,2) = 6
        for (train, test) in &folds {
            assert_eq!(test.len(), 2);
            assert_eq!(train.len(), 2);
        }
    }

    #[test]
    fn test_leave_p_out_equals_loo_when_p1() {
        let folds = leave_p_out(5, 1).unwrap();
        let loo = leave_one_out(5);
        assert_eq!(folds.len(), loo.len());
    }

    #[test]
    fn test_leave_p_out_error_p_zero() {
        assert!(leave_p_out(5, 0).is_err());
    }

    #[test]
    fn test_leave_p_out_error_p_ge_n() {
        assert!(leave_p_out(3, 3).is_err());
    }

    // --- group_k_fold ---

    #[test]
    fn test_group_k_fold_basic() {
        let groups = array![0, 0, 1, 1, 2, 2, 3, 3];
        let folds = group_k_fold(&groups, 2).unwrap();
        assert_eq!(folds.len(), 2);

        // No group should appear in both train and test
        for (train, test) in &folds {
            let train_groups: std::collections::HashSet<usize> =
                train.iter().map(|&i| groups[i]).collect();
            let test_groups: std::collections::HashSet<usize> =
                test.iter().map(|&i| groups[i]).collect();
            assert!(train_groups.is_disjoint(&test_groups));
        }
    }

    #[test]
    fn test_group_k_fold_error_k_gt_groups() {
        let groups = array![0, 0, 1, 1];
        assert!(group_k_fold(&groups, 3).is_err());
    }

    // --- cross_validate ---

    #[test]
    fn test_cross_validate_multi_metric() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let accuracy = |y_true: &Array1<f64>, y_pred: &Array1<f64>| -> Result<f64> {
            let c: usize = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(&a, &b)| a == b)
                .count();
            Ok(c as f64 / y_true.len() as f64)
        };
        let always_half = |_: &Array1<f64>, _: &Array1<f64>| -> Result<f64> { Ok(0.5) };

        let result = cross_validate(
            &x,
            &y,
            2,
            42,
            |_xt, _yt, x_te| Ok(Array1::zeros(x_te.nrows())),
            &[&accuracy, &always_half],
        )
        .unwrap();

        assert_eq!(result.scores.len(), 2);
        assert_eq!(result.mean_scores.len(), 2);
        assert_eq!(result.fit_times.len(), 2);
        assert_eq!(result.score_times.len(), 2);
        // Second metric always returns 0.5
        assert!((result.mean_scores[1] - 0.5).abs() < 1e-10);
    }

    // --- learning_curve ---

    #[test]
    fn test_learning_curve_basic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let (sizes, train_scores, test_scores) = learning_curve(
            &x,
            &y,
            2,
            42,
            &[0.5, 1.0],
            |_xt, _yt, x_te| Ok(Array1::zeros(x_te.nrows())),
            |y_true, y_pred| {
                let c: usize = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&a, &b)| a == b)
                    .count();
                Ok(c as f64 / y_true.len() as f64)
            },
        )
        .unwrap();

        assert_eq!(sizes.len(), 2);
        assert_eq!(train_scores.len(), 2);
        assert_eq!(test_scores.len(), 2);
    }

    // --- validation_curve ---

    #[test]
    fn test_validation_curve_basic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let configs: Vec<
            Box<dyn Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>>,
        > = vec![
            Box::new(|_xt, _yt, x_te| Ok(Array1::zeros(x_te.nrows()))),
            Box::new(|_xt, _yt, x_te| Ok(Array1::from_elem(x_te.nrows(), 1.0))),
        ];

        let (train_scores, test_scores) =
            validation_curve(&x, &y, 2, 42, &configs, |y_true, y_pred| {
                let c: usize = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&a, &b)| a == b)
                    .count();
                Ok(c as f64 / y_true.len() as f64)
            })
            .unwrap();

        assert_eq!(train_scores.len(), 2);
        assert_eq!(test_scores.len(), 2);
    }
}
