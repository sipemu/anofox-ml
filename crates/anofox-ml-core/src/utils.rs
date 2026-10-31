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
    x.select(ndarray::Axis(0), indices)
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

/// A single cross-validation split: `(train_indices, test_indices)`.
pub type CvSplit = (Vec<usize>, Vec<usize>);

/// A borrowed scoring function `(y_true, y_pred) -> score`, as accepted by
/// [`cross_validate`].
pub type ScorerFn<'a, F> = dyn Fn(&Array1<F>, &Array1<F>) -> Result<F> + 'a;

/// Return type of [`learning_curve`]: `(train_sizes_abs, train_scores, test_scores)`.
pub type LearningCurveResult<F> = (Vec<usize>, Vec<Vec<F>>, Vec<Vec<F>>);

/// Return type of [`validation_curve`]: `(train_scores, test_scores)`.
pub type ValidationCurveResult<F> = (Vec<Vec<F>>, Vec<Vec<F>>);

/// Canonical hash key for a label value: `-0.0`/`0.0` collapse, all NaNs collapse.
fn label_key<F: Float>(v: F) -> u64 {
    let f = v.to_f64().unwrap_or(f64::NAN);
    if f.is_nan() {
        f64::NAN.to_bits()
    } else if f == 0.0 {
        0.0f64.to_bits()
    } else {
        f.to_bits()
    }
}

/// Group sample indices by class label.
///
/// Classes are ordered by first appearance in `y` (like sklearn's
/// `StratifiedKFold` encoding) and indices within a class are ascending, so
/// the result is fully deterministic (no hash-map iteration order leaks out).
pub(crate) fn group_by_class<F: Float>(y: &Array1<F>) -> Vec<Vec<usize>> {
    let mut position: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, &v) in y.iter().enumerate() {
        let g = *position.entry(label_key(v)).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }
    groups
}

/// Heuristic: does `y` look like a vector of class labels (rather than a
/// continuous regression target)?
///
/// Returns `true` iff every value is finite and integral **and** the number of
/// distinct values is at most 2 or at most `n / 2` (i.e. on average every class
/// has at least two members). Continuous targets (any non-integral value) and
/// integer-valued targets that are almost all unique (ids, counts with a huge
/// range, ...) are treated as regression targets.
///
/// This is the rule used by [`default_cv_splits`] (and therefore by
/// [`grid_search_cv`], [`randomized_search_cv`], [`cross_val_predict`],
/// [`cross_validate`], [`learning_curve`] and [`validation_curve`]) to choose
/// between stratified and plain shuffled K-fold.
pub fn looks_like_class_labels<F: Float>(y: &Array1<F>) -> bool {
    let n = y.len();
    if n == 0 {
        return false;
    }
    for &v in y.iter() {
        let f = match v.to_f64() {
            Some(f) => f,
            None => return false,
        };
        if !f.is_finite() || f.fract() != 0.0 {
            return false;
        }
    }
    let n_unique = group_by_class(y).len();
    n_unique <= 2 || 2 * n_unique <= n
}

fn check_xy<F: Float>(x: &Array2<F>, y: &Array1<F>) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        )));
    }
    Ok(())
}

fn check_k(k: usize, n: usize) -> Result<()> {
    if k < 2 {
        return Err(RustMlError::InvalidParameter(
            "k must be >= 2 for cross-validation".into(),
        ));
    }
    if k > n {
        return Err(RustMlError::InvalidParameter(format!(
            "k ({}) > number of samples ({})",
            k, n
        )));
    }
    Ok(())
}

/// Turn a per-sample test-fold assignment into `(train, test)` pairs with
/// ascending indices.
fn splits_from_assignment(test_fold: &[usize], k: usize) -> Vec<CvSplit> {
    (0..k)
        .map(|fold| {
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (i, &f) in test_fold.iter().enumerate() {
                if f == fold {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
            (train, test)
        })
        .collect()
}

/// Check user-supplied splits: at least one split, every train/test set
/// non-empty and every index `< n`. If `require_partition`, every sample must
/// appear in exactly one test set (needed by [`cross_val_predict_with_splits`]).
fn validate_splits(splits: &[CvSplit], n: usize, require_partition: bool) -> Result<()> {
    if splits.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "need at least one (train, test) split".into(),
        ));
    }
    let mut seen = vec![0usize; n];
    for (s, (train, test)) in splits.iter().enumerate() {
        if train.is_empty() || test.is_empty() {
            return Err(RustMlError::InvalidParameter(format!(
                "split {} has an empty train or test set",
                s
            )));
        }
        if let Some(&bad) = train.iter().chain(test.iter()).find(|&&i| i >= n) {
            return Err(RustMlError::InvalidParameter(format!(
                "split {} contains index {} >= n_samples ({})",
                s, bad, n
            )));
        }
        for &i in test {
            seen[i] += 1;
        }
    }
    if require_partition && seen.iter().any(|&c| c != 1) {
        return Err(RustMlError::InvalidParameter(
            "test sets must partition the samples (each sample in exactly one test set)".into(),
        ));
    }
    Ok(())
}

/// Generate stratified k-fold cross-validation splits.
///
/// Returns a `Vec` of `(train_indices, test_indices)` tuples, one per fold,
/// with indices in ascending order. Each fold has approximately the same
/// proportion of each class as the full dataset.
///
/// The allocation follows sklearn's `StratifiedKFold(shuffle=True)`: the
/// label vector is sorted by class and dealt round-robin over the `k` folds,
/// which fixes how many samples of each class go to each fold; the samples of
/// every class are then randomly (via `seed`) assigned to their fold slots.
/// Consequently every fold receives `n / k` or `n / k + 1` samples even when
/// some (or all) classes have fewer than `k` members — e.g. a continuous
/// target with all-unique values yields `k` non-empty folds.
pub fn stratified_k_fold<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
) -> Result<Vec<CvSplit>> {
    check_xy(x, y)?;
    if x.nrows() == 0 {
        return Err(RustMlError::EmptyInput("input is empty".into()));
    }
    let n = x.nrows();
    check_k(k, n)?;

    let classes = group_by_class(y);

    // allocation[fold][class]: number of samples of `class` in `fold`.
    let mut allocation = vec![vec![0usize; classes.len()]; k];
    let mut pos = 0usize;
    for (c, members) in classes.iter().enumerate() {
        for _ in 0..members.len() {
            allocation[pos % k][c] += 1;
            pos += 1;
        }
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut test_fold = vec![0usize; n];
    for (c, members) in classes.iter().enumerate() {
        let mut fold_ids: Vec<usize> = (0..k)
            .flat_map(|f| std::iter::repeat_n(f, allocation[f][c]))
            .collect();
        fold_ids.shuffle(&mut rng);
        for (&i, &f) in members.iter().zip(fold_ids.iter()) {
            test_fold[i] = f;
        }
    }

    Ok(splits_from_assignment(&test_fold, k))
}

/// Shuffled (non-stratified) K-fold splitting, like sklearn's
/// `KFold(shuffle=True, random_state=seed)`.
///
/// Samples are randomly permuted with `seed` and cut into `k` folds of size
/// `n / k` or `n / k + 1`. Returned index lists are ascending.
///
/// # Errors
///
/// Returns `InvalidParameter` if `k < 2` or `k > n_samples`.
pub fn shuffled_k_fold(n_samples: usize, k: usize, seed: u64) -> Result<Vec<CvSplit>> {
    check_k(k, n_samples)?;
    let mut perm: Vec<usize> = (0..n_samples).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    perm.shuffle(&mut rng);

    let fold_size = n_samples / k;
    let remainder = n_samples % k;
    let mut test_fold = vec![0usize; n_samples];
    let mut start = 0;
    for fold in 0..k {
        let size = fold_size + usize::from(fold < remainder);
        for &i in &perm[start..start + size] {
            test_fold[i] = fold;
        }
        start += size;
    }
    Ok(splits_from_assignment(&test_fold, k))
}

/// The default CV splitter used by the high-level model-selection helpers.
///
/// Uses [`stratified_k_fold`] when `y` [`looks_like_class_labels`] and
/// [`shuffled_k_fold`] (seeded) otherwise, mirroring sklearn's choice of
/// `StratifiedKFold` for classifiers and `KFold` for regressors.
pub fn default_cv_splits<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
) -> Result<Vec<CvSplit>> {
    if looks_like_class_labels(y) {
        stratified_k_fold(x, y, k, seed)
    } else {
        check_xy(x, y)?;
        if x.nrows() == 0 {
            return Err(RustMlError::EmptyInput("input is empty".into()));
        }
        shuffled_k_fold(x.nrows(), k, seed)
    }
}

/// Fit once on `(x_train, y_train)` and predict both the training and the test
/// rows, by passing `vstack(x_train, x_test)` as the evaluation matrix.
fn fit_predict_train_and_test<F: Float>(
    x_train: &Array2<F>,
    y_train: &Array1<F>,
    x_test: &Array2<F>,
    fit_predict: &impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
) -> Result<(Array1<F>, Array1<F>)> {
    let n_train = x_train.nrows();
    let n_test = x_test.nrows();
    let x_eval = ndarray::concatenate(ndarray::Axis(0), &[x_train.view(), x_test.view()])
        .map_err(|e| RustMlError::ShapeMismatch(e.to_string()))?;
    let pred = fit_predict(x_train, y_train, &x_eval)?;
    if pred.len() != n_train + n_test {
        return Err(RustMlError::ShapeMismatch(format!(
            "fit_predict returned {} predictions for {} rows",
            pred.len(),
            n_train + n_test
        )));
    }
    Ok((
        pred.slice(ndarray::s![..n_train]).to_owned(),
        pred.slice(ndarray::s![n_train..]).to_owned(),
    ))
}

/// Cross-validation score over user-supplied `(train, test)` splits.
///
/// Returns one score per split. Use this to plug in any splitter
/// ([`group_k_fold`](crate::splitters::group_k_fold), [`time_series_split`], [`shuffled_k_fold`], ...).
pub fn cross_val_score_with_splits<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<Vec<F>> {
    check_xy(x, y)?;
    validate_splits(splits, x.nrows(), false)?;
    splits
        .iter()
        .map(|(train, test)| evaluate_fold(x, y, train, test, &fit_predict, &scorer))
        .collect()
}

/// Parallel [`cross_val_score_with_splits`]: folds are evaluated with rayon
/// (the closures must be `Sync`). Results are in split order and identical
/// to the sequential version.
pub fn cross_val_score_with_splits_par<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>> + Sync,
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F> + Sync,
) -> Result<Vec<F>> {
    use rayon::prelude::*;
    check_xy(x, y)?;
    validate_splits(splits, x.nrows(), false)?;
    splits
        .par_iter()
        .map(|(train, test)| evaluate_fold(x, y, train, test, &fit_predict, &scorer))
        .collect()
}

/// Parallel [`grid_search_cv_with_splits`]: every (configuration, fold)
/// pair is evaluated with rayon (the closures must be `Sync`). Results are
/// identical to the sequential version.
pub fn grid_search_cv_with_splits_par<F: Float, P>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
    param_configs: &[P],
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F> + Sync,
) -> Result<GridSearchResult<F>>
where
    P: Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>> + Sync,
{
    use rayon::prelude::*;
    if param_configs.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "param_configs must not be empty".into(),
        ));
    }
    check_xy(x, y)?;
    validate_splits(splits, x.nrows(), false)?;
    let k = splits.len();
    let flat: Vec<F> = (0..param_configs.len() * k)
        .into_par_iter()
        .map(|t| {
            let (train, test) = &splits[t % k];
            evaluate_fold(x, y, train, test, &param_configs[t / k], &scorer)
        })
        .collect::<Result<Vec<F>>>()?;
    let cv_scores: Vec<Vec<F>> = flat.chunks(k).map(|c| c.to_vec()).collect();
    let mean_scores: Vec<F> = cv_scores.iter().map(|s| mean_of(s)).collect();
    let (best_params_index, best_score) = argmax(&mean_scores);
    Ok(GridSearchResult {
        best_score,
        best_params_index,
        cv_scores,
        mean_scores,
    })
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

/// Grid search with cross-validation.
///
/// Evaluates each parameter configuration in `param_configs` using k-fold
/// cross-validation and returns the configuration that achieved the highest mean score.
/// Folds come from [`default_cv_splits`]: stratified for class-label targets,
/// seeded shuffled K-fold for continuous (regression) targets. Use
/// [`grid_search_cv_with_splits`] to supply your own splits.
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
/// the splitter and the individual `fit_predict` / `scorer` calls.
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
    let folds = default_cv_splits(x, y, k, seed)?;
    grid_search_cv_with_splits(x, y, &folds, param_configs, scorer)
}

/// Grid search over user-supplied `(train, test)` splits.
///
/// Same as [`grid_search_cv`] but the cross-validation splits are given
/// explicitly (e.g. from [`group_k_fold`](crate::splitters::group_k_fold), [`time_series_split`] or
/// [`shuffled_k_fold`]).
pub fn grid_search_cv_with_splits<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
    param_configs: &[impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>],
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<GridSearchResult<F>> {
    if param_configs.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "param_configs must not be empty".into(),
        ));
    }
    check_xy(x, y)?;
    validate_splits(splits, x.nrows(), false)?;

    let mut cv_scores: Vec<Vec<F>> = Vec::with_capacity(param_configs.len());
    for fit_predict in param_configs {
        let mut fold_scores = Vec::with_capacity(splits.len());
        for (train_indices, test_indices) in splits {
            let score = evaluate_fold(x, y, train_indices, test_indices, fit_predict, &scorer)?;
            fold_scores.push(score);
        }
        cv_scores.push(fold_scores);
    }

    let mean_scores: Vec<F> = cv_scores.iter().map(|s| mean_of(s)).collect();
    let (best_params_index, best_score) = argmax(&mean_scores);

    Ok(GridSearchResult {
        best_score,
        best_params_index,
        cv_scores,
        mean_scores,
    })
}

fn mean_of<F: Float>(scores: &[F]) -> F {
    let sum: F = scores.iter().copied().sum();
    sum / F::from_usize(scores.len()).unwrap()
}

/// Index and value of the maximum (first one wins on ties; NaNs never win
/// unless everything is NaN). `values` must be non-empty.
fn argmax<F: Float>(values: &[F]) -> (usize, F) {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > values[best] || (values[best].is_nan() && !v.is_nan()) {
            best = i;
        }
    }
    (best, values[best])
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

/// Randomized search with cross-validation.
///
/// Randomly samples `n_iter` parameter configurations from `param_sampler` and
/// evaluates each using k-fold cross-validation (folds from
/// [`default_cv_splits`]: stratified for class labels, shuffled K-fold for
/// continuous targets). Returns the configuration that achieved the highest
/// mean score.
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
/// the splitter and the individual evaluations.
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
    let folds = default_cv_splits(x, y, k, seed)?;
    randomized_search_cv_with_splits(x, y, &folds, seed, n_iter, param_sampler, scorer)
}

/// Randomized search over user-supplied `(train, test)` splits.
///
/// `seed` is only used to derive the per-configuration seeds passed to
/// `param_sampler` (identically to [`randomized_search_cv`]).
pub fn randomized_search_cv_with_splits<F, S, P>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
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
    check_xy(x, y)?;
    validate_splits(splits, x.nrows(), false)?;

    let mut cv_scores: Vec<Vec<F>> = Vec::with_capacity(n_iter);
    let mut mean_scores: Vec<F> = Vec::with_capacity(n_iter);

    for i in 0..n_iter {
        let config_seed = seed
            .wrapping_add(i as u64)
            .wrapping_mul(6364136223846793005);
        let fit_predict = param_sampler(config_seed);

        let mut fold_scores = Vec::with_capacity(splits.len());
        for (train_indices, test_indices) in splits {
            let score = evaluate_fold(x, y, train_indices, test_indices, &fit_predict, &scorer)?;
            fold_scores.push(score);
        }

        mean_scores.push(mean_of(&fold_scores));
        cv_scores.push(fold_scores);
    }

    let (best_params_index, best_score) = argmax(&mean_scores);

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
/// the model trained without it. Folds come from [`default_cv_splits`]
/// (stratified for class labels, shuffled K-fold for continuous targets).
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
    let folds = default_cv_splits(x, y, k, seed)?;
    cross_val_predict_with_splits(x, y, &folds, fit_predict)
}

/// Cross-validated predictions over user-supplied `(train, test)` splits.
///
/// The test sets must partition the samples (every sample in exactly one
/// test set), otherwise `InvalidParameter` is returned.
pub fn cross_val_predict_with_splits<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
) -> Result<Array1<F>> {
    check_xy(x, y)?;
    validate_splits(splits, x.nrows(), true)?;
    let n = x.nrows();
    let mut predictions = Array1::zeros(n);

    for (train_indices, test_indices) in splits {
        let x_train = select_rows(x, train_indices);
        let y_train = select_elements(y, train_indices);
        let x_test = select_rows(x, test_indices);

        let y_pred = fit_predict(&x_train, &y_train, &x_test)?;
        if y_pred.len() != test_indices.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "fit_predict returned {} predictions for {} test rows",
                y_pred.len(),
                test_indices.len()
            )));
        }

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
/// Like [`shuffle_split`] but preserves the class distribution in each split,
/// following sklearn's `StratifiedShuffleSplit`:
///
/// - the test set has exactly `n_test = ceil(test_size * n)` samples (clamped
///   to `1..=n-1`) and the train set the remaining `n - n_test`;
/// - per-class test counts are allocated by the "approximate mode" of the
///   multivariate hypergeometric distribution (floor of the proportional
///   share, remaining slots going to the classes with the largest
///   fractional remainders, ties broken by the seeded RNG);
/// - classes are visited in first-appearance order and a single seeded RNG
///   drives all draws, so the result is fully deterministic for a given `seed`.
///
/// Returned index lists are ascending.
pub fn stratified_shuffle_split<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    n_splits: usize,
    test_size: f64,
    seed: u64,
) -> Result<Vec<CvSplit>> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(RustMlError::InvalidParameter(
            "test_size must be in (0, 1)".into(),
        ));
    }
    if n_splits == 0 {
        return Err(RustMlError::InvalidParameter("n_splits must be > 0".into()));
    }
    check_xy(x, y)?;
    let n = x.nrows();
    if n < 2 {
        return Err(RustMlError::InvalidParameter(
            "need at least 2 samples".into(),
        ));
    }

    let n_test = ((n as f64 * test_size).ceil() as usize).clamp(1, n - 1);
    let classes = group_by_class(y);
    let counts: Vec<usize> = classes.iter().map(Vec::len).collect();

    let mut result = Vec::with_capacity(n_splits);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for _ in 0..n_splits {
        let test_counts = approximate_mode(&counts, n_test, &mut rng);
        let mut test_indices = Vec::with_capacity(n_test);
        let mut train_indices = Vec::with_capacity(n - n_test);

        for (members, &t) in classes.iter().zip(test_counts.iter()) {
            let mut perm = members.clone();
            perm.shuffle(&mut rng);
            test_indices.extend_from_slice(&perm[..t]);
            train_indices.extend_from_slice(&perm[t..]);
        }

        train_indices.sort_unstable();
        test_indices.sort_unstable();
        result.push((train_indices, test_indices));
    }

    Ok(result)
}

/// sklearn's `_approximate_mode`: split `n_draws` across classes with the
/// given `counts` as proportionally as possible, using exact integer
/// arithmetic. Returns per-class draws summing to `n_draws` (`<= counts[c]`).
fn approximate_mode(counts: &[usize], n_draws: usize, rng: &mut rand::rngs::StdRng) -> Vec<usize> {
    let total: usize = counts.iter().sum();
    let mut drawn: Vec<usize> = counts.iter().map(|&c| c * n_draws / total).collect();
    let remainder: Vec<usize> = counts.iter().map(|&c| c * n_draws % total).collect();
    let mut need = n_draws - drawn.iter().sum::<usize>();

    let mut values: Vec<usize> = remainder.iter().copied().filter(|&r| r > 0).collect();
    values.sort_unstable_by(|a, b| b.cmp(a));
    values.dedup();
    for v in values {
        if need == 0 {
            break;
        }
        let mut inds: Vec<usize> = (0..counts.len()).filter(|&c| remainder[c] == v).collect();
        inds.shuffle(rng);
        let take = inds.len().min(need);
        for &c in &inds[..take] {
            drawn[c] += 1;
        }
        need -= take;
    }
    drawn
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
/// and supports multiple scoring functions. Folds come from
/// [`default_cv_splits`] (stratified for class labels, shuffled K-fold for
/// continuous targets).
pub fn cross_validate<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    scorers: &[&ScorerFn<'_, F>],
) -> Result<CrossValidateResult<F>> {
    if scorers.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "need at least one scorer".into(),
        ));
    }
    let folds = default_cv_splits(x, y, k, seed)?;
    cross_validate_with_splits(x, y, &folds, fit_predict, scorers)
}

/// [`cross_validate`] over user-supplied `(train, test)` splits.
pub fn cross_validate_with_splits<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    scorers: &[&ScorerFn<'_, F>],
) -> Result<CrossValidateResult<F>> {
    if scorers.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "need at least one scorer".into(),
        ));
    }
    check_xy(x, y)?;
    validate_splits(splits, x.nrows(), false)?;
    let n_metrics = scorers.len();
    let n_splits = splits.len();

    let mut all_scores: Vec<Vec<F>> = vec![Vec::with_capacity(n_splits); n_metrics];
    let mut fit_times = Vec::with_capacity(n_splits);
    let mut score_times = Vec::with_capacity(n_splits);

    for (train_indices, test_indices) in splits {
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

    let mean_scores: Vec<F> = all_scores.iter().map(|s| mean_of(s)).collect();

    Ok(CrossValidateResult {
        scores: all_scores,
        mean_scores,
        fit_times,
        score_times,
    })
}

/// Scorer that also receives the test fold's sample weights:
/// `(y_true, y_pred, sample_weight) -> score`.
pub type WeightedScorerFn<'a, F> =
    dyn Fn(&Array1<F>, &Array1<F>, Option<&Array1<F>>) -> Result<F> + 'a;

/// [`cross_validate_with_splits`] with per-sample weights, as sklearn's
/// `cross_validate(..., params={"sample_weight": w})` with weighted
/// scoring: `fit_predict(X_train, y_train, w_train, X_test)` receives the
/// training fold's weights and every scorer the test fold's weights.
/// `sample_weight = None` passes `None` to both. Combine with the group
/// splitters (e.g. [`group_k_fold`](crate::group_k_fold)) for grouped CV.
pub fn cross_validate_weighted_with_splits<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    sample_weight: Option<&Array1<F>>,
    splits: &[CvSplit],
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, Option<&Array1<F>>, &Array2<F>) -> Result<Array1<F>>,
    scorers: &[&WeightedScorerFn<'_, F>],
) -> Result<CrossValidateResult<F>> {
    if scorers.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "need at least one scorer".into(),
        ));
    }
    check_xy(x, y)?;
    if let Some(w) = sample_weight {
        crate::class_weight::check_sample_weight(w, y.len())?;
    }
    validate_splits(splits, x.nrows(), false)?;
    let n_metrics = scorers.len();
    let n_splits = splits.len();

    let mut all_scores: Vec<Vec<F>> = vec![Vec::with_capacity(n_splits); n_metrics];
    let mut fit_times = Vec::with_capacity(n_splits);
    let mut score_times = Vec::with_capacity(n_splits);

    for (train_indices, test_indices) in splits {
        let x_train = select_rows(x, train_indices);
        let y_train = select_elements(y, train_indices);
        let x_test = select_rows(x, test_indices);
        let y_test = select_elements(y, test_indices);
        let w_train = sample_weight.map(|w| select_elements(w, train_indices));
        let w_test = sample_weight.map(|w| select_elements(w, test_indices));

        let fit_start = std::time::Instant::now();
        let y_pred = fit_predict(&x_train, &y_train, w_train.as_ref(), &x_test)?;
        fit_times.push(fit_start.elapsed().as_secs_f64());

        let score_start = std::time::Instant::now();
        for (m, scorer) in scorers.iter().enumerate() {
            all_scores[m].push(scorer(&y_test, &y_pred, w_test.as_ref())?);
        }
        score_times.push(score_start.elapsed().as_secs_f64());
    }

    let mean_scores: Vec<F> = all_scores.iter().map(|s| mean_of(s)).collect();
    Ok(CrossValidateResult {
        scores: all_scores,
        mean_scores,
        fit_times,
        score_times,
    })
}

/// Single-metric [`cross_validate_weighted_with_splits`]: per-fold scores.
pub fn cross_val_score_weighted_with_splits<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    sample_weight: Option<&Array1<F>>,
    splits: &[CvSplit],
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, Option<&Array1<F>>, &Array2<F>) -> Result<Array1<F>>,
    scorer: impl Fn(&Array1<F>, &Array1<F>, Option<&Array1<F>>) -> Result<F>,
) -> Result<Vec<F>> {
    let scorer: &WeightedScorerFn<'_, F> = &scorer;
    let res =
        cross_validate_weighted_with_splits(x, y, sample_weight, splits, fit_predict, &[scorer])?;
    Ok(res.scores.into_iter().next().unwrap_or_default())
}

/// Generate a learning curve: scores as a function of training set size.
///
/// For every CV fold (from [`default_cv_splits`]: stratified for class
/// labels, shuffled K-fold for continuous targets) the fold's training indices
/// are shuffled with `seed` and the model is trained on increasing prefixes of
/// that permutation, then evaluated both on the training prefix and on the
/// held-out fold.
///
/// `train_sizes` is a slice of fractions in (0, 1], e.g. `[0.1, 0.3, 0.5, 0.7, 1.0]`.
///
/// The model is fitted **once** per (fold, size): `fit_predict` receives
/// `vstack(X_train, X_test)` as its evaluation matrix and must return one
/// prediction per row (row-wise predictions, as every estimator's `predict`
/// does).
///
/// Returns `(train_sizes_abs, train_scores, test_scores)` where the outer
/// Vecs have length `train_sizes.len()` and the inner ones one score per fold.
/// `train_sizes_abs` is computed from the first fold's training-set size.
pub fn learning_curve<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
    train_sizes: &[f64],
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<LearningCurveResult<F>> {
    check_train_sizes(train_sizes)?;
    let folds = default_cv_splits(x, y, k, seed)?;
    learning_curve_with_splits(x, y, &folds, seed, train_sizes, fit_predict, scorer)
}

fn check_train_sizes(train_sizes: &[f64]) -> Result<()> {
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
    Ok(())
}

/// [`learning_curve`] over user-supplied `(train, test)` splits.
///
/// `seed` drives the shuffling of each split's training indices before the
/// prefixes are taken.
pub fn learning_curve_with_splits<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
    seed: u64,
    train_sizes: &[f64],
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<LearningCurveResult<F>> {
    check_train_sizes(train_sizes)?;
    check_xy(x, y)?;
    validate_splits(splits, x.nrows(), false)?;

    let n_splits = splits.len();
    let mut train_sizes_abs = Vec::with_capacity(train_sizes.len());
    let mut all_train_scores: Vec<Vec<F>> = vec![Vec::with_capacity(n_splits); train_sizes.len()];
    let mut all_test_scores: Vec<Vec<F>> = vec![Vec::with_capacity(n_splits); train_sizes.len()];
    // Decorrelate the prefix shuffle from the splitter's own use of `seed`.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);

    for (fold, (train_indices, test_indices)) in splits.iter().enumerate() {
        let x_test = select_rows(x, test_indices);
        let y_test = select_elements(y, test_indices);
        let mut perm = train_indices.clone();
        perm.shuffle(&mut rng);

        for (si, &frac) in train_sizes.iter().enumerate() {
            let n_train = ((perm.len() as f64 * frac).ceil() as usize)
                .max(1)
                .min(perm.len());
            if fold == 0 {
                train_sizes_abs.push(n_train);
            }
            let sub_train = &perm[..n_train];
            let x_train = select_rows(x, sub_train);
            let y_train = select_elements(y, sub_train);

            let (y_pred_train, y_pred_test) =
                fit_predict_train_and_test(&x_train, &y_train, &x_test, &fit_predict)?;
            all_train_scores[si].push(scorer(&y_train, &y_pred_train)?);
            all_test_scores[si].push(scorer(&y_test, &y_pred_test)?);
        }
    }

    Ok((train_sizes_abs, all_train_scores, all_test_scores))
}

/// Generate a validation curve: scores as a function of a hyperparameter.
///
/// Evaluates performance across different hyperparameter values using
/// k-fold cross-validation (folds from [`default_cv_splits`]: stratified for
/// class labels, shuffled K-fold for continuous targets).
///
/// `param_configs` is a slice of `fit_predict` closures, one per hyperparameter value.
/// Each closure is fitted **once** per fold and receives
/// `vstack(X_train, X_test)` as its evaluation matrix (it must return one
/// prediction per row).
///
/// Returns `(train_scores, test_scores)` where each inner Vec has k fold scores.
pub fn validation_curve<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    k: usize,
    seed: u64,
    param_configs: &[impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>],
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<ValidationCurveResult<F>> {
    if param_configs.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "param_configs must not be empty".into(),
        ));
    }
    let folds = default_cv_splits(x, y, k, seed)?;
    validation_curve_with_splits(x, y, &folds, param_configs, scorer)
}

/// [`validation_curve`] over user-supplied `(train, test)` splits.
pub fn validation_curve_with_splits<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
    param_configs: &[impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>],
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<ValidationCurveResult<F>> {
    if param_configs.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "param_configs must not be empty".into(),
        ));
    }
    check_xy(x, y)?;
    validate_splits(splits, x.nrows(), false)?;
    let n_configs = param_configs.len();
    let n_splits = splits.len();
    let mut train_scores: Vec<Vec<F>> = vec![Vec::with_capacity(n_splits); n_configs];
    let mut test_scores: Vec<Vec<F>> = vec![Vec::with_capacity(n_splits); n_configs];

    for (train_indices, test_indices) in splits {
        let x_train = select_rows(x, train_indices);
        let y_train = select_elements(y, train_indices);
        let x_test = select_rows(x, test_indices);
        let y_test = select_elements(y, test_indices);

        for (ci, fit_predict) in param_configs.iter().enumerate() {
            let (y_pred_train, y_pred_test) =
                fit_predict_train_and_test(&x_train, &y_train, &x_test, fit_predict)?;
            train_scores[ci].push(scorer(&y_train, &y_pred_train)?);
            test_scores[ci].push(scorer(&y_test, &y_pred_test)?);
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
                if seed.is_multiple_of(2) {
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
        let folds = crate::splitters::group_k_fold(&groups, 2).unwrap();
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
        assert!(crate::splitters::group_k_fold(&groups, 3).is_err());
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

    // ---------------------------------------------------------------
    // Regression tests for issue #33
    // ---------------------------------------------------------------

    fn continuous_xy(n: usize) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(n, |i| 0.37 * i as f64 + 0.1);
        (x, y)
    }

    fn assert_valid_partition(folds: &[CvSplit], n: usize) {
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.clone()).collect();
        all_test.sort_unstable();
        assert_eq!(all_test, (0..n).collect::<Vec<_>>());
        for (train, test) in folds {
            assert!(!train.is_empty(), "empty train fold");
            assert!(!test.is_empty(), "empty test fold");
            let mut combined: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
            combined.sort_unstable();
            assert_eq!(combined, (0..n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_stratified_k_fold_all_unique_continuous_y() {
        let (x, y) = continuous_xy(10);
        let folds = stratified_k_fold(&x, &y, 5, 0).unwrap();
        assert_eq!(folds.len(), 5);
        assert_valid_partition(&folds, 10);
        for (train, test) in &folds {
            assert_eq!(test.len(), 2);
            assert_eq!(train.len(), 8);
        }
    }

    #[test]
    fn test_stratified_k_fold_class_smaller_than_k() {
        // class 1 has only 2 members, k = 5
        let x = Array2::from_shape_fn((10, 1), |(i, _)| i as f64);
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let folds = stratified_k_fold(&x, &y, 5, 3).unwrap();
        assert_valid_partition(&folds, 10);
        for (_, test) in &folds {
            assert_eq!(test.len(), 2);
        }
        // the two minority samples land in different folds
        let minority_folds = folds
            .iter()
            .filter(|(_, t)| t.contains(&8) || t.contains(&9))
            .count();
        assert_eq!(minority_folds, 2);
    }

    #[test]
    fn test_looks_like_class_labels() {
        assert!(looks_like_class_labels(&array![0.0, 1.0, 0.0, 1.0]));
        assert!(looks_like_class_labels(&array![0.0, 1.0]));
        assert!(looks_like_class_labels(&array![
            0.0, 1.0, 2.0, 2.0, 1.0, 0.0
        ]));
        assert!(!looks_like_class_labels(&array![0.5, 1.0, 0.0, 1.0]));
        assert!(!looks_like_class_labels(&continuous_xy(10).1));
        // integer-valued but (almost) all unique -> regression
        assert!(!looks_like_class_labels(&array![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        ]));
        assert!(!looks_like_class_labels(&array![0.0, f64::NAN]));
    }

    #[test]
    fn test_shuffled_k_fold_partition_and_determinism() {
        let folds = shuffled_k_fold(11, 3, 7).unwrap();
        assert_valid_partition(&folds, 11);
        let sizes: Vec<usize> = folds.iter().map(|(_, t)| t.len()).collect();
        assert_eq!(sizes, vec![4, 4, 3]);
        assert_eq!(folds, shuffled_k_fold(11, 3, 7).unwrap());
        // shuffled, i.e. not the contiguous k_fold blocks
        assert_ne!(folds, k_fold(11, 3).unwrap());
    }

    #[test]
    fn test_default_cv_splits_regression_target_uses_shuffled_kfold() {
        let (x, y) = continuous_xy(10);
        let folds = default_cv_splits(&x, &y, 5, 1).unwrap();
        assert_eq!(folds, shuffled_k_fold(10, 5, 1).unwrap());
        let y_cls = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let folds = default_cv_splits(&x, &y_cls, 5, 1).unwrap();
        assert_eq!(folds, stratified_k_fold(&x, &y_cls, 5, 1).unwrap());
    }

    /// Mean predictor that errors on an empty training set (as real estimators do).
    fn predict_mean(
        _x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_test: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        if y_train.is_empty() {
            return Err(RustMlError::EmptyInput("empty training set".into()));
        }
        Ok(Array1::from_elem(x_test.nrows(), y_train.mean().unwrap()))
    }

    fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        if y_true.is_empty() {
            return Err(RustMlError::EmptyInput("empty test set".into()));
        }
        Ok(-(y_true - y_pred).mapv(|d| d * d).mean().unwrap())
    }

    #[test]
    fn test_model_selection_helpers_work_on_regression_targets() {
        let (x, y) = continuous_xy(10);
        let configs: Vec<fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>> =
            vec![predict_mean];

        let gs = grid_search_cv(&x, &y, 5, 0, &configs, neg_mse).unwrap();
        assert_eq!(gs.cv_scores[0].len(), 5);

        let rs = randomized_search_cv(&x, &y, 5, 0, 2, |_s| predict_mean, neg_mse).unwrap();
        assert_eq!(rs.cv_scores.len(), 2);

        let preds = cross_val_predict(&x, &y, 5, 0, predict_mean).unwrap();
        assert_eq!(preds.len(), 10);

        let cv = cross_validate(&x, &y, 5, 0, predict_mean, &[&neg_mse]).unwrap();
        assert_eq!(cv.scores[0].len(), 5);

        let (sizes, tr, te) =
            learning_curve(&x, &y, 5, 0, &[0.25, 1.0], predict_mean, neg_mse).unwrap();
        assert_eq!(sizes, vec![2, 8]);
        assert_eq!(tr[0].len(), 5);
        assert_eq!(te[1].len(), 5);

        let (tr, te) = validation_curve(&x, &y, 5, 0, &configs, neg_mse).unwrap();
        assert_eq!(tr[0].len(), 5);
        assert_eq!(te[0].len(), 5);
    }

    #[test]
    fn test_with_splits_variants() {
        let (x, y) = continuous_xy(12);
        let splits = time_series_split(12, 3).unwrap();
        let configs: Vec<fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>> =
            vec![predict_mean, predict_zero];

        let s = cross_val_score_with_splits(&x, &y, &splits, predict_mean, neg_mse).unwrap();
        assert_eq!(s.len(), 3);
        let gs = grid_search_cv_with_splits(&x, &y, &splits, &configs, neg_mse).unwrap();
        assert_eq!(gs.best_params_index, 0);
        assert_eq!(gs.cv_scores[1].len(), 3);
        let rs =
            randomized_search_cv_with_splits(&x, &y, &splits, 0, 3, |_s| predict_mean, neg_mse)
                .unwrap();
        assert_eq!(rs.cv_scores.len(), 3);
        let cv = cross_validate_with_splits(&x, &y, &splits, predict_mean, &[&neg_mse]).unwrap();
        assert_eq!(cv.fit_times.len(), 3);
        let (_, tr, _) =
            learning_curve_with_splits(&x, &y, &splits, 0, &[0.5, 1.0], predict_mean, neg_mse)
                .unwrap();
        assert_eq!(tr[0].len(), 3);
        let (tr, _) = validation_curve_with_splits(&x, &y, &splits, &configs, neg_mse).unwrap();
        assert_eq!(tr.len(), 2);

        // time-series test sets do not partition the samples
        assert!(cross_val_predict_with_splits(&x, &y, &splits, predict_mean).is_err());
        let kf = k_fold(12, 4).unwrap();
        let p = cross_val_predict_with_splits(&x, &y, &kf, predict_mean).unwrap();
        assert_eq!(p.len(), 12);

        // invalid splits
        let bad: Vec<CvSplit> = vec![(vec![], vec![0, 1])];
        assert!(cross_val_score_with_splits(&x, &y, &bad, predict_mean, neg_mse).is_err());
        let bad: Vec<CvSplit> = vec![(vec![0], vec![99])];
        assert!(cross_val_score_with_splits(&x, &y, &bad, predict_mean, neg_mse).is_err());
        assert!(cross_val_score_with_splits(&x, &y, &[], predict_mean, neg_mse).is_err());
    }

    #[test]
    fn test_learning_curve_small_fraction_not_single_class() {
        // Sorted, class-grouped data: without shuffling, a small prefix of a
        // fold's (class-grouped) train indices is single-class.
        let n = 40;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(n, |i| if i < n / 2 { 0.0 } else { 1.0 });
        let seen_classes = std::cell::RefCell::new(Vec::new());
        let fit_predict = |_xt: &Array2<f64>, yt: &Array1<f64>, x_te: &Array2<f64>| {
            let ones = yt.iter().filter(|&&v| v == 1.0).count();
            seen_classes.borrow_mut().push((ones, yt.len() - ones));
            Ok(Array1::zeros(x_te.nrows()))
        };
        learning_curve(&x, &y, 4, 0, &[0.25], fit_predict, accuracy).unwrap();
        let seen = seen_classes.into_inner();
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter().all(|&(ones, zeros)| ones > 0 && zeros > 0),
            "single-class training prefixes: {seen:?}"
        );
    }

    #[test]
    fn test_learning_and_validation_curve_fit_once_per_point() {
        let x = Array2::from_shape_fn((8, 1), |(i, _)| i as f64);
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let calls = std::cell::Cell::new(0usize);
        let fit_predict = |_xt: &Array2<f64>, _yt: &Array1<f64>, x_te: &Array2<f64>| {
            calls.set(calls.get() + 1);
            Ok(Array1::zeros(x_te.nrows()))
        };
        learning_curve(&x, &y, 2, 0, &[0.5, 1.0], fit_predict, accuracy).unwrap();
        assert_eq!(calls.get(), 4); // 2 folds x 2 sizes

        calls.set(0);
        let configs = [fit_predict];
        validation_curve(&x, &y, 2, 0, &configs, accuracy).unwrap();
        assert_eq!(calls.get(), 2); // 2 folds x 1 config
    }

    #[test]
    fn test_stratified_shuffle_split_deterministic() {
        let n = 30;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(n, |i| (i % 3) as f64);
        let first = stratified_shuffle_split(&x, &y, 4, 0.3, 0).unwrap();
        for _ in 0..10 {
            assert_eq!(stratified_shuffle_split(&x, &y, 4, 0.3, 0).unwrap(), first);
        }
        // the four splits differ from each other
        assert_ne!(first[0].1, first[1].1);
    }

    #[test]
    fn test_stratified_shuffle_split_exact_test_size() {
        // classes of size 7 / 4 / 4; the old per-class `ceil(..).max(1)` gave
        // 2 + 1 + 1 = 4 test samples for test_size 0.1 (expected ceil(1.5) = 2).
        let x = Array2::from_shape_fn((15, 1), |(i, _)| i as f64);
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        for &(ts, expected) in &[(0.25, 4usize), (0.2, 3), (0.5, 8), (0.1, 2)] {
            let splits = stratified_shuffle_split(&x, &y, 5, ts, 11).unwrap();
            for (train, test) in &splits {
                assert_eq!(test.len(), expected, "test_size {ts}");
                assert_eq!(train.len(), 15 - expected);
                let mut all: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
                all.sort_unstable();
                assert_eq!(all, (0..15).collect::<Vec<_>>());
            }
        }
        // proportional allocation: 0.4 * 15 = 6 test samples; shares (2.8, 1.6, 1.6)
        // -> class 0 gets 3, one of classes 1/2 gets 2 and the other 1.
        let splits = stratified_shuffle_split(&x, &y, 3, 0.4, 5).unwrap();
        for (_, test) in &splits {
            let c0 = test.iter().filter(|&&i| y[i] == 0.0).count();
            let c1 = test.iter().filter(|&&i| y[i] == 1.0).count();
            let c2 = test.iter().filter(|&&i| y[i] == 2.0).count();
            assert_eq!(c0, 3);
            assert_eq!((c1.min(c2), c1.max(c2)), (1, 2));
        }
    }

    #[test]
    fn test_approximate_mode_sums_to_draws() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let counts = [7usize, 4, 4];
        for draws in 0..=15 {
            let d = approximate_mode(&counts, draws, &mut rng);
            assert_eq!(d.iter().sum::<usize>(), draws);
            for (a, b) in d.iter().zip(counts.iter()) {
                assert!(a <= b);
            }
        }
    }

    #[test]
    fn test_parallel_search_matches_sequential() {
        let x = Array2::from_shape_fn((30, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(30, |i| 2.0 * i as f64 + ((i * 7) % 3) as f64);
        let splits = shuffled_k_fold(30, 5, 3).unwrap();
        let configs: Vec<_> = (0..4)
            .map(|c| {
                move |_: &Array2<f64>, _: &Array1<f64>, xs: &Array2<f64>| -> Result<Array1<f64>> {
                    Ok(xs.column(0).mapv(|v| c as f64 * v))
                }
            })
            .collect();
        let neg_mse = |t: &Array1<f64>, p: &Array1<f64>| -> Result<f64> {
            Ok(-t.iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum::<f64>())
        };
        let seq = grid_search_cv_with_splits(&x, &y, &splits, &configs, neg_mse).unwrap();
        let par = grid_search_cv_with_splits_par(&x, &y, &splits, &configs, neg_mse).unwrap();
        assert_eq!(seq.cv_scores, par.cv_scores);
        assert_eq!(seq.best_params_index, par.best_params_index);
        let a = cross_val_score_with_splits(&x, &y, &splits, &configs[2], neg_mse).unwrap();
        let b = cross_val_score_with_splits_par(&x, &y, &splits, &configs[2], neg_mse).unwrap();
        assert_eq!(a, b);
    }
}
