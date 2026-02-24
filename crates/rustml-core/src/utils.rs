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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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

    #[test]
    fn test_cross_val_score_basic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        // Trivial predictor: always predict 0.0
        let scores = cross_val_score::<f64>(
            &x,
            &y,
            3,
            |_x_train, _y_train, x_test| Ok(Array1::zeros(x_test.nrows())),
            |y_true, y_pred| {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(t, p)| (**t - **p).abs() < 1e-9)
                    .count();
                Ok(correct as f64 / y_true.len() as f64)
            },
        )
        .unwrap();

        assert_eq!(scores.len(), 3);
    }
}
