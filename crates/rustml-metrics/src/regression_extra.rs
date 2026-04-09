use ndarray::Array1;
use rustml_core::{Float, Result, RustMlError};

/// Median Absolute Error: `median(|y_true - y_pred|)`.
///
/// More robust to outliers than Mean Absolute Error (MAE). Returns the median
/// of the absolute differences between true and predicted values.
pub fn median_absolute_error<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_pred)?;

    let mut abs_errors: Vec<F> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .collect();

    abs_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = abs_errors.len();
    let median = if n % 2 == 0 {
        let two = F::from_f64(2.0).unwrap();
        (abs_errors[n / 2 - 1] + abs_errors[n / 2]) / two
    } else {
        abs_errors[n / 2]
    };

    Ok(median)
}

/// Mean Squared Logarithmic Error: `mean((log(1 + y_true) - log(1 + y_pred))^2)`.
///
/// Penalizes under-prediction more than over-prediction relative to the magnitude.
/// Only valid for non-negative values; returns an error if any value is negative.
pub fn mean_squared_log_error<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
) -> Result<F> {
    check_lengths(y_true, y_pred)?;

    let zero = F::zero();
    let one = F::one();

    // Validate non-negative values
    for (i, (&t, &p)) in y_true.iter().zip(y_pred.iter()).enumerate() {
        if t < zero {
            return Err(RustMlError::InvalidParameter(format!(
                "MSLE is undefined for negative values; y_true[{}] = {}",
                i, t
            )));
        }
        if p < zero {
            return Err(RustMlError::InvalidParameter(format!(
                "MSLE is undefined for negative values; y_pred[{}] = {}",
                i, p
            )));
        }
    }

    let n = F::from_usize(y_true.len()).unwrap();
    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            let diff = (one + t).ln() - (one + p).ln();
            diff * diff
        })
        .fold(F::zero(), |acc, v| acc + v);

    Ok(sum / n)
}

fn check_lengths<F: Float>(a: &Array1<F>, b: &Array1<F>) -> Result<()> {
    if a.len() != b.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "y_true length {} != y_pred length {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(RustMlError::EmptyInput("input arrays are empty".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ---------------------------------------------------------------
    // Median Absolute Error tests
    // ---------------------------------------------------------------

    #[test]
    fn test_median_absolute_error_perfect() {
        let y = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(
            median_absolute_error(&y, &y).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_median_absolute_error_odd() {
        // Errors: |1-1.5|=0.5, |2-2.5|=0.5, |3-5|=2.0
        // Sorted: [0.5, 0.5, 2.0], median = 0.5
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.5, 2.5, 5.0];
        assert_abs_diff_eq!(
            median_absolute_error(&y_true, &y_pred).unwrap(),
            0.5,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_median_absolute_error_even() {
        // Errors: |1-2|=1, |2-3|=1, |3-5|=2, |4-7|=3
        // Sorted: [1, 1, 2, 3], median = (1+2)/2 = 1.5
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        let y_pred = array![2.0, 3.0, 5.0, 7.0];
        assert_abs_diff_eq!(
            median_absolute_error(&y_true, &y_pred).unwrap(),
            1.5,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_median_absolute_error_single() {
        let y_true = array![5.0];
        let y_pred = array![3.0];
        assert_abs_diff_eq!(
            median_absolute_error(&y_true, &y_pred).unwrap(),
            2.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_median_absolute_error_robust_to_outlier() {
        // The median is more robust than MAE: one large outlier doesn't dominate.
        // Errors: [0.1, 0.1, 0.1, 0.1, 100.0]
        // Sorted: [0.1, 0.1, 0.1, 0.1, 100.0], median = 0.1
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 2.1, 3.1, 4.1, 105.0];
        assert_abs_diff_eq!(
            median_absolute_error(&y_true, &y_pred).unwrap(),
            0.1,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_median_absolute_error_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];
        assert!(median_absolute_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_median_absolute_error_length_mismatch_error() {
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(median_absolute_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_median_absolute_error_f32() {
        let y_true: Array1<f32> = array![1.0f32, 2.0, 3.0];
        let y_pred: Array1<f32> = array![1.5f32, 2.5, 5.0];
        let result = median_absolute_error(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(result, 0.5f32, epsilon = 1e-6);
    }

    // ---------------------------------------------------------------
    // Mean Squared Logarithmic Error tests
    // ---------------------------------------------------------------

    #[test]
    fn test_msle_perfect() {
        let y = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(
            mean_squared_log_error(&y, &y).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_msle_known_value() {
        // y_true = [3, 5], y_pred = [2.5, 5]
        // (ln(4) - ln(3.5))^2 + (ln(6) - ln(6))^2 = (ln(4/3.5))^2 + 0
        // = (ln(8/7))^2
        let y_true = array![3.0, 5.0];
        let y_pred = array![2.5, 5.0];
        let expected = (4.0_f64.ln() - 3.5_f64.ln()).powi(2) / 2.0;
        assert_abs_diff_eq!(
            mean_squared_log_error(&y_true, &y_pred).unwrap(),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_msle_with_zeros() {
        // Values of zero are valid: log(1 + 0) = 0.
        let y_true = array![0.0, 0.0];
        let y_pred = array![0.0, 0.0];
        assert_abs_diff_eq!(
            mean_squared_log_error(&y_true, &y_pred).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_msle_another_known() {
        // sklearn reference: y_true = [3, 5, 2.5, 7], y_pred = [2.5, 5, 4, 8]
        // Compute each term manually:
        let y_true = array![3.0, 5.0, 2.5, 7.0];
        let y_pred = array![2.5, 5.0, 4.0, 8.0];
        let expected = (
            (4.0_f64.ln() - 3.5_f64.ln()).powi(2)
            + (6.0_f64.ln() - 6.0_f64.ln()).powi(2)
            + (3.5_f64.ln() - 5.0_f64.ln()).powi(2)
            + (8.0_f64.ln() - 9.0_f64.ln()).powi(2)
        ) / 4.0;
        assert_abs_diff_eq!(
            mean_squared_log_error(&y_true, &y_pred).unwrap(),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_msle_negative_y_true_error() {
        let y_true = array![1.0, -1.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(mean_squared_log_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_msle_negative_y_pred_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, -1.0, 3.0];
        assert!(mean_squared_log_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_msle_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];
        assert!(mean_squared_log_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_msle_length_mismatch_error() {
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(mean_squared_log_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_msle_f32() {
        let y_true: Array1<f32> = array![1.0f32, 2.0, 3.0];
        let y_pred: Array1<f32> = array![1.0f32, 2.0, 3.0];
        let result = mean_squared_log_error(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(result, 0.0f32, epsilon = 1e-6);
    }
}
