use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::Array1;

/// Mean Absolute Percentage Error: `mean(|y_true - y_pred| / |y_true|)`.
///
/// Returns an error if any element of `y_true` is zero, since division by zero
/// would be undefined.
pub fn mean_absolute_percentage_error<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
) -> Result<F> {
    check_lengths(y_true, y_pred)?;

    for &v in y_true.iter() {
        if v.abs() < F::from_f64(1e-15).unwrap() {
            return Err(RustMlError::InvalidParameter(
                "MAPE is undefined when y_true contains zero values".into(),
            ));
        }
    }

    let n = F::from_usize(y_true.len()).unwrap();
    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| ((t - p).abs()) / t.abs())
        .fold(F::zero(), |acc, v| acc + v);

    Ok(sum / n)
}

/// Explained Variance Score: `1 - Var(y_true - y_pred) / Var(y_true)`.
///
/// The variance is computed as the population variance (dividing by N, not N-1).
/// If `Var(y_true)` is zero (constant input), returns zero.
pub fn explained_variance_score<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_pred)?;

    let n = F::from_usize(y_true.len()).unwrap();

    // Compute residuals and their mean
    let residuals: Vec<F> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| t - p)
        .collect();

    let res_mean = residuals.iter().copied().fold(F::zero(), |a, b| a + b) / n;
    let var_res = residuals
        .iter()
        .map(|&r| (r - res_mean) * (r - res_mean))
        .fold(F::zero(), |acc, v| acc + v)
        / n;

    // Variance of y_true
    let y_mean = y_true.iter().copied().fold(F::zero(), |a, b| a + b) / n;
    let var_y = y_true
        .iter()
        .map(|&t| (t - y_mean) * (t - y_mean))
        .fold(F::zero(), |acc, v| acc + v)
        / n;

    if var_y == F::zero() {
        return Ok(F::zero());
    }

    Ok(F::one() - var_res / var_y)
}

/// Max Error: `max(|y_true_i - y_pred_i|)`.
///
/// Returns the maximum absolute difference between corresponding elements.
pub fn max_error<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_pred)?;

    let result = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .fold(
            F::zero(),
            |max_val, v| if v > max_val { v } else { max_val },
        );

    Ok(result)
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
    // Mean Absolute Percentage Error tests
    // ---------------------------------------------------------------

    #[test]
    fn test_mape_perfect() {
        let y_true = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(
            mean_absolute_percentage_error(&y_true, &y_true).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mape_known() {
        // |1.0 - 1.5| / 1.0 = 0.5
        // |2.0 - 2.5| / 2.0 = 0.25
        // |4.0 - 3.5| / 4.0 = 0.125
        // mean = (0.5 + 0.25 + 0.125) / 3 = 0.875 / 3
        let y_true = array![1.0, 2.0, 4.0];
        let y_pred = array![1.5, 2.5, 3.5];
        let expected = (0.5 + 0.25 + 0.125) / 3.0;
        assert_abs_diff_eq!(
            mean_absolute_percentage_error(&y_true, &y_pred).unwrap(),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mape_zero_in_y_true_error() {
        let y_true = array![1.0, 0.0, 3.0];
        let y_pred = array![1.5, 2.5, 3.5];
        assert!(mean_absolute_percentage_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_mape_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];
        assert!(mean_absolute_percentage_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_mape_length_mismatch_error() {
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(mean_absolute_percentage_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_mape_negative_values() {
        // MAPE uses |y_true| in denominator, so negative y_true is fine.
        // |(-1.0) - (-1.5)| / |-1.0| = 0.5 / 1.0 = 0.5
        // |(-2.0) - (-2.5)| / |-2.0| = 0.5 / 2.0 = 0.25
        let y_true = array![-1.0, -2.0];
        let y_pred = array![-1.5, -2.5];
        let expected = (0.5 + 0.25) / 2.0;
        assert_abs_diff_eq!(
            mean_absolute_percentage_error(&y_true, &y_pred).unwrap(),
            expected,
            epsilon = 1e-10
        );
    }

    // ---------------------------------------------------------------
    // Explained Variance Score tests
    // ---------------------------------------------------------------

    #[test]
    fn test_evs_perfect() {
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        assert_abs_diff_eq!(
            explained_variance_score(&y_true, &y_true).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evs_known() {
        // y_true = [3, -0.5, 2, 7], y_pred = [2.5, 0.0, 2, 8]
        // residuals = [0.5, -0.5, 0, -1], mean_res = -0.25
        // var_res = ((0.75)^2 + (-0.25)^2 + (0.25)^2 + (-0.75)^2) / 4
        //         = (0.5625 + 0.0625 + 0.0625 + 0.5625) / 4 = 1.25 / 4 = 0.3125
        // y_mean = (3 + (-0.5) + 2 + 7) / 4 = 11.5 / 4 = 2.875
        // var_y = ((0.125)^2 + (-3.375)^2 + (-0.875)^2 + (4.125)^2) / 4
        //       = (0.015625 + 11.390625 + 0.765625 + 17.015625) / 4
        //       = 29.1875 / 4 = 7.296875
        // EVS = 1 - 0.3125 / 7.296875 = 1 - 0.04281... = 0.95718...
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];
        let expected = 1.0 - 0.3125 / 7.296875;
        assert_abs_diff_eq!(
            explained_variance_score(&y_true, &y_pred).unwrap(),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evs_constant_y_true() {
        // Var(y_true) = 0 -> returns 0
        let y_true = array![5.0, 5.0, 5.0];
        let y_pred = array![4.0, 5.0, 6.0];
        assert_abs_diff_eq!(
            explained_variance_score(&y_true, &y_pred).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evs_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];
        assert!(explained_variance_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_evs_length_mismatch_error() {
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(explained_variance_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_evs_with_bias() {
        // Explained variance handles bias (constant offset) differently from R2.
        // y_true = [1, 2, 3], y_pred = [2, 3, 4] (shifted by +1)
        // residuals = [-1, -1, -1], mean_res = -1, var_res = 0
        // var_y = 2/3
        // EVS = 1 - 0 / (2/3) = 1.0
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![2.0, 3.0, 4.0];
        assert_abs_diff_eq!(
            explained_variance_score(&y_true, &y_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    // ---------------------------------------------------------------
    // Max Error tests
    // ---------------------------------------------------------------

    #[test]
    fn test_max_error_perfect() {
        let y_true = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(max_error(&y_true, &y_true).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_error_known() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.5, 2.5, 5.0];
        // max(|0.5|, |0.5|, |2.0|) = 2.0
        assert_abs_diff_eq!(max_error(&y_true, &y_pred).unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_error_negative_diff() {
        let y_true = array![5.0, 10.0, 15.0];
        let y_pred = array![4.0, 3.0, 14.0];
        // max(|1|, |7|, |1|) = 7.0
        assert_abs_diff_eq!(max_error(&y_true, &y_pred).unwrap(), 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_error_single_element() {
        let y_true = array![3.0];
        let y_pred = array![5.0];
        assert_abs_diff_eq!(max_error(&y_true, &y_pred).unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_error_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];
        assert!(max_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_max_error_length_mismatch_error() {
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(max_error(&y_true, &y_pred).is_err());
    }

    // ---------------------------------------------------------------
    // f32 type tests
    // ---------------------------------------------------------------

    #[test]
    fn test_mape_f32() {
        let y_true: Array1<f32> = array![1.0f32, 2.0, 4.0];
        let y_pred: Array1<f32> = array![1.5f32, 2.5, 3.5];
        let result = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
        assert!(result.is_finite());
        assert!(result > 0.0f32);
    }

    #[test]
    fn test_evs_f32() {
        let y_true: Array1<f32> = array![1.0f32, 2.0, 3.0, 4.0];
        let result = explained_variance_score(&y_true, &y_true).unwrap();
        assert_abs_diff_eq!(result, 1.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_max_error_f32() {
        let y_true: Array1<f32> = array![1.0f32, 2.0, 3.0];
        let y_pred: Array1<f32> = array![1.5f32, 2.5, 5.0];
        let result = max_error(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(result, 2.0f32, epsilon = 1e-6);
    }
}
