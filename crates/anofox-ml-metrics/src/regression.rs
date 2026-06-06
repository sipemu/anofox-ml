use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::Array1;

/// Mean Squared Error: `mean((y_true - y_pred)^2)`.
pub fn mse<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_pred)?;
    let n = F::from_usize(y_true.len()).unwrap();
    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p) * (t - p))
        .fold(F::zero(), |acc, v| acc + v);
    Ok(sum / n)
}

/// Mean Absolute Error: `mean(|y_true - y_pred|)`.
pub fn mae<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_pred)?;
    let n = F::from_usize(y_true.len()).unwrap();
    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .fold(F::zero(), |acc, v| acc + v);
    Ok(sum / n)
}

/// R² (coefficient of determination): `1 - SS_res / SS_tot`.
pub fn r2_score<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_pred)?;
    let n = F::from_usize(y_true.len()).unwrap();
    let mean = y_true.iter().copied().fold(F::zero(), |a, b| a + b) / n;

    let ss_res = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p) * (t - p))
        .fold(F::zero(), |acc, v| acc + v);

    let ss_tot = y_true
        .iter()
        .map(|&t| (t - mean) * (t - mean))
        .fold(F::zero(), |acc, v| acc + v);

    if ss_tot == F::zero() {
        return Ok(F::zero());
    }

    Ok(F::one() - ss_res / ss_tot)
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

    #[test]
    fn test_mse_perfect() {
        let y = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(mse(&y, &y).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_known() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.5, 2.5, 3.5];
        assert_abs_diff_eq!(mse(&y_true, &y_pred).unwrap(), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_mae_perfect() {
        let y = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(mae(&y, &y).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mae_known() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.5, 2.5, 3.5];
        assert_abs_diff_eq!(mae(&y_true, &y_pred).unwrap(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_r2_perfect() {
        let y_true = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(r2_score(&y_true, &y_true).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_r2_known() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];
        assert_abs_diff_eq!(
            r2_score(&y_true, &y_pred).unwrap(),
            0.9486081370449679,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_shape_mismatch() {
        let a = array![1.0, 2.0];
        let b = array![1.0, 2.0, 3.0];
        assert!(mse(&a, &b).is_err());
    }

    #[test]
    fn test_empty_input() {
        let a: Array1<f64> = array![];
        let b: Array1<f64> = array![];
        assert!(mse(&a, &b).is_err());
    }

    mod prop_tests {
        use super::*;
        use proptest::collection::vec;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn mse_non_negative(data in vec(proptest::num::f64::NORMAL, 1..100)) {
                let n = data.len();
                let y_true = Array1::from_vec(data.clone());
                // Shift values to create different predictions
                let y_pred = Array1::from_vec(
                    data.iter().map(|&v| v + 1.0).collect::<Vec<_>>()
                );
                let result = mse(&y_true, &y_pred).unwrap();
                prop_assert!(result >= 0.0, "MSE should be non-negative, got {}", result);
                let _ = n;
            }

            #[test]
            fn mse_zero_for_identical(data in vec(proptest::num::f64::NORMAL, 1..100)) {
                let y = Array1::from_vec(data);
                let result = mse(&y, &y).unwrap();
                prop_assert!((result - 0.0).abs() < 1e-10,
                    "MSE(y, y) should be 0, got {}", result);
            }

            #[test]
            fn r2_perfect_for_identical(data in vec(proptest::num::f64::NORMAL, 2..100)) {
                let y = Array1::from_vec(data.clone());
                // Need at least 2 different values for a meaningful r2
                let has_variance = data.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-15);
                prop_assume!(has_variance);

                let result = r2_score(&y, &y).unwrap();
                prop_assert!((result - 1.0).abs() < 1e-10,
                    "R2(y, y) should be 1.0, got {}", result);
            }
        }
    }
}
