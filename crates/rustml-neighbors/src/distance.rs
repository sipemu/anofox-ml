use ndarray::ArrayView1;
use rustml_core::Float;

/// Distance metric for KNN.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DistanceMetric {
    #[default]
    Euclidean,
    Manhattan,
}

/// Compute distance between two points.
pub fn compute_distance<F: Float>(a: &ArrayView1<F>, b: &ArrayView1<F>, metric: DistanceMetric) -> F {
    match metric {
        DistanceMetric::Euclidean => {
            let sum = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y) * (x - y))
                .fold(F::zero(), |acc, v| acc + v);
            sum.sqrt()
        }
        DistanceMetric::Manhattan => a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(F::zero(), |acc, v| acc + v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_euclidean() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Euclidean),
            5.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_manhattan() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Manhattan),
            7.0,
            epsilon = 1e-10
        );
    }
}
