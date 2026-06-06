use ndarray::Array2;
use rustml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};

/// Generates polynomial and interaction features.
///
/// For a feature vector `[a, b]` and `degree=2`:
/// - `interaction_only=false`: `[1, a, b, a^2, ab, b^2]`
/// - `interaction_only=true`:  `[1, a, b, ab]`
///
/// Implements `FitUnsupervised` for pipeline compatibility. The fit step only
/// records the number of input features; all computation happens in `transform`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolynomialFeatures {
    /// Maximum degree of polynomial features.
    pub degree: usize,
    /// If true, only produce interaction features (no `x_i^k` for k > 1).
    pub interaction_only: bool,
}

impl PolynomialFeatures {
    /// Create a new `PolynomialFeatures` with default degree 2 and interaction_only = false.
    pub fn new() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
        }
    }

    /// Set the maximum polynomial degree.
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set whether to produce only interaction features.
    pub fn with_interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted PolynomialFeatures — stores the number of input features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedPolynomialFeatures<F: Float> {
    n_features: usize,
    degree: usize,
    interaction_only: bool,
    /// Pre-computed list of (exponents) for each output column.
    /// Each entry is a Vec of (feature_index, power) pairs.
    combinations: Vec<Vec<(usize, usize)>>,
    _marker: std::marker::PhantomData<F>,
}

/// Enumerate all combinations of features with total degree up to `max_degree`.
///
/// Each combination is represented as a vec of `(feature_index, power)` pairs
/// where power > 0. The bias term (degree 0) is an empty vec.
///
/// Combinations are ordered by total degree, then lexicographically by feature
/// index, matching scikit-learn's convention:
/// - degree 0: `[1]`
/// - degree 1: `[a, b, c, ...]`
/// - degree 2: `[a^2, ab, ac, ..., b^2, bc, ..., c^2, ...]`
fn enumerate_combinations(
    n_features: usize,
    max_degree: usize,
    interaction_only: bool,
) -> Vec<Vec<(usize, usize)>> {
    let mut combos: Vec<Vec<(usize, usize)>> = Vec::new();
    // Degree 0: bias term
    combos.push(vec![]);

    // Helper: enumerate all combinations with exactly `target_degree` total power,
    // starting from feature `start_feature`.
    fn recurse_exact(
        start_feature: usize,
        target_degree: usize,
        n_features: usize,
        interaction_only: bool,
        current: &mut Vec<(usize, usize)>,
        combos: &mut Vec<Vec<(usize, usize)>>,
    ) {
        if target_degree == 0 {
            combos.push(current.clone());
            return;
        }
        for feat in start_feature..n_features {
            let max_power = if interaction_only { 1 } else { target_degree };
            for power in (1..=max_power).rev() {
                current.push((feat, power));
                // Remaining degree allocated to features with index > feat
                let remaining = target_degree - power;
                if remaining == 0 {
                    combos.push(current.clone());
                } else {
                    recurse_exact(
                        feat + 1,
                        remaining,
                        n_features,
                        interaction_only,
                        current,
                        combos,
                    );
                }
                current.pop();
            }
        }
    }

    // Generate degree by degree to ensure correct ordering
    for d in 1..=max_degree {
        let mut current = Vec::new();
        recurse_exact(
            0,
            d,
            n_features,
            interaction_only,
            &mut current,
            &mut combos,
        );
    }

    combos
}

impl<F: Float> FitUnsupervised<F> for PolynomialFeatures {
    type Fitted = FittedPolynomialFeatures<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if self.degree == 0 {
            return Err(RustMlError::InvalidParameter(
                "degree must be at least 1".into(),
            ));
        }

        let n_features = x.ncols();
        let combinations = enumerate_combinations(n_features, self.degree, self.interaction_only);

        Ok(FittedPolynomialFeatures {
            n_features,
            degree: self.degree,
            interaction_only: self.interaction_only,
            combinations,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F: Float> Transform<F> for FittedPolynomialFeatures<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let nrows = x.nrows();
        let ncols_out = self.combinations.len();
        let mut result = Array2::<F>::ones((nrows, ncols_out));

        for (out_col, combo) in self.combinations.iter().enumerate() {
            if combo.is_empty() {
                // Bias term: already 1.0
                continue;
            }
            for i in 0..nrows {
                let mut val = F::one();
                for &(feat, power) in combo {
                    let base = x[[i, feat]];
                    for _ in 0..power {
                        val *= base;
                    }
                }
                result[[i, out_col]] = val;
            }
        }

        Ok(result)
    }
}

impl<F: Float> FittedPolynomialFeatures<F> {
    /// Return the number of input features.
    pub fn n_input_features(&self) -> usize {
        self.n_features
    }

    /// Return the number of output features.
    pub fn n_output_features(&self) -> usize {
        self.combinations.len()
    }

    /// Return the degree.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return whether only interactions are generated.
    pub fn interaction_only(&self) -> bool {
        self.interaction_only
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_degree2_two_features() {
        // [a, b] -> [1, a, b, a^2, ab, b^2]
        let x = array![[2.0, 3.0]];
        let poly = PolynomialFeatures::new();
        let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.ncols(), 6);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10); // 1
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-10); // b
        assert_abs_diff_eq!(out[[0, 3]], 4.0, epsilon = 1e-10); // a^2
        assert_abs_diff_eq!(out[[0, 4]], 6.0, epsilon = 1e-10); // ab
        assert_abs_diff_eq!(out[[0, 5]], 9.0, epsilon = 1e-10); // b^2
    }

    #[test]
    fn test_interaction_only_degree2() {
        // [a, b] -> [1, a, b, ab]
        let x = array![[2.0, 3.0]];
        let poly = PolynomialFeatures::new().with_interaction_only(true);
        let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.ncols(), 4);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10); // 1
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-10); // b
        assert_abs_diff_eq!(out[[0, 3]], 6.0, epsilon = 1e-10); // ab
    }

    #[test]
    fn test_degree3_single_feature() {
        // [a] -> [1, a, a^2, a^3]
        let x = array![[3.0]];
        let poly = PolynomialFeatures::new().with_degree(3);
        let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.ncols(), 4);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10); // 1
        assert_abs_diff_eq!(out[[0, 1]], 3.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(out[[0, 2]], 9.0, epsilon = 1e-10); // a^2
        assert_abs_diff_eq!(out[[0, 3]], 27.0, epsilon = 1e-10); // a^3
    }

    #[test]
    fn test_degree1() {
        // degree=1: [a, b] -> [1, a, b]
        let x = array![[2.0, 3.0]];
        let poly = PolynomialFeatures::new().with_degree(1);
        let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.ncols(), 3);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_degree0_error() {
        let x = array![[1.0, 2.0]];
        let poly = PolynomialFeatures::new().with_degree(0);
        let result = FitUnsupervised::<f64>::fit(&poly, &x);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_rows() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let poly = PolynomialFeatures::new();
        let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.nrows(), 2);
        assert_eq!(out.ncols(), 6);

        // Row 0: [1, 1, 2, 1, 2, 4]
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 3]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 4]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 5]], 4.0, epsilon = 1e-10);

        // Row 1: [1, 3, 4, 9, 12, 16]
        assert_abs_diff_eq!(out[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 2]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 3]], 9.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 4]], 12.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 5]], 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_three_features_degree2() {
        // [a, b, c] -> [1, a, b, c, a^2, ab, ac, b^2, bc, c^2]
        let x = array![[1.0, 2.0, 3.0]];
        let poly = PolynomialFeatures::new();
        let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.ncols(), 10);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10); // 1
        assert_abs_diff_eq!(out[[0, 1]], 1.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(out[[0, 2]], 2.0, epsilon = 1e-10); // b
        assert_abs_diff_eq!(out[[0, 3]], 3.0, epsilon = 1e-10); // c
        assert_abs_diff_eq!(out[[0, 4]], 1.0, epsilon = 1e-10); // a^2
        assert_abs_diff_eq!(out[[0, 5]], 2.0, epsilon = 1e-10); // ab
        assert_abs_diff_eq!(out[[0, 6]], 3.0, epsilon = 1e-10); // ac
        assert_abs_diff_eq!(out[[0, 7]], 4.0, epsilon = 1e-10); // b^2
        assert_abs_diff_eq!(out[[0, 8]], 6.0, epsilon = 1e-10); // bc
        assert_abs_diff_eq!(out[[0, 9]], 9.0, epsilon = 1e-10); // c^2
    }

    #[test]
    fn test_three_features_interaction_only() {
        // [a, b, c] -> [1, a, b, c, ab, ac, bc]
        let x = array![[2.0, 3.0, 5.0]];
        let poly = PolynomialFeatures::new().with_interaction_only(true);
        let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.ncols(), 7);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10); // 1
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-10); // b
        assert_abs_diff_eq!(out[[0, 3]], 5.0, epsilon = 1e-10); // c
        assert_abs_diff_eq!(out[[0, 4]], 6.0, epsilon = 1e-10); // ab
        assert_abs_diff_eq!(out[[0, 5]], 10.0, epsilon = 1e-10); // ac
        assert_abs_diff_eq!(out[[0, 6]], 15.0, epsilon = 1e-10); // bc
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 0));
        let poly = PolynomialFeatures::new();
        assert!(FitUnsupervised::<f64>::fit(&poly, &x).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0]];
        let poly = PolynomialFeatures::new();
        let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_bias_column_all_ones() {
        let x = array![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]];
        let poly = PolynomialFeatures::new();
        let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
        let out = fitted.transform(&x).unwrap();

        // First column (bias) should be all 1.0
        for i in 0..3 {
            assert_abs_diff_eq!(out[[i, 0]], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_n_output_features() {
        let x = array![[1.0, 2.0]];
        let poly = PolynomialFeatures::new();
        let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();

        assert_eq!(fitted.n_input_features(), 2);
        assert_eq!(fitted.n_output_features(), 6);
        assert_eq!(fitted.degree(), 2);
        assert!(!fitted.interaction_only());
    }

    #[test]
    fn test_f32() {
        let x = array![[2.0f32, 3.0]];
        let poly = PolynomialFeatures::new();
        let fitted = FitUnsupervised::<f32>::fit(&poly, &x).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.ncols(), 6);
        assert_abs_diff_eq!(out[[0, 3]], 4.0f32, epsilon = 1e-5); // a^2
        assert_abs_diff_eq!(out[[0, 4]], 6.0f32, epsilon = 1e-5); // ab
        assert_abs_diff_eq!(out[[0, 5]], 9.0f32, epsilon = 1e-5); // b^2
    }

    #[test]
    fn test_default() {
        let poly = PolynomialFeatures::default();
        assert_eq!(poly.degree, 2);
        assert!(!poly.interaction_only);
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        fn make_data(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut values = Vec::with_capacity(rows * cols);
            for i in 0..(rows * cols) {
                let mut h = DefaultHasher::new();
                seed.hash(&mut h);
                (i as u64).hash(&mut h);
                let bits = h.finish();
                let v = (bits as f64 / u64::MAX as f64) * 4.0 - 2.0;
                values.push(v);
            }
            Array2::from_shape_vec((rows, cols), values).unwrap()
        }

        proptest! {
            #[test]
            fn poly_bias_column_all_ones(
                rows in 1..20usize,
                cols in 1..5usize,
                seed in 0u64..10000,
            ) {
                let x = make_data(rows, cols, seed);
                let poly = PolynomialFeatures::new();
                let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
                let out = fitted.transform(&x).unwrap();

                for i in 0..rows {
                    prop_assert!((out[[i, 0]] - 1.0).abs() < 1e-10,
                        "bias column should be 1.0, got {}", out[[i, 0]]);
                }
            }

            #[test]
            fn poly_original_features_preserved(
                rows in 1..20usize,
                cols in 1..5usize,
                seed in 0u64..10000,
            ) {
                let x = make_data(rows, cols, seed);
                let poly = PolynomialFeatures::new();
                let fitted = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
                let out = fitted.transform(&x).unwrap();

                // Columns 1..=cols should be the original features
                for i in 0..rows {
                    for j in 0..cols {
                        prop_assert!((out[[i, 1 + j]] - x[[i, j]]).abs() < 1e-10,
                            "original feature not preserved at ({}, {})", i, j);
                    }
                }
            }
        }
    }
}
