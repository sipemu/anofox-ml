use anofox_ml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};
use ndarray::Array2;

/// Generates polynomial and interaction features.
///
/// Mirrors `sklearn.preprocessing.PolynomialFeatures` (sklearn 1.8). For a
/// feature vector `[a, b]` and `degree=2`:
/// - `interaction_only=false`: `[1, a, b, a^2, ab, b^2]`
/// - `interaction_only=true`:  `[1, a, b, ab]`
///
/// Columns come in sklearn's order: the bias, then degree by degree the
/// combinations (with replacement unless `interaction_only`) of feature
/// indices in lexicographic order. Each column of degree `k` is computed as
/// its degree-`k-1` prefix column times one feature, so `transform` costs one
/// multiplication per output entry.
///
/// Implements `FitUnsupervised` for pipeline compatibility. The fit step only
/// records the number of input features; all computation happens in `transform`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolynomialFeatures {
    /// Maximum degree of polynomial features.
    pub degree: usize,
    /// If true, only produce interaction features (no `x_i^k` for k > 1).
    pub interaction_only: bool,
    /// Prepend the constant column (sklearn's `include_bias`, default true).
    #[serde(default = "default_true")]
    pub include_bias: bool,
    /// Lowest degree of the non-bias terms (sklearn's `degree=(min, max)`).
    #[serde(default)]
    pub min_degree: usize,
}

fn default_true() -> bool {
    true
}

impl PolynomialFeatures {
    /// Create a new `PolynomialFeatures` with default degree 2 and interaction_only = false.
    pub fn new() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
            include_bias: true,
            min_degree: 0,
        }
    }

    /// Set the maximum polynomial degree.
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set the minimum and maximum degree (sklearn's `degree=(min, max)`).
    pub fn with_degree_range(mut self, min_degree: usize, max_degree: usize) -> Self {
        self.min_degree = min_degree;
        self.degree = max_degree;
        self
    }

    /// Set whether to produce only interaction features.
    pub fn with_interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Set whether to include the constant (bias) column.
    pub fn with_include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
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
    min_degree: usize,
    interaction_only: bool,
    include_bias: bool,
    /// `terms[k - 1]` lists the degree-`k` terms as feature-index tuples
    /// (non-decreasing), in output order.
    terms: Vec<Vec<Vec<usize>>>,
    /// `parent[k - 1][t]`: index in the degree-`k-1` list of the prefix of
    /// term `t` (unused for degree 1).
    parent: Vec<Vec<usize>>,
    _marker: std::marker::PhantomData<F>,
}

/// Degree-by-degree terms and the prefix index of each term.
#[allow(clippy::type_complexity)]
fn enumerate_terms(
    n_features: usize,
    max_degree: usize,
    interaction_only: bool,
) -> (Vec<Vec<Vec<usize>>>, Vec<Vec<usize>>) {
    let mut terms: Vec<Vec<Vec<usize>>> = Vec::with_capacity(max_degree);
    let mut parent: Vec<Vec<usize>> = Vec::with_capacity(max_degree);
    if max_degree == 0 {
        return (terms, parent);
    }
    terms.push((0..n_features).map(|j| vec![j]).collect());
    parent.push(vec![0; n_features]);
    for k in 2..=max_degree {
        let prev = &terms[k - 2];
        let mut cur = Vec::new();
        let mut par = Vec::new();
        for (p, t) in prev.iter().enumerate() {
            let last = *t.last().expect("terms are non-empty");
            let first = if interaction_only { last + 1 } else { last };
            for j in first..n_features {
                let mut child = t.clone();
                child.push(j);
                cur.push(child);
                par.push(p);
            }
        }
        terms.push(cur);
        parent.push(par);
    }
    (terms, parent)
}

impl<F: Float> FitUnsupervised<F> for PolynomialFeatures {
    type Fitted = FittedPolynomialFeatures<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if self.min_degree > self.degree {
            return Err(RustMlError::InvalidParameter(format!(
                "min_degree ({}) must be <= degree ({})",
                self.min_degree, self.degree
            )));
        }
        if self.degree == 0 && !self.include_bias {
            return Err(RustMlError::InvalidParameter(
                "degree 0 with include_bias = false would give an empty output".into(),
            ));
        }

        let n_features = x.ncols();
        let (terms, parent) = enumerate_terms(n_features, self.degree, self.interaction_only);

        Ok(FittedPolynomialFeatures {
            n_features,
            degree: self.degree,
            min_degree: self.min_degree,
            interaction_only: self.interaction_only,
            include_bias: self.include_bias,
            terms,
            parent,
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
        let mut result = Array2::<F>::zeros((nrows, self.n_output_features()));
        let mut col = 0;
        if self.include_bias {
            result.column_mut(0).fill(F::one());
            col = 1;
        }
        // Degree-k block from the degree-(k-1) block: one product per entry.
        let mut prev: Array2<F> = x.to_owned();
        for k in 1..=self.degree {
            let block = if k == 1 {
                std::mem::take(&mut prev)
            } else {
                let terms = &self.terms[k - 1];
                let mut b = Array2::<F>::zeros((nrows, terms.len()));
                for (t, (term, &p)) in terms.iter().zip(&self.parent[k - 1]).enumerate() {
                    let j = *term.last().expect("terms are non-empty");
                    let mut bc = b.column_mut(t);
                    bc.assign(&prev.column(p));
                    bc *= &x.column(j);
                }
                b
            };
            if k >= self.min_degree {
                let w = block.ncols();
                result
                    .slice_mut(ndarray::s![.., col..col + w])
                    .assign(&block);
                col += w;
            }
            prev = block;
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
        self.included_terms().count() + usize::from(self.include_bias)
    }

    /// Return the degree.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return whether only interactions are generated.
    pub fn interaction_only(&self) -> bool {
        self.interaction_only
    }

    fn included_terms(&self) -> impl Iterator<Item = &Vec<usize>> {
        self.terms
            .iter()
            .enumerate()
            .filter(move |(k, _)| k + 1 >= self.min_degree)
            .flat_map(|(_, t)| t.iter())
    }

    /// Exponent of each input feature in each output column, shape
    /// `(n_output_features, n_input_features)` (sklearn's `powers_`).
    pub fn powers(&self) -> Array2<usize> {
        let mut p = Array2::<usize>::zeros((self.n_output_features(), self.n_features));
        let offset = usize::from(self.include_bias);
        for (r, term) in self.included_terms().enumerate() {
            for &j in term {
                p[[r + offset, j]] += 1;
            }
        }
        p
    }

    /// Output column names as sklearn's `get_feature_names_out`: `"1"`,
    /// `"x0"`, `"x0^2"`, `"x0 x1"`, ... `input_names` replaces `x0, x1, ...`.
    pub fn feature_names_out(&self, input_names: Option<&[String]>) -> Result<Vec<String>> {
        let names: Vec<String> = match input_names {
            Some(n) if n.len() != self.n_features => {
                return Err(RustMlError::ShapeMismatch(format!(
                    "expected {} input names, got {}",
                    self.n_features,
                    n.len()
                )))
            }
            Some(n) => n.to_vec(),
            None => (0..self.n_features).map(|j| format!("x{j}")).collect(),
        };
        Ok(self
            .powers()
            .rows()
            .into_iter()
            .map(|row| {
                let parts: Vec<String> = row
                    .iter()
                    .enumerate()
                    .filter(|(_, &e)| e > 0)
                    .map(|(j, &e)| {
                        if e == 1 {
                            names[j].clone()
                        } else {
                            format!("{}^{e}", names[j])
                        }
                    })
                    .collect();
                if parts.is_empty() {
                    "1".to_string()
                } else {
                    parts.join(" ")
                }
            })
            .collect())
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
    fn test_degree0_bias_only() {
        // sklearn: PolynomialFeatures(0).fit_transform(X) -> a column of ones;
        // degree 0 without a bias is an error.
        let x = array![[2.0, 3.0], [4.0, 5.0]];
        let fitted =
            FitUnsupervised::<f64>::fit(&PolynomialFeatures::new().with_degree(0), &x).unwrap();
        assert_eq!(fitted.transform(&x).unwrap(), array![[1.0], [1.0]]);
        let poly = PolynomialFeatures::new()
            .with_degree(0)
            .with_include_bias(false);
        assert!(FitUnsupervised::<f64>::fit(&poly, &x).is_err());
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

    #[test]
    fn test_matches_sklearn_names_and_values() {
        // sklearn 1.8 PolynomialFeatures(...).fit(X): get_feature_names_out()
        // and transform(X).
        let x = array![[2.0, 3.0, -1.0], [0.5, -2.0, 4.0]];
        let check = |poly: PolynomialFeatures, names: &[&str], rows: [&[f64]; 2]| {
            let f = FitUnsupervised::<f64>::fit(&poly, &x).unwrap();
            assert_eq!(f.feature_names_out(None).unwrap(), names);
            let t = f.transform(&x).unwrap();
            assert_eq!(t.ncols(), names.len());
            for (i, row) in rows.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    assert_abs_diff_eq!(t[[i, j]], *v, epsilon = 1e-12);
                }
            }
        };
        check(
            PolynomialFeatures::new().with_degree(3),
            &[
                "1", "x0", "x1", "x2", "x0^2", "x0 x1", "x0 x2", "x1^2", "x1 x2", "x2^2", "x0^3",
                "x0^2 x1", "x0^2 x2", "x0 x1^2", "x0 x1 x2", "x0 x2^2", "x1^3", "x1^2 x2",
                "x1 x2^2", "x2^3",
            ],
            [
                &[
                    1.0, 2.0, 3.0, -1.0, 4.0, 6.0, -2.0, 9.0, -3.0, 1.0, 8.0, 12.0, -4.0, 18.0,
                    -6.0, 2.0, 27.0, -9.0, 3.0, -1.0,
                ],
                &[
                    1.0, 0.5, -2.0, 4.0, 0.25, -1.0, 2.0, 4.0, -8.0, 16.0, 0.125, -0.5, 1.0, 2.0,
                    -4.0, 8.0, -8.0, 16.0, -32.0, 64.0,
                ],
            ],
        );
        check(
            PolynomialFeatures::new()
                .with_degree_range(2, 3)
                .with_include_bias(false),
            &[
                "x0^2", "x0 x1", "x0 x2", "x1^2", "x1 x2", "x2^2", "x0^3", "x0^2 x1", "x0^2 x2",
                "x0 x1^2", "x0 x1 x2", "x0 x2^2", "x1^3", "x1^2 x2", "x1 x2^2", "x2^3",
            ],
            [
                &[
                    4.0, 6.0, -2.0, 9.0, -3.0, 1.0, 8.0, 12.0, -4.0, 18.0, -6.0, 2.0, 27.0, -9.0,
                    3.0, -1.0,
                ],
                &[
                    0.25, -1.0, 2.0, 4.0, -8.0, 16.0, 0.125, -0.5, 1.0, 2.0, -4.0, 8.0, -8.0, 16.0,
                    -32.0, 64.0,
                ],
            ],
        );
        check(
            PolynomialFeatures::new()
                .with_degree(3)
                .with_interaction_only(true),
            &["1", "x0", "x1", "x2", "x0 x1", "x0 x2", "x1 x2", "x0 x1 x2"],
            [
                &[1.0, 2.0, 3.0, -1.0, 6.0, -2.0, -3.0, -6.0],
                &[1.0, 0.5, -2.0, 4.0, -1.0, 2.0, -8.0, -4.0],
            ],
        );
        // degree=(2, 2) keeps the bias, as sklearn does.
        check(
            PolynomialFeatures::new()
                .with_degree_range(2, 2)
                .with_interaction_only(true),
            &["1", "x0 x1", "x0 x2", "x1 x2"],
            [&[1.0, 6.0, -2.0, -3.0], &[1.0, -1.0, 2.0, -8.0]],
        );
    }

    #[test]
    fn test_powers_and_custom_names() {
        let x = array![[1.0, 2.0]];
        let f = FitUnsupervised::<f64>::fit(&PolynomialFeatures::new(), &x).unwrap();
        assert_eq!(
            f.powers(),
            array![[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
        );
        let names = ["a".to_string(), "b".to_string()];
        assert_eq!(
            f.feature_names_out(Some(&names)).unwrap(),
            ["1", "a", "b", "a^2", "a b", "b^2"]
        );
    }
}
