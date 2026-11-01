//! Target encoding of categorical features.
//!
//! Mirrors `sklearn.preprocessing.TargetEncoder` (sklearn 1.8). Each category
//! is replaced by a shrunk estimate of the target mean within that category
//! (Micci-Barreca, 2001):
//!
//! * `smooth = Auto`: `λ·mean_c + (1 − λ)·mean_y` with
//!   `λ = var_y·n_c / (var_y·n_c + SS_c / n_c)` (empirical Bayes);
//! * `smooth = Value(m)`: `(sum_c + m·mean_y) / (n_c + m)`.
//!
//! Binary targets are encoded as the probability of the second (larger)
//! class, multiclass targets one-vs-rest per class (`n_features · n_classes`
//! output columns, grouped by feature). Unknown categories map to the target
//! mean. Categories are the distinct values of each column, `NaN` included.
//!
//! `fit(X, y).transform(X)` leaks the target into the training encoding, so
//! use [`TargetEncoder::fit_transform`] for training data: it cross-fits, each
//! fold being encoded with encodings learnt on the other folds. Wrapped in
//! [`Supervised`](anofox_ml_core::Supervised) inside a pipeline, the training
//! data goes through `fit_transform` automatically.

use anofox_ml_core::{
    shuffled_k_fold, stratified_k_fold, Fit, Float, Result, RustMlError, Transform,
};
use ndarray::{Array1, Array2};

/// Kind of target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TargetType {
    /// Infer like sklearn's `type_of_target`: continuous if any value is not
    /// an integer, else binary for at most two distinct values, else
    /// multiclass.
    #[default]
    Auto,
    Continuous,
    Binary,
    Multiclass,
}

/// Shrinkage towards the target mean.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum TargetSmooth {
    /// Empirical-Bayes shrinkage (sklearn's `smooth='auto'`).
    #[default]
    Auto,
    /// Fixed smoothing weight `m >= 0`.
    Value(f64),
}

/// Parameters for `TargetEncoder` (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TargetEncoder {
    pub target_type: TargetType,
    pub smooth: TargetSmooth,
    /// Number of cross-fitting folds in `fit_transform` (stratified for
    /// binary / multiclass targets). Folds are shuffled.
    pub cv: usize,
    /// Seed of the fold shuffle.
    pub random_state: u64,
}

impl Default for TargetEncoder {
    fn default() -> Self {
        Self {
            target_type: TargetType::Auto,
            smooth: TargetSmooth::Auto,
            cv: 5,
            random_state: 0,
        }
    }
}

impl TargetEncoder {
    /// Create a `TargetEncoder` with sklearn's defaults.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_target_type(mut self, target_type: TargetType) -> Self {
        self.target_type = target_type;
        self
    }

    pub fn with_smooth(mut self, smooth: TargetSmooth) -> Self {
        self.smooth = smooth;
        self
    }

    pub fn with_cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
        self
    }
}

/// Fitted `TargetEncoder`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedTargetEncoder<F: Float> {
    /// Sorted distinct values per feature; a `NaN` category comes last.
    categories: Vec<Vec<F>>,
    /// One encoding per output column (`feature · n_targets + target`),
    /// indexed by category.
    encodings: Vec<Vec<f64>>,
    /// Target mean per target column (one for binary / continuous).
    target_mean: Vec<f64>,
    /// Class labels for binary / multiclass targets.
    classes: Option<Vec<F>>,
    target_type: TargetType,
}

/// Category index of `v` in `cats` (sorted, NaN last), if known.
fn category_index<F: Float>(cats: &[F], v: F) -> Option<usize> {
    if v.is_nan() {
        return cats.last().filter(|c| c.is_nan()).map(|_| cats.len() - 1);
    }
    let finite = if cats.last().is_some_and(|c| c.is_nan()) {
        &cats[..cats.len() - 1]
    } else {
        cats
    };
    finite
        .binary_search_by(|c| c.partial_cmp(&v).expect("non-NaN"))
        .ok()
}

fn infer_target_type<F: Float>(y: &Array1<F>) -> TargetType {
    if y.iter().any(|v| v.fract() != F::zero() || !v.is_finite()) {
        return TargetType::Continuous;
    }
    let mut u: Vec<F> = y.to_vec();
    u.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    u.dedup();
    if u.len() <= 2 {
        TargetType::Binary
    } else {
        TargetType::Multiclass
    }
}

/// Encodings of one target column for every feature (sklearn's
/// `_fit_encoding_fast[_auto_smooth]`), fitted on the rows `rows`.
fn fit_encodings(
    ordinal: &[Vec<Option<usize>>],
    n_categories: &[usize],
    y: &[f64],
    rows: &[usize],
    smooth: TargetSmooth,
) -> (Vec<Vec<f64>>, f64) {
    let n = rows.len() as f64;
    let y_mean = rows.iter().map(|&i| y[i]).sum::<f64>() / n;
    let y_var = rows.iter().map(|&i| (y[i] - y_mean).powi(2)).sum::<f64>() / n;
    let enc = n_categories
        .iter()
        .enumerate()
        .map(|(f, &nc)| {
            let mut sums = vec![0.0; nc];
            let mut counts = vec![0.0; nc];
            for &i in rows {
                if let Some(c) = ordinal[f][i] {
                    sums[c] += y[i];
                    counts[c] += 1.0;
                }
            }
            match smooth {
                TargetSmooth::Value(m) => (0..nc)
                    .map(|c| {
                        let den = counts[c] + m;
                        if den == 0.0 {
                            y_mean
                        } else {
                            (sums[c] + m * y_mean) / den
                        }
                    })
                    .collect(),
                TargetSmooth::Auto => {
                    let means: Vec<f64> = (0..nc).map(|c| sums[c] / counts[c]).collect();
                    let mut ss = vec![0.0; nc];
                    for &i in rows {
                        if let Some(c) = ordinal[f][i] {
                            ss[c] += (y[i] - means[c]).powi(2);
                        }
                    }
                    (0..nc)
                        .map(|c| {
                            let lambda =
                                y_var * counts[c] / (y_var * counts[c] + ss[c] / counts[c]);
                            if lambda.is_nan() {
                                y_mean
                            } else {
                                lambda * means[c] + (1.0 - lambda) * y_mean
                            }
                        })
                        .collect()
                }
            }
        })
        .collect();
    (enc, y_mean)
}

impl TargetEncoder {
    /// Learn categories, target columns and the full-data encodings. Returns
    /// the fitted encoder plus what cross-fitting needs.
    #[allow(clippy::type_complexity)]
    fn fit_all<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<(
        FittedTargetEncoder<F>,
        Vec<Vec<Option<usize>>>,
        Vec<usize>,
        Vec<Vec<f64>>,
    )> {
        let (n, d) = x.dim();
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if y.len() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {n} rows, y has {}",
                y.len()
            )));
        }
        if let TargetSmooth::Value(m) = self.smooth {
            if !(m >= 0.0 && m.is_finite()) {
                return Err(RustMlError::InvalidParameter(
                    "smooth must be finite and non-negative".into(),
                ));
            }
        }
        let target_type = match self.target_type {
            TargetType::Auto => infer_target_type(y),
            t => t,
        };
        if y.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "y contains NaN or infinite values".into(),
            ));
        }

        let categories: Vec<Vec<F>> = (0..d)
            .map(|j| {
                let mut c: Vec<F> = x
                    .column(j)
                    .iter()
                    .copied()
                    .filter(|v| !v.is_nan())
                    .collect();
                c.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN"));
                c.dedup();
                if x.column(j).iter().any(|v| v.is_nan()) {
                    c.push(F::nan());
                }
                c
            })
            .collect();
        let n_categories: Vec<usize> = categories.iter().map(Vec::len).collect();
        let ordinal: Vec<Vec<Option<usize>>> = (0..d)
            .map(|j| {
                x.column(j)
                    .iter()
                    .map(|&v| category_index(&categories[j], v))
                    .collect()
            })
            .collect();

        let (classes, targets): (Option<Vec<F>>, Vec<Vec<f64>>) = match target_type {
            TargetType::Continuous => (None, vec![y.iter().map(|v| v.to_f64().unwrap()).collect()]),
            _ => {
                let mut cls: Vec<F> = y.to_vec();
                cls.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                cls.dedup();
                let indicator = |c: F| y.iter().map(|&v| f64::from(u8::from(v == c))).collect();
                let t = if target_type == TargetType::Binary {
                    if cls.len() > 2 {
                        return Err(RustMlError::InvalidParameter(format!(
                            "binary target has {} classes",
                            cls.len()
                        )));
                    }
                    // Probability of the second class (LabelEncoder's 1).
                    vec![match cls.get(1) {
                        Some(&c) => indicator(c),
                        None => vec![0.0; n],
                    }]
                } else {
                    cls.iter().map(|&c| indicator(c)).collect()
                };
                (Some(cls), t)
            }
        };

        let all: Vec<usize> = (0..n).collect();
        let mut per_target = Vec::with_capacity(targets.len());
        let mut target_mean = Vec::with_capacity(targets.len());
        for t in &targets {
            let (enc, mean) = fit_encodings(&ordinal, &n_categories, t, &all, self.smooth);
            per_target.push(enc);
            target_mean.push(mean);
        }
        let encodings = interleave(per_target, d);
        Ok((
            FittedTargetEncoder {
                categories,
                encodings,
                target_mean,
                classes,
                target_type,
            },
            ordinal,
            n_categories,
            targets,
        ))
    }

    /// Fit on `(X, y)` and return the cross-fitted encoding of the training
    /// data (sklearn's `fit_transform`). The fitted encoder itself uses all
    /// the data, as `fit`.
    pub fn fit_transform<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<(FittedTargetEncoder<F>, Array2<F>)> {
        if self.cv < 2 {
            return Err(RustMlError::InvalidParameter(
                "cv must be at least 2".into(),
            ));
        }
        let (fitted, ordinal, n_categories, targets) = self.fit_all(x, y)?;
        let n = x.nrows();
        let folds = if fitted.target_type == TargetType::Continuous {
            shuffled_k_fold(n, self.cv, self.random_state)?
        } else {
            stratified_k_fold(x, y, self.cv, self.random_state)?
        };
        let d = x.ncols();
        let nt = targets.len();
        let mut out = Array2::<F>::zeros((n, d * nt));
        for (train, test) in folds {
            let mut per_target = Vec::with_capacity(nt);
            let mut means = Vec::with_capacity(nt);
            for t in &targets {
                let (enc, mean) = fit_encodings(&ordinal, &n_categories, t, &train, self.smooth);
                per_target.push(enc);
                means.push(mean);
            }
            let enc = interleave(per_target, d);
            for &i in &test {
                for f in 0..d {
                    for t in 0..nt {
                        let col = f * nt + t;
                        let v = match ordinal[f][i] {
                            Some(c) => enc[col][c],
                            None => means[t],
                        };
                        out[[i, col]] = F::from_f64(v).unwrap();
                    }
                }
            }
        }
        Ok((fitted, out))
    }
}

/// Reorder `per_target[t][f]` to output order `f · n_targets + t`.
fn interleave(per_target: Vec<Vec<Vec<f64>>>, d: usize) -> Vec<Vec<f64>> {
    let nt = per_target.len();
    let mut slots: Vec<Option<Vec<f64>>> = vec![None; d * nt];
    for (t, encs) in per_target.into_iter().enumerate() {
        for (f, e) in encs.into_iter().enumerate() {
            slots[f * nt + t] = Some(e);
        }
    }
    slots.into_iter().map(|s| s.expect("filled")).collect()
}

impl<F: Float> Fit<F> for TargetEncoder {
    type Fitted = FittedTargetEncoder<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        Ok(self.fit_all(x, y)?.0)
    }

    /// The pipeline's training output is cross-fitted, as sklearn's
    /// `fit_transform`.
    fn fit_transform_supervised(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<(Self::Fitted, Option<Array2<F>>)> {
        let (fitted, out) = self.fit_transform(x, y)?;
        Ok((fitted, Some(out)))
    }
}

impl<F: Float> Transform<F> for FittedTargetEncoder<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let d = self.categories.len();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {d} features, got {}",
                x.ncols()
            )));
        }
        let nt = self.target_mean.len();
        let mut out = Array2::<F>::zeros((x.nrows(), d * nt));
        for (i, row) in x.rows().into_iter().enumerate() {
            for f in 0..d {
                let cat = category_index(&self.categories[f], row[f]);
                for t in 0..nt {
                    let col = f * nt + t;
                    let v = cat.map_or(self.target_mean[t], |c| self.encodings[col][c]);
                    out[[i, col]] = F::from_f64(v).unwrap();
                }
            }
        }
        Ok(out)
    }
}

impl<F: Float> FittedTargetEncoder<F> {
    /// Categories of each feature (`categories_`; a `NaN` category last).
    pub fn categories(&self) -> &[Vec<F>] {
        &self.categories
    }

    /// Encoding of each category, one vector per output column
    /// (`encodings_`).
    pub fn encodings(&self) -> &[Vec<f64>] {
        &self.encodings
    }

    /// Target mean (`target_mean_`): one value, or one per class for
    /// multiclass targets.
    pub fn target_mean(&self) -> &[f64] {
        &self.target_mean
    }

    /// Class labels (`classes_`) for binary / multiclass targets.
    pub fn classes(&self) -> Option<&[F]> {
        self.classes.as_deref()
    }

    /// The resolved target type (`target_type_`).
    pub fn target_type(&self) -> TargetType {
        self.target_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn data() -> (Array2<f64>, Array1<f64>) {
        let x = array![
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [1.0, f64::NAN],
            [2.0, 2.0],
            [0.0, 1.0],
            [2.0, 3.0],
            [1.0, 2.0],
            [0.0, 3.0],
            [2.0, f64::NAN]
        ];
        let y = array![1.5, 2.0, 3.5, 0.5, 4.0, 1.0, 5.5, 2.5, 0.0, 3.0];
        (x, y)
    }

    fn assert_rows(a: &Array2<f64>, b: &[&[f64]]) {
        assert_eq!(a.nrows(), b.len());
        for (i, row) in b.iter().enumerate() {
            assert_eq!(a.ncols(), row.len());
            for (j, v) in row.iter().enumerate() {
                assert_abs_diff_eq!(a[[i, j]], *v, epsilon = 1e-12);
            }
        }
    }

    fn query() -> Array2<f64> {
        array![[0.0, f64::NAN], [5.0, 2.0], [2.0, 7.0]]
    }

    fn check(f: &FittedTargetEncoder<f64>, enc: &[&[f64]], mean: &[f64], q: &[&[f64]]) {
        assert_eq!(f.encodings().len(), enc.len());
        for (a, b) in f.encodings().iter().zip(enc) {
            assert_eq!(a.len(), b.len());
            for (u, v) in a.iter().zip(b.iter()) {
                assert_abs_diff_eq!(*u, *v, epsilon = 1e-12);
            }
        }
        for (u, v) in f.target_mean().iter().zip(mean) {
            assert_abs_diff_eq!(*u, *v, epsilon = 1e-12);
        }
        assert_rows(&f.transform(&query()).unwrap(), q);
    }

    #[test]
    fn test_continuous_auto_smooth_matches_sklearn() {
        // sklearn 1.8: TargetEncoder().fit(X, y): encodings_, target_mean_,
        // transform(Q) (unknown categories and NaN).
        let (x, y) = data();
        let f = Fit::<f64>::fit(&TargetEncoder::new(), &x, &y).unwrap();
        assert_eq!(f.target_type(), TargetType::Continuous);
        check(
            &f,
            &[
                &[
                    1.186_141_691_860_133_5,
                    2.197_125_226_214_139,
                    3.950_308_709_687_53,
                ],
                &[
                    2.045_500_975_020_893_5,
                    2.792_408_896_994_171,
                    2.513_070_295_672_606_3,
                    1.888_529_737_717_029_8,
                ],
            ],
            &[2.35],
            &[
                &[1.186_141_691_860_133_5, 1.888_529_737_717_029_8],
                &[2.35, 2.792_408_896_994_171],
                &[3.950_308_709_687_53, 2.35],
            ],
        );
    }

    #[test]
    fn test_fixed_smooth_matches_sklearn() {
        let (x, y) = data();
        let te = TargetEncoder::new().with_smooth(TargetSmooth::Value(2.0));
        let f = Fit::<f64>::fit(&te, &x, &y).unwrap();
        check(
            &f,
            &[
                &[1.533_333_333_333_333_2, 2.24, 3.44],
                &[2.14, 2.64, 2.55, 2.05],
            ],
            &[2.35],
            &[
                &[1.533_333_333_333_333_2, 2.05],
                &[2.35, 2.64],
                &[3.44, 2.35],
            ],
        );
    }

    #[test]
    fn test_binary_matches_sklearn() {
        // Labels {2, 5}: encodings are P(y = 5 | category).
        let (x, _) = data();
        let y = array![2.0, 5.0, 5.0, 2.0, 5.0, 2.0, 5.0, 5.0, 2.0, 5.0];
        let f = Fit::<f64>::fit(&TargetEncoder::new(), &x, &y).unwrap();
        assert_eq!(f.target_type(), TargetType::Binary);
        assert_eq!(f.classes().unwrap(), &[2.0, 5.0]);
        check(
            &f,
            &[
                &[0.307_189_542_483_660_15, 0.650_943_396_226_415_1, 1.0],
                &[
                    0.396_226_415_094_339_6,
                    1.0,
                    0.534_246_575_342_465_8,
                    0.534_246_575_342_465_8,
                ],
            ],
            &[0.6],
            &[
                &[0.307_189_542_483_660_15, 0.534_246_575_342_465_8],
                &[0.6, 1.0],
                &[1.0, 0.6],
            ],
        );
    }

    #[test]
    fn test_multiclass_matches_sklearn() {
        let (x, _) = data();
        let y = array![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 2.0];
        let f = Fit::<f64>::fit(&TargetEncoder::new(), &x, &y).unwrap();
        assert_eq!(f.target_type(), TargetType::Multiclass);
        check(
            &f,
            &[
                &[0.454_128_440_366_972_5, 0.324_641_460_234_680_55, 0.0],
                &[
                    0.259_124_087_591_240_9,
                    0.324_641_460_234_680_55,
                    0.324_641_460_234_680_55,
                ],
                &[
                    0.274_509_803_921_568_65,
                    0.349_056_603_773_584_9,
                    0.603_773_584_905_660_4,
                ],
                &[
                    0.324_641_460_234_680_55,
                    0.0,
                    0.425_373_134_328_358_2,
                    0.425_373_134_328_358_2,
                ],
                &[0.0, 1.0, 0.0, 0.0],
                &[
                    0.603_773_584_905_660_4,
                    0.0,
                    0.465_753_424_657_534_2,
                    0.465_753_424_657_534_2,
                ],
            ],
            &[0.3, 0.3, 0.4],
            &[
                &[
                    0.454_128_440_366_972_5,
                    0.259_124_087_591_240_9,
                    0.274_509_803_921_568_65,
                    0.425_373_134_328_358_2,
                    0.0,
                    0.465_753_424_657_534_2,
                ],
                &[0.3, 0.3, 0.4, 0.0, 1.0, 0.0],
                &[
                    0.0,
                    0.324_641_460_234_680_55,
                    0.603_773_584_905_660_4,
                    0.3,
                    0.3,
                    0.4,
                ],
            ],
        );
    }

    #[test]
    fn test_fit_transform_is_cross_fitted() {
        // Each row is encoded by an encoder fitted without its fold.
        let (x, y) = data();
        let te = TargetEncoder::new().with_cv(3).with_random_state(7);
        let (full, out) = te.fit_transform(&x, &y).unwrap();
        let plain = Fit::<f64>::fit(&te, &x, &y).unwrap();
        assert_eq!(full.encodings(), plain.encodings());
        let folds = shuffled_k_fold(x.nrows(), 3, 7).unwrap();
        for (train, test) in folds {
            let xt = x.select(ndarray::Axis(0), &train);
            let yt = y.select(ndarray::Axis(0), &train);
            let f = Fit::<f64>::fit(&te, &xt, &yt).unwrap();
            let enc = f.transform(&x.select(ndarray::Axis(0), &test)).unwrap();
            for (r, &i) in test.iter().enumerate() {
                for j in 0..2 {
                    assert_abs_diff_eq!(out[[i, j]], enc[[r, j]], epsilon = 1e-12);
                }
            }
        }
        assert_ne!(out, plain.transform(&x).unwrap());
    }

    #[test]
    fn test_pipeline_uses_cross_fitting() {
        use anofox_ml_core::{FitTransform, Supervised};
        let (x, y) = data();
        let te = TargetEncoder::new().with_cv(3).with_random_state(1);
        let (_, expected) = te.fit_transform(&x, &y).unwrap();
        let (_, out) = FitTransform::<f64>::fit_transform(&Supervised(te), &x, Some(&y)).unwrap();
        assert_eq!(out, expected);
    }
}
