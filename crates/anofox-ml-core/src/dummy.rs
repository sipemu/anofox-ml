//! Baselines that ignore the features (sklearn's `DummyClassifier` /
//! `DummyRegressor`): a model has to beat these to be worth anything.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{Fit, FitWeighted, Predict, PredictProba, Result, RustMlError};

fn check_xy(x: ArrayView2<f64>, y: ArrayView1<f64>, w: Option<ArrayView1<f64>>) -> Result<()> {
    if y.is_empty() {
        return Err(RustMlError::EmptyInput("y must not be empty".into()));
    }
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "x has {} rows but y has {}",
            x.nrows(),
            y.len()
        )));
    }
    if let Some(w) = w {
        if w.len() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "sample_weight has {} entries for {} rows",
                w.len(),
                y.len()
            )));
        }
        if w.iter().any(|&v| !v.is_finite() || v < 0.0) {
            return Err(RustMlError::InvalidParameter(
                "sample_weight must be finite and non-negative".into(),
            ));
        }
        if w.sum() <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "sample_weight must not sum to zero".into(),
            ));
        }
    }
    Ok(())
}

// ============================================================
// Classifier
// ============================================================

/// How a [`DummyClassifier`] predicts.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum DummyClassifierStrategy {
    /// Always the most frequent class; probabilities one-hot on it.
    MostFrequent,
    /// Always the most frequent class; probabilities are the class priors.
    #[default]
    Prior,
    /// Random classes drawn from the priors; probabilities one-hot on the
    /// draw.
    Stratified,
    /// Random classes, uniformly; probabilities `1 / n_classes`.
    Uniform,
    /// Always the given label, which must occur in the training data.
    Constant(f64),
}

/// Baseline classifier (sklearn's `DummyClassifier`). The random strategies
/// draw from a generator seeded with `seed` at every prediction call, so
/// repeated calls give the same result.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DummyClassifier {
    pub strategy: DummyClassifierStrategy,
    pub seed: u64,
}

impl DummyClassifier {
    pub fn new(strategy: DummyClassifierStrategy) -> Self {
        Self { strategy, seed: 0 }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Fitted [`DummyClassifier`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedDummyClassifier {
    strategy: DummyClassifierStrategy,
    seed: u64,
    classes: Vec<f64>,
    class_prior: Vec<f64>,
}

impl FittedDummyClassifier {
    /// Sorted distinct training labels; the columns of `predict_proba`.
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }

    /// (Weighted) relative class frequencies.
    pub fn class_prior(&self) -> &[f64] {
        &self.class_prior
    }

    /// Index of the most frequent class (the first one on ties, as numpy's
    /// `argmax`).
    fn mode(&self) -> usize {
        self.class_prior
            .iter()
            .enumerate()
            .fold(0, |b, (k, &p)| if p > self.class_prior[b] { k } else { b })
    }

    fn constant_index(&self) -> Option<usize> {
        match self.strategy {
            DummyClassifierStrategy::Constant(c) => self.classes.iter().position(|&v| v == c),
            _ => None,
        }
    }
}

impl FitWeighted<f64> for DummyClassifier {
    type Fitted = FittedDummyClassifier;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedDummyClassifier> {
        check_xy(x, y, sample_weight)?;
        if y.iter().any(|v| v.is_nan()) {
            return Err(RustMlError::InvalidParameter("y contains NaN".into()));
        }
        let mut classes = y.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let mut mass = vec![0.0; classes.len()];
        for (i, &v) in y.iter().enumerate() {
            let k = classes.partition_point(|&c| c < v);
            mass[k] += sample_weight.map_or(1.0, |w| w[i]);
        }
        let total: f64 = mass.iter().sum();
        let class_prior = mass.iter().map(|m| m / total).collect();
        if let DummyClassifierStrategy::Constant(c) = self.strategy {
            if !classes.contains(&c) {
                return Err(RustMlError::InvalidParameter(format!(
                    "the constant {c} does not occur in the training labels {classes:?}"
                )));
            }
        }
        Ok(FittedDummyClassifier {
            strategy: self.strategy,
            seed: self.seed,
            classes,
            class_prior,
        })
    }
}

impl Fit<f64> for DummyClassifier {
    type Fitted = FittedDummyClassifier;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedDummyClassifier> {
        self.fit_weighted_view(x, y, None)
    }
}

impl FittedDummyClassifier {
    /// Class index per row.
    fn indices(&self, n: usize) -> Vec<usize> {
        let k = self.classes.len();
        let mut rng = StdRng::seed_from_u64(self.seed);
        match self.strategy {
            DummyClassifierStrategy::MostFrequent | DummyClassifierStrategy::Prior => {
                vec![self.mode(); n]
            }
            DummyClassifierStrategy::Constant(_) => {
                vec![self.constant_index().expect("checked at fit"); n]
            }
            DummyClassifierStrategy::Uniform => (0..n).map(|_| rng.gen_range(0..k)).collect(),
            DummyClassifierStrategy::Stratified => (0..n)
                .map(|_| {
                    let u: f64 = rng.gen();
                    let mut acc = 0.0;
                    for (c, &p) in self.class_prior.iter().enumerate() {
                        acc += p;
                        if u < acc {
                            return c;
                        }
                    }
                    k - 1
                })
                .collect(),
        }
    }
}

impl Predict<f64> for FittedDummyClassifier {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(self
            .indices(x.nrows())
            .into_iter()
            .map(|k| self.classes[k])
            .collect())
    }
}

impl PredictProba<f64> for FittedDummyClassifier {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let (n, k) = (x.nrows(), self.classes.len());
        Ok(match self.strategy {
            DummyClassifierStrategy::Prior => {
                Array2::from_shape_fn((n, k), |(_, c)| self.class_prior[c])
            }
            DummyClassifierStrategy::Uniform => Array2::from_elem((n, k), 1.0 / k as f64),
            _ => {
                let idx = self.indices(n);
                Array2::from_shape_fn((n, k), |(i, c)| if idx[i] == c { 1.0 } else { 0.0 })
            }
        })
    }
}

// ============================================================
// Regressor
// ============================================================

/// How a [`DummyRegressor`] predicts.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum DummyRegressorStrategy {
    /// The (weighted) mean of the targets.
    #[default]
    Mean,
    /// The median (weighted: sklearn's lower weighted percentile).
    Median,
    /// The quantile at level `q` in [0, 1] (numpy's linear interpolation;
    /// weighted: the lower weighted percentile).
    Quantile(f64),
    /// A fixed value.
    Constant(f64),
}

/// Baseline regressor (sklearn's `DummyRegressor`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DummyRegressor {
    pub strategy: DummyRegressorStrategy,
}

impl DummyRegressor {
    pub fn new(strategy: DummyRegressorStrategy) -> Self {
        Self { strategy }
    }
}

/// Fitted [`DummyRegressor`]: the constant it predicts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedDummyRegressor {
    constant: f64,
}

impl FittedDummyRegressor {
    pub fn constant(&self) -> f64 {
        self.constant
    }
}

/// numpy's default (linear) percentile of `v` at level `q` in [0, 1].
fn linear_quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = q * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

/// sklearn's `_weighted_percentile` (`average=False`, numpy's
/// `inverted_cdf`): the first sorted value whose cumulative weight reaches
/// `q` times the total; leading zero weights are skipped at `q = 0`.
fn weighted_lower_quantile(v: &[f64], w: &[f64], q: f64) -> f64 {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let total: f64 = w.iter().sum();
    let mut target = q * total;
    if target == 0.0 {
        target = f64::from_bits(1); // nextafter(0, 1)
    }
    let mut cum = 0.0;
    for &i in &order {
        cum += w[i];
        if cum >= target {
            return v[i];
        }
    }
    v[*order.last().expect("non-empty")]
}

impl FitWeighted<f64> for DummyRegressor {
    type Fitted = FittedDummyRegressor;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedDummyRegressor> {
        check_xy(x, y, sample_weight)?;
        let yv = y.to_vec();
        let wv = sample_weight.map(|w| w.to_vec());
        let quantile = |q: f64| match &wv {
            Some(w) => weighted_lower_quantile(&yv, w, q),
            None => linear_quantile(&yv, q),
        };
        let constant = match self.strategy {
            DummyRegressorStrategy::Mean => match &wv {
                Some(w) => {
                    yv.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.iter().sum::<f64>()
                }
                None => yv.iter().sum::<f64>() / yv.len() as f64,
            },
            DummyRegressorStrategy::Median => quantile(0.5),
            DummyRegressorStrategy::Quantile(q) => {
                if !(0.0..=1.0).contains(&q) {
                    return Err(RustMlError::InvalidParameter(format!(
                        "quantile must be in [0, 1], got {q}"
                    )));
                }
                quantile(q)
            }
            DummyRegressorStrategy::Constant(c) => c,
        };
        Ok(FittedDummyRegressor { constant })
    }
}

impl Fit<f64> for DummyRegressor {
    type Fitted = FittedDummyRegressor;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedDummyRegressor> {
        self.fit_weighted_view(x, y, None)
    }
}

impl Predict<f64> for FittedDummyRegressor {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(Array1::from_elem(x.nrows(), self.constant))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn classifier_strategies() {
        let x = Array2::zeros((6, 1));
        let y = array![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];
        let prior = DummyClassifier::default().fit(&x, &y).unwrap();
        assert_eq!(prior.classes(), &[1.0, 2.0, 3.0]);
        assert_eq!(prior.predict(&x).unwrap(), Array1::from_elem(6, 2.0));
        assert_eq!(
            prior.predict_proba(&x).unwrap().row(0).to_vec(),
            vec![2.0 / 6.0, 0.5, 1.0 / 6.0]
        );
        let mf = DummyClassifier::new(DummyClassifierStrategy::MostFrequent)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(
            mf.predict_proba(&x).unwrap().row(3).to_vec(),
            vec![0.0, 1.0, 0.0]
        );
        let c = DummyClassifier::new(DummyClassifierStrategy::Constant(3.0))
            .fit(&x, &y)
            .unwrap();
        assert_eq!(c.predict(&x).unwrap(), Array1::from_elem(6, 3.0));
        assert!(DummyClassifier::new(DummyClassifierStrategy::Constant(9.0))
            .fit(&x, &y)
            .is_err());
        // Weighted priors.
        let w = array![1.0, 1.0, 1.0, 10.0, 1.0, 1.0];
        let wp = DummyClassifier::default()
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        assert_eq!(wp.predict(&x).unwrap()[0], 3.0);
    }

    #[test]
    fn random_strategies_follow_their_distribution_and_repeat() {
        let n = 20000;
        let x = Array2::zeros((n, 1));
        let y = Array1::from_shape_fn(n, |i| if i % 4 == 0 { 1.0 } else { 0.0 });
        let s = DummyClassifier::new(DummyClassifierStrategy::Stratified)
            .with_seed(1)
            .fit(&x, &y)
            .unwrap();
        let p = s.predict(&x).unwrap();
        let ones = p.iter().filter(|&&v| v == 1.0).count() as f64 / n as f64;
        assert!((ones - 0.25).abs() < 0.02, "{ones}");
        assert_eq!(p, s.predict(&x).unwrap());
        let u = DummyClassifier::new(DummyClassifierStrategy::Uniform)
            .fit(&x, &y)
            .unwrap();
        let ones = u.predict(&x).unwrap().iter().filter(|&&v| v == 1.0).count() as f64 / n as f64;
        assert!((ones - 0.5).abs() < 0.02, "{ones}");
        assert!(u.predict_proba(&x).unwrap().iter().all(|&v| v == 0.5));
    }

    #[test]
    fn regressor_strategies() {
        let x = Array2::zeros((5, 1));
        let y = array![1.0, 4.0, 2.0, 8.0, 5.0];
        let fit = |s| DummyRegressor::new(s).fit(&x, &y).unwrap().constant();
        assert_eq!(fit(DummyRegressorStrategy::Mean), 4.0);
        assert_eq!(fit(DummyRegressorStrategy::Median), 4.0);
        assert_eq!(fit(DummyRegressorStrategy::Quantile(0.25)), 2.0);
        assert!((fit(DummyRegressorStrategy::Quantile(0.9)) - 6.8).abs() < 1e-12);
        assert_eq!(fit(DummyRegressorStrategy::Constant(-1.0)), -1.0);
        assert!(DummyRegressor::new(DummyRegressorStrategy::Quantile(1.5))
            .fit(&x, &y)
            .is_err());
        let w = array![1.0, 1.0, 1.0, 1.0, 10.0];
        let wm = DummyRegressor::new(DummyRegressorStrategy::Median)
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        assert_eq!(wm.constant(), 5.0);
    }
}
