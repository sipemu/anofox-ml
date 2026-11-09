//! Single extremely randomised trees (sklearn's `ExtraTreeRegressor` /
//! `ExtraTreeClassifier`), built by [`crate::sk_random_tree`] so that a
//! seeded fit equals sklearn's tree node for node.

use crate::sk_random_tree::{fit_random_tree, SkCriterion, SkLimits, SkTree};
use crate::MaxFeatures;
use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// sklearn's tree hyperparameters shared by the random-splitter trees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtraTreeParams {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub min_weight_fraction_leaf: f64,
    /// `None`: all features.
    pub max_features: Option<MaxFeatures>,
    pub min_impurity_decrease: f64,
    pub seed: u32,
}

impl Default for ExtraTreeParams {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_weight_fraction_leaf: 0.0,
            max_features: None,
            min_impurity_decrease: 0.0,
            seed: 0,
        }
    }
}

impl ExtraTreeParams {
    /// sklearn's `BaseDecisionTree._fit` resolution.
    pub(crate) fn limits(
        &self,
        n_samples: usize,
        n_features: usize,
        w: Option<&[f64]>,
    ) -> Result<SkLimits> {
        if self.min_samples_split < 2 || self.min_samples_leaf < 1 {
            return Err(RustMlError::InvalidParameter(
                "min_samples_split must be >= 2 and min_samples_leaf >= 1".into(),
            ));
        }
        if !(0.0..=0.5).contains(&self.min_weight_fraction_leaf) {
            return Err(RustMlError::InvalidParameter(
                "min_weight_fraction_leaf must be in [0, 0.5]".into(),
            ));
        }
        let total = w.map_or(n_samples as f64, |w| w.iter().sum());
        Ok(SkLimits {
            max_depth: self.max_depth.unwrap_or(i32::MAX as usize),
            min_samples_split: self.min_samples_split.max(2 * self.min_samples_leaf),
            min_samples_leaf: self.min_samples_leaf,
            min_weight_leaf: self.min_weight_fraction_leaf * total,
            max_features: self
                .max_features
                .map_or(n_features, |m| m.resolve(n_features)),
            min_impurity_decrease: self.min_impurity_decrease,
        })
    }
}

/// `x` as sklearn's float32 row-major copy.
pub(crate) fn as_f32(x: ArrayView2<f64>) -> Vec<f32> {
    x.iter().map(|&v| v as f32).collect()
}

pub(crate) fn check_weights(w: Option<ArrayView1<f64>>, n: usize) -> Result<Option<Vec<f64>>> {
    match w {
        None => Ok(None),
        Some(w) if w.len() != n => Err(RustMlError::ShapeMismatch(format!(
            "sample_weight has {} entries for {n} rows",
            w.len()
        ))),
        Some(w) if w.iter().any(|v| !(v.is_finite() && *v >= 0.0)) => Err(
            RustMlError::InvalidParameter("sample weights must be finite and >= 0".into()),
        ),
        Some(w) => Ok(Some(w.to_vec())),
    }
}

fn check_x(x: ArrayView2<f64>, n_features: usize) -> Result<()> {
    if x.ncols() != n_features {
        return Err(RustMlError::ShapeMismatch(format!(
            "expected {n_features} features, got {}",
            x.ncols()
        )));
    }
    Ok(())
}

/// sklearn's `ExtraTreeRegressor` (squared error, `max_features = 1.0`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtraTreeRegressor {
    #[serde(flatten)]
    pub params: ExtraTreeParams,
}

/// A fitted [`ExtraTreeRegressor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedExtraTreeRegressor {
    pub tree: SkTree,
}

impl ExtraTreeRegressor {
    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedExtraTreeRegressor> {
        let (n, p) = x.dim();
        if n == 0 || p == 0 || y.len() != n {
            return Err(RustMlError::ShapeMismatch(
                "X and y must be non-empty and aligned".into(),
            ));
        }
        let w = check_weights(sample_weight, n)?;
        let limits = self.params.limits(n, p, w.as_deref())?;
        let mut rng = RandomState::new(self.params.seed);
        let tree = fit_random_tree(
            &as_f32(x),
            n,
            p,
            &y.to_vec(),
            w.as_deref(),
            SkCriterion::SquaredError,
            0,
            limits,
            &mut rng,
        );
        Ok(FittedExtraTreeRegressor { tree })
    }
}

impl FittedExtraTreeRegressor {
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        check_x(x, self.tree.n_features)?;
        Ok(x.rows()
            .into_iter()
            .map(|r| self.tree.value_of(&r.to_vec())[0])
            .collect())
    }

    /// Leaf index of each row (sklearn's `apply`).
    pub fn apply(&self, x: ArrayView2<f64>) -> Result<Array1<usize>> {
        check_x(x, self.tree.n_features)?;
        Ok(x.rows()
            .into_iter()
            .map(|r| self.tree.apply_row(&r.to_vec()))
            .collect())
    }
}

/// sklearn's `ExtraTreeClassifier` (`max_features = "sqrt"`); labels are
/// class codes `0..k` as floats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtraTreeClassifier {
    /// `Gini` or `Entropy` (sklearn's `"gini"`, `"entropy"` / `"log_loss"`).
    pub criterion: SkCriterion,
    #[serde(flatten)]
    pub params: ExtraTreeParams,
}

impl Default for ExtraTreeClassifier {
    fn default() -> Self {
        Self {
            criterion: SkCriterion::Gini,
            params: ExtraTreeParams {
                max_features: Some(MaxFeatures::Sqrt),
                ..Default::default()
            },
        }
    }
}

/// A fitted [`ExtraTreeClassifier`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedExtraTreeClassifier {
    pub tree: SkTree,
    pub classes: Vec<f64>,
}

impl ExtraTreeClassifier {
    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedExtraTreeClassifier> {
        let (n, p) = x.dim();
        if n == 0 || p == 0 || y.len() != n {
            return Err(RustMlError::ShapeMismatch(
                "X and y must be non-empty and aligned".into(),
            ));
        }
        if self.criterion == SkCriterion::SquaredError {
            return Err(RustMlError::InvalidParameter(
                "the classifier's criterion is Gini or Entropy".into(),
            ));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "labels must be finite".into(),
            ));
        }
        let mut classes = y.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let codes: Vec<f64> = y
            .iter()
            .map(|v| classes.partition_point(|c| c < v) as f64)
            .collect();
        let w = check_weights(sample_weight, n)?;
        let limits = self.params.limits(n, p, w.as_deref())?;
        let mut rng = RandomState::new(self.params.seed);
        let tree = fit_random_tree(
            &as_f32(x),
            n,
            p,
            &codes,
            w.as_deref(),
            self.criterion,
            classes.len(),
            limits,
            &mut rng,
        );
        Ok(FittedExtraTreeClassifier { tree, classes })
    }
}

impl FittedExtraTreeClassifier {
    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        check_x(x, self.tree.n_features)?;
        let k = self.classes.len();
        let mut out = Array2::<f64>::zeros((x.nrows(), k));
        for (i, r) in x.rows().into_iter().enumerate() {
            let v = self.tree.value_of(&r.to_vec());
            let s: f64 = v.iter().sum();
            for c in 0..k {
                out[[i, c]] = if s > 0.0 { v[c] / s } else { 0.0 };
            }
        }
        Ok(out)
    }

    /// The class with the largest probability (the first on ties).
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let p = self.predict_proba(x)?;
        Ok(p.rows()
            .into_iter()
            .map(|r| {
                let mut best = 0;
                for c in 1..r.len() {
                    if r[c] > r[best] {
                        best = c;
                    }
                }
                self.classes[best]
            })
            .collect())
    }

    pub fn apply(&self, x: ArrayView2<f64>) -> Result<Array1<usize>> {
        check_x(x, self.tree.n_features)?;
        Ok(x.rows()
            .into_iter()
            .map(|r| self.tree.apply_row(&r.to_vec()))
            .collect())
    }
}
