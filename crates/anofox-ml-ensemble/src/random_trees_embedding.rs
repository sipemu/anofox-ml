//! Totally random trees embedding (sklearn's `ensemble.RandomTreesEmbedding`):
//! a forest of `ExtraTreeRegressor(max_features=1)` fitted on a uniform
//! random target, each row encoded by the one-hot leaf it reaches in every
//! tree (dense here; sklearn returns it sparse by default).
//!
//! sklearn draws the target from `RandomState(seed).uniform(size=n)` and the
//! tree seeds from a fresh `RandomState(seed).randint(2**31 - 1)` per tree,
//! so a seeded fit has sklearn's trees and the same columns.

use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{FitUnsupervisedWeighted, Result, RustMlError, Transform};
use anofox_ml_trees::sk_random_tree::{fit_random_tree, SkCriterion, SkLimits, SkTree};
use ndarray::{Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// sklearn's `RandomTreesEmbedding`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RandomTreesEmbedding {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub min_weight_fraction_leaf: f64,
    pub min_impurity_decrease: f64,
    pub seed: u32,
}

impl Default for RandomTreesEmbedding {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: Some(5),
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_weight_fraction_leaf: 0.0,
            min_impurity_decrease: 0.0,
            seed: 0,
        }
    }
}

/// A fitted [`RandomTreesEmbedding`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedRandomTreesEmbedding {
    pub trees: Vec<SkTree>,
    /// Per tree, its leaf node ids in increasing order (the one-hot
    /// columns), and the first column of each tree.
    leaves: Vec<Vec<usize>>,
    offsets: Vec<usize>,
    n_outputs: usize,
}

impl FitUnsupervisedWeighted<f64> for RandomTreesEmbedding {
    type Fitted = FittedRandomTreesEmbedding;

    fn fit_unsupervised_weighted_view(
        &self,
        x: ArrayView2<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        let (n, p) = x.dim();
        if n == 0 || p == 0 {
            return Err(RustMlError::EmptyInput(
                "RandomTreesEmbedding needs data".into(),
            ));
        }
        if self.n_estimators == 0 || self.min_samples_split < 2 || self.min_samples_leaf < 1 {
            return Err(RustMlError::InvalidParameter(
                "n_estimators >= 1, min_samples_split >= 2, min_samples_leaf >= 1".into(),
            ));
        }
        let w = match sample_weight {
            Some(w) if w.len() != n => {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight has {} entries for {n} rows",
                    w.len()
                )))
            }
            Some(w) => Some(w.to_vec()),
            None => None,
        };
        let mut target_rng = RandomState::new(self.seed);
        let y: Vec<f64> = (0..n).map(|_| target_rng.random_sample()).collect();
        let xs: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let total = w.as_ref().map_or(n as f64, |w| w.iter().sum());
        let limits = SkLimits {
            max_depth: self.max_depth.unwrap_or(i32::MAX as usize),
            min_samples_split: self.min_samples_split.max(2 * self.min_samples_leaf),
            min_samples_leaf: self.min_samples_leaf,
            min_weight_leaf: self.min_weight_fraction_leaf * total,
            max_features: 1,
            min_impurity_decrease: self.min_impurity_decrease,
        };
        let mut forest_rng = RandomState::new(self.seed);
        let mut trees = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            let seed = forest_rng.randint(i32::MAX as u64) as u32;
            let mut tree_rng = RandomState::new(seed);
            trees.push(fit_random_tree(
                &xs,
                n,
                p,
                &y,
                w.as_deref(),
                SkCriterion::SquaredError,
                0,
                limits,
                &mut tree_rng,
            ));
        }
        let leaves: Vec<Vec<usize>> = trees
            .iter()
            .map(|t| {
                (0..t.nodes.len())
                    .filter(|&i| t.nodes[i].left == SkTree::LEAF)
                    .collect()
            })
            .collect();
        let mut offsets = Vec::with_capacity(leaves.len());
        let mut acc = 0;
        for l in &leaves {
            offsets.push(acc);
            acc += l.len();
        }
        Ok(FittedRandomTreesEmbedding {
            trees,
            leaves,
            offsets,
            n_outputs: acc,
        })
    }
}

impl FittedRandomTreesEmbedding {
    fn check(&self, x: ArrayView2<f64>) -> Result<()> {
        let p = self.trees[0].n_features;
        if x.ncols() != p {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {p} features, got {}",
                x.ncols()
            )));
        }
        Ok(())
    }

    /// Leaf node id of each row in each tree (sklearn's `apply`).
    pub fn apply(&self, x: ArrayView2<f64>) -> Result<Array2<usize>> {
        self.check(x)?;
        let mut out = Array2::<usize>::zeros((x.nrows(), self.trees.len()));
        for (i, r) in x.rows().into_iter().enumerate() {
            let row = r.to_vec();
            for (t, tree) in self.trees.iter().enumerate() {
                out[[i, t]] = tree.apply_row(&row);
            }
        }
        Ok(out)
    }
}

impl Transform<f64> for FittedRandomTreesEmbedding {
    /// One-hot leaf indicators, one block of columns per tree.
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let leaves = self.apply(x)?;
        let mut out = Array2::<f64>::zeros((x.nrows(), self.n_outputs));
        for i in 0..x.nrows() {
            for t in 0..self.trees.len() {
                let col = self.leaves[t].binary_search(&leaves[[i, t]]).map_err(|_| {
                    RustMlError::InvalidParameter("row reached an unknown leaf".into())
                })?;
                out[[i, self.offsets[t] + col]] = 1.0;
            }
        }
        Ok(out)
    }
}
