//! Custom objectives (#213): gradients and hessians from the caller, one or
//! several raw scores per row, as LightGBM's `objective="none"` with
//! `Booster.update(fobj)` (`num_class = K` for `K` outputs).

use super::*;
use ndarray::ArrayViewMut2;

/// A loss given by its gradients and hessians in the raw scores.
///
/// The objective holds whatever side data it needs (labels, per-row errors,
/// …); [`LgbmRegressor::fit_custom`] passes it only the raw scores.
///
/// ```
/// use anofox_ml_ensemble::{LgbmCustomObjective, LgbmRegressor};
/// use ndarray::{array, Array1, ArrayView2, ArrayViewMut2};
///
/// /// Squared error against `y`.
/// struct L2(Array1<f64>);
/// impl LgbmCustomObjective for L2 {
///     fn gradients(&self, raw: ArrayView2<f64>, mut g: ArrayViewMut2<f64>, mut h: ArrayViewMut2<f64>) {
///         for i in 0..raw.nrows() {
///             g[[i, 0]] = raw[[i, 0]] - self.0[i];
///             h[[i, 0]] = 1.0;
///         }
///     }
/// }
///
/// let x = array![[0.0], [1.0], [2.0], [3.0]];
/// let model = LgbmRegressor::new()
///     .with_n_estimators(5)
///     .with_min_child_samples(1)
///     .fit_custom(&x, &L2(array![0.0, 0.0, 1.0, 1.0]))
///     .unwrap();
/// assert_eq!(model.predict_raw(&x).unwrap().dim(), (4, 1));
/// ```
pub trait LgbmCustomObjective: Send + Sync {
    /// Raw scores per row: 1 (default) or `K` (one tree per output and
    /// iteration, as LightGBM's multiclass).
    fn n_outputs(&self) -> usize {
        1
    }

    /// The initial raw score of every output (LightGBM: 0 for custom
    /// objectives).
    fn init_score(&self) -> Vec<f64> {
        vec![0.0; self.n_outputs()]
    }

    /// Fill `grad` and `hess` (`n_rows × n_outputs`) from the current raw
    /// scores `raw` (same shape). The engine keeps them as float32, as
    /// LightGBM does.
    fn gradients(&self, raw: ArrayView2<f64>, grad: ArrayViewMut2<f64>, hess: ArrayViewMut2<f64>);
}

/// A booster fitted on a custom objective: its raw scores, without a link.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLgbmBooster {
    /// One tree per iteration for every output.
    tree_sets: Vec<Vec<TreeNode>>,
    bins: Vec<FeatureBins>,
    init_score: Vec<f64>,
    learning_rate: f64,
    n_features: usize,
    feature_gain: Vec<f64>,
    feature_split_count: Vec<u32>,
}

impl FittedLgbmBooster {
    /// Raw scores per row.
    pub fn n_outputs(&self) -> usize {
        self.tree_sets.len()
    }

    /// Boosting iterations kept (training stops early once no output's tree
    /// can split, as in LightGBM).
    pub fn n_iterations(&self) -> usize {
        self.tree_sets.first().map_or(0, Vec::len)
    }

    /// Gain importances, normalized to sum 1.
    pub fn feature_importances(&self) -> Array1<f64> {
        let sum: f64 = self.feature_gain.iter().sum();
        if sum > 0.0 {
            Array1::from_iter(self.feature_gain.iter().map(|g| g / sum))
        } else {
            Array1::zeros(self.n_features)
        }
    }

    /// Number of splits per feature.
    pub fn feature_split_counts(&self) -> &[u32] {
        &self.feature_split_count
    }

    /// The raw scores, `n_rows × n_outputs`.
    pub fn predict_raw<'a, T: LgbmFloat>(
        &self,
        x: impl AsArray<'a, T, Ix2>,
    ) -> Result<Array2<f64>> {
        let x: ArrayView2<T> = x.into();
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let k = self.n_outputs();
        let binned = bin_matrix(x, &self.bins);
        let mut raw = Array2::zeros((x.nrows(), k));
        use rayon::prelude::*;
        raw.as_slice_mut()
            .unwrap()
            .par_chunks_mut(k.max(1))
            .with_min_len(PREDICT_BLOCK_ROWS)
            .enumerate()
            .for_each(|(i, row)| {
                for (c, ts) in self.tree_sets.iter().enumerate() {
                    row[c] = self.init_score[c];
                    for tree in ts {
                        row[c] += self.learning_rate * tree.predict_bins(row_of(&binned, i));
                    }
                }
            });
        Ok(raw)
    }
}

impl LgbmRegressor {
    /// Fit on a custom objective (LightGBM's `objective="none"` with the
    /// gradients of `objective`), one tree per output and iteration.
    ///
    /// The tree parameters, categorical features, monotone constraints,
    /// bagging, GOSS and the feature fractions apply as for the built-in
    /// objectives; `objective`, DART and early stopping do not (the
    /// objective owns the loss, so sample weights belong in its gradients).
    /// As in LightGBM, an output whose tree cannot split adds 0 that
    /// iteration, and training stops once no output's tree can split.
    pub fn fit_custom<'a, T: LgbmFloat>(
        &self,
        x: impl AsArray<'a, T, Ix2>,
        objective: &dyn LgbmCustomObjective,
    ) -> Result<FittedLgbmBooster> {
        let data = self.dataset(x)?;
        with_fit_threads(self.n_jobs, || self.fit_custom_inner(&data, objective))
    }

    /// [`fit_custom`](Self::fit_custom) on a dataset binned by
    /// [`dataset`](Self::dataset) with the same binning settings.
    pub fn fit_custom_dataset(
        &self,
        data: &LgbmDataset,
        objective: &dyn LgbmCustomObjective,
    ) -> Result<FittedLgbmBooster> {
        data.check_settings(&self.binning_settings())?;
        with_fit_threads(self.n_jobs, || self.fit_custom_inner(data, objective))
    }

    fn fit_custom_inner(
        &self,
        data: &LgbmDataset,
        objective: &dyn LgbmCustomObjective,
    ) -> Result<FittedLgbmBooster> {
        let (n, p) = (data.n_rows(), data.n_features());
        let k = objective.n_outputs();
        if n == 0 || p == 0 {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }
        if k == 0 {
            return Err(RustMlError::InvalidParameter(
                "a custom objective needs at least one output".into(),
            ));
        }
        let init_score = objective.init_score();
        if init_score.len() != k {
            return Err(RustMlError::InvalidParameter(format!(
                "init_score has {} entries for {k} outputs",
                init_score.len()
            )));
        }
        if self.boosting_type == BoostingType::Dart {
            return Err(RustMlError::InvalidParameter(
                "custom objectives do not support DART".into(),
            ));
        }
        if !self.monotone_constraints.is_empty() && self.monotone_constraints.len() != p {
            return Err(RustMlError::InvalidParameter(format!(
                "monotone_constraints has {} entries but X has {p} features",
                self.monotone_constraints.len()
            )));
        }
        validate_max_bins(self.max_bins)?;
        validate_num_leaves(self.num_leaves)?;
        self.sampling().validate()?;

        let (binned_x, bins) = (&data.binned, data.bins.clone());
        let binned_rows = binned_x.as_standard_layout().into_owned();
        let bundles = self
            .enable_bundle
            .then(|| FeatureBundles::find(&binned_x))
            .flatten();
        let mut samplers = LgbmSamplers::new(self.sampling(), &binned_x, None);
        let mut feature_rng = feature_generator(1.0, 0);
        let params = LgbmParams {
            has_missing: data.has_missing.clone(),
            fixed_bins: data.fixed_bins.clone(),
            ..self.params()
        };

        let mut feature_gain = vec![0.0; p];
        let mut feature_split_count = vec![0u32; p];
        let mut tree_sets: Vec<Vec<TreeNode>> = vec![Vec::with_capacity(self.n_estimators); k];
        // Raw scores, one column per output (row-major for the objective).
        let mut raw = Array2::from_shape_fn((n, k), |(_, c)| init_score[c]);
        let mut grad = Array2::zeros((n, k));
        let mut hess = Array2::zeros((n, k));
        let mut scores: Vec<Vec<f64>> = (0..k).map(|c| raw.column(c).to_vec()).collect();

        for iter in 0..self.n_estimators {
            objective.gradients(raw.view(), grad.view_mut(), hess.view_mut());
            let (mut all_g, mut all_h): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (0..k)
                .map(|c| {
                    let mut g = grad.column(c).to_vec();
                    let mut h = hess.column(c).to_vec();
                    round_to_f32(&mut g);
                    round_to_f32(&mut h);
                    (g, h)
                })
                .unzip();
            let row_indices = {
                let mut g: Vec<&mut [f64]> = all_g.iter_mut().map(|v| v.as_mut_slice()).collect();
                let mut h: Vec<&mut [f64]> = all_h.iter_mut().map(|v| v.as_mut_slice()).collect();
                samplers.rows(iter, n, &mut g, &mut h)
            };
            let mut any_split = false;
            let mut round = Vec::with_capacity(k);
            for c in 0..k {
                let feature_indices = samplers.tree_features(p);
                let (tree, leaf_rows) = build_lgbm_tree(
                    &binned_x,
                    bundles.as_ref(),
                    &all_g[c],
                    &all_h[c],
                    row_indices.clone(),
                    &feature_indices,
                    &bins,
                    &params,
                    &mut feature_gain,
                    &mut feature_split_count,
                    &mut feature_rng,
                    samplers.node_sampler(),
                );
                if matches!(tree, TreeNode::Leaf { .. }) {
                    // LightGBM keeps an unsplit tree as a zero leaf.
                    round.push(TreeNode::Leaf { value: 0.0 });
                    continue;
                }
                any_split = true;
                add_leaf_scores(
                    &mut scores[c][..],
                    &tree,
                    &leaf_rows,
                    &binned_rows,
                    self.learning_rate,
                );
                round.push(tree);
            }
            // No output could split: LightGBM stops and drops the round
            // (except the first, which holds the initial scores).
            if !any_split {
                if iter == 0 {
                    for (ts, t) in tree_sets.iter_mut().zip(round) {
                        ts.push(t);
                    }
                }
                break;
            }
            for (ts, t) in tree_sets.iter_mut().zip(round) {
                ts.push(t);
            }
            for (c, s) in scores.iter().enumerate() {
                raw.column_mut(c).assign(&ArrayView1::from(s));
            }
        }

        Ok(FittedLgbmBooster {
            tree_sets,
            bins,
            init_score,
            learning_rate: self.learning_rate,
            n_features: p,
            feature_gain,
            feature_split_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gradients `raw - target` per output, or none at all.
    struct Shift {
        target: Vec<f64>,
        flat: bool,
    }

    impl LgbmCustomObjective for Shift {
        fn n_outputs(&self) -> usize {
            self.target.len()
        }
        fn gradients(
            &self,
            raw: ArrayView2<f64>,
            mut g: ArrayViewMut2<f64>,
            mut h: ArrayViewMut2<f64>,
        ) {
            for ((i, c), v) in g.indexed_iter_mut() {
                // The row index makes the target depend on the features.
                let t = if self.flat {
                    0.0
                } else {
                    self.target[c] * i as f64
                };
                *v = raw[[i, c]] - t;
                h[[i, c]] = 1.0;
            }
        }
    }

    fn x() -> Array2<f64> {
        Array2::from_shape_fn((80, 2), |(i, j)| (i * (j + 1)) as f64)
    }

    #[test]
    fn fewer_than_two_leaves_is_rejected() {
        let shift = Shift {
            target: vec![1.0],
            flat: false,
        };
        for bad in [0usize, 1] {
            let r = LgbmRegressor::new()
                .with_num_leaves(bad)
                .fit_custom(&x(), &shift);
            assert!(matches!(r, Err(RustMlError::InvalidParameter(_))), "{bad}");
        }
    }

    #[test]
    fn unsplittable_gradients_stop_after_the_first_round() {
        let shift = Shift {
            target: vec![0.0; 2],
            flat: true,
        };
        let m = LgbmRegressor::new()
            .with_n_estimators(10)
            .fit_custom(&x(), &shift)
            .unwrap();
        assert_eq!(m.n_iterations(), 1);
        assert!(m.predict_raw(&x()).unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn multi_output_fits_each_output_and_round_trips() {
        let shift = Shift {
            target: vec![1.0, -2.0],
            flat: false,
        };
        let m = LgbmRegressor::new()
            .with_n_estimators(50)
            .with_min_child_samples(2)
            .fit_custom(&x(), &shift)
            .unwrap();
        let raw = m.predict_raw(&x()).unwrap();
        assert_eq!(raw.dim(), (80, 2));
        // The outputs move apart, as their targets do.
        assert!(raw[[79, 0]] > 30.0 && raw[[79, 1]] < -60.0);
        let bytes = anofox_ml_core::persistence::to_binary_bytes(&m).unwrap();
        let back: FittedLgbmBooster =
            anofox_ml_core::persistence::from_binary_bytes(&bytes).unwrap();
        assert_eq!(back.predict_raw(&x()).unwrap(), raw);
    }

    #[test]
    fn rejects_dart_and_bad_outputs() {
        let shift = Shift {
            target: vec![],
            flat: true,
        };
        assert!(LgbmRegressor::new().fit_custom(&x(), &shift).is_err());
        let one = Shift {
            target: vec![1.0],
            flat: false,
        };
        let dart = LgbmRegressor::new().with_boosting_type(BoostingType::Dart);
        assert!(dart.fit_custom(&x(), &one).is_err());
    }

    #[test]
    fn predict_raw_is_the_score_before_the_link() {
        let x = x();
        let y = Array1::from_shape_fn(80, |i| (i % 7) as f64);
        let reg = LgbmRegressor::new()
            .with_n_estimators(5)
            .with_objective(LgbmObjective::Poisson)
            .fit(&x, &y)
            .unwrap();
        let raw = reg.predict_raw(&x).unwrap();
        assert_eq!(raw.mapv(f64::exp), reg.predict(&x).unwrap());
        let labels = y.mapv(|v| (v / 3.0).floor());
        let clf = LgbmClassifier::new()
            .with_n_estimators(5)
            .fit(&x, &labels)
            .unwrap();
        let raw = clf.predict_raw(&x).unwrap();
        let proba = clf.predict_proba(&x).unwrap();
        assert_eq!(raw.ncols(), 3);
        let mut row = vec![0.0; 3];
        multiclass_proba(raw.row(0).as_slice().unwrap(), false, &mut row);
        assert!(row
            .iter()
            .zip(proba.row(0))
            .all(|(a, b)| (a - b).abs() < 1e-12));
    }
}
