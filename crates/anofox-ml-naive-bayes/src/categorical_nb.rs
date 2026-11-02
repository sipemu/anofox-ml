//! Categorical Naive Bayes, as sklearn's `CategoricalNB`.
//!
//! Every feature is a categorical variable encoded as a non-negative integer
//! `0, 1, …, n_categories − 1` (e.g. the output of an ordinal encoder).

use anofox_ml_core::{
    Fit, FitWeighted, Float, PartialFit, Predict, PredictLogProba, PredictProba, Result,
    RustMlError,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Axis, Ix1, Ix2};

use crate::common::{
    argmax_labels, check_alpha, check_n_features, check_partial_fit_classes, check_xy,
    class_log_prior, encode_labels, log_proba_from_jll, proba_from_jll, sorted_unique,
    weighted_one_hot,
};

fn default_true() -> bool {
    true
}

/// Minimum number of categories per feature (sklearn's `min_categories`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MinCategories {
    /// The same minimum for every feature.
    All(usize),
    /// One minimum per feature.
    PerFeature(Vec<usize>),
}

/// Categorical Naive Bayes classifier parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CategoricalNB {
    /// Additive (Laplace/Lidstone) smoothing parameter (>= 0).
    pub alpha: f64,
    /// If `false` and `alpha < 1e-10`, alpha is clamped to `1e-10`.
    #[serde(default = "default_true")]
    pub force_alpha: bool,
    /// Whether to learn class priors from the data (`false`: uniform prior).
    #[serde(default = "default_true")]
    pub fit_prior: bool,
    /// Fixed class prior probabilities (sorted class order).
    #[serde(default)]
    pub class_prior: Option<Vec<f64>>,
    /// Minimum number of categories per feature; useful when some categories
    /// do not occur in the training data.
    #[serde(default)]
    pub min_categories: Option<MinCategories>,
}

impl CategoricalNB {
    /// Create a new `CategoricalNB` with sklearn's defaults (`alpha = 1.0`).
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            force_alpha: true,
            fit_prior: true,
            class_prior: None,
            min_categories: None,
        }
    }

    /// Set the additive smoothing parameter.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set `force_alpha` (see field docs).
    pub fn with_force_alpha(mut self, force_alpha: bool) -> Self {
        self.force_alpha = force_alpha;
        self
    }

    /// Set whether class priors are learned.
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    /// Fix the class priors.
    pub fn with_class_prior(mut self, class_prior: Option<Vec<f64>>) -> Self {
        self.class_prior = class_prior;
        self
    }

    /// Set the minimum number of categories per feature.
    pub fn with_min_categories(mut self, min_categories: Option<MinCategories>) -> Self {
        self.min_categories = min_categories;
        self
    }

    /// `X.max(axis=0) + 1`, raised to `min_categories`.
    fn n_categories(&self, codes: &[Vec<usize>], n_features: usize) -> Result<Vec<usize>> {
        let mut n_cat = vec![0usize; n_features];
        for row in codes {
            for (j, &c) in row.iter().enumerate() {
                n_cat[j] = n_cat[j].max(c + 1);
            }
        }
        match &self.min_categories {
            None => {}
            Some(MinCategories::All(m)) => n_cat.iter_mut().for_each(|v| *v = (*v).max(*m)),
            Some(MinCategories::PerFeature(m)) => {
                if m.len() != n_features {
                    return Err(RustMlError::InvalidParameter(format!(
                        "min_categories has {} entries but X has {n_features} features",
                        m.len()
                    )));
                }
                n_cat
                    .iter_mut()
                    .zip(m)
                    .for_each(|(v, &mi)| *v = (*v).max(mi));
            }
        }
        Ok(n_cat)
    }

    /// Incremental fit with optional per-sample weights.
    pub fn partial_fit_weighted<'a, 'b, F: Float>(
        &self,
        state: Option<FittedCategoricalNB<F>>,
        x: impl AsArray<'a, F, Ix2>,
        y: impl AsArray<'b, F, Ix1>,
        classes: Option<&[F]>,
        sample_weight: Option<ArrayView1<F>>,
    ) -> Result<FittedCategoricalNB<F>> {
        let x: ArrayView2<F> = x.into();
        let y: ArrayView1<F> = y.into();
        let alpha = F::from_f64(check_alpha(self.alpha, self.force_alpha)?).unwrap();
        check_xy(x, y, sample_weight)?;
        let n_features = x.ncols();
        let codes = encode_categories(x)?;

        let mut model = match state {
            Some(s) => {
                check_n_features(x, s.category_count.len())?;
                check_partial_fit_classes(&s.class_labels, classes)?;
                s
            }
            None => {
                let class_labels = match classes {
                    Some(c) => sorted_unique(c),
                    None => sorted_unique(&y.to_vec()),
                };
                let k = class_labels.len();
                FittedCategoricalNB {
                    class_labels,
                    class_count: Array1::zeros(k),
                    category_count: vec![Array2::zeros((k, 0)); n_features],
                    feature_log_prob: Vec::new(),
                    log_class_prior: Vec::new(),
                    n_categories: Vec::new(),
                }
            }
        };
        let k = model.class_labels.len();
        let idx = encode_labels(y, &model.class_labels)?;
        let y1h = weighted_one_hot(&idx, k, sample_weight);
        model.class_count = &model.class_count + &y1h.sum_axis(Axis(0));

        // n_categories_ follows the current batch (and min_categories), as
        // sklearn; the count tables only ever grow.
        let n_cat = self.n_categories(&codes, n_features)?;
        for (j, &nc) in n_cat.iter().enumerate() {
            let counts = &mut model.category_count[j];
            if counts.ncols() < nc {
                let mut grown = Array2::<F>::zeros((k, nc));
                grown
                    .slice_mut(ndarray::s![.., ..counts.ncols()])
                    .assign(counts);
                *counts = grown;
            }
            for (i, row) in codes.iter().enumerate() {
                let c = idx[i];
                counts[[c, row[j]]] += y1h[[i, c]];
            }
        }
        model.n_categories = n_cat;

        model.feature_log_prob = model
            .category_count
            .iter()
            .map(|cc| {
                let mut smoothed = cc.mapv(|v| v + alpha);
                for mut r in smoothed.rows_mut() {
                    let log_total = r.sum().ln();
                    r.mapv_inplace(|v| v.ln() - log_total);
                }
                smoothed
            })
            .collect();
        model.log_class_prior = class_log_prior(
            &model.class_count,
            self.fit_prior,
            self.class_prior.as_deref(),
        )?
        .to_vec();
        Ok(model)
    }
}

impl Default for CategoricalNB {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate that `x` holds non-negative integer category codes.
fn encode_categories<F: Float>(x: ArrayView2<F>) -> Result<Vec<Vec<usize>>> {
    x.rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .map(|&v| {
                    if !v.is_finite() || v < F::zero() || v.fract() != F::zero() {
                        Err(RustMlError::InvalidParameter(format!(
                            "CategoricalNB expects non-negative integer category codes, got {v}"
                        )))
                    } else {
                        Ok(v.to_usize().unwrap_or(usize::MAX))
                    }
                })
                .collect()
        })
        .collect()
}

/// Fitted Categorical Naive Bayes classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedCategoricalNB<F: Float> {
    class_labels: Vec<F>,
    class_count: Array1<F>,
    /// Per feature: weighted category counts, shape `(n_classes, n_categories_j)`.
    category_count: Vec<Array2<F>>,
    /// Per feature: `log P(x_j = t | c)`, shape `(n_classes, n_categories_j)`.
    feature_log_prob: Vec<Array2<F>>,
    log_class_prior: Vec<F>,
    n_categories: Vec<usize>,
}

impl<F: Float> FittedCategoricalNB<F> {
    /// Returns the unique sorted class labels.
    pub fn classes(&self) -> &[F] {
        &self.class_labels
    }

    /// Weighted number of samples per class.
    pub fn class_count(&self) -> &Array1<F> {
        &self.class_count
    }

    /// Per-feature category counts (sklearn's `category_count_`).
    pub fn category_count(&self) -> &[Array2<F>] {
        &self.category_count
    }

    /// Per-feature log category probabilities (sklearn's `feature_log_prob_`).
    pub fn feature_log_prob(&self) -> &[Array2<F>] {
        &self.feature_log_prob
    }

    /// Log class priors.
    pub fn log_class_prior(&self) -> &[F] {
        &self.log_class_prior
    }

    /// Number of categories per feature (sklearn's `n_categories_`).
    pub fn n_categories(&self) -> &[usize] {
        &self.n_categories
    }

    /// Joint log-likelihood `Σ_j log P(x_j | c) + log P(c)`. Category codes
    /// outside the range seen during training are an error.
    pub fn joint_log_likelihood<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array2<F>> {
        let x: ArrayView2<F> = x.into();
        check_n_features(x, self.feature_log_prob.len())?;
        let codes = encode_categories(x)?;
        let k = self.class_labels.len();
        let mut jll = Array2::<F>::zeros((x.nrows(), k));
        for (i, row) in codes.iter().enumerate() {
            for (j, &t) in row.iter().enumerate() {
                let flp = &self.feature_log_prob[j];
                if t >= flp.ncols() {
                    return Err(RustMlError::InvalidParameter(format!(
                        "category {t} of feature {j} was not seen during fit \
                         (n_categories = {}); use min_categories",
                        flp.ncols()
                    )));
                }
                for c in 0..k {
                    jll[[i, c]] += flp[[c, t]];
                }
            }
            for c in 0..k {
                jll[[i, c]] += self.log_class_prior[c];
            }
        }
        Ok(jll)
    }
}

impl<F: Float> FitWeighted<F> for CategoricalNB {
    type Fitted = FittedCategoricalNB<F>;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        sample_weight: Option<ArrayView1<F>>,
    ) -> Result<Self::Fitted> {
        self.partial_fit_weighted(None, x, y, None, sample_weight)
    }
}

impl<F: Float> Fit<F> for CategoricalNB {
    type Fitted = FittedCategoricalNB<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        self.partial_fit_weighted(None, x, y, None, None)
    }
}

impl<F: Float> PartialFit<F> for CategoricalNB {
    type Fitted = FittedCategoricalNB<F>;

    fn partial_fit_view(
        &self,
        state: Option<Self::Fitted>,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        classes: Option<&[F]>,
    ) -> Result<Self::Fitted> {
        self.partial_fit_weighted(state, x, y, classes, None)
    }
}

impl<F: Float> Predict<F> for FittedCategoricalNB<F> {
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        let jll = self.joint_log_likelihood(x)?;
        Ok(argmax_labels(&jll, &self.class_labels))
    }
}

impl<F: Float> PredictProba<F> for FittedCategoricalNB<F> {
    fn predict_proba_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        Ok(proba_from_jll(self.joint_log_likelihood(x)?))
    }
}

impl<F: Float> PredictLogProba<F> for FittedCategoricalNB<F> {
    fn predict_log_proba_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        Ok(log_proba_from_jll(self.joint_log_likelihood(x)?))
    }
}

impl<F: Float> anofox_ml_core::ClassifierScore<F> for FittedCategoricalNB<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::assert_close2;
    use ndarray::array;

    fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
        let x = array![
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 2.0, 1.0],
            [2.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0];
        let w = array![1.0, 2.0, 0.5, 1.0, 1.0, 3.0, 1.0, 0.5];
        let t = array![
            [0.0, 1.0, 2.0],
            [2.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
            [3.0, 0.0, 0.0]
        ];
        (x, y, w, t)
    }

    const DEFAULT_PROBA: [&[f64]; 3] = [
        &[0.823271130625686, 0.13721185510428108, 0.03951701427003293],
        &[0.1008878127522195, 0.2017756255044392, 0.6973365617433416],
        &[0.1625487646293889, 0.6501950585175551, 0.18725617685305593],
    ];

    // Reference values generated with scikit-learn 1.8.0.
    #[test]
    fn matches_sklearn_min_categories_weighted() {
        let (x, y, w, t) = data();
        let m: FittedCategoricalNB<f64> = CategoricalNB::new()
            .with_alpha(0.5)
            .with_min_categories(Some(MinCategories::PerFeature(vec![4, 3, 3])))
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        assert_eq!(m.n_categories(), &[4, 3, 3]);
        assert_close2(
            &m.predict_proba(&t).unwrap(),
            &[
                &[
                    0.8842696253717878,
                    0.10918022925508804,
                    0.006550145373124347,
                ],
                &[
                    0.011027591483998742,
                    0.016338839627312408,
                    0.9726335688886889,
                ],
                &[0.2457372116349046, 0.6068204613841522, 0.1474423269809428],
                &[0.16382480775660319, 0.6068204613841522, 0.22935473085924446],
            ],
            1e-12,
            "proba",
        );
        assert_close2(
            &m.predict_log_proba(&t).unwrap(),
            &[
                &[-0.12299325677270811, -2.214755282840585, -5.028268035211139],
                &[
                    -4.5073548300324005,
                    -4.114210206312277,
                    -0.027747867019654926,
                ],
                &[
                    -1.4034925594171117,
                    -0.4995223119309973,
                    -1.9143181831831022,
                ],
                &[
                    -1.8089576675252754,
                    -0.4995223119309973,
                    -1.4724854309040625,
                ],
            ],
            1e-12,
            "log_proba",
        );
    }

    #[test]
    fn matches_sklearn_default_and_partial_fit() {
        let (x, y, _, t) = data();
        let t3 = t.slice(ndarray::s![..3, ..]).to_owned();
        let nb = CategoricalNB::new();
        let m: FittedCategoricalNB<f64> = nb.fit(&x, &y).unwrap();
        assert_close2(&m.predict_proba(&t3).unwrap(), &DEFAULT_PROBA, 1e-12, "fit");
        // Category 3 of feature 0 was never seen and min_categories is unset.
        assert!(m.predict(&t).is_err());

        let a = nb
            .partial_fit(
                None,
                &x.slice(ndarray::s![..4, ..]).to_owned(),
                &y.slice(ndarray::s![..4]).to_owned(),
                Some(&[0.0, 1.0, 2.0]),
            )
            .unwrap();
        let b = nb
            .partial_fit(
                Some(a),
                &x.slice(ndarray::s![4.., ..]).to_owned(),
                &y.slice(ndarray::s![4..]).to_owned(),
                None,
            )
            .unwrap();
        assert_close2(
            &b.predict_proba(&t3).unwrap(),
            &DEFAULT_PROBA,
            1e-12,
            "partial_fit",
        );
    }

    #[test]
    fn rejects_non_integer_or_negative_codes() {
        let (_, y, _, _) = data();
        let mut bad = Array2::<f64>::zeros((8, 3));
        bad[[0, 0]] = 0.5;
        assert!(CategoricalNB::new().fit(&bad, &y).is_err());
        bad[[0, 0]] = -1.0;
        assert!(CategoricalNB::new().fit(&bad, &y).is_err());
    }
}
