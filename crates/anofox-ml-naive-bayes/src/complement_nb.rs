//! Complement Naive Bayes (Rennie et al., 2003), as sklearn's `ComplementNB`.

use anofox_ml_core::{
    Fit, FitWeighted, Float, PartialFit, Predict, PredictLogProba, PredictProba, Result,
};
use ndarray::{Array1, Array2, Axis};

use crate::common::{
    argmax_labels, check_alpha, check_n_features, class_log_prior, log_proba_from_jll,
    proba_from_jll,
};
use crate::discrete::{accumulate, check_non_negative, Counts};

fn default_true() -> bool {
    true
}

/// Complement Naive Bayes classifier parameters (unfitted state).
///
/// Estimates feature weights from the *complement* of each class, which
/// makes it well suited to imbalanced text classification. Feature values
/// must be non-negative (counts / tf-idf).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComplementNB {
    /// Additive (Laplace/Lidstone) smoothing parameter (>= 0).
    pub alpha: f64,
    /// If `false` and `alpha < 1e-10`, alpha is clamped to `1e-10`.
    #[serde(default = "default_true")]
    pub force_alpha: bool,
    /// Only used in the edge case of a single class in the training set.
    #[serde(default = "default_true")]
    pub fit_prior: bool,
    /// Fixed class priors; only used for a single training class (as sklearn).
    #[serde(default)]
    pub class_prior: Option<Vec<f64>>,
    /// Whether to apply the second weight normalisation of Rennie et al.
    #[serde(default)]
    pub norm: bool,
}

impl ComplementNB {
    /// Create a new `ComplementNB` with sklearn's defaults
    /// (`alpha = 1.0`, `norm = false`).
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            force_alpha: true,
            fit_prior: true,
            class_prior: None,
            norm: false,
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

    /// Set whether class priors are learned (single-class edge case only).
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    /// Fix the class priors (single-class edge case only).
    pub fn with_class_prior(mut self, class_prior: Option<Vec<f64>>) -> Self {
        self.class_prior = class_prior;
        self
    }

    /// Enable the second weight normalisation.
    pub fn with_norm(mut self, norm: bool) -> Self {
        self.norm = norm;
        self
    }

    fn finish<F: Float>(&self, counts: Counts<F>) -> Result<FittedComplementNB<F>> {
        let alpha = F::from_f64(check_alpha(self.alpha, self.force_alpha)?).unwrap();
        let log_class_prior = class_log_prior(
            &counts.class_count,
            self.fit_prior,
            self.class_prior.as_deref(),
        )?;
        let feature_all = counts.feature_count.sum_axis(Axis(0));
        // comp_count[c, j] = Σ_c' fc[c', j] + alpha − fc[c, j]
        let mut logged = counts.feature_count.mapv(|v| -v) + &feature_all.mapv(|v| v + alpha);
        for mut row in logged.rows_mut() {
            let log_total = row.sum().ln();
            row.mapv_inplace(|v| v.ln() - log_total);
        }
        let feature_log_prob = if self.norm {
            for mut row in logged.rows_mut() {
                let s = row.sum();
                row.mapv_inplace(|v| v / s);
            }
            logged
        } else {
            logged.mapv(|v| -v)
        };
        Ok(FittedComplementNB {
            class_labels: counts.classes,
            log_class_prior: log_class_prior.to_vec(),
            feature_log_prob,
            class_count: counts.class_count,
            feature_count: counts.feature_count,
        })
    }

    /// Incremental fit with optional per-sample weights.
    pub fn partial_fit_weighted<F: Float>(
        &self,
        state: Option<FittedComplementNB<F>>,
        x: &Array2<F>,
        y: &Array1<F>,
        classes: Option<&[F]>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedComplementNB<F>> {
        check_alpha(self.alpha, self.force_alpha)?;
        check_non_negative(x, "ComplementNB")?;
        let prev = state
            .map(|s| Counts::from_parts(s.class_labels, s.class_count, s.feature_count))
            .transpose()?;
        let counts = accumulate(prev, x, y, classes, sample_weight)?;
        self.finish(counts)
    }
}

impl Default for ComplementNB {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Complement Naive Bayes classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedComplementNB<F: Float> {
    class_labels: Vec<F>,
    log_class_prior: Vec<F>,
    /// Complement feature weights (sklearn's `feature_log_prob_`).
    feature_log_prob: Array2<F>,
    class_count: Array1<F>,
    feature_count: Array2<F>,
}

impl<F: Float> FittedComplementNB<F> {
    /// Returns the unique sorted class labels.
    pub fn classes(&self) -> &[F] {
        &self.class_labels
    }

    /// Log class priors (only used when a single class was seen).
    pub fn log_class_prior(&self) -> &[F] {
        &self.log_class_prior
    }

    /// Complement feature weights, shape `(n_classes, n_features)`.
    pub fn feature_log_prob(&self) -> &Array2<F> {
        &self.feature_log_prob
    }

    /// Weighted number of samples per class.
    pub fn class_count(&self) -> &Array1<F> {
        &self.class_count
    }

    /// Weighted feature totals per class.
    pub fn feature_count(&self) -> &Array2<F> {
        &self.feature_count
    }

    /// Joint log-likelihood `X · feature_log_probᵀ` (plus the log prior when
    /// only one class exists, as sklearn).
    pub fn joint_log_likelihood(&self, x: &Array2<F>) -> Result<Array2<F>> {
        check_n_features(x, self.feature_log_prob.ncols())?;
        let mut jll = x.dot(&self.feature_log_prob.t());
        if self.class_labels.len() == 1 {
            jll.mapv_inplace(|v| v + self.log_class_prior[0]);
        }
        Ok(jll)
    }
}

impl<F: Float> FitWeighted<F> for ComplementNB {
    type Fitted = FittedComplementNB<F>;

    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        self.partial_fit_weighted(None, x, y, None, sample_weight)
    }
}

impl<F: Float> Fit<F> for ComplementNB {
    type Fitted = FittedComplementNB<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.partial_fit_weighted(None, x, y, None, None)
    }
}

impl<F: Float> PartialFit<F> for ComplementNB {
    type Fitted = FittedComplementNB<F>;

    fn partial_fit(
        &self,
        state: Option<Self::Fitted>,
        x: &Array2<F>,
        y: &Array1<F>,
        classes: Option<&[F]>,
    ) -> Result<Self::Fitted> {
        self.partial_fit_weighted(state, x, y, classes, None)
    }
}

impl<F: Float> Predict<F> for FittedComplementNB<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let jll = self.joint_log_likelihood(x)?;
        Ok(argmax_labels(&jll, &self.class_labels))
    }
}

impl<F: Float> PredictProba<F> for FittedComplementNB<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Ok(proba_from_jll(self.joint_log_likelihood(x)?))
    }
}

impl<F: Float> PredictLogProba<F> for FittedComplementNB<F> {
    fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Ok(log_proba_from_jll(self.joint_log_likelihood(x)?))
    }
}

impl<F: Float> anofox_ml_core::ClassifierScore<F> for FittedComplementNB<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::assert_close2;
    use ndarray::array;

    fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
        let x = array![
            [3.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 4.0],
            [0.0, 4.0, 2.0, 0.0],
            [1.0, 3.0, 3.0, 1.0],
            [0.0, 0.0, 5.0, 1.0],
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 2.0]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 2.0];
        let w = array![1.0, 0.5, 2.0, 1.0, 1.5, 1.0, 0.25];
        let t = array![
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 2.0, 1.0],
            [2.0, 2.0, 2.0, 2.0]
        ];
        (x, y, w, t)
    }

    // Reference values generated with scikit-learn 1.8.0.
    #[test]
    fn matches_sklearn_default() {
        let (x, y, _, t) = data();
        let m: FittedComplementNB<f64> = ComplementNB::new().fit(&x, &y).unwrap();
        assert_close2(
            &m.predict_proba(&t).unwrap(),
            &[
                &[0.8611325369719441, 0.04702390282431521, 0.09184356020374071],
                &[0.07314770863373668, 0.6204339108872524, 0.30641838047901004],
                &[0.5792361835319277, 0.2601568076584929, 0.16060700880957976],
            ],
            1e-12,
            "proba",
        );
        assert_close2(
            &m.predict_log_proba(&t).unwrap(),
            &[
                &[-0.14950685263566843, -3.057099235820232, -2.387668581877602],
                &[-2.615274476220743, -0.47733618945843403, -1.182803854406421],
                &[
                    -0.5460449682943427,
                    -1.3464707233833515,
                    -1.8287948370171279,
                ],
            ],
            1e-12,
            "log_proba",
        );
        assert_eq!(m.predict(&t).unwrap(), array![0.0, 1.0, 0.0]);
    }

    #[test]
    fn matches_sklearn_norm_weighted() {
        let (x, y, w, t) = data();
        let m: FittedComplementNB<f64> = ComplementNB::new()
            .with_norm(true)
            .with_alpha(0.5)
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        assert_close2(
            m.feature_log_prob(),
            &[
                &[
                    0.42335347696615794,
                    0.14885730594065766,
                    0.10435303077561198,
                    0.32343618631757254,
                ],
                &[
                    0.18765361209314185,
                    0.4102102341639786,
                    0.16577173166791812,
                    0.23636442207496136,
                ],
                &[
                    0.2349816624219916,
                    0.18022486139798927,
                    0.25439888926240556,
                    0.3303945869176135,
                ],
            ],
            1e-12,
            "feature_log_prob",
        );
        assert_close2(
            &m.predict_proba(&t).unwrap(),
            &[
                &[0.40746739709506224, 0.24790503879459347, 0.3446275641103442],
                &[0.23835502539230033, 0.416644258051206, 0.3450007165564938],
                &[0.33333333333333337, 0.3333333333333333, 0.3333333333333333],
            ],
            1e-12,
            "proba",
        );
        assert_eq!(m.predict(&t).unwrap(), array![0.0, 1.0, 0.0]);
    }

    #[test]
    fn partial_fit_equals_fit_and_rejects_negative() {
        let (x, y, _, t) = data();
        let nb = ComplementNB::new();
        let full: FittedComplementNB<f64> = nb.fit(&x, &y).unwrap();
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
        let d = full.predict_proba(&t).unwrap() - b.predict_proba(&t).unwrap();
        assert!(d.iter().all(|v| v.abs() < 1e-14));
        let r: Result<FittedComplementNB<f64>> = nb.fit(&x.mapv(|v| v - 1.0), &y);
        assert!(r.is_err());
    }
}
