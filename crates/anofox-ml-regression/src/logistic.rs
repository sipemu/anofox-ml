//! Logistic regression classifier.
//!
//! sklearn-compatible `LogisticRegression` for binary and one-vs-rest
//! multiclass problems, with an L2 penalty `C`, `class_weight` and
//! `sample_weight`.
//!
//! Objective (sklearn's parametrisation, penalty on the *summed* loss):
//!
//! ```text
//! min  C Σᵢ sᵢ ℓᵢ(β, b) + ½ ‖β‖²
//! ```
//!
//! with `sᵢ = sample_weightᵢ · class_weight[yᵢ]` and the intercept
//! unpenalised. Every binary sub-fit is delegated to `anofox_regression`'s
//! logit `BinomialRegressor` (penalised IRLS, upstream `lambda = 1/C`).
//!
//! Blocked on anofox-regression (#72): the multinomial (softmax) loss and
//! L1 / elastic-net penalties need upstream support, so
//! [`MultiClass::Multinomial`] and `l1_ratio > 0` return an error and the
//! multiclass default is one-vs-rest.

use crate::convert::{coef_from_col, ndarray_to_col, ndarray_to_mat, upstream_err};
use crate::summary::{glm_summary_from_result, RegressionSummary, StatisticKind};
use anofox_ml_core::{
    ClassifierScore, DecisionFunction, Fit, FitWeighted, Predict, PredictLogProba, PredictProba,
    Result, RustMlError,
};
use anofox_regression::solvers::{
    BinomialRegressor as InnerBinomial, FittedRegressor as _, Regressor as _,
};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

fn blocked(what: &str) -> RustMlError {
    RustMlError::InvalidParameter(format!(
        "{what} is not supported yet: it needs anofox-regression support (see anofox-ml#72)"
    ))
}

/// One binary logistic fit.
struct BinaryFit {
    coef: Array1<f64>,
    intercept: Option<f64>,
    n_iter: usize,
    converged: bool,
    summary: Option<RegressionSummary>,
}

/// Multiclass strategy for `K > 2` classes (binary problems always fit a
/// single logistic model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MultiClass {
    /// Softmax (multinomial) loss — sklearn's default with `lbfgs`. Not
    /// supported yet (needs anofox-regression, #72).
    Multinomial,
    /// One binary model per class; probabilities are normalised
    /// (`sklearn.multiclass.OneVsRestClassifier(LogisticRegression())`).
    #[default]
    OneVsRest,
}

/// Per-class weights (sklearn's `class_weight`). `Balanced` uses
/// `Σw / (n_classes · Σ_{y=class} w)` (`n_samples / (n_classes ·
/// count(class))` without sample weights), as sklearn's logistic regression.
pub use anofox_ml_core::ClassWeight;

/// Logistic regression classifier.
///
/// `predict` returns class labels (the sorted unique values of `y`),
/// `predict_proba` an `(n, K)` matrix (columns follow [`FittedLogisticRegressor::classes`]).
///
/// Defaults follow sklearn (`C = 1.0`, L2, intercept) except for the
/// multiclass strategy, which is one-vs-rest until upstream offers the
/// multinomial loss (#72). Use [`LogisticRegressor::with_no_penalty`] for
/// the unpenalised maximum-likelihood fit (e.g. together with
/// `with_inference(true)` for classical Wald tests / odds-ratio CIs).
#[derive(Debug, Clone)]
pub struct LogisticRegressor {
    c: f64,
    l1_ratio: f64,
    with_intercept: bool,
    threshold: f64,
    max_iter: usize,
    tol: f64,
    multi_class: MultiClass,
    class_weight: Option<ClassWeight>,
    inference: bool,
    confidence_level: f64,
}

impl LogisticRegressor {
    pub fn new() -> Self {
        Self {
            c: 1.0,
            l1_ratio: 0.0,
            with_intercept: true,
            threshold: 0.5,
            max_iter: 1000,
            tol: 1e-8,
            multi_class: MultiClass::OneVsRest,
            class_weight: None,
            inference: false,
            confidence_level: 0.95,
        }
    }

    /// Inverse regularisation strength (sklearn's `C`). Larger C = less
    /// regularisation; `f64::INFINITY` = no penalty. Default 1.0.
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Penalty strength `λ = 1/C` on the summed loss (`λ = 0` = no penalty).
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.c = if lambda == 0.0 {
            f64::INFINITY
        } else {
            1.0 / lambda
        };
        self
    }

    /// Unpenalised maximum-likelihood fit (`C = ∞`).
    pub fn with_no_penalty(mut self) -> Self {
        self.c = f64::INFINITY;
        self
    }

    /// Elastic-net mixing (sklearn's `l1_ratio`): 0 = L2 (default). Values
    /// above 0 are not supported yet (need anofox-regression, #72).
    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    /// Set whether to include an intercept (bias) term. Default: true.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Classification threshold on P(positive class) for binary problems.
    /// Default: 0.5.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Maximum IRLS iterations. Default: 1000.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// IRLS convergence tolerance. Default: 1e-8.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Multiclass strategy for `K > 2`. Default: one-vs-rest.
    pub fn with_multi_class(mut self, multi_class: MultiClass) -> Self {
        self.multi_class = multi_class;
        self
    }

    /// Class weights (sklearn's `class_weight`). Default: none.
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = Some(class_weight);
        self
    }

    /// Compute Wald inference (binary fits only). Default false.
    /// With a finite `C` the covariance is the penalised one
    /// `(XᵀWX + I/C)⁻¹`; use [`Self::with_no_penalty`] for classical SEs.
    pub fn with_inference(mut self, inference: bool) -> Self {
        self.inference = inference;
        self
    }

    /// Confidence level of the summary intervals. Default 0.95.
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Upstream ridge strength for sklearn's `C` on the summed loss.
    fn lambda(&self) -> f64 {
        if self.c.is_infinite() {
            0.0
        } else {
            1.0 / self.c
        }
    }

    fn validate(&self) -> Result<()> {
        if self.c.is_nan() || self.c <= 0.0 {
            return Err(RustMlError::InvalidParameter(format!(
                "C must be positive, got {}",
                self.c
            )));
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err(RustMlError::InvalidParameter(format!(
                "l1_ratio must be in [0, 1], got {}",
                self.l1_ratio
            )));
        }
        if self.l1_ratio > 0.0 && self.c.is_finite() {
            return Err(blocked("an L1 / elastic-net penalty"));
        }
        if self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "max_iter must be >= 1".into(),
            ));
        }
        if self.tol.is_nan() || self.tol <= 0.0 {
            return Err(RustMlError::InvalidParameter("tol must be positive".into()));
        }
        crate::summary::validate_level(self.confidence_level)
    }

    fn fit_binary(
        &self,
        x: &Array2<f64>,
        y01: &Array1<f64>,
        w: &Array1<f64>,
        inference: bool,
    ) -> Result<BinaryFit> {
        let mut b = InnerBinomial::logistic()
            .with_intercept(self.with_intercept)
            .lambda(self.lambda())
            .max_iterations(self.max_iter)
            .tolerance(self.tol)
            .compute_inference(inference)
            .confidence_level(self.confidence_level);
        if w.iter().any(|&v| v != 1.0) {
            b = b.weights(ndarray_to_col(w));
        }
        let inner = b
            .build()
            .fit(&ndarray_to_mat(x), &ndarray_to_col(y01))
            .map_err(upstream_err)?;
        let res = inner.result();
        let summary = if inference {
            Some(glm_summary_from_result(res, StatisticKind::Z, 1.0)?)
        } else {
            None
        };
        Ok(BinaryFit {
            coef: coef_from_col(&res.coefficients),
            intercept: res.intercept,
            n_iter: inner.iterations,
            converged: inner.converged,
            summary,
        })
    }
}

impl Default for LogisticRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted logistic regression classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedLogisticRegressor {
    classes: Vec<f64>,
    /// `(1, p)` for binary problems, `(K, p)` otherwise (sklearn's `coef_`).
    coef: Array2<f64>,
    /// Length 1 (binary) or `K` (sklearn's `intercept_`).
    intercepts: Array1<f64>,
    fit_intercept: bool,
    multi_class: MultiClass,
    threshold: f64,
    n_features: usize,
    n_iter: usize,
    converged: bool,
    #[serde(default)]
    summary: Option<RegressionSummary>,
}

impl FittedLogisticRegressor {
    /// Coefficients of the (binary) model, excluding the intercept. For
    /// multiclass fits this is the first class's row; see [`Self::coef`].
    pub fn coefficients(&self) -> Array1<f64> {
        self.coef.row(0).to_owned()
    }

    /// Intercept of the (binary) model; first class's intercept for
    /// multiclass fits (see [`Self::intercepts`]).
    pub fn intercept(&self) -> Option<f64> {
        if self.fit_intercept {
            Some(self.intercepts[0])
        } else {
            None
        }
    }

    /// Coefficient matrix, `(1, p)` for binary and `(K, p)` for multiclass.
    pub fn coef(&self) -> &Array2<f64> {
        &self.coef
    }

    /// Intercept vector (zeros when fitted without an intercept).
    pub fn intercepts(&self) -> &Array1<f64> {
        &self.intercepts
    }

    /// Sorted class labels; column order of `predict_proba`.
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }

    /// Iterations used (maximum over the one-vs-rest sub-fits).
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Whether the solver met its tolerance.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Coefficient table of a binary fit (requires `with_inference(true)`);
    /// [`RegressionSummary::odds_ratios`] gives odds ratios with CIs.
    pub fn summary(&self) -> Result<&RegressionSummary> {
        self.summary
            .as_ref()
            .ok_or_else(crate::summary::not_computed)
    }

    fn check(&self, x: &Array2<f64>) -> Result<()> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(())
    }

    fn scores(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.check(x)?;
        let mut s = x.dot(&self.coef.t());
        s += &self.intercepts.view().insert_axis(Axis(0));
        Ok(s)
    }

    /// Class probabilities, shape `(n, K)` (columns follow [`Self::classes`]).
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let s = self.scores(x)?;
        let n = s.nrows();
        let k = self.classes.len();
        let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
        let mut out = Array2::zeros((n, k));
        if k == 2 {
            for i in 0..n {
                let p = sigmoid(s[[i, 0]]);
                out[[i, 0]] = 1.0 - p;
                out[[i, 1]] = p;
            }
            return Ok(out);
        }
        for i in 0..n {
            match self.multi_class {
                MultiClass::Multinomial => {
                    let mx = s.row(i).fold(f64::NEG_INFINITY, |m, v| m.max(*v));
                    let mut z = 0.0;
                    for c in 0..k {
                        out[[i, c]] = (s[[i, c]] - mx).exp();
                        z += out[[i, c]];
                    }
                    for c in 0..k {
                        out[[i, c]] /= z;
                    }
                }
                MultiClass::OneVsRest => {
                    let mut z = 0.0;
                    for c in 0..k {
                        out[[i, c]] = sigmoid(s[[i, c]]);
                        z += out[[i, c]];
                    }
                    for c in 0..k {
                        out[[i, c]] = if z > 0.0 {
                            out[[i, c]] / z
                        } else {
                            1.0 / k as f64
                        };
                    }
                }
            }
        }
        Ok(out)
    }

    /// Decision scores: `(n, 1)` log-odds of the positive class for binary
    /// problems (sklearn's 1-D output as a column), `(n, K)` linear scores
    /// otherwise.
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.scores(x)
    }

    /// Classification accuracy.
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        ClassifierScore::score(self, x, y)
    }
}

/// Map `y` to class indices of the sorted unique labels.
fn encode_labels(y: &Array1<f64>) -> Result<(Vec<f64>, Vec<usize>)> {
    if y.iter().any(|v| !v.is_finite()) {
        return Err(RustMlError::InvalidParameter(
            "class labels must be finite".into(),
        ));
    }
    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).expect("finite labels"));
    classes.dedup();
    let idx = y
        .iter()
        .map(|v| {
            classes
                .binary_search_by(|c| c.partial_cmp(v).expect("finite labels"))
                .expect("label present")
        })
        .collect();
    Ok((classes, idx))
}

impl Fit<f64> for LogisticRegressor {
    type Fitted = FittedLogisticRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.fit_weighted(x, y, None)
    }
}

impl FitWeighted<f64> for LogisticRegressor {
    type Fitted = FittedLogisticRegressor;

    fn fit_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Self::Fitted> {
        crate::convert::check_xy(x, y)?;
        self.validate()?;
        if x.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "X contains non-finite values".into(),
            ));
        }
        let (classes, labels) = encode_labels(y)?;
        let k = classes.len();
        if k < 2 {
            return Err(RustMlError::InvalidParameter(
                "logistic regression needs at least two classes".into(),
            ));
        }
        let n = y.len();
        let mut w = crate::convert::resolve_sample_weight(sample_weight, n)?;
        if let Some(cw) = &self.class_weight {
            let per_class: Vec<f64> = match cw {
                ClassWeight::Balanced => {
                    // sklearn >= 1.6: weighted class counts when sample
                    // weights are given.
                    let mut counts = vec![0.0; k];
                    for (i, &l) in labels.iter().enumerate() {
                        counts[l] += w[i];
                    }
                    let total: f64 = counts.iter().sum();
                    counts
                        .iter()
                        .map(|&c| if c > 0.0 { total / (k as f64 * c) } else { 0.0 })
                        .collect()
                }
                ClassWeight::Manual(pairs) => {
                    let mut out = vec![1.0; k];
                    for &(label, weight) in pairs {
                        if !weight.is_finite() || weight < 0.0 {
                            return Err(RustMlError::InvalidParameter(
                                "class weights must be finite and non-negative".into(),
                            ));
                        }
                        if let Some(c) = classes.iter().position(|c| *c == label) {
                            out[c] = weight;
                        }
                    }
                    out
                }
            };
            for i in 0..n {
                w[i] *= per_class[labels[i]];
            }
            if w.sum().is_nan() || w.sum() <= 0.0 {
                return Err(RustMlError::InvalidParameter(
                    "all effective sample weights are zero".into(),
                ));
            }
        }
        let p = x.ncols();
        if self.inference && k > 2 {
            return Err(RustMlError::InvalidParameter(
                "inference is available for binary fits only".into(),
            ));
        }

        let (coef, intercepts, n_iter, converged, summary) = if k == 2 {
            let y01 = Array1::from_iter(labels.iter().map(|&l| l as f64));
            let fit = self.fit_binary(x, &y01, &w, self.inference)?;
            let coef = fit.coef.insert_axis(Axis(0));
            let b = Array1::from_elem(1, fit.intercept.unwrap_or(0.0));
            (coef, b, fit.n_iter, fit.converged, fit.summary)
        } else {
            match self.multi_class {
                MultiClass::Multinomial => return Err(blocked("the multinomial loss")),
                MultiClass::OneVsRest => {
                    let mut coef = Array2::zeros((k, p));
                    let mut b = Array1::zeros(k);
                    let mut it = 0;
                    let mut conv = true;
                    for c in 0..k {
                        let yc =
                            Array1::from_iter(
                                labels.iter().map(|&l| if l == c { 1.0 } else { 0.0 }),
                            );
                        let fit = self.fit_binary(x, &yc, &w, false)?;
                        coef.row_mut(c).assign(&fit.coef);
                        b[c] = fit.intercept.unwrap_or(0.0);
                        it = it.max(fit.n_iter);
                        conv &= fit.converged;
                    }
                    (coef, b, it, conv, None)
                }
            }
        };

        Ok(FittedLogisticRegressor {
            classes,
            coef,
            intercepts,
            fit_intercept: self.with_intercept,
            multi_class: self.multi_class,
            threshold: self.threshold,
            n_features: p,
            n_iter,
            converged,
            summary,
        })
    }
}

impl Predict<f64> for FittedLogisticRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if self.classes.len() == 2 {
            let s = self.scores(x)?;
            return Ok(s.column(0).mapv(|v| {
                let p = 1.0 / (1.0 + (-v).exp());
                if p >= self.threshold {
                    self.classes[1]
                } else {
                    self.classes[0]
                }
            }));
        }
        let proba = self.predict_proba(x)?;
        Ok(Array1::from_iter(proba.rows().into_iter().map(|row| {
            let mut best = 0;
            for c in 1..row.len() {
                if row[c] > row[best] {
                    best = c;
                }
            }
            self.classes[best]
        })))
    }
}

impl PredictProba<f64> for FittedLogisticRegressor {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        FittedLogisticRegressor::predict_proba(self, x)
    }
}

impl PredictLogProba<f64> for FittedLogisticRegressor {}

impl DecisionFunction<f64> for FittedLogisticRegressor {
    fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.scores(x)
    }
}

impl ClassifierScore<f64> for FittedLogisticRegressor {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn toy() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((8, 1), vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])
            .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn test_logistic_basic() {
        let (x, y) = toy();
        let fitted = LogisticRegressor::new().fit(&x, &y).unwrap();
        assert!(fitted.coefficients()[0] > 0.0);
        assert!(fitted.intercept().is_some());
        assert!(fitted.converged());
    }

    #[test]
    fn test_predict_labels() {
        let (x, y) = toy();
        let fitted = LogisticRegressor::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds, y);
    }

    #[test]
    fn test_predict_proba_shape_and_range() {
        let (x, y) = toy();
        let fitted = LogisticRegressor::new().fit(&x, &y).unwrap();
        let probs = fitted.predict_proba(&x).unwrap();
        assert_eq!(probs.dim(), (8, 2));
        for row in probs.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-12);
            assert!(row.iter().all(|p| (0.0..=1.0).contains(p)));
        }
        let df = DecisionFunction::decision_function(&fitted, &x).unwrap();
        assert_eq!(df.dim(), (8, 1));
        let lp = fitted.predict_log_proba(&x).unwrap();
        assert!((lp[[0, 0]].exp() - probs[[0, 0]]).abs() < 1e-12);
    }

    #[test]
    fn test_score() {
        let (x, y) = toy();
        let fitted = LogisticRegressor::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.score(&x, &y).unwrap(), 1.0);
    }

    #[test]
    fn test_l2_regularization() {
        let x = Array2::from_shape_vec((8, 1), vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])
            .unwrap();
        let y = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let no_reg = LogisticRegressor::new()
            .with_no_penalty()
            .fit(&x, &y)
            .unwrap();
        let l2 = LogisticRegressor::new()
            .with_lambda(10.0)
            .fit(&x, &y)
            .unwrap();
        assert!(l2.coefficients()[0].abs() < no_reg.coefficients()[0].abs());
    }

    #[test]
    fn test_arbitrary_binary_labels() {
        let (x, y) = toy();
        let y2 = y.mapv(|v| if v == 1.0 { 7.0 } else { -2.0 });
        let fitted = LogisticRegressor::new().fit(&x, &y2).unwrap();
        assert_eq!(fitted.classes(), &[-2.0, 7.0]);
        assert_eq!(fitted.predict(&x).unwrap(), y2);
    }

    #[test]
    fn test_invalid_params() {
        let (x, y) = toy();
        assert!(LogisticRegressor::new().with_c(0.0).fit(&x, &y).is_err());
        assert!(LogisticRegressor::new()
            .with_l1_ratio(1.5)
            .fit(&x, &y)
            .is_err());
        assert!(LogisticRegressor::new().fit(&x, &Array1::ones(8)).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0.0, 1.0];
        assert!(LogisticRegressor::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_empty_input() {
        let x = Array2::<f64>::zeros((0, 1));
        let y = Array1::<f64>::zeros(0);
        assert!(LogisticRegressor::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_class_weight_equals_sample_weight() {
        let x = Array2::from_shape_vec((8, 1), vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])
            .unwrap();
        let y = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let a = LogisticRegressor::new()
            .with_class_weight(ClassWeight::Manual(vec![(1.0, 3.0)]))
            .fit(&x, &y)
            .unwrap();
        let sw = y.mapv(|v| if v == 1.0 { 3.0 } else { 1.0 });
        let b = LogisticRegressor::new()
            .fit_weighted(&x, &y, Some(&sw))
            .unwrap();
        assert!((a.coefficients()[0] - b.coefficients()[0]).abs() < 1e-10);
        assert!((a.intercept().unwrap() - b.intercept().unwrap()).abs() < 1e-10);
    }
}
