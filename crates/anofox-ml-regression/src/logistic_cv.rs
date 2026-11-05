//! Logistic regression with `C` chosen by cross-validation (sklearn's
//! `LogisticRegressionCV`): unshuffled stratified `cv` folds, the score of
//! each candidate summed over the folds, and a refit on all rows at the
//! best `C`.

use anofox_ml_core::{
    stratified_k_fold_ordered, Fit, FitWeighted, Predict, PredictProba, Result, RustMlError,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::{FittedLogisticRegressor, LogisticRegressor};

/// Candidate inverse regularisation strengths.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CGrid {
    /// `n` values `logspace(−4, 4, n)` (sklearn's integer `Cs`).
    Count(usize),
    Values(Vec<f64>),
}

impl Default for CGrid {
    fn default() -> Self {
        CGrid::Count(10)
    }
}

impl CGrid {
    fn values(&self) -> Result<Vec<f64>> {
        let v = match self {
            CGrid::Count(0) => {
                return Err(RustMlError::InvalidParameter("Cs must be positive".into()))
            }
            CGrid::Count(1) => vec![1e-4],
            CGrid::Count(n) => (0..*n)
                .map(|i| 10f64.powf(-4.0 + 8.0 * i as f64 / (*n - 1) as f64))
                .collect(),
            CGrid::Values(v) => v.clone(),
        };
        if v.is_empty() || v.iter().any(|&c| !(c > 0.0 && c.is_finite())) {
            return Err(RustMlError::InvalidParameter(format!(
                "Cs must be positive, got {v:?}"
            )));
        }
        Ok(v)
    }
}

/// Fold score of a [`LogisticRegressionCV`] candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LogisticCvScoring {
    /// Accuracy (sklearn's default, `LogisticRegression.score`).
    #[default]
    Accuracy,
    /// Negative log loss (`scoring="neg_log_loss"`).
    NegLogLoss,
}

/// Logistic regression with cross-validated `C`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LogisticRegressionCV {
    /// Every other setting of the model (its `C` is replaced).
    #[serde(flatten)]
    pub base: LogisticRegressor,
    pub cs: CGrid,
    pub cv: usize,
    pub scoring: LogisticCvScoring,
}

impl Default for LogisticRegressionCV {
    fn default() -> Self {
        Self {
            base: LogisticRegressor::default(),
            cs: CGrid::default(),
            cv: 5,
            scoring: LogisticCvScoring::Accuracy,
        }
    }
}

/// Fitted [`LogisticRegressionCV`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedLogisticRegressionCV {
    model: FittedLogisticRegressor,
    cs: Vec<f64>,
    c: f64,
    /// `n_folds × n_Cs` fold scores.
    scores: Array2<f64>,
}

impl FittedLogisticRegressionCV {
    /// The refitted model at the best `C`.
    pub fn model(&self) -> &FittedLogisticRegressor {
        &self.model
    }

    /// The best `C` (sklearn's `C_`).
    pub fn c(&self) -> f64 {
        self.c
    }

    pub fn cs(&self) -> &[f64] {
        &self.cs
    }

    /// Fold scores, `n_folds × n_Cs` (sklearn's `scores_`).
    pub fn scores(&self) -> &Array2<f64> {
        &self.scores
    }
}

fn fold_score(
    scoring: LogisticCvScoring,
    m: &FittedLogisticRegressor,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
) -> Result<f64> {
    match scoring {
        LogisticCvScoring::Accuracy => {
            let p = m.predict(x)?;
            Ok(p.iter().zip(y).filter(|(a, b)| a == b).count() as f64 / y.len() as f64)
        }
        LogisticCvScoring::NegLogLoss => {
            let proba = PredictProba::predict_proba(m, x)?;
            let classes = m.classes();
            let eps = f64::EPSILON;
            let mut total = 0.0;
            for (i, &label) in y.iter().enumerate() {
                let k = classes.iter().position(|&c| c == label).ok_or_else(|| {
                    RustMlError::InvalidParameter(format!("label {label} unseen in training fold"))
                })?;
                total -= proba[[i, k]].clamp(eps, 1.0 - eps).ln();
            }
            Ok(-total / y.len() as f64)
        }
    }
}

impl FitWeighted<f64> for LogisticRegressionCV {
    type Fitted = FittedLogisticRegressionCV;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedLogisticRegressionCV> {
        if y.len() != x.nrows() {
            return Err(RustMlError::ShapeMismatch(format!(
                "x has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        let cs = self.cs.values()?;
        let splits = stratified_k_fold_ordered(y, self.cv)?;
        let mut scores = Array2::zeros((splits.len(), cs.len()));
        for (f, (train, test)) in splits.iter().enumerate() {
            let (xt, yt) = (x.select(Axis(0), train), y.select(Axis(0), train));
            let wt = sample_weight.map(|w| w.select(Axis(0), train));
            let (xv, yv) = (x.select(Axis(0), test), y.select(Axis(0), test));
            for (j, &c) in cs.iter().enumerate() {
                let m = self.base.clone().with_c(c).fit_weighted_view(
                    xt.view(),
                    yt.view(),
                    wt.as_ref().map(|w| w.view()),
                )?;
                scores[[f, j]] = fold_score(self.scoring, &m, xv.view(), yv.view())?;
            }
        }
        let sums = scores.sum_axis(Axis(0));
        let best = sums
            .iter()
            .enumerate()
            .fold(0, |b, (j, &v)| if v > sums[b] { j } else { b });
        let c = cs[best];
        let model = self
            .base
            .clone()
            .with_c(c)
            .fit_weighted_view(x, y, sample_weight)?;
        Ok(FittedLogisticRegressionCV {
            model,
            cs,
            c,
            scores,
        })
    }
}

impl Fit<f64> for LogisticRegressionCV {
    type Fitted = FittedLogisticRegressionCV;

    fn fit_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<FittedLogisticRegressionCV> {
        self.fit_weighted_view(x, y, None)
    }
}

impl Predict<f64> for FittedLogisticRegressionCV {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.model.predict_view(x)
    }
}

impl PredictProba<f64> for FittedLogisticRegressionCV {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        PredictProba::predict_proba_view(&self.model, x)
    }
}

impl anofox_ml_core::DecisionFunction<f64> for FittedLogisticRegressionCV {
    fn decision_function_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        anofox_ml_core::DecisionFunction::decision_function_view(&self.model, x)
    }
}
