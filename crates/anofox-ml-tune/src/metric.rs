//! Named scoring metrics for the cross-validated objective.

use anofox_ml_core::{Result, RustMlError};
use anofox_ml_metrics as m;
use ndarray::{Array1, Array2};

/// A metric the search optimises, in its natural direction (losses are
/// minimised, scores maximised).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    R2,
    Mse,
    Rmse,
    Mae,
    MedianAbsoluteError,
    Mape,
    ExplainedVariance,
    Accuracy,
    BalancedAccuracy,
    F1Macro,
    F1Weighted,
    /// Binary: probability of the larger class; multiclass: one-vs-rest,
    /// macro-averaged.
    RocAuc,
    LogLoss,
}

const ALL: [Metric; 13] = [
    Metric::R2,
    Metric::Mse,
    Metric::Rmse,
    Metric::Mae,
    Metric::MedianAbsoluteError,
    Metric::Mape,
    Metric::ExplainedVariance,
    Metric::Accuracy,
    Metric::BalancedAccuracy,
    Metric::F1Macro,
    Metric::F1Weighted,
    Metric::RocAuc,
    Metric::LogLoss,
];

impl Metric {
    pub fn name(self) -> &'static str {
        match self {
            Metric::R2 => "r2",
            Metric::Mse => "mse",
            Metric::Rmse => "rmse",
            Metric::Mae => "mae",
            Metric::MedianAbsoluteError => "median_absolute_error",
            Metric::Mape => "mape",
            Metric::ExplainedVariance => "explained_variance",
            Metric::Accuracy => "accuracy",
            Metric::BalancedAccuracy => "balanced_accuracy",
            Metric::F1Macro => "f1_macro",
            Metric::F1Weighted => "f1_weighted",
            Metric::RocAuc => "roc_auc",
            Metric::LogLoss => "log_loss",
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        ALL.into_iter().find(|m| m.name() == name).ok_or_else(|| {
            let known: Vec<_> = ALL.iter().map(|m| m.name()).collect();
            RustMlError::InvalidParameter(format!("unknown metric '{name}' (known: {known:?})"))
        })
    }

    /// The default metric of an estimator, as scikit-learn's `score`:
    /// accuracy for classifiers, R² for everything else.
    pub fn default_for(kind: &str) -> Self {
        if is_classifier(kind) {
            Metric::Accuracy
        } else {
            Metric::R2
        }
    }

    pub fn maximize(self) -> bool {
        !matches!(
            self,
            Metric::Mse
                | Metric::Rmse
                | Metric::Mae
                | Metric::MedianAbsoluteError
                | Metric::Mape
                | Metric::LogLoss
        )
    }

    /// Whether the metric scores `predict_proba` output instead of `predict`.
    pub fn needs_proba(self) -> bool {
        matches!(self, Metric::RocAuc | Metric::LogLoss)
    }

    /// Score point predictions.
    pub(crate) fn score(self, y: &Array1<f64>, pred: &Array1<f64>) -> Result<f64> {
        match self {
            Metric::R2 => m::r2_score(y, pred),
            Metric::Mse => m::mse(y, pred),
            Metric::Rmse => m::mse(y, pred).map(f64::sqrt),
            Metric::Mae => m::mae(y, pred),
            Metric::MedianAbsoluteError => m::median_absolute_error(y, pred),
            Metric::Mape => m::mean_absolute_percentage_error(y, pred),
            Metric::ExplainedVariance => m::explained_variance_score(y, pred),
            Metric::Accuracy => m::accuracy_score(y, pred),
            Metric::BalancedAccuracy => m::balanced_accuracy_score(y, pred),
            Metric::F1Macro => m::f1_score_avg(y, pred, m::Average::Macro),
            Metric::F1Weighted => m::f1_score_avg(y, pred, m::Average::Weighted),
            Metric::RocAuc | Metric::LogLoss => unreachable!("probability metric"),
        }
    }

    /// Score class probabilities whose columns follow `classes` (the sorted
    /// distinct training labels).
    pub(crate) fn score_proba(
        self,
        y: &Array1<f64>,
        proba: &Array2<f64>,
        classes: &[f64],
    ) -> Result<f64> {
        if proba.ncols() != classes.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "predict_proba returned {} columns for {} classes",
                proba.ncols(),
                classes.len()
            )));
        }
        match self {
            Metric::LogLoss => m::log_loss_multiclass(y, proba, Some(classes), None),
            Metric::RocAuc if classes.len() == 2 => {
                let positive = y.mapv(|v| if v == classes[1] { 1.0 } else { 0.0 });
                m::roc_auc_score(&positive, &proba.column(1).to_owned())
            }
            Metric::RocAuc => m::roc_auc_score_multiclass(
                y,
                proba,
                m::MultiClass::Ovr,
                m::Average::Macro,
                Some(classes),
                None,
            ),
            _ => unreachable!("point metric"),
        }
    }
}

/// Whether a registry kind is a classifier (by name: the registry has no
/// estimator-type tag).
pub fn is_classifier(kind: &str) -> bool {
    kind.ends_with("Classifier")
        || kind.ends_with("NB")
        || matches!(
            kind,
            "Svc"
                | "NuSvc"
                | "LinearSvc"
                | "LogisticRegressor"
                | "LinearDiscriminantAnalysis"
                | "QuadraticDiscriminantAnalysis"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for metric in ALL {
            assert_eq!(Metric::from_name(metric.name()).unwrap(), metric);
        }
        assert!(Metric::from_name("auc")
            .unwrap_err()
            .to_string()
            .contains("roc_auc"));
    }

    #[test]
    fn default_metric_follows_the_estimator_type() {
        assert_eq!(
            Metric::default_for("RandomForestClassifier"),
            Metric::Accuracy
        );
        assert_eq!(Metric::default_for("LogisticRegressor"), Metric::Accuracy);
        assert_eq!(Metric::default_for("GaussianNB"), Metric::Accuracy);
        assert_eq!(
            Metric::default_for("HistGradientBoostingRegressor"),
            Metric::R2
        );
        assert_eq!(Metric::default_for("Svr"), Metric::R2);
    }
}
