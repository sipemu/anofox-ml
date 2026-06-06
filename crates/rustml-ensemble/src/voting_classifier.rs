//! Voting classifier: combines predictions from multiple heterogeneous models.
//!
//! Supports hard voting (majority vote) and soft voting (average probabilities,
//! requires predict_proba on fitted models — not yet implemented).

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};
use std::collections::HashMap;

/// A named estimator for the voting ensemble.
struct NamedEstimator<F: Float> {
    name: String,
    estimator: Box<dyn FitPredictClone<F>>,
}

/// Internal trait combining Fit + Send + Sync for trait objects.
trait FitPredictClone<F: Float>: Send + Sync {
    fn fit_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn PredictBox<F>>>;
}

trait PredictBox<F: Float>: Send + Sync {
    fn predict_box(&self, x: &Array2<F>) -> Result<Array1<F>>;
}

/// Blanket impl: any Fit+Predict type can be a voting member.
impl<F, T> FitPredictClone<F> for T
where
    F: Float,
    T: Fit<F> + Send + Sync,
    T::Fitted: Predict<F> + Send + Sync + 'static,
{
    fn fit_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn PredictBox<F>>> {
        let fitted = Fit::fit(self, x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F, T> PredictBox<F> for T
where
    F: Float,
    T: Predict<F> + Send + Sync,
{
    fn predict_box(&self, x: &Array2<F>) -> Result<Array1<F>> {
        self.predict(x)
    }
}

/// Voting classifier that combines multiple models via majority vote.
pub struct VotingClassifier<F: Float> {
    estimators: Vec<NamedEstimator<F>>,
}

impl<F: Float> VotingClassifier<F> {
    /// Create a new empty VotingClassifier.
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
        }
    }

    /// Add a named estimator to the ensemble.
    pub fn push<T>(mut self, name: impl Into<String>, estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + 'static,
        T::Fitted: Predict<F> + Send + Sync + 'static,
    {
        self.estimators.push(NamedEstimator {
            name: name.into(),
            estimator: Box::new(estimator),
        });
        self
    }
}

/// Fitted voting classifier.
pub struct FittedVotingClassifier<F: Float> {
    fitted_models: Vec<(String, Box<dyn PredictBox<F>>)>,
    n_features: usize,
}

impl<F: Float> FittedVotingClassifier<F> {
    /// Return the names of the constituent estimators.
    pub fn estimator_names(&self) -> Vec<&str> {
        self.fitted_models.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Compute classification accuracy on the given data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<f64> {
        let preds = self.predict(x)?;
        let n = y.len();
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| (p - t).abs() < F::from_f64(1e-9).unwrap())
            .count();
        Ok(correct as f64 / n as f64)
    }
}

impl<F: Float + 'static> Fit<F> for VotingClassifier<F> {
    type Fitted = FittedVotingClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        if self.estimators.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "VotingClassifier needs at least one estimator".into(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }

        let mut fitted_models = Vec::with_capacity(self.estimators.len());
        for est in &self.estimators {
            let fitted = est.estimator.fit_box(x, y)?;
            fitted_models.push((est.name.clone(), fitted));
        }

        Ok(FittedVotingClassifier {
            fitted_models,
            n_features: x.ncols(),
        })
    }
}

impl<F: Float> Predict<F> for FittedVotingClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n = x.nrows();
        let all_preds: Vec<Array1<F>> = self
            .fitted_models
            .iter()
            .map(|(_, model)| model.predict_box(x))
            .collect::<Result<Vec<_>>>()?;

        let mut result = Array1::zeros(n);
        for i in 0..n {
            let mut votes: HashMap<u64, (F, usize)> = HashMap::new();
            for preds in &all_preds {
                let key = preds[i].to_f64().unwrap().to_bits();
                votes
                    .entry(key)
                    .and_modify(|e| e.1 += 1)
                    .or_insert((preds[i], 1));
            }
            result[i] = votes
                .into_values()
                .max_by_key(|&(_, count)| count)
                .unwrap()
                .0;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rustml_trees::DecisionTreeClassifier;

    #[test]
    fn test_voting_classifier_basic() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let vc = VotingClassifier::new()
            .push(
                "tree1",
                DecisionTreeClassifier {
                    max_depth: Some(3),
                    ..Default::default()
                },
            )
            .push(
                "tree2",
                DecisionTreeClassifier {
                    max_depth: Some(2),
                    ..Default::default()
                },
            )
            .push(
                "tree3",
                DecisionTreeClassifier {
                    max_depth: Some(5),
                    ..Default::default()
                },
            );

        let fitted: FittedVotingClassifier<f64> = vc.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (p, t) in preds.iter().zip(y.iter()) {
            assert!((p - t).abs() < 1e-10);
        }
    }

    #[test]
    fn test_voting_classifier_names() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let vc = VotingClassifier::new()
            .push("a", DecisionTreeClassifier::default())
            .push("b", DecisionTreeClassifier::default());

        let fitted: FittedVotingClassifier<f64> = vc.fit(&x, &y).unwrap();
        assert_eq!(fitted.estimator_names(), vec!["a", "b"]);
    }

    #[test]
    fn test_voting_classifier_score() {
        let x = array![[1.0, 0.0], [2.0, 0.0], [10.0, 1.0], [11.0, 1.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let vc = VotingClassifier::new()
            .push("t1", DecisionTreeClassifier::default())
            .push("t2", DecisionTreeClassifier::default());

        let fitted: FittedVotingClassifier<f64> = vc.fit(&x, &y).unwrap();
        let acc = fitted.score(&x, &y).unwrap();
        assert!(acc >= 0.5);
    }

    #[test]
    fn test_voting_classifier_empty_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];
        let vc = VotingClassifier::<f64>::new();
        assert!(vc.fit(&x, &y).is_err());
    }

    #[test]
    fn test_voting_classifier_shape_mismatch() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];
        let vc = VotingClassifier::new().push("t", DecisionTreeClassifier::default());
        assert!(Fit::<f64>::fit(&vc, &x, &y).is_err());
    }
}
