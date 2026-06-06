//! Voting regressor: averages predictions from multiple heterogeneous models.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

/// Internal trait for type-erased fit.
trait FitPredictBox<F: Float>: Send + Sync {
    fn fit_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn PredBox<F>>>;
}

trait PredBox<F: Float>: Send + Sync {
    fn predict_box(&self, x: &Array2<F>) -> Result<Array1<F>>;
}

impl<F, T> FitPredictBox<F> for T
where
    F: Float,
    T: Fit<F> + Send + Sync,
    T::Fitted: Predict<F> + Send + Sync + 'static,
{
    fn fit_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn PredBox<F>>> {
        let fitted = Fit::fit(self, x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F, T> PredBox<F> for T
where
    F: Float,
    T: Predict<F> + Send + Sync,
{
    fn predict_box(&self, x: &Array2<F>) -> Result<Array1<F>> {
        self.predict(x)
    }
}

struct NamedEstimator<F: Float> {
    name: String,
    estimator: Box<dyn FitPredictBox<F>>,
    weight: F,
}

/// Voting regressor that averages predictions from multiple models.
///
/// Supports optional per-estimator weights for weighted averaging.
pub struct VotingRegressor<F: Float> {
    estimators: Vec<NamedEstimator<F>>,
}

impl<F: Float> VotingRegressor<F> {
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
        }
    }

    /// Add a named estimator with weight 1.0.
    pub fn push<T>(self, name: impl Into<String>, estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + 'static,
        T::Fitted: Predict<F> + Send + Sync + 'static,
    {
        self.push_weighted(name, estimator, F::one())
    }

    /// Add a named estimator with a custom weight.
    pub fn push_weighted<T>(mut self, name: impl Into<String>, estimator: T, weight: F) -> Self
    where
        T: Fit<F> + Send + Sync + 'static,
        T::Fitted: Predict<F> + Send + Sync + 'static,
    {
        self.estimators.push(NamedEstimator {
            name: name.into(),
            estimator: Box::new(estimator),
            weight,
        });
        self
    }
}

/// Fitted voting regressor.
pub struct FittedVotingRegressor<F: Float> {
    fitted_models: Vec<(String, Box<dyn PredBox<F>>, F)>,
    total_weight: F,
    n_features: usize,
}

impl<F: Float> FittedVotingRegressor<F> {
    pub fn estimator_names(&self) -> Vec<&str> {
        self.fitted_models
            .iter()
            .map(|(n, _, _)| n.as_str())
            .collect()
    }

    /// Compute R² score on the given data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<f64> {
        let preds = self.predict(x)?;
        let n = y.len();
        let y_mean = y.iter().copied().fold(F::zero(), |a, b| a + b) / F::from_usize(n).unwrap();
        let ss_res: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).to_f64().unwrap().powi(2))
            .sum();
        let ss_tot: f64 = y
            .iter()
            .map(|&t| (t - y_mean).to_f64().unwrap().powi(2))
            .sum();
        Ok(if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        })
    }
}

impl<F: Float + 'static> Fit<F> for VotingRegressor<F> {
    type Fitted = FittedVotingRegressor<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        if self.estimators.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "VotingRegressor needs at least one estimator".into(),
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
        let mut total_weight = F::zero();
        for est in &self.estimators {
            let fitted = est.estimator.fit_box(x, y)?;
            total_weight = total_weight + est.weight;
            fitted_models.push((est.name.clone(), fitted, est.weight));
        }

        Ok(FittedVotingRegressor {
            fitted_models,
            total_weight,
            n_features: x.ncols(),
        })
    }
}

impl<F: Float> Predict<F> for FittedVotingRegressor<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n = x.nrows();
        let mut result = Array1::zeros(n);

        for (_, model, weight) in &self.fitted_models {
            let preds = model.predict_box(x)?;
            for i in 0..n {
                result[i] = result[i] + preds[i] * *weight;
            }
        }

        // Divide by total weight
        let tw = self.total_weight;
        result.mapv_inplace(|v| v / tw);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use rustml_trees::DecisionTreeRegressor;

    #[test]
    fn test_voting_regressor_basic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let vr = VotingRegressor::new()
            .push("t1", DecisionTreeRegressor::default())
            .push(
                "t2",
                DecisionTreeRegressor {
                    max_depth: Some(2),
                    ..Default::default()
                },
            );

        let fitted: FittedVotingRegressor<f64> = vr.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 3.0);
        }
    }

    #[test]
    fn test_voting_regressor_weighted() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let vr = VotingRegressor::new()
            .push_weighted("high", DecisionTreeRegressor::default(), 3.0)
            .push_weighted("low", DecisionTreeRegressor::default(), 1.0);

        let fitted: FittedVotingRegressor<f64> = vr.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_voting_regressor_names() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let vr = VotingRegressor::new()
            .push("a", DecisionTreeRegressor::default())
            .push("b", DecisionTreeRegressor::default());

        let fitted: FittedVotingRegressor<f64> = vr.fit(&x, &y).unwrap();
        assert_eq!(fitted.estimator_names(), vec!["a", "b"]);
    }

    #[test]
    fn test_voting_regressor_score() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let vr = VotingRegressor::new().push("t1", DecisionTreeRegressor::default());

        let fitted: FittedVotingRegressor<f64> = vr.fit(&x, &y).unwrap();
        let r2 = fitted.score(&x, &y).unwrap();
        assert!(r2 > 0.5);
    }

    #[test]
    fn test_voting_regressor_empty_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];
        let vr = VotingRegressor::<f64>::new();
        assert!(vr.fit(&x, &y).is_err());
    }
}
