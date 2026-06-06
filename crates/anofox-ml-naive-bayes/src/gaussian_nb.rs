use anofox_ml_core::{
    Fit, FitWeighted, Float, PartialFit, Predict, PredictLogProba, PredictProba, Result,
    RustMlError,
};
use ndarray::{Array1, Array2, Axis};

/// Gaussian Naive Bayes classifier parameters (unfitted state).
///
/// Uses the type-state pattern: call [`Fit::fit`] to produce a
/// [`FittedGaussianNB`] that can make predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GaussianNB {
    /// Portion of the largest variance of all features that is added to
    /// variances for calculation stability.
    pub var_smoothing: f64,
}

impl GaussianNB {
    /// Create a new `GaussianNB` with default parameters.
    pub fn new() -> Self {
        Self {
            var_smoothing: 1e-9,
        }
    }

    /// Set the variance smoothing factor added to variances for calculation stability.
    pub fn with_var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.var_smoothing = var_smoothing;
        self
    }
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Gaussian Naive Bayes classifier.
///
/// Stores learned class priors, per-class feature means (theta), and
/// per-class feature variances (sigma) needed for prediction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedGaussianNB<F: Float> {
    /// Unique sorted class labels.
    class_labels: Vec<F>,
    /// Prior probability of each class (count / total).
    class_prior: Vec<F>,
    /// Mean of each feature per class, shape `(n_classes, n_features)`.
    theta: Array2<F>,
    /// Variance of each feature per class + var_smoothing, shape `(n_classes, n_features)`.
    sigma: Array2<F>,
    /// Per-class cumulative observation count (weighted). Populated by
    /// `partial_fit`; for a one-shot `fit` it equals `class_prior * n`.
    #[serde(default = "default_class_count")]
    class_count: Vec<F>,
}

fn default_class_count<F: Float>() -> Vec<F> {
    Vec::new()
}

impl<F: Float> FittedGaussianNB<F> {
    /// Returns the prior probability of each class.
    pub fn class_prior(&self) -> &[F] {
        &self.class_prior
    }

    /// Returns the per-class feature means, shape `(n_classes, n_features)`.
    pub fn theta(&self) -> &Array2<F> {
        &self.theta
    }

    /// Returns the per-class feature variances (with smoothing), shape `(n_classes, n_features)`.
    pub fn sigma(&self) -> &Array2<F> {
        &self.sigma
    }

    /// Returns the unique sorted class labels.
    pub fn class_labels(&self) -> &[F] {
        &self.class_labels
    }
}

impl<F: Float> FitWeighted<F> for GaussianNB {
    type Fitted = FittedGaussianNB<F>;

    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        if let Some(w) = sample_weight {
            if w.len() != y.len() {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight len {} != y len {}",
                    w.len(),
                    y.len()
                )));
            }
        }
        if x.is_empty() || y.is_empty() {
            return Err(RustMlError::EmptyInput(
                "training data must not be empty".into(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }

        let smoothing = F::from_f64(self.var_smoothing).unwrap();
        let n_features = x.ncols();

        let mut class_labels: Vec<F> = y.to_vec();
        class_labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        class_labels.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-12).unwrap());
        let n_classes = class_labels.len();

        // Total weight across all samples.
        let total_w: F = match sample_weight {
            Some(w) => w.iter().copied().fold(F::zero(), |a, b| a + b),
            None => F::from_usize(x.nrows()).unwrap(),
        };

        let mut theta = Array2::<F>::zeros((n_classes, n_features));
        let mut sigma = Array2::<F>::zeros((n_classes, n_features));
        let mut class_prior = Vec::with_capacity(n_classes);

        for (ci, &label) in class_labels.iter().enumerate() {
            // Weighted mean per class.
            let mut wsum = F::zero();
            let mut wmean = vec![F::zero(); n_features];
            for i in 0..x.nrows() {
                if (y[i] - label).abs() >= F::from_f64(1e-12).unwrap() {
                    continue;
                }
                let wi = sample_weight.map(|w| w[i]).unwrap_or(F::one());
                wsum = wsum + wi;
                for j in 0..n_features {
                    wmean[j] = wmean[j] + wi * x[[i, j]];
                }
            }
            let wsum_safe = if wsum == F::zero() {
                F::from_f64(1e-12).unwrap()
            } else {
                wsum
            };
            for j in 0..n_features {
                wmean[j] = wmean[j] / wsum_safe;
            }
            // Weighted variance per class.
            let mut wvar = vec![F::zero(); n_features];
            for i in 0..x.nrows() {
                if (y[i] - label).abs() >= F::from_f64(1e-12).unwrap() {
                    continue;
                }
                let wi = sample_weight.map(|w| w[i]).unwrap_or(F::one());
                for j in 0..n_features {
                    let d = x[[i, j]] - wmean[j];
                    wvar[j] = wvar[j] + wi * d * d;
                }
            }
            for j in 0..n_features {
                wvar[j] = wvar[j] / wsum_safe + smoothing;
                theta[[ci, j]] = wmean[j];
                sigma[[ci, j]] = wvar[j];
            }
            class_prior.push(wsum / total_w);
        }

        let class_count: Vec<F> = class_prior.iter().map(|p| *p * total_w).collect();
        Ok(FittedGaussianNB {
            class_labels,
            class_prior,
            theta,
            sigma,
            class_count,
        })
    }
}

impl<F: Float> Fit<F> for GaussianNB {
    type Fitted = FittedGaussianNB<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(RustMlError::EmptyInput(
                "training data must not be empty".into(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }

        let n_samples = F::from_usize(x.nrows()).unwrap();
        let n_features = x.ncols();
        let smoothing = F::from_f64(self.var_smoothing).unwrap();

        // Extract unique class labels and sort them.
        let mut class_labels: Vec<F> = y.to_vec();
        class_labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        class_labels.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-12).unwrap());

        let n_classes = class_labels.len();
        let mut theta = Array2::<F>::zeros((n_classes, n_features));
        let mut sigma = Array2::<F>::zeros((n_classes, n_features));
        let mut class_prior = Vec::with_capacity(n_classes);

        for (ci, &label) in class_labels.iter().enumerate() {
            // Gather row indices belonging to this class.
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &val)| (val - label).abs() < F::from_f64(1e-12).unwrap())
                .map(|(i, _)| i)
                .collect();

            let count = F::from_usize(mask.len()).unwrap();
            class_prior.push(count / n_samples);

            // Build sub-matrix for this class.
            let class_x = x.select(Axis(0), &mask);

            // Compute per-feature mean.
            let mean = class_x.mean_axis(Axis(0)).unwrap();

            // Compute per-feature variance: E[(x - mu)^2].
            let diff = &class_x - &mean;
            let var = diff.mapv(|v| v * v).mean_axis(Axis(0)).unwrap();

            theta.row_mut(ci).assign(&mean);
            sigma.row_mut(ci).assign(&(var + smoothing));
        }

        let class_count: Vec<F> = class_prior.iter().map(|p| *p * n_samples).collect();
        Ok(FittedGaussianNB {
            class_labels,
            class_prior,
            theta,
            sigma,
            class_count,
        })
    }
}

impl<F: Float> PartialFit<F> for GaussianNB {
    type Fitted = FittedGaussianNB<F>;

    fn partial_fit(
        &self,
        state: Option<Self::Fitted>,
        x: &Array2<F>,
        y: &Array1<F>,
        classes: Option<&[F]>,
    ) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(RustMlError::EmptyInput(
                "training data must not be empty".into(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        let smoothing = F::from_f64(self.var_smoothing).unwrap();
        let n_features = x.ncols();
        let eps = F::from_f64(1e-12).unwrap();

        // Determine the global class set. On the first call (state == None)
        // either honour `classes` or derive from the batch. On subsequent
        // calls reuse `state.class_labels`.
        let (class_labels, mut counts, mut means, mut vars) = if let Some(s) = state {
            // Recover raw (unsmoothed) variance from sigma.
            let mut raw_var = s.sigma.clone();
            for v in raw_var.iter_mut() {
                *v = *v - smoothing;
                if *v < F::zero() {
                    *v = F::zero();
                }
            }
            (s.class_labels, s.class_count, s.theta, raw_var)
        } else {
            let labels: Vec<F> = if let Some(c) = classes {
                c.to_vec()
            } else {
                let mut v = y.to_vec();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v.dedup_by(|a, b| (*a - *b).abs() < eps);
                v
            };
            let n_classes = labels.len();
            let counts = vec![F::zero(); n_classes];
            let means = Array2::<F>::zeros((n_classes, n_features));
            let vars = Array2::<F>::zeros((n_classes, n_features));
            (labels, counts, means, vars)
        };

        // Process the batch class-by-class with Chan-Golub-LeVeque parallel
        // variance combination: stable for streamed updates.
        for (ci, &label) in class_labels.iter().enumerate() {
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &v)| (v - label).abs() < eps)
                .map(|(i, _)| i)
                .collect();
            if mask.is_empty() {
                continue;
            }
            let batch_count = F::from_usize(mask.len()).unwrap();
            let batch_x = x.select(Axis(0), &mask);
            let batch_mean = batch_x.mean_axis(Axis(0)).unwrap();
            let batch_diff = &batch_x - &batch_mean;
            let batch_var = batch_diff.mapv(|v| v * v).mean_axis(Axis(0)).unwrap();

            let old_count = counts[ci];
            let new_count = old_count + batch_count;

            if old_count == F::zero() {
                for j in 0..n_features {
                    means[[ci, j]] = batch_mean[j];
                    vars[[ci, j]] = batch_var[j];
                }
            } else {
                for j in 0..n_features {
                    let m_old = means[[ci, j]];
                    let m_batch = batch_mean[j];
                    let v_old = vars[[ci, j]];
                    let v_batch = batch_var[j];
                    let m_new = (old_count * m_old + batch_count * m_batch) / new_count;
                    // Parallel variance: var_new = (n_old*v_old + n_batch*v_batch
                    //                              + n_old*(m_old - m_new)² + n_batch*(m_batch - m_new)²) / n_new
                    let d_old = m_old - m_new;
                    let d_batch = m_batch - m_new;
                    let v_new = (old_count * v_old
                        + batch_count * v_batch
                        + old_count * d_old * d_old
                        + batch_count * d_batch * d_batch)
                        / new_count;
                    means[[ci, j]] = m_new;
                    vars[[ci, j]] = v_new;
                }
            }
            counts[ci] = new_count;
        }

        // Final prior and smoothed sigma.
        let total: F = counts.iter().fold(F::zero(), |a, b| a + *b);
        let class_prior: Vec<F> = if total > F::zero() {
            counts.iter().map(|c| *c / total).collect()
        } else {
            vec![F::zero(); counts.len()]
        };
        let mut sigma = vars;
        for v in sigma.iter_mut() {
            *v = *v + smoothing;
        }
        Ok(FittedGaussianNB {
            class_labels,
            class_prior,
            theta: means,
            sigma,
            class_count: counts,
        })
    }
}

impl<F: Float> Predict<F> for FittedGaussianNB<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.theta.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.theta.ncols(),
                x.ncols()
            )));
        }

        let two = F::from_f64(2.0).unwrap();
        let two_pi = two * F::from_f64(std::f64::consts::PI).unwrap();
        let half = F::from_f64(0.5).unwrap();

        let mut predictions = Array1::<F>::zeros(x.nrows());

        for (i, sample) in x.rows().into_iter().enumerate() {
            let mut best_class = self.class_labels[0];
            let mut best_log_posterior = F::neg_infinity();

            for (ci, &label) in self.class_labels.iter().enumerate() {
                let log_prior = self.class_prior[ci].ln();

                // Sum of log-likelihoods over features:
                // -0.5 * log(2*pi*sigma) - 0.5 * (x - mu)^2 / sigma
                let mut log_likelihood = F::zero();
                for j in 0..x.ncols() {
                    let mu = self.theta[[ci, j]];
                    let var = self.sigma[[ci, j]];
                    let diff = sample[j] - mu;
                    log_likelihood =
                        log_likelihood - half * (two_pi * var).ln() - half * diff * diff / var;
                }

                let log_posterior = log_prior + log_likelihood;
                if log_posterior > best_log_posterior {
                    best_log_posterior = log_posterior;
                    best_class = label;
                }
            }

            predictions[i] = best_class;
        }

        Ok(predictions)
    }
}

impl<F: Float> PredictProba<F> for FittedGaussianNB<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.theta.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.theta.ncols(),
                x.ncols()
            )));
        }
        let two = F::from_f64(2.0).unwrap();
        let two_pi = two * F::from_f64(std::f64::consts::PI).unwrap();
        let half = F::from_f64(0.5).unwrap();
        let n_classes = self.class_labels.len();
        let n = x.nrows();
        let mut out = Array2::<F>::zeros((n, n_classes));
        for (i, sample) in x.rows().into_iter().enumerate() {
            let mut logs = vec![F::zero(); n_classes];
            let mut max_l = F::neg_infinity();
            for ci in 0..n_classes {
                let mut log_post = self.class_prior[ci].ln();
                for j in 0..x.ncols() {
                    let mu = self.theta[[ci, j]];
                    let var = self.sigma[[ci, j]];
                    let diff = sample[j] - mu;
                    log_post = log_post - half * (two_pi * var).ln() - half * diff * diff / var;
                }
                logs[ci] = log_post;
                if log_post > max_l {
                    max_l = log_post;
                }
            }
            let mut z = F::zero();
            for ci in 0..n_classes {
                let e = (logs[ci] - max_l).exp();
                out[[i, ci]] = e;
                z = z + e;
            }
            for ci in 0..n_classes {
                out[[i, ci]] = out[[i, ci]] / z;
            }
        }
        Ok(out)
    }
}

impl<F: Float> PredictLogProba<F> for FittedGaussianNB<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_partial_fit_two_halves_matches_one_shot() {
        // GaussianNB partial_fit on two halves should produce identical
        // theta/sigma/prior to one-shot fit on the union.
        let x = array![
            [1.0_f64, 2.0],
            [1.5, 2.5],
            [0.8, 1.7],
            [1.2, 1.9],
            [5.0, 5.0],
            [5.5, 5.5],
            [4.8, 4.9],
            [5.1, 5.3]
        ];
        let y = array![0.0_f64, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::new();
        let one_shot: FittedGaussianNB<f64> = Fit::fit(&nb, &x, &y).unwrap();

        let classes = [0.0_f64, 1.0];
        let h1 = x.slice(ndarray::s![0..4, ..]).to_owned();
        let y1 = y.slice(ndarray::s![0..4]).to_owned();
        let h2 = x.slice(ndarray::s![4..8, ..]).to_owned();
        let y2 = y.slice(ndarray::s![4..8]).to_owned();
        let s1 = PartialFit::partial_fit(&nb, None, &h1, &y1, Some(&classes)).unwrap();
        let s2 = PartialFit::partial_fit(&nb, Some(s1), &h2, &y2, Some(&classes)).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(one_shot.theta[[i, j]], s2.theta[[i, j]], epsilon = 1e-9);
                assert_abs_diff_eq!(one_shot.sigma[[i, j]], s2.sigma[[i, j]], epsilon = 1e-9);
            }
            assert_abs_diff_eq!(one_shot.class_prior[i], s2.class_prior[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_two_class_well_separated() {
        // Two well-separated Gaussian clusters.
        let x_train = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [0.9, 0.9],
            [1.0, 1.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [9.9, 9.9],
            [10.0, 10.2]
        ];
        let y_train = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Verify class labels.
        assert_eq!(fitted.class_labels(), &[0.0, 1.0]);

        // Verify equal priors.
        assert_abs_diff_eq!(fitted.class_prior()[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.class_prior()[1], 0.5, epsilon = 1e-10);

        // Predict points near each cluster center.
        let x_test = array![[1.0, 1.0], [10.0, 10.0], [0.5, 0.5], [9.5, 9.5]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[2], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predictions_known_data() {
        // Simple 1-D data: class 0 centred at 0, class 1 centred at 5.
        let x_train = array![[0.0], [0.5], [-0.5], [5.0], [5.5], [4.5]];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Means should be approximately 0.0 and 5.0.
        assert_abs_diff_eq!(fitted.theta()[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.theta()[[1, 0]], 5.0, epsilon = 1e-10);

        // A point at 1.0 should be class 0; at 4.0 should be class 1.
        let x_test = array![[1.0], [4.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let nb = GaussianNB::default();
        let result: Result<FittedGaussianNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());

        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_fit() {
        // X has 2 rows, y has 3 elements.
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 0.0];

        let nb = GaussianNB::default();
        let result: Result<FittedGaussianNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());

        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![0.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Predict with wrong number of features (3 instead of 2).
        let x_test = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&x_test);
        assert!(result.is_err());

        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_zero_variance_feature() {
        // One feature column is constant — var_smoothing should prevent NaN.
        let x_train = array![
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
            [10.0, 5.0],
            [11.0, 5.0],
            [12.0, 5.0]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Variance for the second feature should be the smoothing value, not zero.
        assert!(fitted.sigma()[[0, 1]] > 0.0);
        assert!(fitted.sigma()[[1, 1]] > 0.0);

        // Predictions should still work without NaN.
        let x_test = array![[2.0, 5.0], [11.0, 5.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
        // Verify no NaN in predictions.
        for &p in preds.iter() {
            assert!(!p.is_nan(), "prediction should not be NaN");
        }
    }

    #[test]
    fn test_highly_imbalanced_classes() {
        // 50 class-0 samples and 2 class-1 samples.
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for i in 0..50 {
            x_data.push(i as f64 * 0.1);
            x_data.push(i as f64 * 0.1);
            y_data.push(0.0);
        }
        // Class-1 samples far away.
        x_data.push(100.0);
        x_data.push(100.0);
        y_data.push(1.0);
        x_data.push(101.0);
        x_data.push(101.0);
        y_data.push(1.0);

        let x_train = Array2::from_shape_vec((52, 2), x_data).unwrap();
        let y_train = Array1::from_vec(y_data);

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Should not crash, and priors should reflect the imbalance.
        assert_abs_diff_eq!(fitted.class_prior()[0], 50.0 / 52.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.class_prior()[1], 2.0 / 52.0, epsilon = 1e-10);

        // A point near class-1 cluster should still be predicted as class 1.
        let x_test = array![[100.5, 100.5]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_many_classes() {
        // 10 different classes, each with 3 samples.
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for class in 0..10 {
            let center = class as f64 * 100.0;
            for offset in &[-0.1, 0.0, 0.1] {
                x_data.push(center + offset);
                x_data.push(center + offset);
                y_data.push(class as f64);
            }
        }
        let x_train = Array2::from_shape_vec((30, 2), x_data).unwrap();
        let y_train = Array1::from_vec(y_data);

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        assert_eq!(fitted.class_labels().len(), 10);

        // Predict at each class center; verify predictions are valid labels.
        let mut test_data = Vec::new();
        for class in 0..10 {
            test_data.push(class as f64 * 100.0);
            test_data.push(class as f64 * 100.0);
        }
        let x_test = Array2::from_shape_vec((10, 2), test_data).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        for &p in preds.iter() {
            assert!(
                fitted.class_labels().contains(&p),
                "prediction {} is not a valid class label",
                p
            );
        }
    }

    #[test]
    fn test_f32_support() {
        let x_train: Array2<f32> = array![[1.0f32, 1.0], [1.1, 1.1], [10.0, 10.0], [10.1, 10.1]];
        let y_train: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f32> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test: Array2<f32> = array![[1.0f32, 1.0], [10.0, 10.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(preds[1], 1.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_shape_mismatch_error() {
        // x has 3 rows but y has 2 elements.
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0.0, 1.0];

        let nb = GaussianNB::default();
        let result: Result<FittedGaussianNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(msg)) => {
                assert!(msg.contains("3"), "error should mention row count");
                assert!(msg.contains("2"), "error should mention y length");
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_predict_wrong_features() {
        let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y_train = array![0.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Predict with 2 features instead of 3.
        let x_test = array![[1.0, 2.0]];
        let result = fitted.predict(&x_test);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(msg)) => {
                assert!(msg.contains("3"), "error should mention expected features");
                assert!(msg.contains("2"), "error should mention actual features");
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_prior_probabilities() {
        // 6 class-0 and 4 class-1 samples.
        let x_train = array![
            [1.0],
            [1.1],
            [1.2],
            [0.9],
            [0.8],
            [1.3],
            [10.0],
            [10.1],
            [10.2],
            [9.9]
        ];
        let y_train = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Priors should reflect 6/10 and 4/10.
        assert_abs_diff_eq!(fitted.class_prior()[0], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.class_prior()[1], 0.4, epsilon = 1e-10);

        // Priors should sum to 1.
        let prior_sum: f64 = fitted.class_prior().iter().sum();
        assert_abs_diff_eq!(prior_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_sample_per_class() {
        // Two classes, one sample each.
        let x_train = array![[0.0, 0.0], [10.0, 10.0]];
        let y_train = array![0.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Means should match the single sample.
        assert_abs_diff_eq!(fitted.theta()[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.theta()[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.theta()[[1, 0]], 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.theta()[[1, 1]], 10.0, epsilon = 1e-10);

        // Should still predict correctly (variance is just smoothing).
        let x_test = array![[0.1, 0.1], [9.9, 9.9]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_three_features_two_classes() {
        // Well-separated 3D data.
        let x_train = array![
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9],
            [1.0, 2.0, 3.2],
            [20.0, 21.0, 22.0],
            [20.1, 21.1, 22.1],
            [19.9, 20.9, 21.9],
            [20.0, 21.0, 22.2]
        ];
        let y_train = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Verify theta has the right shape.
        assert_eq!(fitted.theta().shape(), &[2, 3]);
        assert_eq!(fitted.sigma().shape(), &[2, 3]);

        // Predict near each cluster.
        let x_test = array![[1.0, 2.0, 3.0], [20.0, 21.0, 22.0], [10.0, 11.0, 12.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
        // The midpoint could go either way, but it should be a valid label.
        assert!(preds[2] == 0.0 || preds[2] == 1.0);
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        /// Generate well-separated two-class Gaussian data.
        ///
        /// Class 0 is centred at -5.0, class 1 at 5.0 on every feature,
        /// with small deterministic perturbations derived from a hash so
        /// that proptest does not rely on external RNGs.
        fn make_separated_data(
            n_per_class: usize,
            n_features: usize,
            seed: u64,
        ) -> (Array2<f64>, Array1<f64>) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let n_samples = n_per_class * 2;
            let mut x_data = Vec::with_capacity(n_samples * n_features);
            let mut y_data = Vec::with_capacity(n_samples);

            for i in 0..n_samples {
                let class = if i < n_per_class { 0.0 } else { 1.0 };
                let center = if i < n_per_class { -5.0 } else { 5.0 };
                for j in 0..n_features {
                    let mut h = DefaultHasher::new();
                    seed.hash(&mut h);
                    (i as u64).hash(&mut h);
                    (j as u64).hash(&mut h);
                    let bits = h.finish();
                    // Small perturbation in [-0.5, 0.5]
                    let noise = (bits as f64 / u64::MAX as f64) - 0.5;
                    x_data.push(center + noise);
                }
                y_data.push(class);
            }

            let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
            let y = Array1::from_vec(y_data);
            (x, y)
        }

        proptest! {
            #[test]
            fn well_separated_accuracy_above_80(
                n_per_class in 5..30usize,
                n_features in 1..5usize,
                seed in 0u64..1000,
            ) {
                let (x, y) = make_separated_data(n_per_class, n_features, seed);

                let nb = GaussianNB::default();
                let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x, &y).unwrap();
                let preds = fitted.predict(&x).unwrap();

                let correct = preds.iter()
                    .zip(y.iter())
                    .filter(|(&p, &t)| (p - t).abs() < 1e-10)
                    .count();
                let accuracy = correct as f64 / y.len() as f64;

                prop_assert!(
                    accuracy >= 0.8,
                    "accuracy was {:.3} (expected >= 0.8), n_per_class={}, n_features={}, seed={}",
                    accuracy, n_per_class, n_features, seed
                );
            }
        }
    }
}

impl<F: Float> anofox_ml_core::ClassifierScore<F> for FittedGaussianNB<F> {}
