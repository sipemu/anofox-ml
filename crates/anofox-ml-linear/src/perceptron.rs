//! The perceptron (sklearn's `Perceptron`): [`SgdClassifier`] with the
//! perceptron loss, a constant learning rate `eta0` and no penalty by
//! default, exactly as sklearn defines it.

use anofox_ml_core::{ClassWeight, Fit, FitWeighted, Result};
use ndarray::{ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::sgd_common::{LearningRate, Penalty};
use crate::{ClassifierLoss, FittedSgdClassifier, SgdClassifier};

/// Perceptron classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Perceptron {
    pub penalty: Penalty,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub max_iter: usize,
    pub tol: Option<f64>,
    pub shuffle: bool,
    pub eta0: f64,
    pub early_stopping: bool,
    pub validation_fraction: f64,
    pub n_iter_no_change: usize,
    pub class_weight: Option<ClassWeight>,
    pub seed: u64,
}

impl Default for Perceptron {
    fn default() -> Self {
        Self {
            penalty: Penalty::None,
            alpha: 1e-4,
            l1_ratio: 0.15,
            max_iter: 1000,
            tol: Some(1e-3),
            shuffle: true,
            eta0: 1.0,
            early_stopping: false,
            validation_fraction: 0.1,
            n_iter_no_change: 5,
            class_weight: None,
            seed: 0,
        }
    }
}

impl Perceptron {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_tol(mut self, tol: impl Into<Option<f64>>) -> Self {
        self.tol = tol.into();
        self
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn with_eta0(mut self, eta0: f64) -> Self {
        self.eta0 = eta0;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_class_weight(mut self, class_weight: Option<ClassWeight>) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// The equivalent [`SgdClassifier`].
    pub fn to_sgd(&self) -> SgdClassifier {
        SgdClassifier::new()
            .with_loss(ClassifierLoss::Perceptron)
            .with_penalty(self.penalty)
            .with_alpha(self.alpha)
            .with_l1_ratio(self.l1_ratio)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_shuffle(self.shuffle)
            .with_eta0(self.eta0)
            .with_learning_rate(LearningRate::Constant)
            .with_early_stopping(self.early_stopping)
            .with_validation_fraction(self.validation_fraction)
            .with_n_iter_no_change(self.n_iter_no_change)
            .with_class_weight(self.class_weight.clone())
            .with_seed(self.seed)
    }
}

impl FitWeighted<f64> for Perceptron {
    type Fitted = FittedSgdClassifier<f64>;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        self.to_sgd().fit_weighted_view(x, y, sample_weight)
    }
}

impl Fit<f64> for Perceptron {
    type Fitted = FittedSgdClassifier<f64>;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        self.to_sgd().fit_view(x, y)
    }
}
