//! Bernoulli restricted Boltzmann machine trained by stochastic maximum
//! likelihood / persistent contrastive divergence (sklearn's
//! `neural_network.BernoulliRBM`), drawing from numpy's `RandomState` in
//! sklearn's order so that a seeded fit equals sklearn's.

use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// sklearn's `BernoulliRBM`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BernoulliRbm {
    pub n_components: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub n_iter: usize,
    /// `random_state` (an int seed).
    pub seed: u32,
}

impl Default for BernoulliRbm {
    fn default() -> Self {
        Self {
            n_components: 256,
            learning_rate: 0.1,
            batch_size: 10,
            n_iter: 10,
            seed: 0,
        }
    }
}

/// A fitted [`BernoulliRbm`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedBernoulliRbm {
    /// Weights, `n_components × n_features` (sklearn's `components_`).
    pub components: Array2<f64>,
    pub intercept_hidden: Array1<f64>,
    pub intercept_visible: Array1<f64>,
    /// The persistent chain's hidden states, `batch_size × n_components`.
    pub h_samples: Array2<f64>,
    learning_rate: f64,
    seed: u32,
    /// The training stream, continued by `partial_fit` and `gibbs`
    /// (sklearn's `random_state_`).
    rng: RandomState,
}

/// `scipy.special.expit`.
fn expit(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// `rng.uniform(size=p.shape) < p`, row-major, as 0 / 1.
fn bernoulli(p: &Array2<f64>, rng: &mut RandomState) -> Array2<f64> {
    let mut out = p.clone();
    for v in out.iter_mut() {
        *v = if rng.random_sample() < *v { 1.0 } else { 0.0 };
    }
    out
}

impl FittedBernoulliRbm {
    fn check(&self, x: ArrayView2<f64>) -> Result<()> {
        if x.ncols() != self.components.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.components.ncols(),
                x.ncols()
            )));
        }
        Ok(())
    }

    /// `P(h = 1 | v)` (sklearn's `_mean_hiddens`, also `transform`).
    fn mean_hiddens(&self, v: ArrayView2<f64>) -> Array2<f64> {
        let mut p = v.dot(&self.components.t());
        p += &self.intercept_hidden;
        p.mapv_inplace(expit);
        p
    }

    fn sample_visibles(&mut self, h: &Array2<f64>) -> Array2<f64> {
        let mut p = h.dot(&self.components);
        p += &self.intercept_visible;
        p.mapv_inplace(expit);
        bernoulli(&p, &mut self.rng)
    }

    /// One SML step on a mini-batch (sklearn's `_fit`).
    fn step(&mut self, v_pos: ArrayView2<f64>) {
        let h_pos = self.mean_hiddens(v_pos);
        let h_samples = self.h_samples.clone();
        let v_neg = self.sample_visibles(&h_samples);
        let mut h_neg = self.mean_hiddens(v_neg.view());
        let lr = self.learning_rate / v_pos.nrows() as f64;
        let mut update = v_pos.t().dot(&h_pos).reversed_axes();
        update -= &h_neg.t().dot(&v_neg);
        self.components.scaled_add(lr, &update);
        let dh = &h_pos.sum_axis(Axis(0)) - &h_neg.sum_axis(Axis(0));
        self.intercept_hidden.scaled_add(lr, &dh);
        let dv = &v_pos.sum_axis(Axis(0)) - &v_neg.sum_axis(Axis(0));
        self.intercept_visible.scaled_add(lr, &dv);
        for v in h_neg.iter_mut() {
            if self.rng.random_sample() < *v {
                *v = 1.0;
            }
        }
        h_neg.mapv_inplace(f64::floor);
        self.h_samples = h_neg;
    }

    /// One more mini-batch (sklearn's `partial_fit` after the first call).
    pub fn partial_fit(&mut self, x: ArrayView2<f64>) -> Result<()> {
        self.check(x)?;
        self.step(x);
        Ok(())
    }

    /// One Gibbs step `v → h → v` from the model's stream (sklearn's
    /// `gibbs`).
    pub fn gibbs(&mut self, v: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.check(v)?;
        let h = bernoulli(&self.mean_hiddens(v), &mut self.rng);
        Ok(self.sample_visibles(&h))
    }

    /// `F(v) = −v·b_v − Σ_j log(1 + exp(W_j·v + b_h,j))`.
    fn free_energy(&self, v: ArrayView2<f64>) -> Array1<f64> {
        let lin = v.dot(&self.components.t()) + &self.intercept_hidden;
        let soft = lin.mapv(|t| {
            // np.logaddexp(0, t)
            if t > 0.0 {
                t + (-t).exp().ln_1p()
            } else {
                t.exp().ln_1p()
            }
        });
        -v.dot(&self.intercept_visible) - soft.sum_axis(Axis(1))
    }

    /// Pseudo-likelihood of `x` with one random bit flipped per row, drawn
    /// from a fresh `RandomState(seed)` as sklearn's `score_samples`.
    pub fn score_samples(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.check(x)?;
        let (n, p) = x.dim();
        let mut rng = RandomState::new(self.seed);
        let mut flipped = x.to_owned();
        for i in 0..n {
            let j = rng.randint(p as u64) as usize;
            flipped[[i, j]] = 1.0 - flipped[[i, j]];
        }
        let (fe, fe_) = (self.free_energy(x), self.free_energy(flipped.view()));
        Ok((&fe_ - &fe).mapv(|d| {
            // -p · logaddexp(0, -d)
            let t = -d;
            let lae = if t > 0.0 {
                t + (-t).exp().ln_1p()
            } else {
                t.exp().ln_1p()
            };
            -(p as f64) * lae
        }))
    }
}

impl BernoulliRbm {
    /// Weights from `rng.normal(0, 0.01)`, zero intercepts and chain.
    fn init(&self, x: ArrayView2<f64>, mut rng: RandomState) -> Result<FittedBernoulliRbm> {
        let (n, p) = x.dim();
        if n == 0 || p == 0 {
            return Err(RustMlError::EmptyInput("BernoulliRBM needs data".into()));
        }
        if self.n_components == 0 || self.batch_size == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_components and batch_size must be at least 1".into(),
            ));
        }
        let components =
            Array2::from_shape_simple_fn((self.n_components, p), || rng.normal(0.0, 0.01));
        Ok(FittedBernoulliRbm {
            components,
            intercept_hidden: Array1::zeros(self.n_components),
            intercept_visible: Array1::zeros(p),
            h_samples: Array2::zeros((self.batch_size, self.n_components)),
            learning_rate: self.learning_rate,
            seed: self.seed,
            rng,
        })
    }

    /// sklearn's first `partial_fit` call: initialise from `RandomState(seed)`
    /// and take one step on `x` with the same stream, which later calls
    /// continue.
    pub fn partial_fit_first(&self, x: ArrayView2<f64>) -> Result<FittedBernoulliRbm> {
        let mut fitted = self.init(x, RandomState::new(self.seed))?;
        fitted.step(x);
        Ok(fitted)
    }
}

impl FitUnsupervised<f64> for BernoulliRbm {
    type Fitted = FittedBernoulliRbm;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let mut fitted = self.init(x, RandomState::new(self.seed))?;
        // sklearn's gen_even_slices(n_batches * batch_size, n_batches, n_samples=n).
        let n_batches = n.div_ceil(self.batch_size);
        let slices: Vec<(usize, usize)> = (0..n_batches)
            .map(|b| (b * self.batch_size, ((b + 1) * self.batch_size).min(n)))
            .filter(|(s, e)| s < e)
            .collect();
        for _ in 0..self.n_iter {
            for &(s, e) in &slices {
                fitted.step(x.slice(ndarray::s![s..e, ..]));
            }
        }
        // sklearn's `fit` trains with a local generator; `partial_fit` and
        // `gibbs` afterwards start `random_state_` afresh from the seed.
        fitted.rng = RandomState::new(self.seed);
        Ok(fitted)
    }
}

impl Transform<f64> for FittedBernoulliRbm {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.check(x)?;
        Ok(self.mean_hiddens(x))
    }
}
