//! Training engine shared by [`crate::MlpClassifier`] and
//! [`crate::MlpRegressor`] — a line-by-line port of sklearn's
//! `BaseMultilayerPerceptron._fit_stochastic` / `_fit_lbfgs`.

use anofox_ml_core::optim::{minimize_lbfgsb, LbfgsbOptions};
use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::{s, Array1, Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::activation::Activation;
use crate::network::{
    backprop, n_params, pack, pack_grads, unpack, DenseLayer, LayerGradients, OutputKind,
};
use crate::solver::{LearningRateSchedule, Solver, StochasticOptimizer};
use crate::utils::{select_rows, shuffle_indices};

/// Hyperparameters common to both MLP estimators.
#[derive(Debug, Clone)]
pub(crate) struct TrainConfig {
    pub hidden_layer_sizes: Vec<usize>,
    pub activation: Activation,
    pub solver: Solver,
    pub learning_rate: f64,
    pub learning_rate_schedule: LearningRateSchedule,
    pub power_t: f64,
    pub momentum: f64,
    pub nesterovs_momentum: bool,
    pub beta_1: f64,
    pub beta_2: f64,
    pub epsilon: f64,
    pub alpha: f64,
    pub batch_size: Option<usize>,
    pub max_iter: usize,
    pub tol: f64,
    pub n_iter_no_change: usize,
    pub early_stopping: bool,
    pub validation_fraction: f64,
    pub shuffle: bool,
    pub max_fun: usize,
    pub initial_weights: Option<Vec<(Array2<f64>, Array1<f64>)>>,
}

impl TrainConfig {
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(RustMlError::InvalidParameter(m));
        if self.hidden_layer_sizes.is_empty() {
            return bad("hidden_layer_sizes must not be empty".into());
        }
        for (i, &s) in self.hidden_layer_sizes.iter().enumerate() {
            if s == 0 {
                return bad(format!("hidden_layer_sizes[{i}] must be > 0"));
            }
        }
        if self.alpha.is_nan() || self.alpha < 0.0 {
            return bad(format!("alpha must be >= 0, got {}", self.alpha));
        }
        if self.max_iter == 0 {
            return bad("max_iter must be >= 1".into());
        }
        if self.solver != Solver::Lbfgs
            && (self.learning_rate.is_nan() || self.learning_rate <= 0.0)
        {
            return bad(format!(
                "learning_rate must be > 0, got {}",
                self.learning_rate
            ));
        }
        if !(0.0..=1.0).contains(&self.momentum) {
            return bad(format!("momentum must be in [0, 1], got {}", self.momentum));
        }
        if !(0.0..1.0).contains(&self.beta_1) || !(0.0..1.0).contains(&self.beta_2) {
            return bad("beta_1 and beta_2 must be in [0, 1)".into());
        }
        if self.early_stopping
            && !(self.validation_fraction > 0.0 && self.validation_fraction < 1.0)
        {
            return bad(format!(
                "validation_fraction must be in (0, 1), got {}",
                self.validation_fraction
            ));
        }
        if self.batch_size == Some(0) {
            return bad("batch_size must be >= 1".into());
        }
        Ok(())
    }

    fn new_optimizer<F: Float>(&self, layers: &[DenseLayer<F>]) -> StochasticOptimizer<F> {
        match self.solver {
            Solver::Sgd => {
                let (vel_w, vel_b) = StochasticOptimizer::zeros_like(layers);
                StochasticOptimizer::Sgd {
                    learning_rate_init: self.learning_rate,
                    learning_rate: self.learning_rate,
                    schedule: self.learning_rate_schedule,
                    momentum: self.momentum,
                    nesterov: self.nesterovs_momentum,
                    power_t: self.power_t,
                    vel_w,
                    vel_b,
                }
            }
            _ => {
                let (m_w, m_b) = StochasticOptimizer::zeros_like(layers);
                let (v_w, v_b) = StochasticOptimizer::zeros_like(layers);
                StochasticOptimizer::Adam {
                    learning_rate_init: self.learning_rate,
                    beta_1: self.beta_1,
                    beta_2: self.beta_2,
                    epsilon: self.epsilon,
                    t: 0,
                    m_w,
                    v_w,
                    m_b,
                    v_b,
                }
            }
        }
    }
}

/// Mutable training state (sklearn's fitted attributes).
#[derive(Debug, Clone)]
pub(crate) struct TrainState<F: Float> {
    pub layers: Vec<DenseLayer<F>>,
    pub loss_curve: Vec<f64>,
    pub loss: f64,
    pub n_iter: usize,
    pub t: usize,
    pub best_loss: Option<f64>,
    pub no_improvement_count: usize,
    pub validation_scores: Vec<f64>,
    pub best_validation_score: Option<f64>,
    pub optimizer: Option<StochasticOptimizer<F>>,
}

/// Fresh state with Glorot-uniform (or injected) weights; `rng` is the
/// estimator RNG (consumed for initialisation first, as in sklearn).
pub(crate) fn initialize<F: Float>(
    cfg: &TrainConfig,
    n_features: usize,
    n_outputs: usize,
    rng: &mut StdRng,
) -> Result<TrainState<F>> {
    let mut sizes = Vec::with_capacity(cfg.hidden_layer_sizes.len() + 2);
    sizes.push(n_features);
    sizes.extend_from_slice(&cfg.hidden_layer_sizes);
    sizes.push(n_outputs);
    let layers = match &cfg.initial_weights {
        Some(init) => {
            if init.len() != sizes.len() - 1 {
                return Err(RustMlError::InvalidParameter(format!(
                    "initial_weights has {} layers, network has {}",
                    init.len(),
                    sizes.len() - 1
                )));
            }
            let mut layers = Vec::with_capacity(init.len());
            for (i, (w, b)) in init.iter().enumerate() {
                if w.dim() != (sizes[i], sizes[i + 1]) || b.len() != sizes[i + 1] {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "initial_weights[{i}]: expected W {:?} and b ({}), got {:?} and ({})",
                        (sizes[i], sizes[i + 1]),
                        sizes[i + 1],
                        w.dim(),
                        b.len()
                    )));
                }
                layers.push(DenseLayer {
                    weights: w.mapv(|v| F::from_f64(v).unwrap()),
                    biases: b
                        .mapv(|v| F::from_f64(v).unwrap())
                        .insert_axis(ndarray::Axis(0)),
                });
            }
            layers
        }
        None => (0..sizes.len() - 1)
            .map(|i| DenseLayer::new(sizes[i], sizes[i + 1], cfg.activation, rng))
            .collect(),
    };
    let stochastic = cfg.solver != Solver::Lbfgs;
    Ok(TrainState {
        layers,
        loss_curve: Vec::new(),
        loss: f64::NAN,
        n_iter: 0,
        t: 0,
        best_loss: if stochastic && !cfg.early_stopping {
            Some(f64::INFINITY)
        } else {
            None
        },
        no_improvement_count: 0,
        validation_scores: Vec::new(),
        best_validation_score: if cfg.early_stopping {
            Some(f64::NEG_INFINITY)
        } else {
            None
        },
        optimizer: None,
    })
}

/// Validation scorer: `(layers, x_val, y_val_targets) -> score` (higher is
/// better; accuracy for classifiers, R² for regressors).
pub(crate) type Scorer<'a, F> = dyn Fn(&[DenseLayer<F>], ArrayView2<F>, ArrayView2<F>) -> f64 + 'a;

/// Train `state` on `(x, y)` (`y` already encoded as network targets).
/// `strata` (class index per row) stratifies the early-stopping split.
#[allow(clippy::too_many_arguments)]
pub(crate) fn train<F: Float>(
    cfg: &TrainConfig,
    state: &mut TrainState<F>,
    kind: OutputKind,
    x: ArrayView2<F>,
    y: ArrayView2<F>,
    rng: &mut StdRng,
    incremental: bool,
    strata: Option<&[usize]>,
    scorer: &Scorer<'_, F>,
) -> Result<()> {
    match cfg.solver {
        Solver::Lbfgs => {
            if incremental {
                return Err(RustMlError::InvalidParameter(
                    "partial_fit is only available for stochastic solvers (sgd, adam)".into(),
                ));
            }
            fit_lbfgs(cfg, state, kind, x, y)
        }
        _ => fit_stochastic(cfg, state, kind, x, y, rng, incremental, strata, scorer),
    }
}

fn fit_lbfgs<F: Float>(
    cfg: &TrainConfig,
    state: &mut TrainState<F>,
    kind: OutputKind,
    x: ArrayView2<F>,
    y: ArrayView2<F>,
) -> Result<()> {
    let mut p0 = Vec::with_capacity(n_params(&state.layers));
    pack(&state.layers, &mut p0);
    let mut work = state.layers.clone();
    let mut grads = LayerGradients::zeros_like(&work);
    let opts = LbfgsbOptions {
        max_iter: cfg.max_iter,
        max_fun: cfg.max_fun,
        pgtol: cfg.tol,
        ..Default::default()
    };
    let res = minimize_lbfgsb(
        |p, g| {
            unpack(&mut work, p);
            let loss = backprop(
                &work,
                cfg.activation,
                kind,
                x.view(),
                y.view(),
                cfg.alpha,
                &mut grads,
            );
            pack_grads(&grads, g);
            loss
        },
        &p0,
        None,
        &opts,
    )?;
    unpack(&mut state.layers, &res.x);
    state.n_iter = res.n_iter;
    state.loss = res.fun;
    Ok(())
}

/// Split row indices into (train, validation) like sklearn's
/// `train_test_split(test_size=validation_fraction, stratify=...)`:
/// `ceil(fraction · n)` validation rows.
fn validation_split(
    n: usize,
    fraction: f64,
    strata: Option<&[usize]>,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<usize>) {
    let n_val = ((fraction * n as f64).ceil() as usize).clamp(1, n.saturating_sub(1).max(1));
    let mut perm: Vec<usize> = (0..n).collect();
    shuffle_indices(&mut perm, rng);
    let mut val = Vec::with_capacity(n_val);
    if let Some(strata) = strata {
        // Proportional allocation per class (rounded down), remainder
        // filled from the shuffled leftovers.
        let n_classes = strata.iter().copied().max().map_or(0, |m| m + 1);
        let mut counts = vec![0usize; n_classes];
        for &c in strata {
            counts[c] += 1;
        }
        let quota: Vec<usize> = counts
            .iter()
            .map(|&c| ((c as f64) * n_val as f64 / n as f64).floor() as usize)
            .collect();
        let mut taken = vec![0usize; n_classes];
        let mut is_val = vec![false; n];
        for &i in &perm {
            let c = strata[i];
            if taken[c] < quota[c] {
                taken[c] += 1;
                is_val[i] = true;
                val.push(i);
            }
        }
        for &i in &perm {
            if val.len() >= n_val {
                break;
            }
            if !is_val[i] {
                is_val[i] = true;
                val.push(i);
            }
        }
        let train = perm.iter().copied().filter(|&i| !is_val[i]).collect();
        (train, val)
    } else {
        val.extend_from_slice(&perm[..n_val]);
        (perm[n_val..].to_vec(), val)
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_stochastic<F: Float>(
    cfg: &TrainConfig,
    state: &mut TrainState<F>,
    kind: OutputKind,
    x: ArrayView2<F>,
    y: ArrayView2<F>,
    rng: &mut StdRng,
    incremental: bool,
    strata: Option<&[usize]>,
    scorer: &Scorer<'_, F>,
) -> Result<()> {
    if !incremental || state.optimizer.is_none() {
        state.optimizer = Some(cfg.new_optimizer(&state.layers));
    }
    if cfg.early_stopping && incremental {
        return Err(RustMlError::InvalidParameter(
            "partial_fit does not support early_stopping=true".into(),
        ));
    }
    let split;
    let (x_train, y_train, val): (_, _, Option<(Array2<F>, Array2<F>)>) = if cfg.early_stopping {
        let (tr, va) = validation_split(x.nrows(), cfg.validation_fraction, strata, rng);
        if va.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "the validation set is too small; increase validation_fraction or the dataset size"
                    .into(),
            ));
        }
        split = (select_rows(x, &tr), select_rows(y, &tr));
        (
            split.0.view(),
            split.1.view(),
            Some((select_rows(x, &va), select_rows(y, &va))),
        )
    } else {
        (x.view(), y.view(), None)
    };

    let n = x_train.nrows();
    let batch_size = cfg.batch_size.unwrap_or(n).clamp(1, n);
    let full_batch = batch_size >= n;
    let mut idx: Vec<usize> = (0..n).collect();
    let mut grads = LayerGradients::zeros_like(&state.layers);
    let mut best_layers = if cfg.early_stopping {
        Some(state.layers.clone())
    } else {
        None
    };
    let optimizer = state.optimizer.as_mut().unwrap();

    state.n_iter = 0;
    for _ in 0..cfg.max_iter {
        if cfg.shuffle {
            shuffle_indices(&mut idx, rng);
        }
        let mut accumulated = 0.0;
        let mut start = 0;
        while start < n {
            let end = (start + batch_size).min(n);
            let loss = if full_batch {
                // Full batch: the gradient is permutation invariant — no copy.
                backprop(
                    &state.layers,
                    cfg.activation,
                    kind,
                    x_train.view(),
                    y_train.view(),
                    cfg.alpha,
                    &mut grads,
                )
            } else if cfg.shuffle {
                let xb = select_rows(x_train.view(), &idx[start..end]);
                let yb = select_rows(y_train.view(), &idx[start..end]);
                backprop(
                    &state.layers,
                    cfg.activation,
                    kind,
                    xb.view(),
                    yb.view(),
                    cfg.alpha,
                    &mut grads,
                )
            } else {
                backprop(
                    &state.layers,
                    cfg.activation,
                    kind,
                    x_train.slice(s![start..end, ..]),
                    y_train.slice(s![start..end, ..]),
                    cfg.alpha,
                    &mut grads,
                )
            };
            accumulated += loss * (end - start) as f64;
            optimizer.update(&mut state.layers, &grads);
            start = end;
        }
        state.n_iter += 1;
        state.loss = accumulated / n as f64;
        state.t += n;
        state.loss_curve.push(state.loss);

        // sklearn `_update_no_improvement_count`
        if let Some((xv, yv)) = &val {
            let score = scorer(&state.layers, xv.view(), yv.view());
            state.validation_scores.push(score);
            let best = state.best_validation_score.unwrap_or(f64::NEG_INFINITY);
            if score.is_nan() || score < best + cfg.tol {
                state.no_improvement_count += 1;
            } else {
                state.no_improvement_count = 0;
            }
            if score > best {
                state.best_validation_score = Some(score);
                best_layers = Some(state.layers.clone());
            }
        } else {
            let best = state.best_loss.unwrap_or(f64::INFINITY);
            if state.loss > best - cfg.tol {
                state.no_improvement_count += 1;
            } else {
                state.no_improvement_count = 0;
            }
            if state.loss < best {
                state.best_loss = Some(state.loss);
            }
        }

        optimizer.iteration_ends(state.t);
        if state.no_improvement_count > cfg.n_iter_no_change {
            if optimizer.trigger_stopping() {
                break;
            }
            state.no_improvement_count = 0;
        }
        if incremental {
            break;
        }
    }
    if let Some(best) = best_layers {
        state.layers = best;
    }
    let finite = state.layers.iter().all(|l| {
        l.weights
            .iter()
            .chain(l.biases.iter())
            .all(|v| v.is_finite())
    });
    if !finite {
        return Err(RustMlError::InvalidParameter(
            "solver produced non-finite parameter weights; the input data may contain large \
             values and need to be preprocessed"
                .into(),
        ));
    }
    Ok(())
}

/// Estimator RNG for one `fit` / `partial_fit` call (sklearn re-creates
/// `check_random_state(random_state)` on every call).
pub(crate) fn rng_for(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}
