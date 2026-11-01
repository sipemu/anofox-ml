use anofox_ml_core::{
    DecisionFunction, Fit, FitWeighted, Float, Predict, PredictLogProba, PredictProba, Result,
    RustMlError,
};
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::kernel::{kernel_matrix, SvmKernel};
use crate::solver::{self, KernelSource, QKind, QMatrix, SolverParams};

/// Per-class weights for the SVC penalty: sample `i` gets the upper bound
/// `C · w_{y_i}` (sklearn's `class_weight`). A `Manual` label that is not a
/// class of `y` is an error, as in sklearn.
pub use anofox_ml_core::ClassWeight;

/// Shape of [`FittedSvc::decision_function`] for more than two classes
/// (sklearn's `decision_function_shape`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum DecisionFunctionShape {
    /// `(n_samples, n_classes)`: one-vs-one votes plus a confidence term
    /// in `(-1/3, 1/3)` (sklearn's `_ovr_decision_function`).
    #[default]
    Ovr,
    /// `(n_samples, n_classes·(n_classes−1)/2)`: raw pairwise decision
    /// values, ordered `(0,1), (0,2), …, (1,2), …`; positive favours the
    /// first class of the pair.
    Ovo,
}

fn default_shape() -> DecisionFunctionShape {
    DecisionFunctionShape::Ovr
}

/// Support Vector Classifier with kernel support (unfitted state).
///
/// Every pair of classes is solved as a binary problem (one-vs-one, as
/// libsvm / sklearn) with libsvm's SMO solver (second-order
/// working-set selection, shrinking, kernel column cache), and predictions are
/// made by majority vote. Uses the type-state pattern: call [`Fit::fit`] to
/// produce a [`FittedSvc`] that can make predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Svc {
    /// Regularization parameter. Larger values mean less regularization.
    pub c: f64,
    /// Kernel function to use (default: RBF with `gamma = 'scale'`).
    pub kernel: SvmKernel,
    /// Maximum number of SMO iterations per binary sub-problem (libsvm
    /// caps it at `max(10⁷, 100 l)` by default).
    pub max_iter: usize,
    /// Tolerance of the stopping criterion (maximal violating pair gap,
    /// libsvm / sklearn default 1e-3).
    pub tol: f64,
    /// Use libsvm's shrinking heuristic (default true).
    #[serde(default = "crate::solver::default_true")]
    pub shrinking: bool,
    /// Kernel cache size in MB (default 200).
    #[serde(default = "crate::solver::default_cache_size")]
    pub cache_size: f64,
    /// Random seed (used for the internal cross-validation of
    /// `probability = true`).
    pub seed: u64,
    /// Optional class weights (per-sample upper bound `C · w_y`).
    #[serde(default)]
    pub class_weight: Option<ClassWeight>,
    /// Fit Platt-scaling models (5-fold internal CV per class pair) so that
    /// `predict_proba` is available.
    #[serde(default)]
    pub probability: bool,
    /// Layout of `decision_function` for more than two classes.
    #[serde(default = "default_shape")]
    pub decision_function_shape: DecisionFunctionShape,
    /// With `Ovr` shape and more than two classes, break voting ties by the
    /// decision-function confidence instead of the lowest class index.
    #[serde(default)]
    pub break_ties: bool,
}

impl Svc {
    /// Create a new `Svc` with default parameters.
    pub fn new() -> Self {
        Self {
            c: 1.0,
            kernel: SvmKernel::default(),
            max_iter: 10_000_000,
            tol: 1e-3,
            shrinking: true,
            cache_size: 200.0,
            seed: 0,
            class_weight: None,
            probability: false,
            decision_function_shape: DecisionFunctionShape::Ovr,
            break_ties: false,
        }
    }

    /// Set the regularization parameter C.
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the kernel function.
    pub fn with_kernel(mut self, kernel: SvmKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for the stopping criterion.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Enable or disable the shrinking heuristic.
    pub fn with_shrinking(mut self, shrinking: bool) -> Self {
        self.shrinking = shrinking;
        self
    }

    /// Set the kernel cache size in MB.
    pub fn with_cache_size(mut self, cache_size: f64) -> Self {
        self.cache_size = cache_size;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the class weights.
    pub fn with_class_weight(mut self, class_weight: Option<ClassWeight>) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Enable Platt-scaled probability estimates.
    pub fn with_probability(mut self, probability: bool) -> Self {
        self.probability = probability;
        self
    }

    /// Set the multi-class decision-function layout.
    pub fn with_decision_function_shape(mut self, shape: DecisionFunctionShape) -> Self {
        self.decision_function_shape = shape;
        self
    }

    /// Break voting ties by decision-function confidence.
    pub fn with_break_ties(mut self, break_ties: bool) -> Self {
        self.break_ties = break_ties;
        self
    }

    /// Validate parameters before fitting.
    fn validate(&self) -> Result<()> {
        if !(self.c.is_finite() && self.c > 0.0) {
            return Err(RustMlError::InvalidParameter("C must be positive".into()));
        }
        if self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "max_iter must be at least 1".into(),
            ));
        }
        if self.tol <= 0.0 {
            return Err(RustMlError::InvalidParameter("tol must be positive".into()));
        }
        self.kernel.validate(true)?;
        if self.break_ties && self.decision_function_shape == DecisionFunctionShape::Ovo {
            return Err(RustMlError::InvalidParameter(
                "break_ties must be false when decision_function_shape is Ovo".into(),
            ));
        }
        Ok(())
    }
}

impl Default for Svc {
    fn default() -> Self {
        Self::new()
    }
}

/// One binary (one-vs-one) sub-model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
struct PairModel<F: Float> {
    /// Index of the "positive" class (decision > 0 votes for it).
    class_i: usize,
    /// Index of the "negative" class.
    class_j: usize,
    /// `(position in FittedSvc::support, alpha · y)` for the pair's SVs.
    coef: Vec<(usize, F)>,
    /// Bias term.
    bias: F,
    /// Platt-scaling sigmoid `P(class_i) = 1 / (1 + exp(A f + B))`.
    prob_a: f64,
    prob_b: f64,
}

/// Fitted Support Vector Classifier.
///
/// Holds the union of the support vectors of all one-vs-one sub-problems
/// (ordered by class, then by training index, as sklearn's `support_`) and
/// one sub-model per class pair.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedSvc<F: Float> {
    /// Unique sorted class labels.
    class_labels: Vec<F>,
    /// Kernel with gamma resolved.
    kernel: SvmKernel,
    /// Training indices of the support vectors.
    support: Vec<usize>,
    /// Support vectors (empty for the precomputed kernel).
    support_vectors: Array2<F>,
    /// Number of support vectors per class.
    n_support: Vec<usize>,
    /// One sub-model per class pair `(i, j)`, `i < j`, in lexicographic order.
    pairs: Vec<PairModel<F>>,
    /// Number of input columns expected at prediction time.
    n_features_in: usize,
    decision_function_shape: DecisionFunctionShape,
    break_ties: bool,
    probability: bool,
}

impl<F: Float> FittedSvc<F> {
    /// Returns the class labels.
    pub fn class_labels(&self) -> &[F] {
        &self.class_labels
    }

    /// Returns the support vectors (union over all class pairs, sklearn's
    /// `support_vectors_`). Empty for the precomputed kernel.
    pub fn support_vectors(&self) -> Array2<F> {
        self.support_vectors.clone()
    }

    /// Training-set indices of the support vectors (sklearn's `support_`).
    pub fn support(&self) -> &[usize] {
        &self.support
    }

    /// Total number of (distinct) support vectors.
    pub fn n_support(&self) -> usize {
        self.support.len()
    }

    /// Number of support vectors per class (sklearn's `n_support_`).
    pub fn n_support_per_class(&self) -> &[usize] {
        &self.n_support
    }

    /// Kernel used by the fitted model (gamma resolved).
    pub fn kernel(&self) -> &SvmKernel {
        &self.kernel
    }

    /// Pairwise intercepts in libsvm sign convention (sklearn's
    /// `intercept_` for more than two classes; negated for two classes).
    pub fn intercepts(&self) -> Vec<F> {
        self.pairs.iter().map(|p| p.bias).collect()
    }

    /// Platt-scaling parameters `(A, B)` per class pair (sklearn's
    /// `probA_`, `probB_`), or `None` if fitted without `probability`.
    pub fn prob_params(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        self.probability.then(|| {
            (
                self.pairs.iter().map(|p| p.prob_a).collect(),
                self.pairs.iter().map(|p| p.prob_b).collect(),
            )
        })
    }

    /// Pairwise decision values (libsvm convention: positive favours the
    /// first class of the pair), shape `(n_samples, n_pairs)`.
    fn pairwise_decision(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput(
                "prediction input must not be empty".into(),
            ));
        }
        if x.ncols() != self.n_features_in {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features_in,
                x.ncols()
            )));
        }
        let k = if self.kernel.is_precomputed() {
            x.select(Axis(1), &self.support)
        } else {
            kernel_matrix(x, &self.support_vectors, &self.kernel)
        };
        let mut dec = Array2::<F>::zeros((x.nrows(), self.pairs.len()));
        for (p, pair) in self.pairs.iter().enumerate() {
            for i in 0..x.nrows() {
                let mut f = pair.bias;
                for &(pos, c) in &pair.coef {
                    f += c * k[[i, pos]];
                }
                dec[[i, p]] = f;
            }
        }
        Ok(dec)
    }

    /// Decision function.
    ///
    /// * Two classes: shape `(n_samples, 1)`; positive values favour
    ///   `class_labels[1]` (sklearn returns the same values with shape
    ///   `(n_samples,)`).
    /// * More classes, `Ovr` (default): shape `(n_samples, n_classes)` with
    ///   one-vs-one votes plus a confidence term, as sklearn.
    /// * More classes, `Ovo`: shape `(n_samples, n_classes·(n_classes−1)/2)`.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let dec = self.pairwise_decision(x)?;
        let k = self.class_labels.len();
        if k == 2 {
            return Ok(dec.mapv(|v| -v));
        }
        match self.decision_function_shape {
            DecisionFunctionShape::Ovo => Ok(dec),
            DecisionFunctionShape::Ovr => Ok(ovr_decision_function(&dec, k)),
        }
    }

    /// Platt-scaled class probabilities with Wu–Lin–Weng pairwise coupling
    /// (libsvm). Requires `probability = true` at fit time.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if !self.probability {
            return Err(RustMlError::InvalidParameter(
                "predict_proba is not available when fitted with probability=false".into(),
            ));
        }
        let dec = self.pairwise_decision(x)?;
        let k = self.class_labels.len();
        let min_prob = 1e-7;
        let mut out = Array2::<F>::zeros((x.nrows(), k));
        let mut r = vec![vec![0.0f64; k]; k];
        for i in 0..x.nrows() {
            for (p, pair) in self.pairs.iter().enumerate() {
                let f = dec[[i, p]].to_f64().unwrap();
                let v =
                    sigmoid_predict(f, pair.prob_a, pair.prob_b).clamp(min_prob, 1.0 - min_prob);
                r[pair.class_i][pair.class_j] = v;
                r[pair.class_j][pair.class_i] = 1.0 - v;
            }
            let probs = if k == 2 {
                vec![r[0][1], r[1][0]]
            } else {
                multiclass_probability(&r)
            };
            for (c, &v) in probs.iter().enumerate() {
                out[[i, c]] = F::from_f64(v).unwrap();
            }
        }
        Ok(out)
    }
}

/// sklearn's `_ovr_decision_function(dec < 0, -dec, n_classes)`.
fn ovr_decision_function<F: Float>(dec: &Array2<F>, k: usize) -> Array2<F> {
    let n = dec.nrows();
    let mut votes = Array2::<F>::zeros((n, k));
    let mut conf = Array2::<F>::zeros((n, k));
    let three = F::from_f64(3.0).unwrap();
    let mut p = 0;
    for i in 0..k {
        for j in (i + 1)..k {
            for s in 0..n {
                let d = dec[[s, p]];
                // confidences = -dec: sum_conf[i] -= -d, sum_conf[j] += -d
                conf[[s, i]] += d;
                conf[[s, j]] -= d;
                if d < F::zero() {
                    votes[[s, j]] += F::one();
                } else {
                    votes[[s, i]] += F::one();
                }
            }
            p += 1;
        }
    }
    let transformed = conf.mapv(|c| c / (three * (c.abs() + F::one())));
    votes + transformed
}

impl<F: Float> Predict<F> for FittedSvc<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let k = self.class_labels.len();
        if self.break_ties && k > 2 && self.decision_function_shape == DecisionFunctionShape::Ovr {
            let d = self.decision_function(x)?;
            return Ok(d.map_axis(Axis(1), |row| {
                let mut best = 0;
                for (c, &v) in row.iter().enumerate() {
                    if v > row[best] {
                        best = c;
                    }
                }
                self.class_labels[best]
            }));
        }
        let dec = self.pairwise_decision(x)?;
        // libsvm voting: dec > 0 votes for class_i, otherwise class_j; ties
        // go to the lowest class index.
        let mut out = Array1::<F>::zeros(x.nrows());
        let mut votes = vec![0usize; k];
        for i in 0..x.nrows() {
            votes.iter_mut().for_each(|v| *v = 0);
            for (p, pair) in self.pairs.iter().enumerate() {
                if dec[[i, p]] > F::zero() {
                    votes[pair.class_i] += 1;
                } else {
                    votes[pair.class_j] += 1;
                }
            }
            let mut best = 0;
            for c in 1..k {
                if votes[c] > votes[best] {
                    best = c;
                }
            }
            out[i] = self.class_labels[best];
        }
        Ok(out)
    }
}

impl<F: Float> DecisionFunction<F> for FittedSvc<F> {
    fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>> {
        FittedSvc::decision_function(self, x)
    }
}

impl<F: Float> PredictProba<F> for FittedSvc<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        FittedSvc::predict_proba(self, x)
    }
}

/// `log(predict_proba)`, as sklearn's `SVC.predict_log_proba`.
impl<F: Float> PredictLogProba<F> for FittedSvc<F> {}

impl<F: Float> anofox_ml_core::ClassifierScore<F> for FittedSvc<F> {}

/// Extract unique sorted class labels from y.
fn extract_class_labels<F: Float>(y: &Array1<F>) -> Vec<F> {
    let mut labels: Vec<F> = y.to_vec();
    labels.sort_by(anofox_ml_core::labels::label_cmp);
    labels.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-12).unwrap());
    labels
}

/// Per-class weights for `class_labels` (sklearn's `compute_class_weight`).
fn class_weights<F: Float>(
    cw: Option<&ClassWeight>,
    labels: &[F],
    class_idx: &[usize],
) -> Result<Vec<f64>> {
    let k = labels.len();
    match cw {
        None => Ok(vec![1.0; k]),
        Some(ClassWeight::Balanced) => {
            let mut counts = vec![0usize; k];
            for &c in class_idx {
                counts[c] += 1;
            }
            let n = class_idx.len() as f64;
            Ok(counts.iter().map(|&c| n / (k as f64 * c as f64)).collect())
        }
        Some(ClassWeight::Manual(map)) => {
            let mut w = vec![1.0; k];
            for &(label, weight) in map {
                if !(weight.is_finite() && weight >= 0.0) {
                    return Err(RustMlError::InvalidParameter(format!(
                        "class weight for {label} must be finite and >= 0"
                    )));
                }
                let pos = labels
                    .iter()
                    .position(|l| (l.to_f64().unwrap() - label).abs() < 1e-12)
                    .ok_or_else(|| {
                        RustMlError::InvalidParameter(format!(
                            "class_weight label {label} is not a class of y"
                        ))
                    })?;
                w[pos] = weight;
            }
            Ok(w)
        }
    }
}

// ---------------------------------------------------------------------------
// Simplified SMO on a precomputed (sub-)Gram matrix with per-sample upper
// bounds C_i. (A full libsvm-style solver is tracked separately.)
// ---------------------------------------------------------------------------

/// Training data of one binary sub-problem: feature rows (kernel evaluated
/// on demand) or a precomputed Gram matrix.
enum PairData {
    Rows(Array2<f64>),
    Gram(Array2<f64>),
}

impl PairData {
    fn subset(&self, idx: &[usize]) -> PairData {
        match self {
            PairData::Rows(x) => PairData::Rows(x.select(Axis(0), idx)),
            PairData::Gram(k) => PairData::Gram(k.select(Axis(0), idx).select(Axis(1), idx)),
        }
    }

    fn source<'a>(&'a self, kernel: &'a SvmKernel) -> KernelSource<'a> {
        match self {
            PairData::Rows(x) => KernelSource::Data { x, kernel },
            PairData::Gram(k) => KernelSource::Precomputed(k),
        }
    }

    fn kernel_value(&self, kernel: &SvmKernel, i: usize, j: usize) -> f64 {
        match self {
            PairData::Rows(x) => kernel.compute(&x.row(i), &x.row(j)),
            PairData::Gram(k) => k[[i, j]],
        }
    }
}

/// Dual formulation of the binary sub-problems.
#[derive(Clone, Copy)]
pub(crate) enum Formulation {
    /// C-SVC (per-sample upper bounds `C · w`).
    C,
    /// ν-SVC with the given ν (upper bounds 1, libsvm's `solve_nu_svc`).
    Nu(f64),
}

/// Solver settings shared by the sub-problems.
struct SvcSolve<'a> {
    kernel: &'a SvmKernel,
    params: SolverParams,
    cache_size: f64,
    formulation: Formulation,
}

/// C-SVC dual of one binary sub-problem with ±1 labels `y` and per-sample
/// upper bounds `c`. Returns `(alpha, bias)` with
/// `f(x) = Σ α_i y_i K(x_i, x) + b`.
fn train_binary(data: &PairData, y: &[f64], c: &[f64], solve: &SvcSolve) -> (Vec<f64>, f64) {
    let l = y.len();
    let mut q = QMatrix::new(
        QKind::Classification,
        data.source(solve.kernel),
        y.to_vec(),
        solve.cache_size,
    );
    match solve.formulation {
        Formulation::C => {
            let res = solver::solve(
                &mut q,
                vec![-1.0; l],
                vec![0.0; l],
                c.to_vec(),
                &solve.params,
            );
            (res.alpha, -res.rho)
        }
        Formulation::Nu(nu) => {
            // libsvm solve_nu_svc: feasible start with Σ_{y=±1} α = ν l / 2.
            let (mut sum_pos, mut sum_neg) = (nu * l as f64 / 2.0, nu * l as f64 / 2.0);
            let alpha0: Vec<f64> = y
                .iter()
                .map(|&yi| {
                    let s = if yi > 0.0 { &mut sum_pos } else { &mut sum_neg };
                    let a = s.min(1.0);
                    *s -= a;
                    a
                })
                .collect();
            let params = SolverParams {
                nu: true,
                ..solve.params
            };
            let res = solver::solve(&mut q, vec![0.0; l], alpha0, vec![1.0; l], &params);
            let r = res.r;
            (res.alpha.iter().map(|a| a / r).collect(), -res.rho / r)
        }
    }
}

// ---------------------------------------------------------------------------
// Probability estimates (libsvm: svm_binary_svc_probability, sigmoid_train,
// multiclass_probability).
// ---------------------------------------------------------------------------

/// Platt's sigmoid fit with Lin, Lin & Weng's (2007) Newton method, exactly
/// as libsvm's `sigmoid_train`. Labels are `> 0` for the positive class.
fn sigmoid_train(dec: &[f64], labels: &[f64]) -> (f64, f64) {
    let l = dec.len();
    let prior1 = labels.iter().filter(|&&v| v > 0.0).count() as f64;
    let prior0 = l as f64 - prior1;
    let max_iter = 100;
    let min_step = 1e-10;
    let sigma = 1e-12;
    let eps = 1e-5;
    let hi_target = (prior1 + 1.0) / (prior1 + 2.0);
    let lo_target = 1.0 / (prior0 + 2.0);
    let t: Vec<f64> = labels
        .iter()
        .map(|&v| if v > 0.0 { hi_target } else { lo_target })
        .collect();
    let mut a = 0.0;
    let mut b = ((prior0 + 1.0) / (prior1 + 1.0)).ln();
    let objective = |a: f64, b: f64| -> f64 {
        let mut f = 0.0;
        for i in 0..l {
            let fapb = dec[i] * a + b;
            if fapb >= 0.0 {
                f += t[i] * fapb + (1.0 + (-fapb).exp()).ln();
            } else {
                f += (t[i] - 1.0) * fapb + (1.0 + fapb.exp()).ln();
            }
        }
        f
    };
    let mut fval = objective(a, b);
    for _ in 0..max_iter {
        let (mut h11, mut h22, mut h21, mut g1, mut g2) = (sigma, sigma, 0.0, 0.0, 0.0);
        for i in 0..l {
            let fapb = dec[i] * a + b;
            let (p, q) = if fapb >= 0.0 {
                let e = (-fapb).exp();
                (e / (1.0 + e), 1.0 / (1.0 + e))
            } else {
                let e = fapb.exp();
                (1.0 / (1.0 + e), e / (1.0 + e))
            };
            let d2 = p * q;
            h11 += dec[i] * dec[i] * d2;
            h22 += d2;
            h21 += dec[i] * d2;
            let d1 = t[i] - p;
            g1 += dec[i] * d1;
            g2 += d1;
        }
        if g1.abs() < eps && g2.abs() < eps {
            break;
        }
        let det = h11 * h22 - h21 * h21;
        let da = -(h22 * g1 - h21 * g2) / det;
        let db = -(-h21 * g1 + h11 * g2) / det;
        let gd = g1 * da + g2 * db;
        let mut stepsize = 1.0;
        while stepsize >= min_step {
            let na = a + stepsize * da;
            let nb = b + stepsize * db;
            let newf = objective(na, nb);
            if newf < fval + 0.0001 * stepsize * gd {
                a = na;
                b = nb;
                fval = newf;
                break;
            }
            stepsize /= 2.0;
        }
        if stepsize < min_step {
            break;
        }
    }
    (a, b)
}

fn sigmoid_predict(f: f64, a: f64, b: f64) -> f64 {
    let fapb = f * a + b;
    if fapb >= 0.0 {
        (-fapb).exp() / (1.0 + (-fapb).exp())
    } else {
        1.0 / (1.0 + fapb.exp())
    }
}

/// Wu, Lin & Weng (2004) pairwise coupling, method 2 (libsvm's
/// `multiclass_probability`). `r[i][j]` estimates `P(i | i or j)`.
fn multiclass_probability(r: &[Vec<f64>]) -> Vec<f64> {
    let k = r.len();
    let max_iter = 100.max(k);
    let eps = 0.005 / k as f64;
    let mut p = vec![1.0 / k as f64; k];
    let mut q = vec![vec![0.0; k]; k];
    for t in 0..k {
        for j in 0..t {
            q[t][t] += r[j][t] * r[j][t];
            q[t][j] = q[j][t];
        }
        for j in (t + 1)..k {
            q[t][t] += r[j][t] * r[j][t];
            q[t][j] = -r[j][t] * r[t][j];
        }
    }
    let mut qp = vec![0.0; k];
    for _ in 0..max_iter {
        let mut pqp = 0.0;
        for t in 0..k {
            qp[t] = (0..k).map(|j| q[t][j] * p[j]).sum();
            pqp += p[t] * qp[t];
        }
        let max_error = (0..k).map(|t| (qp[t] - pqp).abs()).fold(0.0, f64::max);
        if max_error < eps {
            break;
        }
        for t in 0..k {
            let diff = (-qp[t] + pqp) / q[t][t];
            p[t] += diff;
            pqp = (pqp + diff * (diff * q[t][t] + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff);
            for j in 0..k {
                qp[j] = (qp[j] + diff * q[t][j]) / (1.0 + diff);
                p[j] /= 1.0 + diff;
            }
        }
    }
    p
}

/// libsvm's `svm_binary_svc_probability`: 5-fold CV decision values on the
/// pair sub-problem, then a Platt sigmoid fit.
fn binary_svc_probability(
    data: &PairData,
    y: &[f64],
    c: &[f64],
    solve: &SvcSolve,
    rng: &mut StdRng,
) -> (f64, f64) {
    let l = y.len();
    let nr_fold = 5;
    let mut perm: Vec<usize> = (0..l).collect();
    for i in 0..l {
        let j = rng.gen_range(i..l);
        perm.swap(i, j);
    }
    let mut dec = vec![0.0f64; l];
    for fold in 0..nr_fold {
        let begin = fold * l / nr_fold;
        let end = (fold + 1) * l / nr_fold;
        let train: Vec<usize> = perm[..begin]
            .iter()
            .chain(perm[end..].iter())
            .copied()
            .collect();
        let test = &perm[begin..end];
        let p_count = train.iter().filter(|&&t| y[t] > 0.0).count();
        let n_count = train.len() - p_count;
        if p_count == 0 && n_count == 0 {
            test.iter().for_each(|&t| dec[t] = 0.0);
        } else if n_count == 0 {
            test.iter().for_each(|&t| dec[t] = 1.0);
        } else if p_count == 0 {
            test.iter().for_each(|&t| dec[t] = -1.0);
        } else {
            let y_tr: Vec<f64> = train.iter().map(|&t| y[t]).collect();
            let c_tr: Vec<f64> = train.iter().map(|&t| c[t]).collect();
            let (alpha, b) = train_binary(&data.subset(&train), &y_tr, &c_tr, solve);
            for &t in test {
                let mut f = b;
                for (pos, &tr) in train.iter().enumerate() {
                    if alpha[pos] > 0.0 {
                        f += alpha[pos] * y_tr[pos] * data.kernel_value(solve.kernel, tr, t);
                    }
                }
                dec[t] = f;
            }
        }
    }
    sigmoid_train(&dec, y)
}

impl Svc {
    /// Shared fit with optional sample weights (`C_i = C · w_{y_i} · sw_i`).
    /// ν-SVC fit (used by [`crate::NuSvc`]): every class pair is solved as
    /// libsvm's ν dual; `self.c` and class weights are ignored.
    pub(crate) fn fit_nu<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        nu: f64,
    ) -> Result<FittedSvc<F>> {
        self.fit_impl(x, y, None, Formulation::Nu(nu))
    }

    fn fit_impl<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
        formulation: Formulation,
    ) -> Result<FittedSvc<F>> {
        self.validate()?;

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
        if self.kernel.is_precomputed() && x.nrows() != x.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "precomputed kernel: X must be a square Gram matrix, got {}x{}",
                x.nrows(),
                x.ncols()
            )));
        }
        if let Some(w) = sample_weight {
            if w.len() != y.len() {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight len {} != y len {}",
                    w.len(),
                    y.len()
                )));
            }
            if w.iter().any(|v| !v.is_finite() || *v < F::zero()) {
                return Err(RustMlError::InvalidParameter(
                    "sample_weight must be finite and non-negative".into(),
                ));
            }
        }

        let class_labels = extract_class_labels(y);
        let k_classes = class_labels.len();
        if k_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 distinct classes for classification".into(),
            ));
        }
        let eps = F::from_f64(1e-12).unwrap();
        let class_idx: Vec<usize> = y
            .iter()
            .map(|&v| {
                class_labels
                    .iter()
                    .position(|&c| (c - v).abs() < eps)
                    .unwrap()
            })
            .collect();
        let weights = class_weights(self.class_weight.as_ref(), &class_labels, &class_idx)?;
        let c = self.c;
        let upper: Vec<F> = (0..y.len())
            .map(|i| {
                let sw = sample_weight.map_or(F::one(), |w| w[i]);
                F::from_f64(c * weights[class_idx[i]]).unwrap() * sw
            })
            .collect();

        let kernel = self.kernel.resolve(x);
        let x64: Array2<f64> = x.mapv(|v| v.to_f64().unwrap());
        let solve = SvcSolve {
            kernel: &kernel,
            params: SolverParams {
                eps: self.tol,
                shrinking: self.shrinking,
                max_iter: Some(self.max_iter),
                nu: false,
            },
            cache_size: self.cache_size,
            formulation,
        };
        let by_class: Vec<Vec<usize>> = (0..k_classes)
            .map(|c| (0..y.len()).filter(|&i| class_idx[i] == c).collect())
            .collect();

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut is_sv = vec![false; y.len()];
        // `coef` holds global training indices until the support set is known.
        let mut pairs: Vec<PairModel<F>> = Vec::new();
        for i in 0..k_classes {
            for j in (i + 1)..k_classes {
                let sub: Vec<usize> = by_class[i]
                    .iter()
                    .chain(by_class[j].iter())
                    .copied()
                    .collect();
                let y_sub: Vec<f64> = sub
                    .iter()
                    .map(|&s| if class_idx[s] == i { 1.0 } else { -1.0 })
                    .collect();
                let c_sub: Vec<f64> = sub.iter().map(|&s| upper[s].to_f64().unwrap()).collect();
                let data = if kernel.is_precomputed() {
                    PairData::Gram(x64.select(Axis(0), &sub).select(Axis(1), &sub))
                } else {
                    PairData::Rows(x64.select(Axis(0), &sub))
                };
                let (prob_a, prob_b) = if self.probability {
                    binary_svc_probability(&data, &y_sub, &c_sub, &solve, &mut rng)
                } else {
                    (0.0, 0.0)
                };
                let (alpha, bias) = train_binary(&data, &y_sub, &c_sub, &solve);
                if !bias.is_finite() || alpha.iter().any(|a| !a.is_finite()) {
                    return Err(RustMlError::InvalidParameter(
                        "the dual coefficients or intercept are not finite (degenerate \
                         problem, e.g. nu too large); preprocess the data or change the \
                         parameters"
                            .into(),
                    ));
                }
                let bias = F::from_f64(bias).unwrap();
                let mut coef = Vec::new();
                for (pos, &s) in sub.iter().enumerate() {
                    if alpha[pos] > 0.0 {
                        is_sv[s] = true;
                        coef.push((s, F::from_f64(alpha[pos] * y_sub[pos]).unwrap()));
                    }
                }
                pairs.push(PairModel {
                    class_i: i,
                    class_j: j,
                    coef,
                    bias,
                    prob_a,
                    prob_b,
                });
            }
        }

        // Union of support vectors ordered by class, then training index.
        let mut support = Vec::new();
        let mut n_support = vec![0usize; k_classes];
        let mut position = vec![usize::MAX; y.len()];
        for (c, members) in by_class.iter().enumerate() {
            for &s in members {
                if is_sv[s] {
                    position[s] = support.len();
                    support.push(s);
                    n_support[c] += 1;
                }
            }
        }
        let support_vectors = if kernel.is_precomputed() {
            Array2::zeros((0, x.ncols()))
        } else {
            x.select(Axis(0), &support)
        };
        for pair in &mut pairs {
            for entry in &mut pair.coef {
                entry.0 = position[entry.0];
            }
        }

        Ok(FittedSvc {
            class_labels,
            kernel,
            support,
            support_vectors,
            n_support,
            pairs,
            n_features_in: x.ncols(),
            decision_function_shape: self.decision_function_shape,
            break_ties: self.break_ties,
            probability: self.probability,
        })
    }
}

impl<F: Float> Fit<F> for Svc {
    type Fitted = FittedSvc<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.fit_impl(x, y, None, Formulation::C)
    }
}

impl<F: Float> FitWeighted<F> for Svc {
    type Fitted = FittedSvc<F>;

    /// Per-sample weights rescale the upper bound: `C_i = C · w_{y_i} · sw_i`.
    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        self.fit_impl(x, y, sample_weight, Formulation::C)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn well_separated_data() -> (Array2<f64>, Array1<f64>) {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.1],
            [0.1, 0.5],
            [0.2, 0.3],
            [5.0, 5.0],
            [5.5, 5.1],
            [5.1, 5.5],
            [5.2, 5.3]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn test_binary_linear_kernel_f64() {
        let (x, y) = well_separated_data();
        let svc = Svc::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binary_rbf_kernel_f64() {
        let (x, y) = well_separated_data();
        let svc = Svc::new()
            .with_kernel(SvmKernel::rbf(0.5))
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binary_polynomial_kernel_f64() {
        let (x, y) = well_separated_data();
        let svc = Svc::new()
            .with_kernel(SvmKernel::poly(2, 1.0, 1.0))
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binary_rbf_kernel_f32() {
        let x: Array2<f32> = array![
            [0.0, 0.0],
            [0.5, 0.1],
            [0.1, 0.5],
            [0.2, 0.3],
            [5.0, 5.0],
            [5.5, 5.1],
            [5.1, 5.5],
            [5.2, 5.3]
        ];
        let y: Array1<f32> = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let svc = Svc::new()
            .with_kernel(SvmKernel::rbf(0.5))
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f32> = svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0_f32, epsilon = 1e-5);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0_f32, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_support_vectors() {
        let (x, y) = well_separated_data();
        let svc = Svc::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let sv = fitted.support_vectors();
        let n_sv = fitted.n_support();
        assert_eq!(sv.nrows(), n_sv);
        assert!(n_sv > 0, "should have at least one support vector");
        assert!(
            n_sv <= x.nrows(),
            "cannot have more SVs than training samples"
        );
    }

    #[test]
    fn test_decision_function() {
        let (x, y) = well_separated_data();
        let svc = Svc::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let scores = fitted.decision_function(&x).unwrap();
        assert_eq!(scores.nrows(), x.nrows());
        assert_eq!(scores.ncols(), 1); // binary

        // Class 0 should have negative scores; class 1 positive.
        for i in 0..4 {
            assert!(scores[[i, 0]] < 0.0, "expected negative for class 0");
        }
        for i in 4..8 {
            assert!(scores[[i, 0]] > 0.0, "expected positive for class 1");
        }
    }

    #[test]
    fn test_multiclass_svc() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 0.0],
            [5.1, 0.1],
            [5.2, 0.0],
            [0.0, 5.0],
            [0.1, 5.1],
            [0.0, 5.2]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let svc = Svc::new()
            .with_kernel(SvmKernel::rbf(0.5))
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        assert_eq!(fitted.class_labels(), &[0.0, 1.0, 2.0]);

        let preds = fitted.predict(&x).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 3..6 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
        for i in 6..9 {
            assert_abs_diff_eq!(preds[i], 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let svc = Svc::new();
        let result: Result<FittedSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 0.0];

        let svc = Svc::new();
        let result: Result<FittedSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let (x, y) = well_separated_data();
        let svc = Svc::new().with_kernel(SvmKernel::Linear).with_c(10.0);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let x_bad = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_c() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let svc = Svc::new().with_c(-1.0);
        let result: Result<FittedSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_gamma() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let svc = Svc::new().with_kernel(SvmKernel::rbf(-0.5));
        let result: Result<FittedSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_single_class_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 0.0];

        let svc = Svc::new();
        let result: Result<FittedSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_builder_pattern() {
        let svc = Svc::new()
            .with_c(0.5)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(500)
            .with_tol(1e-3)
            .with_seed(42);
        assert_eq!(svc.c, 0.5);
        assert_eq!(svc.max_iter, 500);
        assert_eq!(svc.tol, 1e-3);
        assert_eq!(svc.seed, 42);
        assert!(matches!(svc.kernel, SvmKernel::Linear));
    }

    #[test]
    fn test_default() {
        let svc = Svc::default();
        assert_eq!(svc.c, 1.0);
        assert_eq!(svc.max_iter, 10_000_000);
        assert_eq!(svc.tol, 1e-3);
        assert!(svc.shrinking);
        assert_eq!(svc.cache_size, 200.0);
        assert_eq!(svc.seed, 0);
        assert_eq!(
            svc.kernel,
            SvmKernel::Rbf {
                gamma: crate::Gamma::Scale
            }
        );
    }

    #[test]
    fn test_decision_function_empty_input() {
        let (x, y) = well_separated_data();
        let svc = Svc::new().with_kernel(SvmKernel::Linear).with_c(10.0);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let x_empty = Array2::<f64>::zeros((0, 2));
        let result = fitted.decision_function(&x_empty);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }
}

/// Comparisons against scikit-learn 1.8.0 (`SVC(tol=1e-8, ...)`) on a
/// 3-class, 30-sample synthetic problem.
#[cfg(test)]
mod sklearn_parity {
    use super::*;
    use crate::Gamma;
    use ndarray::array;

    fn data() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let x = array![
            [1.41, 0.32],
            [0.78, 1.79],
            [1.49, -0.78],
            [0.76, -0.12],
            [-0.08, 0.33],
            [0.12, 1.16],
            [0.61, 0.1],
            [0.36, 0.27],
            [1.2, -0.16],
            [0.25, -0.68],
            [0.46, 1.02],
            [3.19, -0.09],
            [4.32, -0.66],
            [2.54, 0.35],
            [3.73, 1.68],
            [2.62, 0.8],
            [1.79, -1.08],
            [2.22, 0.63],
            [3.48, 1.46],
            [2.19, 0.26],
            [0.16, 1.36],
            [-0.37, 4.06],
            [0.59, 2.15],
            [-0.0, 3.12],
            [-0.29, 2.33],
            [0.28, 2.81],
            [0.59, 1.56],
            [0.98, 2.84],
            [1.05, 2.74],
            [0.49, 2.21]
        ];
        let mut y = Array1::zeros(30);
        for i in 0..30 {
            y[i] = (i / 10) as f64;
        }
        let t = array![
            [0.5, 0.5],
            [2.0, 1.0],
            [1.0, 2.0],
            [1.3, 1.1],
            [3.0, -1.0],
            [-1.0, 3.0]
        ];
        (x, y, t)
    }

    fn separated() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.3],
            [0.2, 0.8],
            [1.0, 0.4],
            [4.0, 0.0],
            [4.5, 0.6],
            [5.0, 0.2],
            [4.2, 1.0],
            [2.0, 4.0],
            [2.5, 4.6],
            [1.8, 5.0],
            [2.4, 3.8]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let t = array![[1.0, 1.0], [3.5, 0.5], [2.0, 3.0], [2.5, 2.0]];
        (x, y, t)
    }

    fn assert_close(a: &Array2<f64>, e: &[&[f64]], tol: f64, what: &str) {
        assert_eq!(a.nrows(), e.len(), "{what}");
        for (i, row) in e.iter().enumerate() {
            assert_eq!(a.ncols(), row.len(), "{what}");
            for (j, &v) in row.iter().enumerate() {
                assert!(
                    (a[[i, j]] - v).abs() <= tol,
                    "{what}[{i},{j}]: got {}, expected {v}",
                    a[[i, j]]
                );
            }
        }
    }

    #[test]
    fn gamma_scale_matches_sklearn() {
        let (x, y, _) = data();
        let m: FittedSvc<f64> = Svc::new().fit(&x, &y).unwrap();
        // sklearn SVC()._gamma on this data.
        match m.kernel() {
            SvmKernel::Rbf {
                gamma: Gamma::Value(g),
            } => assert!((g - 0.30898497406071146).abs() < 1e-14, "{g}"),
            k => panic!("unexpected kernel {k:?}"),
        }
    }

    #[test]
    fn ovo_shapes_and_class_order() {
        let (x, y, t) = data();
        let m: FittedSvc<f64> = Svc::new().fit(&x, &y).unwrap();
        assert_eq!(m.pairs.len(), 3);
        let pairs: Vec<(usize, usize)> = m.pairs.iter().map(|p| (p.class_i, p.class_j)).collect();
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2)]);
        assert_eq!(m.decision_function(&t).unwrap().dim(), (6, 3));
        let ovo: FittedSvc<f64> = Svc::new()
            .with_decision_function_shape(DecisionFunctionShape::Ovo)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(ovo.decision_function(&t).unwrap().dim(), (6, 3));
        // support_ is grouped by class and sorted within each class.
        let sup = m.support();
        assert_eq!(m.n_support_per_class().iter().sum::<usize>(), sup.len());
        let mut off = 0;
        for (c, &n) in m.n_support_per_class().iter().enumerate() {
            let block = &sup[off..off + n];
            assert!(block.windows(2).all(|w| w[0] < w[1]));
            assert!(block.iter().all(|&i| y[i] == c as f64));
            off += n;
        }
        // Binary: one column, positive favours class_labels[1].
        let yb = y.mapv(|v| if v == 1.0 { 1.0 } else { 0.0 });
        let b: FittedSvc<f64> = Svc::new().fit(&x, &yb).unwrap();
        let d = b.decision_function(&x).unwrap();
        let p = b.predict(&x).unwrap();
        assert_eq!(d.ncols(), 1);
        for i in 0..x.nrows() {
            assert_eq!(p[i], if d[[i, 0]] >= 0.0 { 1.0 } else { 0.0 });
        }
    }

    #[test]
    fn ovr_transform_matches_sklearn() {
        // sklearn's ovo decision values (SVC(kernel='rbf', tol=1e-8) on
        // `data()`) fed through our transform must give sklearn's ovr output.
        let ovo = array![
            [1.1366567442342639, 0.9370257855164766, -0.06420023111184084],
            [-0.37317404913661645, 0.427095036367066, 0.6565335291340861],
            [0.6620216144738548, -0.6897381323896608, -0.892937933616744],
            [
                0.5153836190494122,
                0.32924969977806906,
                -0.018786041728202613
            ],
            [-0.8972266591452541, 0.3392913343527705, 1.0899151362221828],
            [
                0.15331748172635315,
                -0.8763403622557985,
                -0.8584016571089679
            ]
        ];
        let ovr = ovr_decision_function(&ovo, 3);
        assert_close(
            &ovr,
            &[
                &[2.2248857854467383, -0.18187717917733068, 0.8446508534743369],
                &[1.0170540890900959, 2.1691060638971464, -0.17335600395113593],
                &[0.9910103232319308, -0.20286812358766682, 2.2042682377448],
                &[2.152628946216867, -0.11606053629185177, 0.9210295643218179],
                &[0.8806250563563883, 2.221743942481423, -0.19611431248392866],
                &[0.8601251152456593, -0.16763757247960118, 2.2114449318047824],
            ],
            1e-14,
            "ovr",
        );
    }

    /// Model whose pairwise decision values are fixed constants (no SVs).
    fn constant_model(biases: [f64; 3], break_ties: bool) -> FittedSvc<f64> {
        let pairs = [(0, 1), (0, 2), (1, 2)]
            .iter()
            .zip(biases.iter())
            .map(|(&(i, j), &b)| PairModel {
                class_i: i,
                class_j: j,
                coef: Vec::new(),
                bias: b,
                prob_a: 0.0,
                prob_b: 0.0,
            })
            .collect();
        FittedSvc {
            class_labels: vec![10.0, 20.0, 30.0],
            kernel: SvmKernel::Linear,
            support: Vec::new(),
            support_vectors: Array2::zeros((0, 1)),
            n_support: vec![0, 0, 0],
            pairs,
            n_features_in: 1,
            decision_function_shape: DecisionFunctionShape::Ovr,
            break_ties,
            probability: false,
        }
    }

    #[test]
    fn voting_tie_breaking() {
        let x = array![[0.0]];
        // (0,1) -> 0, (0,2) -> 2, (1,2) -> 1: three-way tie; libsvm gives
        // the lowest class index.
        let m = constant_model([0.5, -0.1, 0.9], false);
        assert_eq!(m.predict(&x).unwrap()[0], 10.0);
        // Same tie, but class 1 has the largest summed confidence:
        // conf = [0.1 + (-0.1), -0.1 + 2.0, 0.1 - 2.0] -> class 1 wins with
        // break_ties.
        let m = constant_model([0.1, -0.1, 2.0], true);
        let d = m.decision_function(&x).unwrap();
        assert!(d.row(0).iter().all(|&v| v > 0.5 && v < 1.5));
        assert_eq!(m.predict(&x).unwrap()[0], 20.0);
        let m = constant_model([0.1, -0.1, 2.0], false);
        assert_eq!(m.predict(&x).unwrap()[0], 10.0);
        // A zero decision value votes for the second class (libsvm `> 0`).
        // (0,1) = 0 -> class 1, (0,2) -> class 0, (1,2) -> class 1.
        let m = constant_model([0.0, 1.0, 1.0], false);
        assert_eq!(m.predict(&x).unwrap()[0], 20.0);
        let m = constant_model([0.0, 0.0, 1.0], false);
        assert_eq!(m.predict(&x).unwrap()[0], 20.0);
    }

    #[test]
    fn well_separated_predictions_match_sklearn() {
        let (x, y, t) = separated();
        // sklearn: SVC(C=10, tol=1e-8, kernel=...).predict(T) == [0, 1, 2, 2]
        for k in [SvmKernel::Linear, SvmKernel::rbf(0.5)] {
            let m: FittedSvc<f64> = Svc::new()
                .with_kernel(k)
                .with_c(10.0)
                .with_tol(1e-8)
                .with_max_iter(10_000)
                .fit(&x, &y)
                .unwrap();
            assert_eq!(m.predict(&t).unwrap(), array![0.0, 1.0, 2.0, 2.0]);
            assert_eq!(m.predict(&x).unwrap(), y);
        }
        // RBF ovr decision function is close to sklearn's (the simplified
        // SMO solver is only approximately optimal; see the ignored exact test).
        let m: FittedSvc<f64> = Svc::new()
            .with_kernel(SvmKernel::rbf(0.5))
            .with_c(10.0)
            .with_tol(1e-8)
            .with_max_iter(10_000)
            .fit(&x, &y)
            .unwrap();
        assert_close(
            &m.decision_function(&t).unwrap(),
            &[
                &[2.207271797413894, -0.15133313254751832, 0.8505532596656649],
                &[-0.14969666361607104, 2.2058677536797933, 0.8518614645077831],
                &[-0.12384856061618367, 0.8819424850541991, 2.1775414642617847],
                &[-0.04972018752290213, 1.0000072566295017, 2.0497149340357703],
            ],
            0.05,
            "rbf ovr",
        );
    }

    #[test]
    fn probability_close_to_sklearn() {
        let (x, y, t) = separated();
        let m: FittedSvc<f64> = Svc::new()
            .with_kernel(SvmKernel::rbf(0.5))
            .with_c(10.0)
            .with_probability(true)
            .with_max_iter(10_000)
            .fit(&x, &y)
            .unwrap();
        let p = m.predict_proba(&t).unwrap();
        // sklearn SVC(probability=True, random_state=0). The internal CV
        // folds differ (different RNG), so only compare loosely.
        let sk = [
            [0.6923434102173265, 0.1553553048195688, 0.15230128496310455],
            [0.14189508995087916, 0.7144363996534185, 0.1436685103957025],
            [0.2028829909285462, 0.213387051537204, 0.5837299575342498],
            [0.2915247233106682, 0.3462496338157718, 0.36222564287355996],
        ];
        for i in 0..4 {
            assert!((p.row(i).sum() - 1.0).abs() < 1e-9);
            for c in 0..3 {
                assert!((p[[i, c]] - sk[i][c]).abs() < 0.25, "{p}");
            }
        }
        // On the three clear points the most probable class agrees.
        for (i, &c) in [0usize, 1, 2].iter().enumerate() {
            let row = p.row(i);
            let best = (0..3)
                .max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap())
                .unwrap();
            assert_eq!(best, c);
        }
        let (a, b) = m.prob_params().unwrap();
        assert_eq!((a.len(), b.len()), (3, 3));
        assert!(a.iter().all(|&v| v < 0.0), "A must be negative: {a:?}");
        // predict_proba without probability=true is an error.
        let m: FittedSvc<f64> = Svc::new().fit(&x, &y).unwrap();
        assert!(m.predict_proba(&t).is_err());
    }

    #[test]
    fn binary_probability_is_platt_of_decision() {
        let (x, y, t) = separated();
        let yb = y.mapv(|v| if v == 1.0 { 1.0 } else { 0.0 });
        let m: FittedSvc<f64> = Svc::new()
            .with_kernel(SvmKernel::rbf(0.5))
            .with_probability(true)
            .fit(&x, &yb)
            .unwrap();
        let p = m.predict_proba(&t).unwrap();
        let d = m.decision_function(&t).unwrap();
        let (a, b) = m.prob_params().unwrap();
        for i in 0..t.nrows() {
            // libsvm: P(class 0) = sigmoid(A * f + B) with f = -decision.
            let p0 = sigmoid_predict(-d[[i, 0]], a[0], b[0]).clamp(1e-7, 1.0 - 1e-7);
            assert!((p[[i, 0]] - p0).abs() < 1e-12);
            assert!((p[[i, 0]] + p[[i, 1]] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn sigmoid_train_minimises_platt_objective() {
        // Reference: scipy BFGS on libsvm's regularised Platt objective.
        let dec = [
            -2.1, -1.3, -0.7, -0.2, 0.1, 0.4, 0.9, 1.5, 2.2, -0.5, 0.3, 1.1,
        ];
        let lab = [
            -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
        ];
        let (a, b) = sigmoid_train(&dec, &lab);
        assert!((a - -1.0595723156237895).abs() < 1e-4, "{a}");
        assert!((b - 0.1637970621652391).abs() < 1e-4, "{b}");
    }

    #[test]
    fn pairwise_coupling_solves_wu_lin_weng_method_2() {
        let r = vec![
            vec![0.0, 0.7, 0.6],
            vec![0.3, 0.0, 0.2],
            vec![0.4, 0.8, 0.0],
        ];
        let p = multiclass_probability(&r);
        // Exact solution of min p'Qp s.t. sum(p) = 1 (numpy linear solve);
        // libsvm stops at max error < 0.005 / k.
        let e = [0.4898989898989899, 0.14646464646464646, 0.36363636363636365];
        for (a, b) in p.iter().zip(e.iter()) {
            assert!((a - b).abs() < 2e-3, "{p:?}");
        }
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn precomputed_kernel_equals_linear() {
        let (x, y, t) = data();
        let lin: FittedSvc<f64> = Svc::new()
            .with_kernel(SvmKernel::Linear)
            .fit(&x, &y)
            .unwrap();
        let gram = x.dot(&x.t());
        let pre: FittedSvc<f64> = Svc::new()
            .with_kernel(SvmKernel::Precomputed)
            .fit(&gram, &y)
            .unwrap();
        assert_eq!(pre.support(), lin.support());
        let kt = t.dot(&x.t());
        let a = lin.decision_function(&t).unwrap();
        let b = pre.decision_function(&kt).unwrap();
        assert!((&a - &b).iter().all(|d| d.abs() < 1e-12));
        assert_eq!(lin.predict(&t).unwrap(), pre.predict(&kt).unwrap());
        // Wrong shapes are rejected.
        assert!(pre.predict(&t).is_err());
        let r: Result<FittedSvc<f64>> = Svc::new().with_kernel(SvmKernel::Precomputed).fit(&x, &y);
        assert!(r.is_err());
    }

    #[test]
    fn class_weight_is_a_per_sample_upper_bound() {
        let (x, y, t) = data();
        let base = Svc::new().with_kernel(SvmKernel::rbf(0.5)).with_c(2.0);
        let cw: FittedSvc<f64> = base
            .clone()
            .with_class_weight(Some(ClassWeight::Manual(vec![(0.0, 3.0), (2.0, 0.5)])))
            .fit(&x, &y)
            .unwrap();
        let sw = y.mapv(|v| match v as i64 {
            0 => 3.0,
            2 => 0.5,
            _ => 1.0,
        });
        let swm: FittedSvc<f64> = base.fit_weighted(&x, &y, Some(&sw)).unwrap();
        let d = cw.decision_function(&t).unwrap() - swm.decision_function(&t).unwrap();
        assert!(d.iter().all(|v| v.abs() < 1e-12));
        // Balanced weights: n / (k * n_c) = 1 for equal class sizes.
        let idx: Vec<usize> = (0..30).map(|i| i / 10).collect();
        let w = class_weights(Some(&ClassWeight::Balanced), &[0.0, 1.0, 2.0], &idx).unwrap();
        assert_eq!(w, vec![1.0, 1.0, 1.0]);
        let idx = [0usize, 0, 0, 1];
        let w = class_weights(Some(&ClassWeight::Balanced), &[0.0, 1.0], &idx).unwrap();
        assert_eq!(w, vec![4.0 / 6.0, 2.0]);
        // Unknown labels in the map are rejected.
        let r: Result<FittedSvc<f64>> = Svc::new()
            .with_class_weight(Some(ClassWeight::Manual(vec![(7.0, 2.0)])))
            .fit(&x, &y);
        assert!(r.is_err());
    }

    #[test]
    fn heavy_class_weight_moves_the_boundary() {
        // Overlapping 1-D classes: up-weighting class 1 must not shrink the
        // region predicted as class 1.
        let x = array![[0.0], [0.4], [0.8], [1.2], [0.6], [1.0], [1.4], [1.8]];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let grid = Array2::from_shape_fn((41, 1), |(i, _)| i as f64 * 0.05);
        let count = |m: &FittedSvc<f64>| {
            m.predict(&grid)
                .unwrap()
                .iter()
                .filter(|&&p| p == 1.0)
                .count()
        };
        let plain: FittedSvc<f64> = Svc::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(0.5)
            .fit(&x, &y)
            .unwrap();
        let heavy: FittedSvc<f64> = Svc::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(0.5)
            .with_class_weight(Some(ClassWeight::Manual(vec![(1.0, 10.0)])))
            .fit(&x, &y)
            .unwrap();
        assert!(
            count(&heavy) >= count(&plain),
            "{} < {}",
            count(&heavy),
            count(&plain)
        );
    }

    #[test]
    fn break_ties_with_ovo_is_rejected() {
        let (x, y, _) = data();
        let r: Result<FittedSvc<f64>> = Svc::new()
            .with_decision_function_shape(DecisionFunctionShape::Ovo)
            .with_break_ties(true)
            .fit(&x, &y);
        assert!(r.is_err());
    }

    #[test]
    fn sigmoid_and_poly_kernels_fit() {
        let (x, y, t) = separated();
        for k in [SvmKernel::sigmoid(0.1, 0.0), SvmKernel::poly(2, 0.5, 1.0)] {
            let m: FittedSvc<f64> = Svc::new().with_kernel(k).fit(&x, &y).unwrap();
            assert_eq!(m.predict(&t).unwrap().len(), 4);
        }
    }

    /// Exact parity with libsvm's optimum. Requires the libsvm-style
    /// working-set solver (issue #62); the simplified SMO only reaches an
    /// approximate optimum on overlapping classes.
    #[test]
    fn exact_parity_with_sklearn_rbf_scale() {
        let (x, y, t) = data();
        let m: FittedSvc<f64> = Svc::new().with_tol(1e-8).fit(&x, &y).unwrap();
        assert_eq!(
            m.support(),
            &[0, 1, 2, 4, 5, 8, 9, 10, 12, 13, 14, 16, 17, 19, 20, 21, 22, 24, 26, 28, 29]
        );
        assert_eq!(m.n_support_per_class(), &[7, 7, 7]);
        assert_eq!(m.predict(&t).unwrap(), array![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let ovo: FittedSvc<f64> = Svc::new()
            .with_tol(1e-8)
            .with_decision_function_shape(DecisionFunctionShape::Ovo)
            .fit(&x, &y)
            .unwrap();
        assert_close(
            &ovo.decision_function(&t).unwrap(),
            &[
                &[1.1366567442342639, 0.9370257855164766, -0.06420023111184084],
                &[-0.37317404913661645, 0.427095036367066, 0.6565335291340861],
                &[0.6620216144738548, -0.6897381323896608, -0.892937933616744],
                &[
                    0.5153836190494122,
                    0.32924969977806906,
                    -0.018786041728202613,
                ],
                &[-0.8972266591452541, 0.3392913343527705, 1.0899151362221828],
                &[
                    0.15331748172635315,
                    -0.8763403622557985,
                    -0.8584016571089679,
                ],
            ],
            1e-6,
            "ovo",
        );
        let yb = y.mapv(|v| if v == 1.0 { 1.0 } else { 0.0 });
        let b: FittedSvc<f64> = Svc::new()
            .with_kernel(SvmKernel::rbf(0.5))
            .with_tol(1e-8)
            .fit(&x, &yb)
            .unwrap();
        assert_close(
            &b.decision_function(&t).unwrap(),
            &[
                &[-1.1507742796892788],
                &[0.5740728445792119],
                &[-0.9417442825287918],
                &[-0.45285576267819094],
                &[0.7418059241763877],
                &[-0.7312684766435076],
            ],
            1e-6,
            "binary",
        );
    }
}
