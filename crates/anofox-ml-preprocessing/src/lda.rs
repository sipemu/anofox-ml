//! Latent Dirichlet allocation by variational Bayes (sklearn's
//! `decomposition.LatentDirichletAllocation`, batch or online updates).
//!
//! The topic-word parameters start from `RandomState(seed).gamma(100, 0.01)`
//! and every training E-step from fresh `gamma(100, 0.01)` document draws of
//! the same stream, in sklearn's order; the digamma is sklearn's own
//! (`_online_lda_fast.psi`). A seeded fit therefore equals sklearn's up to
//! floating-point summation order.

use crate::special::ln_gamma;
use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// sklearn's `learning_method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LdaLearning {
    /// All documents per EM step.
    #[default]
    Batch,
    /// Mini-batches of `batch_size` with the `learning_offset` /
    /// `learning_decay` step size.
    Online,
}

/// sklearn's `LatentDirichletAllocation`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LatentDirichletAllocation {
    pub n_components: usize,
    /// `None`: `1 / n_components`.
    pub doc_topic_prior: Option<f64>,
    /// `None`: `1 / n_components`.
    pub topic_word_prior: Option<f64>,
    pub learning_method: LdaLearning,
    pub learning_decay: f64,
    pub learning_offset: f64,
    pub max_iter: usize,
    pub batch_size: usize,
    /// Evaluate the perplexity every this many iterations (`<= 0`: never)
    /// and stop once it changes by less than `perp_tol`.
    pub evaluate_every: i64,
    pub perp_tol: f64,
    pub mean_change_tol: f64,
    pub max_doc_update_iter: usize,
    pub seed: u32,
}

impl Default for LatentDirichletAllocation {
    fn default() -> Self {
        Self {
            n_components: 10,
            doc_topic_prior: None,
            topic_word_prior: None,
            learning_method: LdaLearning::Batch,
            learning_decay: 0.7,
            learning_offset: 10.0,
            max_iter: 10,
            batch_size: 128,
            evaluate_every: -1,
            perp_tol: 1e-1,
            mean_change_tol: 1e-3,
            max_doc_update_iter: 100,
            seed: 0,
        }
    }
}

/// A fitted [`LatentDirichletAllocation`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedLatentDirichletAllocation {
    /// Variational topic-word parameters, `n_components × n_features`
    /// (sklearn's `components_`).
    pub components: Array2<f64>,
    exp_dirichlet_component: Array2<f64>,
    pub doc_topic_prior: f64,
    pub topic_word_prior: f64,
    pub n_iter: usize,
    /// Perplexity of the training data at the end (sklearn's `bound_`).
    pub bound: f64,
    mean_change_tol: f64,
    max_doc_update_iter: usize,
}

/// sklearn's Cython digamma (`_online_lda_fast.psi`).
fn psi(mut x: f64) -> f64 {
    const EULER: f64 = 0.577_215_664_901_532_9;
    if x <= 1e-6 {
        return -EULER - 1.0 / x;
    }
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let mut r = 1.0 / x;
    result += x.ln() - 0.5 * r;
    r *= r;
    result -= r * ((1.0 / 12.0) - r * ((1.0 / 120.0) - r * (1.0 / 252.0)));
    result
}

/// `ψ(a_ij) − ψ(Σ_j a_ij)` per row (`_dirichlet_expectation_2d`).
fn dirichlet_expectation_2d(a: &Array2<f64>) -> Array2<f64> {
    let mut out = a.clone();
    for mut row in out.rows_mut() {
        let total: f64 = row.iter().sum();
        let pt = psi(total);
        row.mapv_inplace(|v| psi(v) - pt);
    }
    out
}

/// `exp(_dirichlet_expectation_2d(a))`.
fn exp_dirichlet(a: &Array2<f64>) -> Array2<f64> {
    dirichlet_expectation_2d(a).mapv(f64::exp)
}

/// The non-zero columns and counts of one document.
fn nonzero(row: ndarray::ArrayView1<f64>) -> (Vec<usize>, Vec<f64>) {
    row.iter()
        .enumerate()
        .filter(|(_, v)| **v != 0.0)
        .map(|(j, v)| (j, *v))
        .unzip()
}

/// sklearn's `_update_doc_distribution`: the per-document variational
/// updates and, with `stats`, the sufficient statistics.
fn update_docs(
    x: ArrayView2<f64>,
    exp_topic_word: &Array2<f64>,
    doc_topic_prior: f64,
    max_doc_update_iter: usize,
    mean_change_tol: f64,
    stats: bool,
    rng: Option<&mut RandomState>,
) -> (Array2<f64>, Option<Array2<f64>>) {
    let (n, k) = (x.nrows(), exp_topic_word.nrows());
    let mut doc_topic = match rng {
        Some(r) => Array2::from_shape_simple_fn((n, k), || r.gamma(100.0, 0.01)),
        None => Array2::ones((n, k)),
    };
    let exp_doc_topic = exp_dirichlet(&doc_topic);
    let mut suff = stats.then(|| Array2::<f64>::zeros(exp_topic_word.dim()));
    let eps = f64::EPSILON;
    for d in 0..n {
        let (ids, cnts) = nonzero(x.row(d));
        let mut theta = doc_topic.row(d).to_owned();
        let mut exp_theta = exp_doc_topic.row(d).to_owned();
        let beta = exp_topic_word.select(Axis(1), &ids); // k × |ids|
        let norm = |exp_theta: &Array1<f64>| exp_theta.dot(&beta) + eps;
        for _ in 0..max_doc_update_iter {
            let last = theta.clone();
            let ratio =
                Array1::from_iter(cnts.iter().zip(norm(&exp_theta).iter()).map(|(c, p)| c / p));
            theta = &exp_theta * &beta.dot(&ratio);
            // _dirichlet_expectation_1d: adds the prior in place.
            theta.mapv_inplace(|v| v + doc_topic_prior);
            let pt = psi(theta.sum());
            exp_theta = theta.mapv(|v| (psi(v) - pt).exp());
            let change = last
                .iter()
                .zip(&theta)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / k as f64;
            if change < mean_change_tol {
                break;
            }
        }
        doc_topic.row_mut(d).assign(&theta);
        if let Some(s) = suff.as_mut() {
            let ratio =
                Array1::from_iter(cnts.iter().zip(norm(&exp_theta).iter()).map(|(c, p)| c / p));
            for t in 0..k {
                for (c, &j) in ids.iter().enumerate() {
                    s[[t, j]] += exp_theta[t] * ratio[c];
                }
            }
        }
    }
    (doc_topic, suff)
}

fn check_counts(x: ArrayView2<f64>) -> Result<()> {
    if x.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
        return Err(RustMlError::InvalidParameter(
            "LatentDirichletAllocation needs non-negative finite counts".into(),
        ));
    }
    Ok(())
}

impl FittedLatentDirichletAllocation {
    fn e_step(
        &self,
        x: ArrayView2<f64>,
        stats: bool,
        rng: Option<&mut RandomState>,
    ) -> (Array2<f64>, Option<Array2<f64>>) {
        let (doc_topic, suff) = update_docs(
            x,
            &self.exp_dirichlet_component,
            self.doc_topic_prior,
            self.max_doc_update_iter,
            self.mean_change_tol,
            stats,
            rng,
        );
        (doc_topic, suff.map(|s| s * &self.exp_dirichlet_component))
    }

    fn check(&self, x: ArrayView2<f64>) -> Result<()> {
        if x.ncols() != self.components.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.components.ncols(),
                x.ncols()
            )));
        }
        check_counts(x)
    }

    /// The unnormalised document-topic parameters (`normalize=False`).
    pub fn transform_unnormalized(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.check(x)?;
        Ok(self.e_step(x, false, None).0)
    }

    /// The variational lower bound of `x` (sklearn's `score`).
    pub fn score(&self, x: ArrayView2<f64>) -> Result<f64> {
        let doc_topic = self.transform_unnormalized(x)?;
        Ok(self.approx_bound(x, &doc_topic))
    }

    /// `exp(−bound / word count)` (sklearn's `perplexity`).
    pub fn perplexity(&self, x: ArrayView2<f64>) -> Result<f64> {
        let doc_topic = self.transform_unnormalized(x)?;
        Ok(self.perplexity_of(x, &doc_topic))
    }

    fn perplexity_of(&self, x: ArrayView2<f64>, doc_topic: &Array2<f64>) -> f64 {
        (-self.approx_bound(x, doc_topic) / x.sum()).exp()
    }

    /// sklearn's `_approx_bound` without sub-sampling.
    fn approx_bound(&self, x: ArrayView2<f64>, doc_topic: &Array2<f64>) -> f64 {
        fn loglik(prior: f64, distr: &Array2<f64>, dirichlet: &Array2<f64>, size: usize) -> f64 {
            let mut score: f64 = distr
                .iter()
                .zip(dirichlet)
                .map(|(d, e)| (prior - d) * e)
                .sum();
            score += distr
                .iter()
                .map(|&d| ln_gamma(d) - ln_gamma(prior))
                .sum::<f64>();
            score += distr
                .rows()
                .into_iter()
                .map(|r| ln_gamma(prior * size as f64) - ln_gamma(r.sum()))
                .sum::<f64>();
            score
        }
        let dir_doc = dirichlet_expectation_2d(doc_topic);
        let dir_comp = dirichlet_expectation_2d(&self.components);
        let mut score = 0.0;
        for d in 0..x.nrows() {
            let (ids, cnts) = nonzero(x.row(d));
            for (c, &j) in ids.iter().enumerate() {
                // logsumexp over topics
                let temp: Vec<f64> = (0..dir_doc.ncols())
                    .map(|t| dir_doc[[d, t]] + dir_comp[[t, j]])
                    .collect();
                let m = temp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let lse = m + temp.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
                score += cnts[c] * lse;
            }
        }
        score += loglik(
            self.doc_topic_prior,
            doc_topic,
            &dir_doc,
            self.components.nrows(),
        );
        score += loglik(
            self.topic_word_prior,
            &self.components,
            &dir_comp,
            self.components.ncols(),
        );
        score
    }
}

impl FitUnsupervised<f64> for LatentDirichletAllocation {
    type Fitted = FittedLatentDirichletAllocation;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<Self::Fitted> {
        let (n, p) = x.dim();
        if n == 0 || p == 0 {
            return Err(RustMlError::EmptyInput("LDA needs documents".into()));
        }
        check_counts(x)?;
        if self.n_components == 0 || self.batch_size == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_components and batch_size must be at least 1".into(),
            ));
        }
        let k = self.n_components;
        let mut rng = RandomState::new(self.seed);
        let components = Array2::from_shape_simple_fn((k, p), || rng.gamma(100.0, 0.01));
        let mut m = FittedLatentDirichletAllocation {
            exp_dirichlet_component: exp_dirichlet(&components),
            components,
            doc_topic_prior: self.doc_topic_prior.unwrap_or(1.0 / k as f64),
            topic_word_prior: self.topic_word_prior.unwrap_or(1.0 / k as f64),
            n_iter: 0,
            bound: f64::NAN,
            mean_change_tol: self.mean_change_tol,
            max_doc_update_iter: self.max_doc_update_iter,
        };
        let mut n_batch_iter = 1.0;
        let mut last_bound: Option<f64> = None;
        for i in 0..self.max_iter {
            let batches: Vec<(usize, usize)> = match self.learning_method {
                LdaLearning::Batch => vec![(0, n)],
                LdaLearning::Online => (0..n)
                    .step_by(self.batch_size)
                    .map(|s| (s, (s + self.batch_size).min(n)))
                    .collect(),
            };
            for (s, e) in batches {
                let xb = x.slice(ndarray::s![s..e, ..]);
                let (_, suff) = m.e_step(xb, true, Some(&mut rng));
                let suff = suff.expect("sufficient statistics");
                match self.learning_method {
                    LdaLearning::Batch => m.components = suff + m.topic_word_prior,
                    LdaLearning::Online => {
                        let w = (self.learning_offset + n_batch_iter).powf(-self.learning_decay);
                        let ratio = n as f64 / (e - s) as f64;
                        m.components *= 1.0 - w;
                        m.components += &((suff * ratio + m.topic_word_prior) * w);
                    }
                }
                m.exp_dirichlet_component = exp_dirichlet(&m.components);
                n_batch_iter += 1.0;
            }
            if self.evaluate_every > 0 && (i as i64 + 1) % self.evaluate_every == 0 {
                let (doc_topic, _) = m.e_step(x, false, None);
                let bound = m.perplexity_of(x, &doc_topic);
                if let Some(last) = last_bound {
                    if last != 0.0 && (last - bound).abs() < self.perp_tol {
                        break;
                    }
                }
                last_bound = Some(bound);
            }
            m.n_iter += 1;
        }
        let (doc_topic, _) = m.e_step(x, false, None);
        m.bound = m.perplexity_of(x, &doc_topic);
        Ok(m)
    }
}

impl Transform<f64> for FittedLatentDirichletAllocation {
    /// Document-topic distributions, rows summing to 1.
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut t = self.transform_unnormalized(x)?;
        for mut row in t.rows_mut() {
            let s = row.sum();
            row.mapv_inplace(|v| v / s);
        }
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn psi_is_sklearns() {
        // sklearn's `_dirichlet_expectation_2d([[2.5, 1.5]])[0, 0]`, i.e.
        // psi(2.5) - psi(4); its asymptotic series is ~1e-9 off scipy's psi.
        assert_eq!(super::psi(2.5) - super::psi(4.0), -0.552_961_026_677_109_9);
    }
}
