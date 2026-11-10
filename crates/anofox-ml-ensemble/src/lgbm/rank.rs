//! Learning to rank (#207): LightGBM's `lambdarank` (LambdaRank with
//! NDCG) and `rank_xendcg` objectives over query groups, and the NDCG@k
//! metric.

use super::*;

/// LightGBM's ranking parameters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LgbmRankingParams {
    /// Pairs are formed with the top `truncation_level` documents of a
    /// query (`lambdarank_truncation_level`, default 30).
    pub truncation_level: usize,
    /// Normalize each query's lambdas (`lambdarank_norm`, default true).
    pub norm: bool,
    /// Steepness of the pairwise sigmoid (`sigmoid`, default 1).
    pub sigmoid: f64,
    /// Gain of each integer label (`label_gain`); empty: `2^l - 1` for
    /// labels 0..30.
    pub label_gain: Vec<f64>,
    /// NDCG@k used for early stopping on an eval set (LightGBM's first
    /// `eval_at`, default 1).
    pub eval_at: usize,
}

impl Default for LgbmRankingParams {
    fn default() -> Self {
        Self {
            truncation_level: 30,
            norm: true,
            sigmoid: 1.0,
            label_gain: Vec::new(),
            eval_at: 1,
        }
    }
}

impl LgbmRankingParams {
    /// The label gains (LightGBM's `DefaultLabelGain`).
    fn gains(&self) -> Vec<f64> {
        if !self.label_gain.is_empty() {
            return self.label_gain.clone();
        }
        std::iter::once(0.0)
            .chain((1..31).map(|i| ((1i64 << i) - 1) as f64))
            .collect()
    }
}

/// LightGBM's `DCGCalculator::kMaxPosition`.
const MAX_POSITION: usize = 10_000;

/// `1 / log2(2 + i)` (LightGBM's DCG discount).
fn discounts() -> Vec<f64> {
    (0..MAX_POSITION)
        .map(|i| 1.0 / (2.0 + i as f64).log2())
        .collect()
}

/// Query boundaries from the group sizes, checked against `n` rows.
pub(super) fn query_boundaries(group: &[usize], n: usize) -> Result<Vec<usize>> {
    let mut b = Vec::with_capacity(group.len() + 1);
    b.push(0);
    for &g in group {
        if g > MAX_POSITION {
            return Err(RustMlError::InvalidParameter(format!(
                "a query has {g} rows, more than LightGBM's {MAX_POSITION}"
            )));
        }
        b.push(b.last().unwrap() + g);
    }
    if *b.last().unwrap() != n {
        return Err(RustMlError::ShapeMismatch(format!(
            "the groups hold {} rows but there are {n}",
            b.last().unwrap()
        )));
    }
    Ok(b)
}

/// Labels must be integers indexing `label_gain` (LightGBM's `CheckLabel`).
pub(super) fn check_labels(y: ArrayView1<f64>, params: &LgbmRankingParams) -> Result<()> {
    let n_gains = params.gains().len();
    match y
        .iter()
        .find(|&&v| v < 0.0 || v.fract() != 0.0 || v as usize >= n_gains)
    {
        Some(v) => Err(RustMlError::InvalidParameter(format!(
            "ranking labels must be integers in [0, {n_gains}), got {v}"
        ))),
        None => Ok(()),
    }
}

/// The DCG of the best order of `labels` at `k` (`CalMaxDCGAtK`).
fn max_dcg_at(k: usize, labels: &[f64], gains: &[f64], discount: &[f64]) -> f64 {
    let mut count = vec![0usize; gains.len()];
    for &l in labels {
        count[l as usize] += 1;
    }
    let mut top = gains.len() - 1;
    let mut ret = 0.0;
    for d in discount.iter().take(k.min(labels.len())) {
        while top > 0 && count[top] == 0 {
            top -= 1;
        }
        ret += d * gains[top];
        count[top] = count[top].saturating_sub(1);
    }
    ret
}

/// The order of `scores`, highest first, ties in row order
/// (`std::stable_sort` with `>`).
fn order_by_score(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx
}

/// The gradients of LightGBM's ranking objectives; `rank_xendcg` keeps one
/// generator per query across iterations.
pub(super) struct RankingObjective {
    boundaries: Vec<usize>,
    labels: Vec<f64>,
    gains: Vec<f64>,
    discount: Vec<f64>,
    params: LgbmRankingParams,
    kind: RankKind,
}

enum RankKind {
    LambdaRank {
        inverse_max_dcg: Vec<f64>,
        table: Vec<f64>,
        min_input: f64,
        max_input: f64,
        idx_factor: f64,
    },
    Xendcg {
        rands: Vec<LgbmRandom>,
    },
}

/// Bins of LightGBM's sigmoid table.
const SIGMOID_BINS: usize = 1024 * 1024;

impl RankingObjective {
    pub(super) fn new(
        objective: LgbmObjective,
        labels: &[f64],
        boundaries: Vec<usize>,
        params: &LgbmRankingParams,
        objective_seed: u32,
    ) -> Result<Self> {
        let gains = params.gains();
        let discount = discounts();
        let kind = match objective {
            LgbmObjective::LambdaRank => {
                if params.sigmoid <= 0.0 {
                    return Err(RustMlError::InvalidParameter(
                        "sigmoid must be positive".into(),
                    ));
                }
                let inverse_max_dcg = boundaries
                    .windows(2)
                    .map(|w| {
                        let m = max_dcg_at(
                            params.truncation_level,
                            &labels[w[0]..w[1]],
                            &gains,
                            &discount,
                        );
                        if m > 0.0 {
                            1.0 / m
                        } else {
                            m
                        }
                    })
                    .collect();
                // LightGBM's `ConstructSigmoidTable`.
                let min_input = -50.0 / params.sigmoid / 2.0;
                let max_input = -min_input;
                let idx_factor = SIGMOID_BINS as f64 / (max_input - min_input);
                let table = (0..SIGMOID_BINS)
                    .map(|i| {
                        let score = i as f64 / idx_factor + min_input;
                        1.0 / (1.0 + (score * params.sigmoid).exp())
                    })
                    .collect();
                RankKind::LambdaRank {
                    inverse_max_dcg,
                    table,
                    min_input,
                    max_input,
                    idx_factor,
                }
            }
            LgbmObjective::RankXendcg => RankKind::Xendcg {
                rands: (0..boundaries.len() - 1)
                    .map(|q| LgbmRandom(objective_seed.wrapping_add(q as u32)))
                    .collect(),
            },
            _ => unreachable!("not a ranking objective"),
        };
        Ok(Self {
            boundaries,
            labels: labels.to_vec(),
            gains,
            discount,
            params: params.clone(),
            kind,
        })
    }

    /// Gradients and hessians (float32 values) of every row at `scores`.
    pub(super) fn gradients(&mut self, scores: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = scores.len();
        let mut grad = vec![0.0f32; n];
        let mut hess = vec![0.0f32; n];
        let Self {
            boundaries,
            labels,
            gains,
            discount,
            params,
            kind,
        } = self;
        use rayon::prelude::*;
        let mut queries: Vec<(usize, &mut [f32], &mut [f32])> = Vec::new();
        let (mut g_rest, mut h_rest) = (&mut grad[..], &mut hess[..]);
        for (q, w) in boundaries.windows(2).enumerate() {
            let (g, gr) = std::mem::take(&mut g_rest).split_at_mut(w[1] - w[0]);
            let (h, hr) = std::mem::take(&mut h_rest).split_at_mut(w[1] - w[0]);
            (g_rest, h_rest) = (gr, hr);
            queries.push((q, g, h));
        }
        match kind {
            RankKind::LambdaRank {
                inverse_max_dcg,
                table,
                min_input,
                max_input,
                idx_factor,
            } => {
                let sigmoid = |s: f64| {
                    if s <= *min_input {
                        table[0]
                    } else if s >= *max_input {
                        table[SIGMOID_BINS - 1]
                    } else {
                        table[((s - *min_input) * *idx_factor) as usize]
                    }
                };
                queries.into_par_iter().for_each(|(q, g, h)| {
                    let (a, b) = (boundaries[q], boundaries[q + 1]);
                    lambdarank_query(
                        &labels[a..b],
                        &scores[a..b],
                        inverse_max_dcg[q],
                        gains,
                        discount,
                        params,
                        &sigmoid,
                        g,
                        h,
                    );
                });
            }
            RankKind::Xendcg { rands } => {
                queries
                    .into_par_iter()
                    .zip(rands.par_iter_mut())
                    .for_each(|((q, g, h), rand)| {
                        let (a, b) = (boundaries[q], boundaries[q + 1]);
                        xendcg_query(&labels[a..b], &scores[a..b], rand, g, h);
                    });
            }
        }
        (
            grad.into_iter().map(f64::from).collect(),
            hess.into_iter().map(f64::from).collect(),
        )
    }

    /// Mean NDCG@k over the queries of `scores` (LightGBM's `ndcg@k`; a
    /// query without relevant documents counts as 1).
    pub(super) fn ndcg(&self, labels: &[f64], scores: &[f64], boundaries: &[usize]) -> f64 {
        ndcg_at(
            self.params.eval_at,
            labels,
            scores,
            boundaries,
            &self.gains,
            &self.discount,
        )
    }
}

/// Mean NDCG@k of `scores` over the queries `boundaries`.
fn ndcg_at(
    k: usize,
    labels: &[f64],
    scores: &[f64],
    boundaries: &[usize],
    gains: &[f64],
    discount: &[f64],
) -> f64 {
    let mut total = 0.0;
    for w in boundaries.windows(2) {
        let (l, s) = (&labels[w[0]..w[1]], &scores[w[0]..w[1]]);
        let max = max_dcg_at(k, l, gains, discount);
        if max <= 0.0 {
            total += 1.0;
            continue;
        }
        let dcg: f64 = order_by_score(s)
            .iter()
            .take(k)
            .zip(discount)
            .map(|(&i, d)| gains[l[i] as usize] * d)
            .sum();
        total += dcg * (1.0 / max);
    }
    total / (boundaries.len() - 1).max(1) as f64
}

/// LightGBM's `LambdarankNDCG::GetGradientsForOneQuery`, accumulating in
/// float32 as LightGBM does.
#[allow(clippy::too_many_arguments)]
fn lambdarank_query(
    labels: &[f64],
    scores: &[f64],
    inverse_max_dcg: f64,
    gains: &[f64],
    discount: &[f64],
    params: &LgbmRankingParams,
    sigmoid: &(dyn Fn(f64) -> f64 + Sync),
    lambdas: &mut [f32],
    hessians: &mut [f32],
) {
    let cnt = labels.len();
    if cnt == 0 {
        return;
    }
    let sorted = order_by_score(scores);
    let best = scores[sorted[0]];
    let worst = scores[sorted[cnt - 1]];
    let mut sum_lambdas = 0.0f64;
    let s = params.sigmoid;
    for i in 0..(cnt - 1).min(params.truncation_level) {
        for j in i + 1..cnt {
            let (li, lj) = (labels[sorted[i]], labels[sorted[j]]);
            if li == lj {
                continue;
            }
            let (high_rank, low_rank) = if li > lj { (i, j) } else { (j, i) };
            let (high, low) = (sorted[high_rank], sorted[low_rank]);
            let delta_score = scores[high] - scores[low];
            let dcg_gap = gains[labels[high] as usize] - gains[labels[low] as usize];
            let paired_discount = (discount[high_rank] - discount[low_rank]).abs();
            let mut delta_ndcg = dcg_gap * paired_discount * inverse_max_dcg;
            if params.norm && best != worst {
                delta_ndcg /= f64::from(0.01f32) + delta_score.abs();
            }
            let mut p_lambda = sigmoid(delta_score);
            let mut p_hessian = p_lambda * (1.0 - p_lambda);
            p_lambda *= -s * delta_ndcg;
            p_hessian *= s * s * delta_ndcg;
            lambdas[low] -= p_lambda as f32;
            hessians[low] += p_hessian as f32;
            lambdas[high] += p_lambda as f32;
            hessians[high] += p_hessian as f32;
            sum_lambdas -= 2.0 * p_lambda;
        }
    }
    if params.norm && sum_lambdas > 0.0 {
        let factor = (1.0 + sum_lambdas).log2() / sum_lambdas;
        for (l, h) in lambdas.iter_mut().zip(hessians.iter_mut()) {
            *l = (f64::from(*l) * factor) as f32;
            *h = (f64::from(*h) * factor) as f32;
        }
    }
}

/// LightGBM's `RankXENDCG::GetGradientsForOneQuery`.
fn xendcg_query(
    labels: &[f64],
    scores: &[f64],
    rand: &mut LgbmRandom,
    lambdas: &mut [f32],
    hessians: &mut [f32],
) {
    let cnt = labels.len();
    if cnt <= 1 {
        return;
    }
    // Softmax (LightGBM's `Common::Softmax`).
    let wmax = scores.iter().copied().fold(scores[0], f64::max);
    let mut rho: Vec<f64> = scores.iter().map(|&s| (s - wmax).exp()).collect();
    let wsum: f64 = rho.iter().sum();
    rho.iter_mut().for_each(|r| *r /= wsum);
    let mut params: Vec<f64> = labels
        .iter()
        .map(|&l| 2f64.powi(l as i32) - f64::from(rand.next_float()))
        .collect();
    let inv_denominator = 1.0 / params.iter().sum::<f64>().max(1e-15f32 as f64);
    let mut sum_l1 = 0.0;
    for i in 0..cnt {
        let term = -params[i] * inv_denominator + rho[i];
        lambdas[i] = term as f32;
        params[i] = term / (1.0 - rho[i]);
        sum_l1 += params[i];
    }
    let mut sum_l2 = 0.0;
    for i in 0..cnt {
        let term = rho[i] * (sum_l1 - params[i]);
        lambdas[i] += term as f32;
        params[i] = term / (1.0 - rho[i]);
        sum_l2 += params[i];
    }
    for i in 0..cnt {
        lambdas[i] += (rho[i] * (sum_l2 - params[i])) as f32;
        hessians[i] = (rho[i] * (1.0 - rho[i])) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndcg_is_one_for_the_ideal_order() {
        let (gains, discount) = (LgbmRankingParams::default().gains(), discounts());
        let labels = [3.0, 0.0, 1.0, 2.0];
        let ideal = [4.0, 1.0, 2.0, 3.0];
        let reversed = [1.0, 4.0, 3.0, 2.0];
        let b = [0, 4];
        assert!((ndcg_at(4, &labels, &ideal, &b, &gains, &discount) - 1.0).abs() < 1e-15);
        assert!(ndcg_at(4, &labels, &reversed, &b, &gains, &discount) < 0.7);
        // A query without relevant documents counts as 1.
        assert_eq!(
            ndcg_at(2, &[0.0, 0.0], &[1.0, 2.0], &[0, 2], &gains, &discount),
            1.0
        );
    }

    #[test]
    fn ranker_early_stops_on_eval_ndcg() {
        let n = 400;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * (3 + 2 * j)) % 29) as f64 / 7.0);
        let y = Array1::from_shape_fn(n, |i| (x[[i, 0]] / 1.5).floor().min(3.0));
        let group = vec![20usize; n / 20];
        let (xe, ye) = (
            x.slice(ndarray::s![..100, ..]).to_owned(),
            y.slice(ndarray::s![..100]).to_owned(),
        );
        let eval_group = vec![20usize; 5];
        let opts = LgbmFitOptions {
            group: Some(&group),
            eval_set: Some((&xe, &ye)),
            eval_group: Some(&eval_group),
            ..Default::default()
        };
        let m = LgbmRegressor::ranker()
            .with_n_estimators(200)
            .with_early_stopping(Some(5))
            .fit_with_eval(&x, &y, &opts)
            .unwrap();
        assert!(m.best_iteration() < 200);
        let missing = LgbmFitOptions {
            eval_group: None,
            ..opts
        };
        assert!(LgbmRegressor::ranker()
            .fit_with_eval(&x, &y, &missing)
            .is_err());
        let bad_labels = y.mapv(|v| v + 0.5);
        let plain = LgbmFitOptions {
            group: Some(&group),
            ..Default::default()
        };
        assert!(LgbmRegressor::ranker()
            .fit_with_eval(&x, &bad_labels, &plain)
            .is_err());
    }
}
