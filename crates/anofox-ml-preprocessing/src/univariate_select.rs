//! Univariate feature selection with a configurable rule (sklearn's
//! `GenericUnivariateSelect`, `SelectPercentile`, `SelectFpr`, `SelectFdr`,
//! `SelectFwe`), using the scoring functions of [`SelectKBest`].

use anofox_ml_core::{Fit, Float, Result, RustMlError};
use ndarray::{ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::select_k_best::{univariate_scores, FittedSelectKBest, ScoringFunction};
#[cfg(doc)]
use crate::SelectKBest;

/// Which features a [`GenericUnivariateSelect`] keeps.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SelectionMode {
    /// The `k` best scores.
    KBest(usize),
    /// The best `percentile` percent of the scores (sklearn's
    /// `SelectPercentile`; ties at the threshold fill up to
    /// `⌊n · percentile / 100⌋` features in index order).
    Percentile(f64),
    /// p-value below `alpha` (false positive rate).
    Fpr(f64),
    /// Benjamini–Hochberg at false discovery rate `alpha`.
    Fdr(f64),
    /// p-value below `alpha / n_features` (family-wise error, Bonferroni).
    Fwe(f64),
}

impl Default for SelectionMode {
    fn default() -> Self {
        SelectionMode::Percentile(10.0)
    }
}

/// Univariate feature selector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenericUnivariateSelect {
    pub scoring_fn: ScoringFunction,
    pub mode: SelectionMode,
}

impl Default for GenericUnivariateSelect {
    fn default() -> Self {
        Self {
            scoring_fn: ScoringFunction::FClassif,
            mode: SelectionMode::default(),
        }
    }
}

impl GenericUnivariateSelect {
    pub fn new(scoring_fn: ScoringFunction, mode: SelectionMode) -> Self {
        Self { scoring_fn, mode }
    }
}

/// numpy's linear percentile (`q` in [0, 100]) of non-NaN values.
fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = q / 100.0 * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

/// Indices kept by `mode` given the scores and p-values (sklearn's
/// `_get_support_mask`; NaN scores count as the lowest float).
pub(crate) fn support(
    mode: SelectionMode,
    scores: &[f64],
    pvalues: Option<&[f64]>,
) -> Result<Vec<usize>> {
    let n = scores.len();
    let clean: Vec<f64> = scores
        .iter()
        .map(|&s| if s.is_nan() { f64::MIN } else { s })
        .collect();
    let need_p = || {
        pvalues.ok_or_else(|| {
            RustMlError::InvalidParameter(format!(
                "selection mode {mode:?} needs p-values; use FClassif, FRegression or Chi2"
            ))
        })
    };
    let check_alpha = |a: f64| {
        if (0.0..=1.0).contains(&a) {
            Ok(a)
        } else {
            Err(RustMlError::InvalidParameter(format!(
                "alpha must be in [0, 1], got {a}"
            )))
        }
    };
    Ok(match mode {
        SelectionMode::KBest(k) => {
            if k > n {
                return Err(RustMlError::InvalidParameter(format!(
                    "k ({k}) exceeds the number of features ({n})"
                )));
            }
            let mut order: Vec<usize> = (0..n).collect();
            // Stable ascending, last k: higher indices win ties.
            order.sort_by(|&a, &b| clean[a].total_cmp(&clean[b]));
            let mut keep = order[n - k..].to_vec();
            keep.sort_unstable();
            keep
        }
        SelectionMode::Percentile(p) => {
            if !(0.0..=100.0).contains(&p) {
                return Err(RustMlError::InvalidParameter(format!(
                    "percentile must be in [0, 100], got {p}"
                )));
            }
            if p == 100.0 {
                return Ok((0..n).collect());
            }
            if p == 0.0 {
                return Ok(Vec::new());
            }
            let threshold = percentile(&clean, 100.0 - p);
            let mut mask: Vec<bool> = clean.iter().map(|&s| s > threshold).collect();
            let max_feats = (n as f64 * p / 100.0) as usize;
            let mut room = max_feats.saturating_sub(mask.iter().filter(|&&m| m).count());
            for (j, &s) in clean.iter().enumerate() {
                if room == 0 {
                    break;
                }
                if s == threshold {
                    mask[j] = true;
                    room -= 1;
                }
            }
            (0..n).filter(|&j| mask[j]).collect()
        }
        SelectionMode::Fpr(alpha) => {
            let (p, a) = (need_p()?, check_alpha(alpha)?);
            (0..n).filter(|&j| p[j] < a).collect()
        }
        SelectionMode::Fwe(alpha) => {
            let (p, a) = (need_p()?, check_alpha(alpha)?);
            (0..n).filter(|&j| p[j] < a / n as f64).collect()
        }
        SelectionMode::Fdr(alpha) => {
            let (p, a) = (need_p()?, check_alpha(alpha)?);
            let mut sv = p.to_vec();
            sv.sort_by(f64::total_cmp);
            let cut = sv
                .iter()
                .enumerate()
                .filter(|&(i, &v)| v <= a / n as f64 * (i + 1) as f64)
                .map(|(_, &v)| v)
                .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
            match cut {
                Some(c) => (0..n).filter(|&j| p[j] <= c).collect(),
                None => Vec::new(),
            }
        }
    })
}

impl<F: Float> Fit<F> for GenericUnivariateSelect {
    type Fitted = FittedSelectKBest<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<FittedSelectKBest<F>> {
        let (n, d) = x.dim();
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if !matches!(self.scoring_fn, ScoringFunction::Variance) && y.len() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {n} samples but y has {} elements",
                y.len()
            )));
        }
        let (scores, pvalues) = univariate_scores(&self.scoring_fn, x, y)?;
        let s: Vec<f64> = scores.iter().map(|v| v.to_f64().unwrap()).collect();
        let p: Option<Vec<f64>> = pvalues
            .as_ref()
            .map(|p| p.iter().map(|v| v.to_f64().unwrap()).collect());
        let selected = support(self.mode, &s, p.as_deref())?;
        Ok(FittedSelectKBest::from_parts(scores, pvalues, selected, d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_on_known_scores() {
        let s = [5.0, 1.0, 3.0, 3.0, f64::NAN];
        assert_eq!(
            support(SelectionMode::KBest(2), &s, None).unwrap(),
            vec![0, 3]
        );
        assert_eq!(
            support(SelectionMode::Percentile(40.0), &s, None).unwrap(),
            vec![0, 2]
        );
        let p = [0.001, 0.2, 0.011, 0.04, 0.9];
        assert_eq!(
            support(SelectionMode::Fpr(0.05), &s, Some(&p)).unwrap(),
            vec![0, 2, 3]
        );
        assert_eq!(
            support(SelectionMode::Fwe(0.05), &s, Some(&p)).unwrap(),
            vec![0]
        );
        // BH: sorted 0.001, 0.011, 0.04, 0.2, 0.9 vs 0.01, 0.02, 0.03, ...
        assert_eq!(
            support(SelectionMode::Fdr(0.05), &s, Some(&p)).unwrap(),
            vec![0, 2]
        );
        assert!(support(SelectionMode::Fdr(0.05), &s, None).is_err());
    }
}
