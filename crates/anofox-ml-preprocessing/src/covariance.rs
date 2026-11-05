//! Covariance estimation (sklearn's `sklearn.covariance`):
//! [`EmpiricalCovariance`], [`ShrunkCovariance`], [`LedoitWolf`], [`Oas`],
//! the robust [`MinCovDet`] (FastMCD) and the outlier detector
//! [`EllipticEnvelope`] built on it.
//!
//! Every estimator fits to a [`FittedCovariance`]: location, covariance,
//! precision (the symmetric pseudo-inverse, scipy's `pinvh`), squared
//! Mahalanobis distances and the Gaussian log-likelihood `score`.
//!
//! `MinCovDet` follows sklearn's FastMCD step by step (random starts,
//! concentration steps, consistency correction and reweighting) but draws
//! its random subsets from its own generator; on data where FastMCD finds
//! the optimal subset, the results agree with sklearn.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::special::gamma_q;

fn check_x(x: ArrayView2<f64>, min_rows: usize) -> Result<()> {
    let (n, d) = x.dim();
    if n < min_rows || d == 0 {
        return Err(RustMlError::EmptyInput(format!(
            "need at least {min_rows} rows and one column, got {n} × {d}"
        )));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(RustMlError::InvalidParameter("x must be finite".into()));
    }
    Ok(())
}

/// Mean of the rows, or zeros when `assume_centered`.
fn location_of(x: ArrayView2<f64>, assume_centered: bool) -> Array1<f64> {
    if assume_centered {
        Array1::zeros(x.ncols())
    } else {
        x.mean_axis(Axis(0)).expect("non-empty")
    }
}

/// Maximum-likelihood covariance (sklearn's `empirical_covariance`).
fn empirical(x: ArrayView2<f64>, assume_centered: bool) -> Array2<f64> {
    let n = x.nrows() as f64;
    if assume_centered {
        x.t().dot(&x) / n
    } else {
        let xc = &x - &location_of(x, false);
        xc.t().dot(&xc) / n
    }
}

/// `(1 − s) Σ + s μ I` with `μ = tr(Σ) / p`.
fn shrink(cov: &Array2<f64>, s: f64) -> Array2<f64> {
    let p = cov.nrows();
    let mu = cov.diag().sum() / p as f64;
    let mut out = cov * (1.0 - s);
    for i in 0..p {
        out[[i, i]] += s * mu;
    }
    out
}

fn symmetric_eigen(a: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>)> {
    let n = a.nrows();
    let m = Mat::<f64>::from_fn(n, n, |i, j| 0.5 * (a[[i, j]] + a[[j, i]]));
    let eig = SelfAdjointEigen::new(m.as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("eigendecomposition failed: {e:?}")))?;
    let vals = (0..n).map(|i| eig.S().column_vector()[i]).collect();
    let vecs = Array2::from_shape_fn((n, n), |(i, j)| eig.U()[(i, j)]);
    Ok((vals, vecs))
}

/// scipy's `pinvh`: eigenvalues below `max(n) · eps · max|λ|` are dropped.
fn pinvh(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (vals, vecs) = symmetric_eigen(a)?;
    let n = a.nrows();
    let max_abs = vals.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let cutoff = n as f64 * f64::EPSILON * max_abs;
    let mut out = Array2::zeros((n, n));
    for (k, &l) in vals.iter().enumerate() {
        if l.abs() > cutoff {
            let v = vecs.column(k);
            for i in 0..n {
                for j in 0..n {
                    out[[i, j]] += v[i] * v[j] / l;
                }
            }
        }
    }
    Ok(out)
}

/// sklearn's `fast_logdet`: `log det` of a positive-definite matrix, `−∞`
/// otherwise.
fn logdet(a: &Array2<f64>) -> Result<f64> {
    let (vals, _) = symmetric_eigen(a)?;
    if vals.iter().any(|&l| l <= 0.0) {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(vals.iter().map(|l| l.ln()).sum())
}

/// Squared Mahalanobis distances of the rows of `x`.
fn mahalanobis(x: ArrayView2<f64>, location: &Array1<f64>, precision: &Array2<f64>) -> Array1<f64> {
    let xc = &x - location;
    (xc.dot(precision) * &xc).sum_axis(Axis(1))
}

/// A fitted covariance model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedCovariance {
    location: Array1<f64>,
    covariance: Array2<f64>,
    precision: Array2<f64>,
    /// Shrinkage coefficient (ShrunkCovariance, LedoitWolf, OAS).
    shrinkage: Option<f64>,
    /// MinCovDet: the reweighted support and the training distances.
    support: Option<Vec<bool>>,
    dist: Option<Array1<f64>>,
    raw_location: Option<Array1<f64>>,
    raw_covariance: Option<Array2<f64>>,
    raw_support: Option<Vec<bool>>,
}

impl FittedCovariance {
    fn new(location: Array1<f64>, covariance: Array2<f64>, shrinkage: Option<f64>) -> Result<Self> {
        let precision = pinvh(&covariance)?;
        Ok(Self {
            location,
            covariance,
            precision,
            shrinkage,
            support: None,
            dist: None,
            raw_location: None,
            raw_covariance: None,
            raw_support: None,
        })
    }

    pub fn location(&self) -> &Array1<f64> {
        &self.location
    }

    pub fn covariance(&self) -> &Array2<f64> {
        &self.covariance
    }

    pub fn precision(&self) -> &Array2<f64> {
        &self.precision
    }

    pub fn shrinkage(&self) -> Option<f64> {
        self.shrinkage
    }

    /// MinCovDet: rows used for the final (reweighted) estimate.
    pub fn support(&self) -> Option<&[bool]> {
        self.support.as_deref()
    }

    /// MinCovDet: squared Mahalanobis distances of the training rows.
    pub fn dist(&self) -> Option<&Array1<f64>> {
        self.dist.as_ref()
    }

    /// MinCovDet: the raw (unreweighted) estimate.
    pub fn raw_location(&self) -> Option<&Array1<f64>> {
        self.raw_location.as_ref()
    }

    pub fn raw_covariance(&self) -> Option<&Array2<f64>> {
        self.raw_covariance.as_ref()
    }

    pub fn raw_support(&self) -> Option<&[bool]> {
        self.raw_support.as_deref()
    }

    fn check(&self, x: ArrayView2<f64>) -> Result<()> {
        if x.ncols() != self.location.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.location.len(),
                x.ncols()
            )));
        }
        Ok(())
    }

    /// Squared Mahalanobis distances of the rows of `x`.
    pub fn mahalanobis(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.check(x)?;
        Ok(mahalanobis(x, &self.location, &self.precision))
    }

    /// Gaussian log-likelihood of `x` under the model (sklearn's `score`):
    /// `(−tr(S P) + log det P − p log 2π) / 2`, `S` the covariance of `x`
    /// around the fitted location.
    pub fn score(&self, x: ArrayView2<f64>) -> Result<f64> {
        self.check(x)?;
        let s = empirical((&x - &self.location).view(), true);
        let p = self.precision.nrows() as f64;
        Ok((-(&s * &self.precision).sum() + logdet(&self.precision)?
            - p * (2.0 * std::f64::consts::PI).ln())
            / 2.0)
    }
}

/// `n × 1` squared Mahalanobis distances (the registry's `transform`).
impl Transform<f64> for FittedCovariance {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        Ok(self.mahalanobis(x)?.insert_axis(Axis(1)))
    }
}

// ============================================================
// Empirical and shrunk
// ============================================================

/// Maximum-likelihood covariance.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EmpiricalCovariance {
    pub assume_centered: bool,
}

impl FitUnsupervised<f64> for EmpiricalCovariance {
    type Fitted = FittedCovariance;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedCovariance> {
        check_x(x, 1)?;
        FittedCovariance::new(
            location_of(x, self.assume_centered),
            empirical(x, self.assume_centered),
            None,
        )
    }
}

/// Covariance shrunk towards `μ I` by a fixed `shrinkage` in [0, 1].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShrunkCovariance {
    pub assume_centered: bool,
    pub shrinkage: f64,
}

impl Default for ShrunkCovariance {
    fn default() -> Self {
        Self {
            assume_centered: false,
            shrinkage: 0.1,
        }
    }
}

impl FitUnsupervised<f64> for ShrunkCovariance {
    type Fitted = FittedCovariance;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedCovariance> {
        check_x(x, 1)?;
        if !(0.0..=1.0).contains(&self.shrinkage) {
            return Err(RustMlError::InvalidParameter(format!(
                "shrinkage must be in [0, 1], got {}",
                self.shrinkage
            )));
        }
        let cov = shrink(&empirical(x, self.assume_centered), self.shrinkage);
        FittedCovariance::new(
            location_of(x, self.assume_centered),
            cov,
            Some(self.shrinkage),
        )
    }
}

/// Ledoit–Wolf shrinkage (sklearn's `LedoitWolf`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LedoitWolf {
    pub assume_centered: bool,
}

/// sklearn's `ledoit_wolf_shrinkage` (single block).
fn ledoit_wolf_shrinkage(x: ArrayView2<f64>, assume_centered: bool) -> f64 {
    let (n, p) = x.dim();
    if p == 1 {
        return 0.0;
    }
    let xc = if assume_centered {
        x.to_owned()
    } else {
        &x - &location_of(x, false)
    };
    let x2 = xc.mapv(|v| v * v);
    let nf = n as f64;
    let emp_cov_trace = x2.sum_axis(Axis(0)) / nf;
    let mu = emp_cov_trace.sum() / p as f64;
    let beta_ = x2.t().dot(&x2).sum();
    let delta_ = xc.t().dot(&xc).mapv(|v| v * v).sum() / (nf * nf);
    let beta = (beta_ / nf - delta_) / (p as f64 * nf);
    let delta = (delta_ - 2.0 * mu * emp_cov_trace.sum() + p as f64 * mu * mu) / p as f64;
    let beta = beta.min(delta);
    if beta == 0.0 {
        0.0
    } else {
        beta / delta
    }
}

impl FitUnsupervised<f64> for LedoitWolf {
    type Fitted = FittedCovariance;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedCovariance> {
        check_x(x, 1)?;
        let s = ledoit_wolf_shrinkage(x, self.assume_centered);
        let cov = shrink(&empirical(x, self.assume_centered), s);
        FittedCovariance::new(location_of(x, self.assume_centered), cov, Some(s))
    }
}

/// Oracle Approximating Shrinkage (sklearn's `OAS`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Oas {
    pub assume_centered: bool,
}

impl FitUnsupervised<f64> for Oas {
    type Fitted = FittedCovariance;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedCovariance> {
        check_x(x, 1)?;
        let (n, p) = x.dim();
        let emp = empirical(x, self.assume_centered);
        let s = if p == 1 {
            0.0
        } else {
            let alpha = emp.mapv(|v| v * v).mean().expect("non-empty");
            let mu = emp.diag().sum() / p as f64;
            let num = alpha + mu * mu;
            let den = (n as f64 + 1.0) * (alpha - mu * mu / p as f64);
            if den == 0.0 {
                1.0
            } else {
                (num / den).min(1.0)
            }
        };
        FittedCovariance::new(
            location_of(x, self.assume_centered),
            shrink(&emp, s),
            Some(s),
        )
    }
}

// ============================================================
// Minimum covariance determinant
// ============================================================

/// χ² CDF with `k` degrees of freedom.
fn chi2_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0 - gamma_q(k / 2.0, x / 2.0)
    }
}

/// χ² quantile by bisection on the CDF (to the last bits).
fn chi2_ppf(p: f64, k: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, k.max(1.0));
    while chi2_cdf(hi, k) < p {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break;
        }
        if chi2_cdf(mid, k) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// sklearn's `_consistency_factor`: makes the covariance of an `alpha`
/// fraction of Gaussian data consistent.
fn consistency_factor(p: usize, alpha: f64) -> f64 {
    let q = chi2_ppf(alpha, p as f64);
    alpha / chi2_cdf(q, p as f64 + 2.0)
}

/// Indices of the `h` smallest values (numpy's `argpartition(..)[:h]` as a
/// set; ties broken by index).
fn smallest(dist: &Array1<f64>, h: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..dist.len()).collect();
    order.sort_by(|&a, &b| dist[a].total_cmp(&dist[b]).then(a.cmp(&b)));
    order.truncate(h);
    order
}

#[derive(Clone)]
struct CStep {
    location: Array1<f64>,
    covariance: Array2<f64>,
    det: f64,
    support: Vec<usize>,
    dist: Array1<f64>,
}

fn estimate(x: ArrayView2<f64>, support: &[usize]) -> Result<(Array1<f64>, Array2<f64>, f64)> {
    let xs = x.select(Axis(0), support);
    let loc = xs.mean_axis(Axis(0)).expect("non-empty");
    let cov = empirical(xs.view(), false);
    let det = logdet(&cov)?;
    Ok((loc, cov, det))
}

/// sklearn's `_c_step`: concentration steps from a random subset or from
/// initial estimates.
fn c_step(
    x: ArrayView2<f64>,
    h: usize,
    max_iter: usize,
    init: Option<(&Array1<f64>, &Array2<f64>)>,
    rng: &mut StdRng,
) -> Result<CStep> {
    let n = x.nrows();
    let mut dist = Array1::from_elem(n, f64::INFINITY);
    let mut support = match init {
        None => {
            let mut perm: Vec<usize> = (0..n).collect();
            perm.shuffle(rng);
            perm.truncate(h);
            perm
        }
        Some((loc, cov)) => {
            dist = mahalanobis(x, loc, &pinvh(cov)?);
            smallest(&dist, h)
        }
    };
    let (mut location, mut covariance, mut det) = estimate(x, &support)?;
    let mut precision = if det.is_infinite() {
        pinvh(&covariance)?
    } else {
        Array2::zeros((0, 0))
    };
    let mut previous: Option<CStep> = None;
    let mut remaining = max_iter;
    let mut previous_det = f64::INFINITY;
    while det < previous_det && remaining > 0 && !det.is_infinite() {
        previous = Some(CStep {
            location: location.clone(),
            covariance: covariance.clone(),
            det,
            support: support.clone(),
            dist: dist.clone(),
        });
        previous_det = det;
        precision = pinvh(&covariance)?;
        dist = mahalanobis(x, &location, &precision);
        support = smallest(&dist, h);
        (location, covariance, det) = estimate(x, &support)?;
        remaining -= 1;
    }
    let previous_dist = dist.clone();
    if precision.nrows() == 0 {
        precision = pinvh(&covariance)?;
    }
    let dist = mahalanobis(x, &location, &precision);
    let current = CStep {
        location,
        covariance,
        det,
        support,
        dist,
    };
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-8 + 1e-5 * b.abs() || a == b;
    let result = if det.is_infinite() || close(det, previous_det) || remaining == 0 {
        current
    } else if det > previous_det {
        let mut p = previous.expect("an iteration ran");
        p.dist = previous_dist;
        p
    } else {
        current
    };
    Ok(result)
}

/// sklearn's `select_candidates`: run `c_step` from each start and keep
/// the `select` with the smallest determinants.
fn select_candidates(
    x: ArrayView2<f64>,
    h: usize,
    starts: Starts<'_>,
    select: usize,
    n_iter: usize,
    rng: &mut StdRng,
) -> Result<Vec<CStep>> {
    let mut all = Vec::new();
    match starts {
        Starts::Random(n_trials) => {
            for _ in 0..n_trials {
                all.push(c_step(x, h, n_iter, None, rng)?);
            }
        }
        Starts::From(estimates) => {
            for (loc, cov) in estimates {
                all.push(c_step(x, h, n_iter, Some((loc, cov)), rng)?);
            }
        }
    }
    let mut order: Vec<usize> = (0..all.len()).collect();
    order.sort_by(|&a, &b| all[a].det.total_cmp(&all[b].det));
    Ok(order
        .into_iter()
        .take(select)
        .map(|i| all[i].clone())
        .collect())
}

enum Starts<'a> {
    Random(usize),
    From(&'a [(Array1<f64>, Array2<f64>)]),
}

/// sklearn's `fast_mcd`: raw location, covariance, support and distances.
fn fast_mcd(
    x: ArrayView2<f64>,
    support_fraction: Option<f64>,
    rng: &mut StdRng,
) -> Result<(Array1<f64>, Array2<f64>, Vec<bool>, Array1<f64>)> {
    let (n, p) = x.dim();
    let h = match support_fraction {
        None => ((0.5 * (n + p + 1) as f64).ceil() as usize).min(n),
        Some(f) => (f * n as f64) as usize,
    };
    if h == 0 || h > n {
        return Err(RustMlError::InvalidParameter(format!(
            "support size {h} is not in [1, {n}]"
        )));
    }
    let to_mask = |idx: &[usize]| {
        let mut m = vec![false; n];
        for &i in idx {
            m[i] = true;
        }
        m
    };
    if p == 1 {
        let col: Vec<f64> = x.column(0).to_vec();
        let (location, support) = if h < n {
            let mut s = col.clone();
            s.sort_by(f64::total_cmp);
            let diff: Vec<f64> = (0..n - h).map(|i| s[h + i] - s[i]).collect();
            let min = diff.iter().cloned().fold(f64::INFINITY, f64::min);
            let starts: Vec<usize> = (0..diff.len()).filter(|&i| diff[i] == min).collect();
            let loc =
                0.5 * starts.iter().map(|&i| s[h + i] + s[i]).sum::<f64>() / starts.len() as f64;
            let dev = Array1::from_iter(col.iter().map(|v| (v - loc).abs()));
            (loc, to_mask(&smallest(&dev, h)))
        } else {
            (col.iter().sum::<f64>() / n as f64, vec![true; n])
        };
        let sup: Vec<f64> = (0..n).filter(|&i| support[i]).map(|i| col[i]).collect();
        let m = sup.iter().sum::<f64>() / sup.len() as f64;
        let var = sup.iter().map(|v| (v - m).powi(2)).sum::<f64>() / sup.len() as f64;
        let location = Array1::from_elem(1, location);
        let covariance = Array2::from_elem((1, 1), var);
        let dist = mahalanobis(x, &location, &pinvh(&covariance)?);
        return Ok((location, covariance, support, dist));
    }
    let estimates_of = |v: &[CStep]| -> Vec<(Array1<f64>, Array2<f64>)> {
        v.iter()
            .map(|c| (c.location.clone(), c.covariance.clone()))
            .collect()
    };
    if n > 500 {
        let n_subsets = n / 300;
        let n_sub = n / n_subsets;
        let mut shuffle: Vec<usize> = (0..n).collect();
        shuffle.shuffle(rng);
        let h_sub = (n_sub as f64 * (h as f64 / n as f64)).ceil() as usize;
        let n_trials = (500 / n_subsets).max(10);
        let mut best = Vec::new();
        for i in 0..n_subsets {
            let rows = &shuffle[i * n_sub..(i + 1) * n_sub];
            let sub = x.select(Axis(0), rows);
            best.extend(select_candidates(
                sub.view(),
                h_sub,
                Starts::Random(n_trials),
                10,
                2,
                rng,
            )?);
        }
        let n_merged = n.min(1500);
        let h_merged = (n_merged as f64 * (h as f64 / n as f64)).ceil() as usize;
        let n_best_merged = if n > 1500 { 10 } else { 1 };
        let mut sel: Vec<usize> = (0..n).collect();
        sel.shuffle(rng);
        sel.truncate(n_merged);
        let merged_x = x.select(Axis(0), &sel);
        let starts = estimates_of(&best);
        let merged = select_candidates(
            merged_x.view(),
            h_merged,
            Starts::From(&starts),
            n_best_merged,
            30,
            rng,
        )?;
        if n < 1500 {
            let c = &merged[0];
            let mut support = vec![false; n];
            let mut dist = Array1::zeros(n);
            for (k, &i) in sel.iter().enumerate() {
                dist[i] = c.dist[k];
            }
            for &k in &c.support {
                support[sel[k]] = true;
            }
            return Ok((c.location.clone(), c.covariance.clone(), support, dist));
        }
        let starts = estimates_of(&merged);
        let full = select_candidates(x, h, Starts::From(&starts), 1, 30, rng)?;
        let c = &full[0];
        return Ok((
            c.location.clone(),
            c.covariance.clone(),
            to_mask(&c.support),
            c.dist.clone(),
        ));
    }
    let best = select_candidates(x, h, Starts::Random(30), 10, 2, rng)?;
    let starts = estimates_of(&best);
    let full = select_candidates(x, h, Starts::From(&starts), 1, 30, rng)?;
    let c = &full[0];
    Ok((
        c.location.clone(),
        c.covariance.clone(),
        to_mask(&c.support),
        c.dist.clone(),
    ))
}

/// Robust covariance by the minimum covariance determinant (FastMCD), with
/// sklearn's consistency correction and reweighting.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MinCovDet {
    pub assume_centered: bool,
    /// Fraction of rows in the raw support (`None`: `(n + p + 1) / 2`).
    pub support_fraction: Option<f64>,
    pub seed: u64,
}

impl FitUnsupervised<f64> for MinCovDet {
    type Fitted = FittedCovariance;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedCovariance> {
        check_x(x, 2)?;
        if let Some(f) = self.support_fraction {
            if !(f > 0.0 && f <= 1.0) {
                return Err(RustMlError::InvalidParameter(format!(
                    "support_fraction must be in (0, 1], got {f}"
                )));
            }
        }
        let (n, p) = x.dim();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let (mut raw_location, mut raw_covariance, raw_support, mut dist) =
            fast_mcd(x, self.support_fraction, &mut rng)?;
        if self.assume_centered {
            raw_location = Array1::zeros(p);
            let rows: Vec<usize> = (0..n).filter(|&i| raw_support[i]).collect();
            raw_covariance = empirical(x.select(Axis(0), &rows).view(), true);
            dist = mahalanobis(x, &raw_location, &pinvh(&raw_covariance)?);
        }
        // correct_covariance: only the distances are rescaled (sklearn's
        // `fit` discards the corrected covariance).
        let n_support = raw_support.iter().filter(|&&s| s).count();
        if n_support < n && raw_covariance.iter().all(|v| v.abs() <= 1e-8) {
            return Err(RustMlError::InvalidParameter(
                "the covariance of the support is 0; increase support_fraction".into(),
            ));
        }
        dist /= consistency_factor(p, n_support as f64 / n as f64);
        // reweight_covariance.
        let threshold = chi2_ppf(1.0 - 0.025, p as f64);
        let mask: Vec<usize> = (0..n).filter(|&i| dist[i] < threshold).collect();
        let xm = x.select(Axis(0), &mask);
        let location = location_of(xm.view(), self.assume_centered);
        let covariance =
            empirical(xm.view(), self.assume_centered) * consistency_factor(p, 1.0 - 0.025);
        let mut fitted = FittedCovariance::new(location, covariance, None)?;
        let mut support = vec![false; n];
        for &i in &mask {
            support[i] = true;
        }
        fitted.dist = Some(mahalanobis(x, &fitted.location, &fitted.precision));
        fitted.support = Some(support);
        fitted.raw_location = Some(raw_location);
        fitted.raw_covariance = Some(raw_covariance);
        fitted.raw_support = Some(raw_support);
        Ok(fitted)
    }
}

// ============================================================
// Elliptic envelope
// ============================================================

/// Outlier detection under a Gaussian assumption: a robust (MCD) fit, with
/// the `contamination` fraction of training rows farthest from it as
/// outliers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EllipticEnvelope {
    pub assume_centered: bool,
    pub support_fraction: Option<f64>,
    /// Expected fraction of outliers, in (0, 0.5].
    pub contamination: f64,
    pub seed: u64,
}

impl Default for EllipticEnvelope {
    fn default() -> Self {
        Self {
            assume_centered: false,
            support_fraction: None,
            contamination: 0.1,
            seed: 0,
        }
    }
}

/// Fitted [`EllipticEnvelope`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedEllipticEnvelope {
    mcd: FittedCovariance,
    offset: f64,
}

/// numpy's linear percentile.
fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = q / 100.0 * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

impl FitUnsupervised<f64> for EllipticEnvelope {
    type Fitted = FittedEllipticEnvelope;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedEllipticEnvelope> {
        if !(self.contamination > 0.0 && self.contamination <= 0.5) {
            return Err(RustMlError::InvalidParameter(format!(
                "contamination must be in (0, 0.5], got {}",
                self.contamination
            )));
        }
        let mcd = MinCovDet {
            assume_centered: self.assume_centered,
            support_fraction: self.support_fraction,
            seed: self.seed,
        }
        .fit_view(x)?;
        let neg: Vec<f64> = mcd
            .dist
            .as_ref()
            .expect("MCD distances")
            .iter()
            .map(|d| -d)
            .collect();
        let offset = percentile(&neg, 100.0 * self.contamination);
        Ok(FittedEllipticEnvelope { mcd, offset })
    }
}

impl FittedEllipticEnvelope {
    /// The robust covariance fit.
    pub fn covariance(&self) -> &FittedCovariance {
        &self.mcd
    }

    /// Threshold on [`score_samples`](Self::score_samples) (sklearn's
    /// `offset_`).
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Negative squared Mahalanobis distances.
    pub fn score_samples(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(-self.mcd.mahalanobis(x)?)
    }

    /// `score_samples − offset`: negative for outliers.
    pub fn decision_function(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(self.score_samples(x)? - self.offset)
    }

    /// `1` for inliers, `−1` for outliers.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(self
            .decision_function(x)?
            .mapv(|v| if v >= 0.0 { 1.0 } else { -1.0 }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn chi2_quantiles() {
        // scipy.stats.chi2.ppf(0.975, 2) and .ppf(0.5, 3)
        assert!((chi2_ppf(0.975, 2.0) - 7.377_758_908_227_871).abs() < 1e-10);
        assert!((chi2_ppf(0.5, 3.0) - 2.365_973_884_375_338).abs() < 1e-10);
    }

    #[test]
    fn empirical_and_shrunk() {
        let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0]];
        let e = EmpiricalCovariance::default().fit(&x).unwrap();
        assert!((e.covariance()[[0, 0]] - 1.25).abs() < 1e-12);
        let id = e.covariance().dot(e.precision());
        assert!((id[[0, 0]] - 1.0).abs() < 1e-10 && id[[0, 1]].abs() < 1e-10);
        let s = ShrunkCovariance {
            shrinkage: 1.0,
            ..Default::default()
        }
        .fit(&x)
        .unwrap();
        assert!(s.covariance()[[0, 1]].abs() < 1e-15);
        assert!(LedoitWolf::default().fit(&x).unwrap().shrinkage().unwrap() >= 0.0);
    }
}
