//! Shared numerical helpers for the centroid-based estimators and the
//! mixture models: squared Euclidean distances, GEMM-based nearest-centroid
//! assignment, greedy k-means++ seeding and `logsumexp`.
//!
//! All functions assume `x` is in standard (row-major) layout; estimator
//! entry points guarantee that via [`crate::standard_layout`].

use anofox_ml_core::Float;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;

/// Rows per block for the blocked GEMM / parallel loops.
const CHUNK: usize = 256;

/// Squared Euclidean distance between two slices (4-accumulator pattern so
/// the loop auto-vectorises).
#[inline]
pub(crate) fn squared_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    let n = a.len();
    let chunks = n / 4;
    let mut acc0 = F::zero();
    let mut acc1 = F::zero();
    let mut acc2 = F::zero();
    let mut acc3 = F::zero();
    let mut i = 0;
    for _ in 0..chunks {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
        i += 4;
    }
    for j in i..n {
        let d = a[j] - b[j];
        acc0 += d * d;
    }
    (acc0 + acc1) + (acc2 + acc3)
}

/// Squared Euclidean distance between two (possibly non-contiguous) rows.
#[inline]
pub(crate) fn sq_dist_rows<F: Float>(a: ArrayView1<F>, b: ArrayView1<F>) -> F {
    match (a.as_slice(), b.as_slice()) {
        (Some(a), Some(b)) => squared_euclidean(a, b),
        _ => a
            .iter()
            .zip(b.iter())
            .fold(F::zero(), |acc, (&p, &q)| acc + (p - q) * (p - q)),
    }
}

/// Squared L2 norm of every row.
pub(crate) fn row_norms_sq<F: Float>(x: ArrayView2<F>) -> Array1<F> {
    x.rows()
        .into_iter()
        .map(|r| r.iter().fold(F::zero(), |acc, &v| acc + v * v))
        .collect()
}

/// Numerically stable `log Σ exp(v)`. Returns `-inf` for an all `-inf` input.
pub(crate) fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY || !m.is_finite() {
        return m;
    }
    let s: f64 = v.iter().map(|x| (x - m).exp()).sum();
    m + s.ln()
}

/// Nearest centroid of every row of `x`, together with the squared distance.
///
/// Distances are computed blockwise as `‖x‖² − 2·X·Cᵀ + ‖c‖²` (one GEMM per
/// block of [`CHUNK`] rows, dispatched to `matrixmultiply` for f32/f64), in
/// parallel over row blocks. Distances are clamped at zero; callers that need
/// exact distances (inertia) recompute them for the chosen centroid. Ties are
/// broken towards the lower centroid index.
pub(crate) fn assign_nearest<F: Float>(
    x: ArrayView2<F>,
    x_sq: &Array1<F>,
    centroids: &Array2<F>,
) -> (Vec<usize>, Vec<F>) {
    let n = x.nrows();
    let k = centroids.nrows();
    let c_sq = row_norms_sq(centroids.view());
    let ct = centroids.t();
    let two = F::from_f64(2.0).unwrap();
    let n_chunks = n.div_ceil(CHUNK);
    let blocks: Vec<(Vec<usize>, Vec<F>)> = (0..n_chunks)
        .into_par_iter()
        .map(|b| {
            let lo = b * CHUNK;
            let hi = (lo + CHUNK).min(n);
            let g = x.slice(s![lo..hi, ..]).dot(&ct);
            let mut labels = Vec::with_capacity(hi - lo);
            let mut dists = Vec::with_capacity(hi - lo);
            for (r, i) in (lo..hi).enumerate() {
                let mut best = 0usize;
                let mut best_d = F::infinity();
                for c in 0..k {
                    let d = x_sq[i] - two * g[[r, c]] + c_sq[c];
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                labels.push(best);
                dists.push(if best_d > F::zero() {
                    best_d
                } else {
                    F::zero()
                });
            }
            (labels, dists)
        })
        .collect();
    let mut labels = Vec::with_capacity(n);
    let mut dists = Vec::with_capacity(n);
    for (l, d) in blocks {
        labels.extend(l);
        dists.extend(d);
    }
    (labels, dists)
}

/// Exact squared distance of every row to the given centroid row.
fn sq_dist_to_all<F: Float>(x: ArrayView2<F>, c: ArrayView1<F>) -> Vec<F> {
    let c = c.to_owned();
    let cs = c.as_slice().unwrap();
    (0..x.nrows())
        .into_par_iter()
        .with_min_len(CHUNK)
        .map(|i| {
            let r = x.row(i);
            match r.as_slice() {
                Some(rs) => squared_euclidean(rs, cs),
                None => squared_euclidean(&r.to_vec(), cs),
            }
        })
        .collect()
}

/// Index `i` such that `cum[i-1] < r <= cum[i]` (numpy `searchsorted`,
/// side='left'), clipped to the last index.
fn search_cumulative<F: Float>(cum: &[F], r: F) -> usize {
    cum.partition_point(|&c| c < r).min(cum.len() - 1)
}

/// Greedy k-means++ seeding (Arthur & Vassilvitskii 2007; sklearn's
/// `_kmeans_plusplus`).
///
/// The first centre is drawn with probability ∝ `w_i` (uniform without
/// weights). Every further step draws `2 + ⌊ln k⌋` candidates with
/// probability ∝ `w_i·D²(x_i)` and keeps the one that minimises the resulting
/// potential `Σ w_i·D²(x_i)`. Returns the centres and the chosen row indices.
pub(crate) fn kmeans_plusplus<F: Float>(
    x: ArrayView2<F>,
    n_clusters: usize,
    rng: &mut StdRng,
    sample_weight: Option<ArrayView1<F>>,
) -> (Array2<F>, Vec<usize>) {
    let n = x.nrows();
    let d = x.ncols();
    let mut centers = Array2::<F>::zeros((n_clusters, d));
    let mut indices = Vec::with_capacity(n_clusters);
    if n_clusters == 0 || n == 0 {
        return (centers, indices);
    }
    let w = |i: usize| sample_weight.map_or(F::one(), |sw| sw[i]);
    let n_local_trials = 2 + (n_clusters as f64).ln().floor() as usize;

    // First centre.
    let first = match sample_weight {
        Some(sw) => {
            let mut cum = Vec::with_capacity(n);
            let mut acc = F::zero();
            for &v in sw.iter() {
                acc += v;
                cum.push(acc);
            }
            if acc > F::zero() {
                let r = F::from_f64(rng.gen::<f64>()).unwrap() * acc;
                search_cumulative(&cum, r)
            } else {
                rng.gen_range(0..n)
            }
        }
        None => rng.gen_range(0..n),
    };
    centers.row_mut(0).assign(&x.row(first));
    indices.push(first);

    let mut closest = sq_dist_to_all(x, x.row(first));
    let mut pot = (0..n).fold(F::zero(), |acc, i| acc + closest[i] * w(i));

    for c in 1..n_clusters {
        let mut cum = Vec::with_capacity(n);
        let mut acc = F::zero();
        for (i, &v) in closest.iter().enumerate() {
            acc += v * w(i);
            cum.push(acc);
        }
        let candidates: Vec<usize> = (0..n_local_trials)
            .map(|_| {
                let r = F::from_f64(rng.gen::<f64>()).unwrap() * pot;
                search_cumulative(&cum, r)
            })
            .collect();

        let mut best: Option<(F, usize, Vec<F>)> = None;
        for &cand in &candidates {
            let mut dist = sq_dist_to_all(x, x.row(cand));
            let mut cand_pot = F::zero();
            for i in 0..n {
                if closest[i] < dist[i] {
                    dist[i] = closest[i];
                }
                cand_pot += dist[i] * w(i);
            }
            if best.as_ref().is_none_or(|(p, _, _)| cand_pot < *p) {
                best = Some((cand_pot, cand, dist));
            }
        }
        let (best_pot, best_idx, best_dist) = best.unwrap();
        pot = best_pot;
        closest = best_dist;
        centers.row_mut(c).assign(&x.row(best_idx));
        indices.push(best_idx);
    }
    (centers, indices)
}

/// `k` distinct row indices drawn uniformly without replacement (partial
/// Fisher–Yates), or ∝ `sample_weight` when given (sequential weighted
/// sampling without replacement, falling back to uniform once the positive
/// weight is exhausted).
pub(crate) fn sample_without_replacement<F: Float>(
    n: usize,
    k: usize,
    rng: &mut StdRng,
    sample_weight: Option<ArrayView1<F>>,
) -> Vec<usize> {
    match sample_weight {
        None => {
            let mut idx: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = i + rng.gen_range(0..(n - i));
                idx.swap(i, j);
            }
            idx.truncate(k);
            idx
        }
        Some(sw) => {
            let mut w: Vec<f64> = sw.iter().map(|v| v.to_f64().unwrap().max(0.0)).collect();
            let mut out = Vec::with_capacity(k);
            for _ in 0..k {
                let total: f64 = w.iter().sum();
                let pick = if total > 0.0 {
                    let r = rng.gen::<f64>() * total;
                    let mut acc = 0.0;
                    let mut pick = None;
                    for (i, &wi) in w.iter().enumerate() {
                        acc += wi;
                        if wi > 0.0 && acc >= r {
                            pick = Some(i);
                            break;
                        }
                    }
                    pick.unwrap_or_else(|| w.iter().rposition(|&v| v > 0.0).unwrap())
                } else {
                    let rest: Vec<usize> = (0..n).filter(|i| !out.contains(i)).collect();
                    rest[rng.gen_range(0..rest.len())]
                };
                w[pick] = 0.0;
                out.push(pick);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;

    #[test]
    fn gemm_assignment_matches_brute_force() {
        let mut rng = StdRng::seed_from_u64(1);
        let x = Array2::from_shape_fn((600, 5), |_| rng.gen::<f64>() * 10.0);
        let c = Array2::from_shape_fn((7, 5), |_| rng.gen::<f64>() * 10.0);
        let (labels, dists) = assign_nearest(x.view(), &row_norms_sq(x.view()), &c);
        for i in 0..x.nrows() {
            let mut best = (0, f64::INFINITY);
            for k in 0..c.nrows() {
                let d = sq_dist_rows(x.row(i), c.row(k));
                if d < best.1 {
                    best = (k, d);
                }
            }
            assert_eq!(labels[i], best.0);
            assert!((dists[i] - best.1).abs() < 1e-9);
        }
    }

    #[test]
    fn kmeans_plusplus_picks_distinct_far_points() {
        let x = array![
            [0.0, 0.0],
            [0.0, 0.1],
            [100.0, 0.0],
            [100.0, 0.1],
            [0.0, 100.0]
        ];
        for seed in 0..20 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (_, idx) = kmeans_plusplus(x.view(), 3, &mut rng, None);
            let mut groups: Vec<usize> = idx
                .iter()
                .map(|&i| match i {
                    0 | 1 => 0,
                    2 | 3 => 1,
                    _ => 2,
                })
                .collect();
            groups.sort();
            assert_eq!(groups, vec![0, 1, 2], "seed {seed}: {idx:?}");
        }
    }

    #[test]
    fn kmeans_plusplus_never_picks_zero_weight() {
        let x = array![[0.0], [1.0], [2.0], [50.0], [100.0]];
        let w = array![1.0, 0.0, 1.0, 0.0, 1.0];
        for seed in 0..20 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (_, idx) = kmeans_plusplus(x.view(), 3, &mut rng, Some(w.view()));
            for i in idx {
                assert!(w[i] > 0.0);
            }
        }
    }

    #[test]
    fn logsumexp_stable() {
        assert!((logsumexp(&[1000.0, 1000.0]) - (1000.0 + 2f64.ln())).abs() < 1e-12);
        assert_eq!(logsumexp(&[f64::NEG_INFINITY]), f64::NEG_INFINITY);
    }
}
