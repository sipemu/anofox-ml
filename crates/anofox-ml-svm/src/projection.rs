//! Exact Euclidean projections / proximal maps onto "box ∩ hyperplane" sets
//! used by the projected-gradient (FISTA) dual solvers.
//!
//! Every set handled here has the form
//! `{ w : Σ_i w_i = target, lo ≤ w_i ≤ hi }` (optionally combined with a
//! separable ℓ1 term). By the KKT conditions the minimiser of
//! `½‖w − v‖² (+ κ‖w‖₁)` over that set is `w_i = φ(v_i − τ)` for a scalar
//! shift `τ` (the multiplier of the equality constraint), where `φ` is the
//! separable box-(soft-)clipping map. `S(τ) = Σ_i φ(v_i − τ)` is continuous,
//! piecewise linear and non-increasing, so `τ` is found by bisection followed
//! by a final linear interpolation on the bracketing segment.

use anofox_ml_core::Float;

#[inline]
pub(crate) fn clip<F: Float>(value: F, lo: F, hi: F) -> F {
    if value > hi {
        hi
    } else if value < lo {
        lo
    } else {
        value
    }
}

/// Find `τ` such that `Σ_i phi(v_i − τ) = target`, where `phi` is continuous
/// and non-decreasing, given a bracket with `S(tau_lo) ≥ target ≥ S(tau_hi)`.
fn find_shift<F: Float>(
    v: &[F],
    target: F,
    mut tau_lo: F,
    mut tau_hi: F,
    phi: impl Fn(F) -> F,
) -> F {
    let sum_at = |tau: F| v.iter().fold(F::zero(), |acc, &vi| acc + phi(vi - tau));
    let two = F::one() + F::one();
    let mut s_lo = sum_at(tau_lo);
    let mut s_hi = sum_at(tau_hi);
    // 200 halvings shrink any finite f64 bracket to rounding level; stop
    // early once the midpoint no longer moves.
    for _ in 0..200 {
        let mid = (tau_lo + tau_hi) / two;
        if mid <= tau_lo || mid >= tau_hi {
            break;
        }
        let s_mid = sum_at(mid);
        if s_mid == target {
            return mid;
        }
        if s_mid > target {
            tau_lo = mid;
            s_lo = s_mid;
        } else {
            tau_hi = mid;
            s_hi = s_mid;
        }
    }
    // S is linear on the (tiny) final bracket unless a breakpoint lies inside;
    // interpolate to remove the residual bisection error.
    let denom = s_lo - s_hi;
    if denom > F::zero() {
        let t = (s_lo - target) / denom;
        tau_lo + clip(t, F::zero(), F::one()) * (tau_hi - tau_lo)
    } else {
        (tau_lo + tau_hi) / two
    }
}

fn min_max<F: Float>(v: &[F]) -> (F, F) {
    v.iter().fold((v[0], v[0]), |(lo, hi), &x| {
        (if x < lo { x } else { lo }, if x > hi { x } else { hi })
    })
}

/// In-place Euclidean projection of `v` onto
/// `{ α : Σ α_i = target, 0 ≤ α_i ≤ upper }`.
///
/// The set must be non-empty (`0 ≤ target ≤ n · upper`).
pub(crate) fn project_capped_simplex<F: Float>(v: &mut [F], target: F, upper: F) {
    if v.is_empty() {
        return;
    }
    let zero = F::zero();
    let (vmin, vmax) = min_max(v);
    // S(vmin − upper) = n·upper ≥ target, S(vmax) = 0 ≤ target.
    let tau = find_shift(v, target, vmin - upper, vmax, |t| clip(t, zero, upper));
    for val in v.iter_mut() {
        *val = clip(*val - tau, zero, upper);
    }
}

/// In-place proximal map of `κ‖w‖₁` restricted to
/// `{ w : Σ w_i = 0, −c ≤ w_i ≤ c }`, i.e.
/// `argmin_w ½‖w − v‖² + κ‖w‖₁` over that set.
///
/// With `κ = 0` this is the exact Euclidean projection onto the box ∩
/// hyperplane. The solution is `w_i = clip(soft(v_i − τ, κ), −c, c)`.
pub(crate) fn prox_l1_box_zero_sum<F: Float>(v: &mut [F], kappa: F, c: F) {
    if v.is_empty() {
        return;
    }
    let zero = F::zero();
    let phi = |t: F| {
        let s = if t > kappa {
            t - kappa
        } else if t < -kappa {
            t + kappa
        } else {
            zero
        };
        clip(s, -c, c)
    };
    let (vmin, vmax) = min_max(v);
    // S(vmin − κ − c) = n·c ≥ 0, S(vmax + κ + c) = −n·c ≤ 0.
    let tau = find_shift(v, zero, vmin - kappa - c, vmax + kappa + c, phi);
    for val in v.iter_mut() {
        *val = phi(*val - tau);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capped_simplex_hits_target_and_box() {
        let mut v = vec![3.0, -1.0, 0.2, 0.25, 0.9, 0.1];
        let upper = 0.3;
        project_capped_simplex(&mut v, 1.0, upper);
        let s: f64 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "sum = {s}");
        assert!(v.iter().all(|&a| (-1e-15..=upper + 1e-15).contains(&a)));
        // Box active on the largest entries.
        assert!((v[0] - upper).abs() < 1e-12);
    }

    #[test]
    fn capped_simplex_is_projection() {
        // Projection of a feasible point is the point itself.
        let mut v: Vec<f64> = vec![0.25, 0.25, 0.2, 0.3];
        let orig = v.clone();
        project_capped_simplex(&mut v, 1.0, 0.3);
        for (a, b) in v.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn prox_zero_sum_box() {
        let mut v = vec![5.0, 0.3, -0.2, 0.05, 1.7, -0.4];
        prox_l1_box_zero_sum(&mut v, 0.0, 1.0);
        let s: f64 = v.iter().sum();
        assert!(s.abs() < 1e-12, "sum = {s}");
        assert!(v.iter().all(|&w| w.abs() <= 1.0 + 1e-15));
        // With κ > 0 the sum constraint must still hold.
        let mut v = vec![5.0, 0.3, -0.2, 0.05, 1.7, -0.4];
        prox_l1_box_zero_sum(&mut v, 0.1, 1.0);
        let s: f64 = v.iter().sum();
        assert!(s.abs() < 1e-12, "sum = {s}");
    }
}
