//! Small special-function and 1-D optimisation toolkit used by the feature
//! selectors (p-values, k-NN mutual information) and transformers
//! (normal quantile mapping, power-transform λ search).
//!
//! Everything is computed in `f64`. Accuracy targets are ~1e-13 relative,
//! which is far below what the callers (and their sklearn reference values)
//! need.

use std::f64::consts::PI;

/// `ln Γ(x)` for `x > 0` (Lanczos approximation, g = 7, n = 9).
pub(crate) fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = COEF[0];
    let t = x + G + 0.5;
    for (i, &c) in COEF.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Digamma function ψ(x) for `x > 0` (recurrence up to x ≥ 10, then the
/// asymptotic expansion).
pub(crate) fn digamma(mut x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let mut result = 0.0;
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // ψ(x) ~ ln x − 1/(2x) − Σ B_{2k}/(2k x^{2k})
    let series = inv2
        * (1.0 / 12.0
            - inv2
                * (1.0 / 120.0
                    - inv2
                        * (1.0 / 252.0
                            - inv2
                                * (1.0 / 240.0 - inv2 * (1.0 / 132.0 - inv2 * 691.0 / 32760.0)))));
    result + x.ln() - 0.5 * inv - series
}

/// Continued fraction for the regularised incomplete beta (Numerical Recipes
/// `betacf`, modified Lentz).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 10_000;
    const EPS: f64 = 1e-16;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularised incomplete beta function `I_x(a, b)`.
pub(crate) fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Survival function of the F distribution with `(d1, d2)` degrees of
/// freedom (`scipy.stats.f.sf`).
pub(crate) fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
    if f.is_nan() {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    if f.is_infinite() {
        return 0.0;
    }
    beta_inc(0.5 * d2, 0.5 * d1, d2 / (d2 + d1 * f))
}

/// Regularised upper incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a)`.
pub(crate) fn gamma_q(a: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let gln = ln_gamma(a);
    if x < a + 1.0 {
        // Series for P(a, x).
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..100_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-17 {
                break;
            }
        }
        1.0 - sum * (-x + a * x.ln() - gln).exp()
    } else {
        // Continued fraction for Q(a, x) (modified Lentz).
        const FPMIN: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..100_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-17 {
                break;
            }
        }
        (-x + a * x.ln() - gln).exp() * h
    }
}

/// Survival function of the χ² distribution with `k` degrees of freedom.
pub(crate) fn chi2_sf(x: f64, k: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    gamma_q(0.5 * k, 0.5 * x)
}

/// Standard normal CDF Φ(z).
pub(crate) fn norm_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    // Φ(z) = ½ erfc(−z/√2) and erfc(t) = Q(½, t²) for t ≥ 0.
    let tail = 0.5 * gamma_q(0.5, 0.5 * z * z);
    if z < 0.0 {
        tail
    } else {
        1.0 - tail
    }
}

/// Standard normal quantile Φ⁻¹(p) (Acklam's rational approximation refined
/// by one Halley step against [`norm_cdf`], giving ~machine precision).
pub(crate) fn norm_ppf(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.02425;
    let x = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - p_low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    // Halley refinement against the accurate CDF. The tail probability is
    // computed directly on the side of `x` to avoid cancellation.
    let refine = |x: f64| {
        let tail = 0.5 * gamma_q(0.5, 0.5 * x * x);
        let e = if x < 0.0 { tail - p } else { (1.0 - p) - tail };
        let u = e * (2.0 * PI).sqrt() * (0.5 * x * x).exp();
        x - u / (1.0 + 0.5 * x * u)
    };
    refine(refine(x))
}

// ---------------------------------------------------------------------------
// 1-D minimisers (ports of scipy.optimize.brent / fminbound)
// ---------------------------------------------------------------------------

/// Port of `scipy.optimize.bracket`. Returns `(xa, xb, xc, fa, fb, fc)` or
/// `None` if no valid bracket was found (scipy then falls back to the best
/// point found).
#[allow(clippy::type_complexity)]
fn bracket(
    f: &impl Fn(f64) -> f64,
    mut xa: f64,
    mut xb: f64,
) -> std::result::Result<(f64, f64, f64, f64, f64, f64), (f64, f64)> {
    const GOLD: f64 = 1.618_034;
    const VERYSMALL: f64 = 1e-21;
    const GROW_LIMIT: f64 = 110.0;
    const MAXITER: usize = 1000;
    let mut fa = f(xa);
    let mut fb = f(xb);
    if fa < fb {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut xc = xb + GOLD * (xb - xa);
    let mut fc = f(xc);
    let mut iter = 0;
    while fc < fb {
        let tmp1 = (xb - xa) * (fb - fc);
        let tmp2 = (xb - xc) * (fb - fa);
        let val = tmp2 - tmp1;
        let denom = if val.abs() < VERYSMALL {
            2.0 * VERYSMALL
        } else {
            2.0 * val
        };
        let mut w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
        let wlim = xb + GROW_LIMIT * (xc - xb);
        if iter > MAXITER {
            break;
        }
        iter += 1;
        let mut fw;
        if (w - xc) * (xb - w) > 0.0 {
            fw = f(w);
            if fw < fc {
                xa = xb;
                xb = w;
                fa = fb;
                fb = fw;
                break;
            } else if fw > fb {
                xc = w;
                fc = fw;
                break;
            }
            w = xc + GOLD * (xc - xb);
            fw = f(w);
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
            fw = f(w);
        } else if (w - wlim) * (xc - w) > 0.0 {
            fw = f(w);
            if fw < fc {
                xb = xc;
                xc = w;
                w = xc + GOLD * (xc - xb);
                fb = fc;
                fc = fw;
                fw = f(w);
            }
        } else {
            w = xc + GOLD * (xc - xb);
            fw = f(w);
        }
        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = fw;
    }
    let cond1 = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
    let cond2 = (xa < xb && xb < xc) || (xc < xb && xb < xa);
    let cond3 = xa.is_finite() && xb.is_finite() && xc.is_finite();
    if cond1 && cond2 && cond3 {
        Ok((xa, xb, xc, fa, fb, fc))
    } else {
        // Best finite point seen in the final triple.
        let mut best = (xb, fb);
        for (x, fx) in [(xa, fa), (xc, fc)] {
            if fx < best.1 && x.is_finite() {
                best = (x, fx);
            }
        }
        Err(best)
    }
}

/// Port of `scipy.optimize.brent(func, brack=(xa, xb))` (tol = 1.48e-8,
/// maxiter = 500). Returns the minimiser.
pub(crate) fn brent_minimize(f: impl Fn(f64) -> f64, xa: f64, xb: f64) -> f64 {
    const TOL: f64 = 1.48e-8;
    const MINTOL: f64 = 1.0e-11;
    const CG: f64 = 0.381_966_0;
    const MAXITER: usize = 500;
    let (xa, xb, xc, _fa, fb, _fc) = match bracket(&f, xa, xb) {
        Ok(b) => b,
        Err((x, _)) => return x,
    };
    let (mut x, mut w, mut v) = (xb, xb, xb);
    let (mut fx, mut fw, mut fv) = (fb, fb, fb);
    let (mut a, mut b) = if xa < xc { (xa, xc) } else { (xc, xa) };
    let mut deltax: f64 = 0.0;
    let mut rat: f64 = 0.0;
    for _ in 0..MAXITER {
        let tol1 = TOL * x.abs() + MINTOL;
        let tol2 = 2.0 * tol1;
        let xmid = 0.5 * (a + b);
        if (x - xmid).abs() < (tol2 - 0.5 * (b - a)) {
            break;
        }
        if deltax.abs() <= tol1 {
            deltax = if x >= xmid { a - x } else { b - x };
            rat = CG * deltax;
        } else {
            let tmp1 = (x - w) * (fx - fv);
            let mut tmp2 = (x - v) * (fx - fw);
            let mut p = (x - v) * tmp2 - (x - w) * tmp1;
            tmp2 = 2.0 * (tmp2 - tmp1);
            if tmp2 > 0.0 {
                p = -p;
            }
            tmp2 = tmp2.abs();
            let dx_temp = deltax;
            deltax = rat;
            if p > tmp2 * (a - x) && p < tmp2 * (b - x) && p.abs() < (0.5 * tmp2 * dx_temp).abs() {
                rat = p / tmp2;
                let u = x + rat;
                if (u - a) < tol2 || (b - u) < tol2 {
                    rat = if xmid - x >= 0.0 { tol1 } else { -tol1 };
                }
            } else {
                deltax = if x >= xmid { a - x } else { b - x };
                rat = CG * deltax;
            }
        }
        let u = if rat.abs() < tol1 {
            if rat >= 0.0 {
                x + tol1
            } else {
                x - tol1
            }
        } else {
            x + rat
        };
        let fu = f(u);
        if fu > fx {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        } else {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }
    }
    x
}

/// Port of `scipy.optimize.fminbound(func, x1, x2, xtol)` (bounded Brent,
/// maxfun = 500). Returns the minimiser.
pub(crate) fn fminbound(f: impl Fn(f64) -> f64, x1: f64, x2: f64, xatol: f64) -> f64 {
    const MAXFUN: usize = 500;
    let sqrt_eps = 2.2e-16_f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let mut nfc = fulc;
    let mut xf = fulc;
    let mut rat: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut fx = f(xf);
    let mut num = 1;
    let mut ffulc = fx;
    let mut fnfc = fx;
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;
    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let d = xm - xf;
                    let si = if d == 0.0 { 1.0 } else { d.signum() };
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = if rat == 0.0 { 1.0 } else { rat.signum() };
        let x = xf + si * rat.abs().max(tol1);
        let fu = f(x);
        num += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= MAXFUN {
            break;
        }
    }
    xf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rtol: f64) -> bool {
        (a - b).abs() <= rtol * b.abs().max(1e-300)
    }

    #[test]
    fn special_functions_match_scipy() {
        // scipy.special.gammaln(4.5), digamma(3), digamma(0.7)
        assert!(close(ln_gamma(4.5), 2.453_736_570_842_442, 1e-13));
        assert!(close(digamma(3.0), 0.922_784_335_098_467_1, 1e-13));
        assert!(close(digamma(0.7), -1.220_023_553_697_934_7, 1e-12));
        // scipy.stats.f.sf(3.2, 2, 17)
        assert!(close(f_sf(3.2, 2.0, 17.0), 0.066_142_566_666_421_3, 1e-9));
        // scipy.stats.chi2.sf(5.3, 3), chi2.sf(25.0, 2)
        assert!(close(chi2_sf(5.3, 3.0), 0.151_102_391_098_458_24, 1e-10));
        assert!(close(chi2_sf(25.0, 2.0), 3.726_653_172_078_671e-6, 1e-10));
        // scipy.stats.norm.cdf(-1.3), norm.ppf(0.01), norm.ppf(1e-7)
        assert!(close(norm_cdf(-1.3), 0.096_800_484_585_610_35, 1e-12));
        assert!(close(norm_ppf(0.01), -2.326_347_874_040_841, 1e-12));
        assert!(close(norm_ppf(1e-7), -5.199_337_582_192_816_5, 1e-12));
        assert!(close(norm_ppf(0.975), 1.959_963_984_540_054, 1e-12));
    }

    #[test]
    fn minimisers_find_parabola_minimum() {
        let f = |x: f64| (x - 0.7).powi(2) + 1.0;
        assert!((brent_minimize(f, -2.0, 2.0) - 0.7).abs() < 1e-7);
        assert!((fminbound(f, -5.0, 5.0, 1.48e-8) - 0.7).abs() < 1e-7);
        // Minimum outside the initial bracket.
        let g = |x: f64| (x - 7.5).powi(2);
        assert!((brent_minimize(g, -2.0, 2.0) - 7.5).abs() < 1e-6);
    }
}
