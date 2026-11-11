//! A float32 port of the reference LAPACK routines CatBoost's LDA calls
//! (`ssygst` + `ssyev`, from its bundled f2c CLAPACK and CBLAS).
//!
//! CatBoost's f2c `ilaenv` declares its name buffers as `char[1]`, so no
//! routine name ever matches and every block size is 1: `ssygst`, `ssytrd`
//! and `sorgqr` always run their unblocked kernels (`ssygs2`, `ssytd2`,
//! `sorg2r`), all with unit strides. Only those paths are ported.
//!
//! The arithmetic follows the C code operation by operation, including the
//! places where f2c evaluates in double (`sqrt`, `r_sign` and the `REAL
//! FUNCTION`s return `double`). Matrices are column-major; indices in the
//! comments are LAPACK's 1-based ones.

/// `slamch`: relative machine epsilon ("E", 2^-24).
const EPS: f32 = f32::EPSILON / 2.0;
/// `slamch("P")`: epsilon times the base (2^-23).
const PREC: f32 = f32::EPSILON;
/// `slamch("S")`: the safe minimum (2^-126).
const SAFMIN: f32 = f32::MIN_POSITIVE;

/// Column-major index of (i, j), 0-based.
#[inline]
fn at(ld: usize, i: usize, j: usize) -> usize {
    i + j * ld
}

/// f2c `r_sign`: |a| with the sign of b, in double.
#[inline]
fn r_sign(a: f32, b: f32) -> f64 {
    let x = f64::from(a.abs());
    if b >= 0.0 {
        x
    } else {
        -x
    }
}

/// f2c's `max`/`dmax` macro: `a >= b ? a : b`.
#[inline]
fn fmax(a: f32, b: f32) -> f32 {
    if a >= b {
        a
    } else {
        b
    }
}

#[inline]
fn fmin(a: f32, b: f32) -> f32 {
    if a <= b {
        a
    } else {
        b
    }
}

fn sscal(sa: f32, x: &mut [f32]) {
    for v in x {
        *v = sa * *v;
    }
}

fn saxpy(sa: f32, x: &[f32], y: &mut [f32]) {
    if sa == 0.0 {
        return;
    }
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += sa * xi;
    }
}

/// `sdot` (unit strides; its unrolling keeps the sequential order).
fn sdot(x: &[f32], y: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (&a, &b) in x.iter().zip(y) {
        s += a * b;
    }
    s
}

/// `snrm2`: scaled sum of squares; the final `scale * sqrt(ssq)` is double.
fn snrm2(x: &[f32]) -> f32 {
    match x.len() {
        0 => 0.0,
        1 => x[0].abs(),
        _ => {
            let (mut scale, mut ssq) = (0.0f32, 1.0f32);
            for &v in x {
                if v != 0.0 {
                    let absxi = v.abs();
                    if scale < absxi {
                        let r = scale / absxi;
                        ssq = ssq * (r * r) + 1.0;
                        scale = absxi;
                    } else {
                        let r = absxi / scale;
                        ssq += r * r;
                    }
                }
            }
            (f64::from(scale) * f64::from(ssq).sqrt()) as f32
        }
    }
}

/// `slapy2`: sqrt(x^2 + y^2) without overflow.
fn slapy2(x: f32, y: f32) -> f32 {
    let (xabs, yabs) = (x.abs(), y.abs());
    let w = fmax(xabs, yabs);
    let z = fmin(xabs, yabs);
    if z == 0.0 {
        w
    } else {
        let r = z / w;
        (f64::from(w) * f64::from(r * r + 1.0).sqrt()) as f32
    }
}

/// `ssymv`, lower, unit strides: y = alpha A x + beta y (A is `n` x `n`
/// at `a` with leading dimension `lda`).
fn ssymv_lower(n: usize, alpha: f32, a: &[f32], lda: usize, x: &[f32], beta: f32, y: &mut [f32]) {
    if n == 0 || (alpha == 0.0 && beta == 1.0) {
        return;
    }
    if beta != 1.0 {
        if beta == 0.0 {
            y[..n].fill(0.0);
        } else {
            sscal(beta, &mut y[..n]);
        }
    }
    if alpha == 0.0 {
        return;
    }
    for j in 0..n {
        let temp1 = alpha * x[j];
        let mut temp2 = 0.0f32;
        y[j] += temp1 * a[at(lda, j, j)];
        for i in j + 1..n {
            y[i] += temp1 * a[at(lda, i, j)];
            temp2 += a[at(lda, i, j)] * x[i];
        }
        y[j] += alpha * temp2;
    }
}

/// `ssyr2`, lower, unit strides: A += alpha (x y' + y x').
fn ssyr2_lower(n: usize, alpha: f32, x: &[f32], y: &[f32], a: &mut [f32], lda: usize) {
    if n == 0 || alpha == 0.0 {
        return;
    }
    for j in 0..n {
        if x[j] != 0.0 || y[j] != 0.0 {
            let temp1 = alpha * y[j];
            let temp2 = alpha * x[j];
            for i in j..n {
                let k = at(lda, i, j);
                a[k] = a[k] + x[i] * temp1 + y[i] * temp2;
            }
        }
    }
}

/// `strsv`, lower, no transpose, non-unit, unit stride: x = A^-1 x.
fn strsv_lower(n: usize, a: &[f32], lda: usize, x: &mut [f32]) {
    for j in 0..n {
        if x[j] != 0.0 {
            x[j] /= a[at(lda, j, j)];
            let temp = x[j];
            for i in j + 1..n {
                x[i] -= temp * a[at(lda, i, j)];
            }
        }
    }
}

/// `ssygs2` with itype 1, lower: A = inv(L) A inv(L') where L is the lower
/// triangle of `b` (CatBoost passes its matrix unfactored).
fn ssygs2_lower(n: usize, a: &mut [f32], b: &[f32]) {
    let ld = n;
    for k in 0..n {
        let mut akk = a[at(ld, k, k)];
        let bkk = b[at(ld, k, k)];
        akk /= bkk * bkk;
        a[at(ld, k, k)] = akk;
        if k + 1 < n {
            let m = n - k - 1;
            let col = at(ld, k + 1, k);
            let bcol = &b[col..col + m];
            sscal(1.0 / bkk, &mut a[col..col + m]);
            let ct = akk * -0.5;
            saxpy(ct, bcol, &mut a[col..col + m]);
            {
                let (left, right) = a.split_at_mut(at(ld, 0, k + 1));
                ssyr2_lower(m, -1.0, &left[col..col + m], bcol, &mut right[k + 1..], ld);
            }
            saxpy(ct, bcol, &mut a[col..col + m]);
            strsv_lower(m, &b[at(ld, k + 1, k + 1)..], ld, &mut a[col..col + m]);
        }
    }
}

/// `slarfg`: the elementary reflector H with H' (alpha; x) = (beta; 0).
/// Returns tau; `alpha` and `x` are overwritten.
fn slarfg(n: usize, alpha: &mut f32, x: &mut [f32]) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    let x = &mut x[..n - 1];
    let mut xnorm = snrm2(x);
    if xnorm == 0.0 {
        return 0.0;
    }
    let mut beta = -r_sign(slapy2(*alpha, xnorm), *alpha) as f32;
    let safmin = SAFMIN / EPS;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            sscal(rsafmn, x);
            beta *= rsafmn;
            *alpha *= rsafmn;
            if beta.abs() >= safmin {
                break;
            }
        }
        xnorm = snrm2(x);
        beta = -r_sign(slapy2(*alpha, xnorm), *alpha) as f32;
    }
    let tau = (beta - *alpha) / beta;
    sscal(1.0 / (*alpha - beta), x);
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = beta;
    tau
}

/// `ssytd2`, lower: reduce A to tridiagonal form Q' A Q = T.
fn ssytd2_lower(n: usize, a: &mut [f32], d: &mut [f32], e: &mut [f32], tau: &mut [f32]) {
    let ld = n;
    if n == 0 {
        return;
    }
    for i in 0..n - 1 {
        let m = n - i - 1;
        let col = at(ld, i + 1, i);
        let taui = {
            let (alpha, x) = a[col..col + m].split_first_mut().expect("m >= 1");
            slarfg(m, alpha, x)
        };
        e[i] = a[col];
        if taui != 0.0 {
            a[col] = 1.0;
            let (left, right) = a.split_at_mut(at(ld, 0, i + 1));
            let v = &mut left[col..col + m];
            let blk = &mut right[i + 1..];
            ssymv_lower(m, taui, blk, ld, v, 0.0, &mut tau[i..i + m]);
            let alpha = taui * -0.5 * sdot(&tau[i..i + m], v);
            saxpy(alpha, v, &mut tau[i..i + m]);
            ssyr2_lower(m, -1.0, v, &tau[i..i + m], blk, ld);
            v[0] = e[i];
        }
        d[i] = a[at(ld, i, i)];
        tau[i] = taui;
    }
    d[n - 1] = a[at(ld, n - 1, n - 1)];
}

/// `ilaslc`: the last non-zero column of the `m` x `n` matrix at `c`.
fn ilaslc(m: usize, n: usize, c: &[f32], ldc: usize) -> usize {
    if n == 0 || c[at(ldc, 0, n - 1)] != 0.0 || c[at(ldc, m - 1, n - 1)] != 0.0 {
        return n;
    }
    for j in (0..n).rev() {
        if (0..m).any(|i| c[at(ldc, i, j)] != 0.0) {
            return j + 1;
        }
    }
    0
}

/// `slarf`, left: C = (I - tau v v') C for the `m` x `n` matrix `c`.
fn slarf_left(
    m: usize,
    n: usize,
    v: &[f32],
    tau: f32,
    c: &mut [f32],
    ldc: usize,
    work: &mut [f32],
) {
    if tau == 0.0 {
        return;
    }
    let mut lastv = m;
    while lastv > 0 && v[lastv - 1] == 0.0 {
        lastv -= 1;
    }
    if lastv == 0 {
        return;
    }
    let lastc = ilaslc(lastv, n, c, ldc);
    if lastc == 0 {
        return;
    }
    // sgemv("T"): work = C' v.
    work[..lastc].fill(0.0);
    for j in 0..lastc {
        let mut temp = 0.0f32;
        for i in 0..lastv {
            temp += c[at(ldc, i, j)] * v[i];
        }
        work[j] += 1.0 * temp;
    }
    // sger: C -= tau v work'.
    let alpha = -tau;
    if alpha == 0.0 {
        return;
    }
    for j in 0..lastc {
        if work[j] != 0.0 {
            let temp = alpha * work[j];
            for i in 0..lastv {
                c[at(ldc, i, j)] += v[i] * temp;
            }
        }
    }
}

/// `sorg2r`: the `m` x `n` matrix Q with orthonormal columns from `k`
/// reflectors (stored below the diagonal of `a`, leading dimension `lda`).
fn sorg2r(m: usize, n: usize, k: usize, a: &mut [f32], lda: usize, tau: &[f32], work: &mut [f32]) {
    if n == 0 {
        return;
    }
    for j in k..n {
        for l in 0..m {
            a[at(lda, l, j)] = 0.0;
        }
        a[at(lda, j, j)] = 1.0;
    }
    for i in (0..k).rev() {
        if i + 1 < n {
            a[at(lda, i, i)] = 1.0;
            let (left, right) = a.split_at_mut(at(lda, 0, i + 1));
            let v = &left[at(lda, i, i)..at(lda, i, i) + (m - i)];
            slarf_left(m - i, n - i - 1, v, tau[i], &mut right[i..], lda, work);
        }
        if i + 1 < m {
            let s = at(lda, i + 1, i);
            sscal(-tau[i], &mut a[s..s + (m - i - 1)]);
        }
        a[at(lda, i, i)] = 1.0 - tau[i];
        for l in 0..i {
            a[at(lda, l, i)] = 0.0;
        }
    }
}

/// `sorgtr`, lower: the orthogonal Q of `ssytrd` from its reflectors.
fn sorgtr_lower(n: usize, a: &mut [f32], tau: &[f32], work: &mut [f32]) {
    let ld = n;
    if n == 0 {
        return;
    }
    // Shift the reflectors one column to the right; first row and column
    // become those of the identity.
    for j in (1..n).rev() {
        a[at(ld, 0, j)] = 0.0;
        for i in j + 1..n {
            a[at(ld, i, j)] = a[at(ld, i, j - 1)];
        }
    }
    a[0] = 1.0;
    for i in 1..n {
        a[at(ld, i, 0)] = 0.0;
    }
    if n > 1 {
        sorg2r(n - 1, n - 1, n - 1, &mut a[at(ld, 1, 1)..], ld, tau, work);
    }
}

/// `slaev2`: eigen decomposition of [[a, b], [b, c]]: (rt1, rt2, cs1, sn1).
fn slaev2(a: f32, b: f32, c: f32) -> (f32, f32, f32, f32) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        let r = ab / adf;
        (f64::from(adf) * f64::from(r * r + 1.0).sqrt()) as f32
    } else if adf < ab {
        let r = adf / ab;
        (f64::from(ab) * f64::from(r * r + 1.0).sqrt()) as f32
    } else {
        (f64::from(ab) * 2.0f64.sqrt()) as f32
    };
    let (rt1, rt2, sgn1);
    if sm < 0.0 {
        rt1 = (sm - rt) * 0.5;
        sgn1 = -1;
        rt2 = acmx / rt1 * acmn - b / rt1 * b;
    } else if sm > 0.0 {
        rt1 = (sm + rt) * 0.5;
        sgn1 = 1;
        rt2 = acmx / rt1 * acmn - b / rt1 * b;
    } else {
        rt1 = rt * 0.5;
        rt2 = rt * -0.5;
        sgn1 = 1;
    }
    let (cs, sgn2) = if df >= 0.0 {
        (df + rt, 1)
    } else {
        (df - rt, -1)
    };
    let acs = cs.abs();
    let (mut cs1, mut sn1);
    if acs > ab {
        let ct = -tb / cs;
        sn1 = (1.0 / f64::from(ct * ct + 1.0).sqrt()) as f32;
        cs1 = ct * sn1;
    } else if ab == 0.0 {
        cs1 = 1.0;
        sn1 = 0.0;
    } else {
        let tn = -cs / tb;
        cs1 = (1.0 / f64::from(tn * tn + 1.0).sqrt()) as f32;
        sn1 = tn * cs1;
    }
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
}

/// `slartg`: a plane rotation with [cs sn; -sn cs] [f; g] = [r; 0].
fn slartg(f: f32, g: f32) -> (f32, f32, f32) {
    // pow(2, (int)(log(2^-102) / log(2) / 2)) = 2^-51.
    const SAFMN2: f32 = 1.0 / 2_251_799_813_685_248.0;
    const SAFMX2: f32 = 1.0 / SAFMN2;
    if g == 0.0 {
        return (1.0, 0.0, f);
    }
    if f == 0.0 {
        return (0.0, 1.0, g);
    }
    let hyp = |f1: f32, g1: f32| (f64::from(f1 * f1 + g1 * g1).sqrt()) as f32;
    let (mut f1, mut g1) = (f, g);
    let scale = fmax(f1.abs(), g1.abs());
    let (mut cs, mut sn, mut r);
    if scale >= SAFMX2 {
        let mut count = 0;
        loop {
            count += 1;
            f1 *= SAFMN2;
            g1 *= SAFMN2;
            if fmax(f1.abs(), g1.abs()) < SAFMX2 {
                break;
            }
        }
        r = hyp(f1, g1);
        cs = f1 / r;
        sn = g1 / r;
        for _ in 0..count {
            r *= SAFMX2;
        }
    } else if scale <= SAFMN2 {
        let mut count = 0;
        loop {
            count += 1;
            f1 *= SAFMX2;
            g1 *= SAFMX2;
            if fmax(f1.abs(), g1.abs()) > SAFMN2 {
                break;
            }
        }
        r = hyp(f1, g1);
        cs = f1 / r;
        sn = g1 / r;
        for _ in 0..count {
            r *= SAFMN2;
        }
    } else {
        r = hyp(f1, g1);
        cs = f1 / r;
        sn = g1 / r;
    }
    if f.abs() > g.abs() && cs < 0.0 {
        cs = -cs;
        sn = -sn;
        r = -r;
    }
    (cs, sn, r)
}

/// `slasr` with side "R", pivot "V": apply the `nrot` rotations (c, s) to
/// columns `col..col + nrot + 1` of the `m`-row matrix `z`, forward or
/// backward.
fn slasr_rv(forward: bool, m: usize, c: &[f32], s: &[f32], z: &mut [f32], ldz: usize, col: usize) {
    let nrot = c.len();
    let mut rotate = |j: usize| {
        let (ctemp, stemp) = (c[j], s[j]);
        if ctemp != 1.0 || stemp != 0.0 {
            for i in 0..m {
                let k1 = at(ldz, i, col + j + 1);
                let k0 = at(ldz, i, col + j);
                let temp = z[k1];
                z[k1] = ctemp * temp - stemp * z[k0];
                z[k0] = stemp * temp + ctemp * z[k0];
            }
        }
    };
    if forward {
        (0..nrot).for_each(&mut rotate);
    } else {
        (0..nrot).rev().for_each(&mut rotate);
    }
}

/// `slanst("I")` (= the 1-norm) of the tridiagonal (d, e).
fn slanst_one(d: &[f32], e: &[f32]) -> f32 {
    let n = d.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return d[0].abs();
    }
    let mut anorm = fmax(d[0].abs() + e[0].abs(), e[n - 2].abs() + d[n - 1].abs());
    for i in 1..n - 1 {
        anorm = fmax(anorm, d[i].abs() + e[i].abs() + e[i - 1].abs());
    }
    anorm
}

/// `slascl`: multiply `x` by cto / cfrom without over/underflow. `lower`
/// scales the lower triangle of the `n` x `n` matrix `x` (type "L"),
/// otherwise every entry (type "G").
fn slascl(cfrom: f32, cto: f32, x: &mut [f32], lower: Option<usize>) {
    let smlnum = SAFMIN;
    let bignum = 1.0 / smlnum;
    let (mut cfromc, mut ctoc) = (cfrom, cto);
    loop {
        let cfrom1 = cfromc * smlnum;
        let (mul, done);
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
            }
        }
        match lower {
            Some(n) => {
                for j in 0..n {
                    for i in j..n {
                        x[at(n, i, j)] *= mul;
                    }
                }
            }
            None => x.iter_mut().for_each(|v| *v *= mul),
        }
        if done {
            return;
        }
    }
}

/// `ssteqr` with compz "V": eigenvalues (into `d`, ascending) and vectors
/// (`z` = Q times the tridiagonal's eigenvectors) of the tridiagonal (d, e).
/// Returns `info`.
fn ssteqr(n: usize, d: &mut [f32], e: &mut [f32], z: &mut [f32], work: &mut [f32]) -> usize {
    let ldz = n;
    if n <= 1 {
        return 0;
    }
    let eps = EPS;
    let eps2 = eps * eps;
    let safmin = SAFMIN;
    let safmax = 1.0 / safmin;
    let ssfmax = (f64::from(safmax).sqrt() / 3.0) as f32;
    let ssfmin = (f64::from(safmin).sqrt() / f64::from(eps2)) as f32;
    let nmaxit = n * 30;
    let mut jtot = 0;
    // 1-based indices, as the C code: d(i) = d[i - 1].
    let mut l1 = 1usize;
    let nm1 = n - 1;
    let mut info = 0;
    'outer: loop {
        // L10
        if l1 > n {
            break;
        }
        if l1 > 1 {
            e[l1 - 2] = 0.0;
        }
        let mut m = n;
        if l1 <= nm1 {
            for mm in l1..=nm1 {
                let tst = e[mm - 1].abs();
                if tst == 0.0 {
                    m = mm;
                    break;
                }
                let bound = f64::from(d[mm - 1].abs()).sqrt()
                    * f64::from(d[mm].abs()).sqrt()
                    * f64::from(eps);
                if f64::from(tst) <= bound {
                    e[mm - 1] = 0.0;
                    m = mm;
                    break;
                }
            }
        }
        // L30
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue;
        }
        let anorm = slanst_one(&d[l - 1..lend], &e[l - 1..lend - 1]);
        let mut iscale = 0;
        if anorm == 0.0 {
            continue;
        }
        if anorm > ssfmax {
            iscale = 1;
            slascl(anorm, ssfmax, &mut d[l - 1..lend], None);
            slascl(anorm, ssfmax, &mut e[l - 1..lend - 1], None);
        } else if anorm < ssfmin {
            iscale = 2;
            slascl(anorm, ssfmin, &mut d[l - 1..lend], None);
            slascl(anorm, ssfmin, &mut e[l - 1..lend - 1], None);
        }
        if d[lend - 1].abs() < d[l - 1].abs() {
            lend = lsv;
            l = lendsv;
        }
        if lend > l {
            // QL iteration. L40
            loop {
                let mut m = lend;
                if l != lend {
                    for mm in l..lend {
                        let tst = e[mm - 1].abs() * e[mm - 1].abs();
                        if tst <= eps2 * d[mm - 1].abs() * d[mm].abs() + safmin {
                            m = mm;
                            break;
                        }
                    }
                }
                // L60
                if m < lend {
                    e[m - 1] = 0.0;
                }
                let mut p = d[l - 1];
                if m == l {
                    // L80
                    d[l - 1] = p;
                    l += 1;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if m == l + 1 {
                    let (rt1, rt2, c, s) = slaev2(d[l - 1], e[l - 1], d[l]);
                    work[l - 1] = c;
                    work[n - 1 + l - 1] = s;
                    let (wc, ws) = work.split_at(n - 1);
                    slasr_rv(false, n, &wc[l - 1..l], &ws[l - 1..l], z, ldz, l - 1);
                    d[l - 1] = rt1;
                    d[l] = rt2;
                    e[l - 1] = 0.0;
                    l += 2;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                let mut g = (d[l] - p) / (e[l - 1] * 2.0);
                let mut r = slapy2(g, 1.0);
                g = (f64::from(d[m - 1] - p) + f64::from(e[l - 1]) / (f64::from(g) + r_sign(r, g)))
                    as f32;
                let (mut s, mut c) = (1.0f32, 1.0f32);
                p = 0.0;
                for i in (l..m).rev() {
                    let f = s * e[i - 1];
                    let b = c * e[i - 1];
                    (c, s, r) = slartg(g, f);
                    if i != m - 1 {
                        e[i] = r;
                    }
                    g = d[i] - p;
                    r = (d[i - 1] - g) * s + c * 2.0 * b;
                    p = s * r;
                    d[i] = g + p;
                    g = c * r - b;
                    work[i - 1] = c;
                    work[n - 1 + i - 1] = -s;
                }
                let mm = m - l + 1;
                let (wc, ws) = work.split_at(n - 1);
                slasr_rv(
                    false,
                    n,
                    &wc[l - 1..l - 1 + mm - 1],
                    &ws[l - 1..l - 1 + mm - 1],
                    z,
                    ldz,
                    l - 1,
                );
                d[l - 1] -= p;
                e[l - 1] = g;
            }
        } else {
            // QR iteration. L90
            loop {
                let mut m = lend;
                if l != lend {
                    let mut mm = l;
                    while mm > lend {
                        let tst = e[mm - 2].abs() * e[mm - 2].abs();
                        if tst <= eps2 * d[mm - 1].abs() * d[mm - 2].abs() + safmin {
                            m = mm;
                            break;
                        }
                        mm -= 1;
                    }
                }
                // L110
                if m > lend {
                    e[m - 2] = 0.0;
                }
                let mut p = d[l - 1];
                if m == l {
                    // L130
                    d[l - 1] = p;
                    if l == lend {
                        break;
                    }
                    l -= 1;
                    if l >= lend {
                        continue;
                    }
                    break;
                }
                if m + 1 == l {
                    let (rt1, rt2, c, s) = slaev2(d[l - 2], e[l - 2], d[l - 1]);
                    work[m - 1] = c;
                    work[n - 1 + m - 1] = s;
                    let (wc, ws) = work.split_at(n - 1);
                    slasr_rv(true, n, &wc[m - 1..m], &ws[m - 1..m], z, ldz, l - 2);
                    d[l - 2] = rt1;
                    d[l - 1] = rt2;
                    e[l - 2] = 0.0;
                    if l < lend + 2 {
                        break;
                    }
                    l -= 2;
                    continue;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                let mut g = (d[l - 2] - p) / (e[l - 2] * 2.0);
                let mut r = slapy2(g, 1.0);
                g = (f64::from(d[m - 1] - p) + f64::from(e[l - 2]) / (f64::from(g) + r_sign(r, g)))
                    as f32;
                let (mut s, mut c) = (1.0f32, 1.0f32);
                p = 0.0;
                let lm1 = l - 1;
                for i in m..=lm1 {
                    let f = s * e[i - 1];
                    let b = c * e[i - 1];
                    (c, s, r) = slartg(g, f);
                    if i != m {
                        e[i - 2] = r;
                    }
                    g = d[i - 1] - p;
                    r = (d[i] - g) * s + c * 2.0 * b;
                    p = s * r;
                    d[i - 1] = g + p;
                    g = c * r - b;
                    work[i - 1] = c;
                    work[n - 1 + i - 1] = s;
                }
                let mm = l - m + 1;
                let (wc, ws) = work.split_at(n - 1);
                slasr_rv(
                    true,
                    n,
                    &wc[m - 1..m - 1 + mm - 1],
                    &ws[m - 1..m - 1 + mm - 1],
                    z,
                    ldz,
                    m - 1,
                );
                d[l - 1] -= p;
                e[lm1 - 1] = g;
            }
        }
        // L140
        if iscale == 1 {
            slascl(ssfmax, anorm, &mut d[lsv - 1..lendsv], None);
            slascl(ssfmax, anorm, &mut e[lsv - 1..lendsv - 1], None);
        } else if iscale == 2 {
            slascl(ssfmin, anorm, &mut d[lsv - 1..lendsv], None);
            slascl(ssfmin, anorm, &mut e[lsv - 1..lendsv - 1], None);
        }
        if jtot < nmaxit {
            continue 'outer;
        }
        info = e[..n - 1].iter().filter(|&&v| v != 0.0).count();
        return info;
    }
    // L160: selection sort, swapping the vectors.
    for ii in 2..=n {
        let i = ii - 1;
        let mut k = i;
        let mut p = d[i - 1];
        for j in ii..=n {
            if d[j - 1] < p {
                k = j;
                p = d[j - 1];
            }
        }
        if k != i {
            d[k - 1] = d[i - 1];
            d[i - 1] = p;
            for r in 0..n {
                z.swap(at(ldz, r, i - 1), at(ldz, r, k - 1));
            }
        }
    }
    info
}

/// `ssyev("V", "L")`: eigenvalues (ascending) of the symmetric `n` x `n`
/// matrix `a`, which is overwritten by its eigenvectors (column-major).
/// Returns `info`.
fn ssyev(n: usize, a: &mut [f32], w: &mut [f32]) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        w[0] = a[0];
        a[0] = 1.0;
        return 0;
    }
    let smlnum = SAFMIN / PREC;
    let bignum = 1.0 / smlnum;
    let rmin = f64::from(smlnum).sqrt() as f32;
    let rmax = f64::from(bignum).sqrt() as f32;
    // slansy("M", "L").
    let mut anrm = 0.0f32;
    for j in 0..n {
        for i in j..n {
            anrm = fmax(anrm, a[at(n, i, j)].abs());
        }
    }
    let mut sigma = 0.0f32;
    let iscale = if anrm > 0.0 && anrm < rmin {
        sigma = rmin / anrm;
        true
    } else if anrm > rmax {
        sigma = rmax / anrm;
        true
    } else {
        false
    };
    if iscale {
        slascl(1.0, sigma, a, Some(n));
    }
    let mut e = vec![0.0f32; n];
    let mut tau = vec![0.0f32; n];
    ssytd2_lower(n, a, w, &mut e, &mut tau);
    let mut work = vec![0.0f32; n.max(2 * n - 2)];
    sorgtr_lower(n, a, &tau, &mut work);
    // ssteqr's work is ssyev's tau region onwards (2n - 2 used).
    let mut work = vec![0.0f32; 2 * n];
    work[..n].copy_from_slice(&tau);
    let info = ssteqr(n, w, &mut e, a, &mut work);
    if iscale {
        let imax = if info == 0 { n } else { info - 1 };
        sscal(1.0 / sigma, &mut w[..imax]);
    }
    info
}

/// CatBoost's `CalculateProjection`: `ssygst(1, "L")` of `a` by the
/// (unfactored) `b`, then `ssyev("V", "L")`. `a` is overwritten by the
/// eigenvectors (column-major, ascending eigenvalues); returns the
/// eigenvalues.
pub(crate) fn sygst_syev(n: usize, a: &mut [f32], b: &[f32]) -> Vec<f32> {
    ssygs2_lower(n, a, b);
    let mut w = vec![0.0f32; n];
    ssyev(n, a, &mut w);
    w
}

/// `isamax`: the index of the first entry of largest absolute value.
fn isamax(x: &[f32]) -> usize {
    let mut best = 0;
    let mut smax = x[0].abs();
    for (i, v) in x.iter().enumerate().skip(1) {
        if v.abs() > smax {
            best = i;
            smax = v.abs();
        }
    }
    best
}

/// `sgetf2`: the LU factorization with partial pivoting of the `n` x `n`
/// matrix `a` (CatBoost's `sgetrf` with block size 1). Returns the pivots.
fn sgetf2(n: usize, a: &mut [f32]) -> Vec<usize> {
    let ld = n;
    let mut ipiv = vec![0; n];
    for j in 0..n {
        let jp = j + isamax(&a[at(ld, j, j)..at(ld, n - 1, j) + 1]);
        ipiv[j] = jp;
        if a[at(ld, jp, j)] != 0.0 {
            if jp != j {
                for c in 0..n {
                    a.swap(at(ld, j, c), at(ld, jp, c));
                }
            }
            if j + 1 < n {
                let pivot = a[at(ld, j, j)];
                let col = at(ld, j + 1, j)..at(ld, n - 1, j) + 1;
                if pivot.abs() >= SAFMIN {
                    sscal(1.0 / pivot, &mut a[col]);
                } else {
                    a[col].iter_mut().for_each(|v| *v /= pivot);
                }
            }
        }
        if j + 1 < n {
            // sger: A[j+1.., j+1..] -= a[j+1.., j] a[j, j+1..].
            for c in j + 1..n {
                let yv = a[at(ld, j, c)];
                if yv != 0.0 {
                    let temp = -1.0 * yv;
                    for i in j + 1..n {
                        let x = a[at(ld, i, j)];
                        a[at(ld, i, c)] += x * temp;
                    }
                }
            }
        }
    }
    ipiv
}

/// `strti2("U", "N")`: the inverse of the upper triangle of `a` in place.
fn strti2_upper(n: usize, a: &mut [f32]) {
    let ld = n;
    for j in 0..n {
        a[at(ld, j, j)] = 1.0 / a[at(ld, j, j)];
        let ajj = -a[at(ld, j, j)];
        // strmv("U", "N", "N") of column j by the leading j x j triangle.
        for k in 0..j {
            let xk = a[at(ld, k, j)];
            if xk != 0.0 {
                for i in 0..k {
                    let v = a[at(ld, i, k)];
                    a[at(ld, i, j)] += xk * v;
                }
                a[at(ld, k, j)] *= a[at(ld, k, k)];
            }
        }
        sscal(ajj, &mut a[at(ld, 0, j)..at(ld, 0, j) + j]);
    }
}

/// CatBoost's `InverseMatrix`: `sgetrf` + `sgetri` (block size 1) of the
/// `n` x `n` matrix `a`, in place.
pub(crate) fn inverse(n: usize, a: &mut [f32]) {
    if n == 0 {
        return;
    }
    let ld = n;
    let ipiv = sgetf2(n, a);
    // strtri stops at a zero diagonal (singular): sgetri leaves `a` as is.
    if (0..n).any(|j| a[at(ld, j, j)] == 0.0) {
        return;
    }
    strti2_upper(n, a);
    let mut work = vec![0.0f32; n];
    for j in (0..n).rev() {
        for i in j + 1..n {
            work[i] = a[at(ld, i, j)];
            a[at(ld, i, j)] = 0.0;
        }
        // sgemv("N"): a[:, j] -= a[:, j+1..] work[j+1..].
        for c in j + 1..n {
            if work[c] != 0.0 {
                let temp = -1.0 * work[c];
                for i in 0..n {
                    let v = a[at(ld, i, c)];
                    a[at(ld, i, j)] += temp * v;
                }
            }
        }
    }
    for j in (0..n.saturating_sub(1)).rev() {
        let jp = ipiv[j];
        if jp != j {
            for i in 0..n {
                a.swap(at(ld, i, j), at(ld, i, jp));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(line: &str) -> Vec<f32> {
        line.split_whitespace()
            .map(|h| f32::from_bits(u32::from_str_radix(h, 16).unwrap()))
            .collect()
    }

    /// Bit for bit against CatBoost's own f2c CLAPACK (`ssygst` + `ssyev`
    /// with its `lwork`), on LDA-shaped and edge-case matrices.
    #[test]
    fn sygst_syev_matches_catboost_clapack() {
        let golden = include_str!("lapack_golden.txt");
        let mut lines = golden.lines();
        let mut cases = 0;
        while let Some(n) = lines.next() {
            let n: usize = n.parse().unwrap();
            let mut a = bits(lines.next().unwrap());
            let b = bits(lines.next().unwrap());
            let want_a = bits(lines.next().unwrap());
            let want_w = bits(lines.next().unwrap());
            assert_eq!(lines.next().unwrap(), "0 0");
            let w = sygst_syev(n, &mut a, &b);
            let same = |x: &[f32], y: &[f32]| {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
            };
            assert!(
                same(&w, &want_w),
                "case {cases} (n = {n}): eigenvalues {w:?} vs {want_w:?}"
            );
            assert!(
                same(&a, &want_a),
                "case {cases} (n = {n}): eigenvectors differ"
            );
            cases += 1;
        }
        assert_eq!(cases, 31);
    }

    /// `sgetrf` + `sgetri` bit for bit against CatBoost's CLAPACK.
    #[test]
    fn inverse_matches_catboost_clapack() {
        let golden = include_str!("lapack_inv_golden.txt");
        let mut lines = golden.lines();
        let mut cases = 0;
        while let Some(n) = lines.next() {
            let n: usize = n.parse().unwrap();
            let mut a = bits(lines.next().unwrap());
            let want = bits(lines.next().unwrap());
            inverse(n, &mut a);
            assert!(
                a.iter().zip(&want).all(|(p, q)| p.to_bits() == q.to_bits()),
                "case {cases} (n = {n})"
            );
            cases += 1;
        }
        assert!(cases >= 10);
    }
}
