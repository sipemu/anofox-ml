//! Canonical Correlation Analysis.
//!
//! A port of `sklearn.cross_decomposition.CCA` (sklearn 1.8: `_PLS` with
//! `deflation_mode="canonical"`, `mode="B"`, NIPALS):
//!
//! 1. Centre X and Y and (with `scale`) divide by their standard deviations
//!    (ddof = 1; constant columns keep scale 1).
//! 2. For each component, the power method finds the first pair of
//!    canonical directions on the current (deflated) matrices:
//!    `w = pinv(X) y_score` normalised, `c = pinv(Y) (X w)` normalised,
//!    until `‖Δw‖² < tol`. Signs follow `_svd_flip_1d` (largest |w| entry
//!    positive).
//! 3. Scores `t = X w`, `u = Y c`; both blocks are deflated by their own
//!    scores (`X -= t pᵀ`, `p = Xᵀ t / tᵀt`).
//! 4. Rotations `W (Pᵀ W)⁻¹` map new data to scores (`transform`).
//!
//! Weights, loadings, rotations and scores match sklearn.

use anofox_ml_core::{Result, RustMlError};
use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView2, AsArray, Axis, Ix2};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cca {
    pub n_components: usize,
    /// Standardise X and Y (sklearn `scale`, default true).
    pub scale: bool,
    pub max_iter: usize,
    pub tol: f64,
}

impl Cca {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            scale: true,
            max_iter: 500,
            tol: 1e-6,
        }
    }
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }

    pub fn fit(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<FittedCca> {
        fit_nipals(
            x,
            y,
            self.n_components,
            self.scale,
            self.max_iter,
            self.tol,
            true,
        )
    }
}

/// sklearn's `PLSCanonical` (`_PLS` with `deflation_mode="canonical"`,
/// `mode="A"`, NIPALS): as [`Cca`], with the power method's weights
/// `Xᵀu / uᵀu` and `Yᵀt / tᵀt` instead of the pseudo-inverses.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlsCanonical {
    pub n_components: usize,
    /// Standardise X and Y (sklearn `scale`, default true).
    pub scale: bool,
    pub max_iter: usize,
    pub tol: f64,
}

impl PlsCanonical {
    pub fn new(n_components: usize) -> Self {
        let d = Cca::new(n_components);
        Self {
            n_components,
            scale: d.scale,
            max_iter: d.max_iter,
            tol: d.tol,
        }
    }

    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// The fitted weights, loadings, rotations and scores, as [`FittedCca`].
    pub fn fit(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<FittedCca> {
        fit_nipals(
            x,
            y,
            self.n_components,
            self.scale,
            self.max_iter,
            self.tol,
            false,
        )
    }
}

/// The two-block NIPALS of sklearn's `_PLS.fit` with canonical deflation,
/// `mode="B"` (CCA) or `"A"` (PLSCanonical).
fn fit_nipals(
    x: &Array2<f64>,
    y: &Array2<f64>,
    k: usize,
    scale: bool,
    max_iter: usize,
    tol: f64,
    mode_b: bool,
) -> Result<FittedCca> {
    let (n, dx) = x.dim();
    let dy = y.ncols();
    if y.nrows() != n {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {n} rows but Y has {}",
            y.nrows()
        )));
    }
    if k == 0 || k > dx.min(dy).min(n) {
        return Err(RustMlError::InvalidParameter(format!(
            "n_components must be in 1..={}",
            dx.min(dy).min(n)
        )));
    }
    if n < 2 {
        return Err(RustMlError::EmptyInput("need at least 2 samples".into()));
    }
    let (mut xk, x_mean, x_std) = center_scale(x, scale);
    let (mut yk, y_mean, y_std) = center_scale(y, scale);
    let eps = f64::EPSILON;

    let mut x_weights = Array2::<f64>::zeros((dx, k));
    let mut y_weights = Array2::<f64>::zeros((dy, k));
    let mut x_loadings = Array2::<f64>::zeros((dx, k));
    let mut y_loadings = Array2::<f64>::zeros((dy, k));
    let mut n_iter = Vec::with_capacity(k);
    for c in 0..k {
        // Columns of Y that are numerically zero are set to exactly 0.
        for mut col in yk.columns_mut() {
            if col.iter().all(|v| v.abs() < 10.0 * eps) {
                col.fill(0.0);
            }
        }
        let (mut w, mut cw, it) = power_method(&xk, &yk, max_iter, tol, mode_b)?;
        // _svd_flip_1d
        let mut best = 0;
        for j in 1..w.len() {
            if w[j].abs() > w[best].abs() {
                best = j;
            }
        }
        if w[best] < 0.0 {
            w.mapv_inplace(|v| -v);
            cw.mapv_inplace(|v| -v);
        }
        let t = xk.dot(&w);
        let u = yk.dot(&cw); // norm_y_weights: y_ss = 1
        let p = xk.t().dot(&t) / t.dot(&t);
        let q = yk.t().dot(&u) / u.dot(&u);
        xk -= &outer(&t, &p);
        yk -= &outer(&u, &q);
        x_weights.column_mut(c).assign(&w);
        y_weights.column_mut(c).assign(&cw);
        x_loadings.column_mut(c).assign(&p);
        y_loadings.column_mut(c).assign(&q);
        n_iter.push(it);
    }
    let x_rotations = x_weights.dot(&pinv(&x_loadings.t().dot(&x_weights))?);
    let y_rotations = y_weights.dot(&pinv(&y_loadings.t().dot(&y_weights))?);
    let mut fitted = FittedCca {
        x_mean,
        y_mean,
        x_std,
        y_std,
        x_weights,
        y_weights,
        x_loadings,
        y_loadings,
        x_rotations,
        y_rotations,
        canonical_correlations: Array1::zeros(k),
        n_iter,
    };
    let (xs, ys) = (fitted.transform_x(x)?, fitted.transform_y(y)?);
    fitted.canonical_correlations = (0..k)
        .map(|c| pearson(xs.column(c), ys.column(c)))
        .collect();
    Ok(fitted)
}

/// sklearn `_center_scale_xy` for one block.
fn center_scale(a: &Array2<f64>, scale: bool) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mean = a.mean_axis(Axis(0)).expect("rows > 0");
    let std = if scale {
        a.std_axis(Axis(0), 1.0)
            .mapv(|s| if s == 0.0 { 1.0 } else { s })
    } else {
        Array1::ones(a.ncols())
    };
    ((a - &mean) / &std, mean, std)
}

fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    Array2::from_shape_fn((a.len(), b.len()), |(i, j)| a[i] * b[j])
}

fn pearson(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
    let (ma, mb) = (a.mean().unwrap(), b.mean().unwrap());
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}

/// sklearn `_pinv2_old`: pseudo-inverse with the singular-value cutoff
/// `max(s) · 1e6 · eps`.
fn pinv(a: &Array2<f64>) -> Result<Array2<f64>> {
    let m = Mat::<f64>::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]]);
    let svd = Svd::new_thin(m.as_ref())
        .map_err(|e| RustMlError::InvalidParameter(format!("CCA SVD failed: {e:?}")))?;
    let (u, s, v) = (svd.U(), svd.S().column_vector(), svd.V());
    let smax = (0..s.nrows()).map(|i| s[i]).fold(0.0, f64::max);
    let cond = smax * 1e6 * f64::EPSILON;
    Ok(Array2::from_shape_fn((a.ncols(), a.nrows()), |(i, j)| {
        (0..s.nrows())
            .filter(|&c| s[c] > cond)
            .map(|c| v[(i, c)] * u[(j, c)] / s[c])
            .sum()
    }))
}

/// sklearn `_get_first_singular_vectors_power_method` with
/// `norm_y_weights=True`, `mode="B"` (pseudo-inverses) or `"A"`.
fn power_method(
    x: &Array2<f64>,
    y: &Array2<f64>,
    max_iter: usize,
    tol: f64,
    mode_b: bool,
) -> Result<(Array1<f64>, Array1<f64>, usize)> {
    let eps = f64::EPSILON;
    let mut y_score = y
        .columns()
        .into_iter()
        .find(|c| c.iter().any(|v| v.abs() > eps))
        .ok_or_else(|| RustMlError::InvalidParameter("Y residual is constant".into()))?
        .to_owned();
    let (x_pinv, y_pinv) = if mode_b {
        (Some(pinv(x)?), Some(pinv(y)?))
    } else {
        (None, None)
    };
    let mut w_old: Option<Array1<f64>> = None;
    let (mut w, mut c) = (Array1::zeros(x.ncols()), Array1::zeros(y.ncols()));
    let mut n_iter = max_iter;
    for it in 0..max_iter {
        w = match &x_pinv {
            Some(p) => p.dot(&y_score),
            None => x.t().dot(&y_score) / y_score.dot(&y_score),
        };
        w /= w.dot(&w).sqrt() + eps;
        let x_score = x.dot(&w);
        c = match &y_pinv {
            Some(p) => p.dot(&x_score),
            None => y.t().dot(&x_score) / x_score.dot(&x_score),
        };
        c /= c.dot(&c).sqrt() + eps;
        y_score = y.dot(&c) / (c.dot(&c) + eps);
        let diff = match &w_old {
            Some(o) => (&w - o).mapv(|v| v * v).sum(),
            None => (&w - 100.0).mapv(|v| v * v).sum(),
        };
        if diff < tol || y.ncols() == 1 {
            n_iter = it + 1;
            break;
        }
        w_old = Some(w.clone());
    }
    Ok((w, c, n_iter))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedCca {
    pub x_mean: Array1<f64>,
    pub y_mean: Array1<f64>,
    pub x_std: Array1<f64>,
    pub y_std: Array1<f64>,
    /// `(n_features_x, k)` (sklearn `x_weights_`).
    pub x_weights: Array2<f64>,
    pub y_weights: Array2<f64>,
    pub x_loadings: Array2<f64>,
    pub y_loadings: Array2<f64>,
    pub x_rotations: Array2<f64>,
    pub y_rotations: Array2<f64>,
    /// Correlation of each pair of training scores.
    pub canonical_correlations: Array1<f64>,
    /// Power-method iterations per component (sklearn `n_iter_`).
    pub n_iter: Vec<usize>,
}

impl FittedCca {
    /// X scores of new data: `((X − mean) / std) x_rotations`.
    pub fn transform_x<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
        if x.ncols() != self.x_mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} X features, got {}",
                self.x_mean.len(),
                x.ncols()
            )));
        }
        Ok(((&x - &self.x_mean) / &self.x_std).dot(&self.x_rotations))
    }

    /// Y scores of new data: `((Y − mean) / std) y_rotations`.
    pub fn transform_y<'a>(&self, y: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let y: ArrayView2<f64> = y.into();
        if y.ncols() != self.y_mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} Y features, got {}",
                self.y_mean.len(),
                y.ncols()
            )));
        }
        Ok(((&y - &self.y_mean) / &self.y_std).dot(&self.y_rotations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_sklearn() {
        // sklearn 1.8 CCA(2).fit(A, B): x_weights_, y_weights_, n_iter_ and
        // transform(A, B) rows.
        let a = Array2::from_shape_fn((30, 3), |(i, j)| {
            ((i * (j + 2)) as f64 * 0.37).sin() + j as f64 * 0.1 * i as f64
        });
        let b = Array2::from_shape_fn((30, 2), |(i, j)| {
            a[[i, j]] * 0.5 + ((i + j) as f64 * 0.9).cos()
        });
        let f = Cca::new(2).fit(&a, &b).unwrap();
        let xw = [
            -0.196_363_781_903_187_87,
            0.950_591_961_615_063_4,
            0.976_819_652_074_914_3,
            0.168_331_363_465_421_44,
            0.085_232_813_381_467_35,
            0.260_843_774_291_617_65,
        ];
        let yw = [
            -0.449_282_013_763_073_64,
            0.893_389_988_811_714_8,
            0.893_389_988_811_715_4,
            0.449_282_013_763_073_5,
        ];
        for (g, w) in f.x_weights.iter().zip(xw) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
        for (g, w) in f.y_weights.iter().zip(yw) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
        assert_eq!(f.n_iter, vec![8, 2]);
        let xs = f.transform_x(&a).unwrap();
        let ys = f.transform_y(&b).unwrap();
        for (g, w) in xs
            .row(0)
            .iter()
            .zip([-1.431_608_874_686_806_8, -0.575_169_199_861_631_4])
        {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
        for (g, w) in ys
            .row(5)
            .iter()
            .zip([0.159_884_416_254_659_2, -0.860_474_347_645_636_8])
        {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
    }
}

/// sklearn's `PLSSVD`: the leading singular vectors of `XᵀY` of the
/// centred (and scaled) blocks, signs by `svd_flip` (largest |u| entry
/// positive).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlsSvd {
    pub n_components: usize,
    /// Standardise X and Y (sklearn `scale`, default true).
    pub scale: bool,
}

/// A fitted [`PlsSvd`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedPlsSvd {
    pub x_mean: Array1<f64>,
    pub y_mean: Array1<f64>,
    pub x_std: Array1<f64>,
    pub y_std: Array1<f64>,
    /// `(n_features_x, k)` (sklearn `x_weights_`).
    pub x_weights: Array2<f64>,
    pub y_weights: Array2<f64>,
}

impl PlsSvd {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            scale: true,
        }
    }

    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    pub fn fit(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<FittedPlsSvd> {
        let (n, dx) = x.dim();
        let dy = y.ncols();
        if y.nrows() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {n} rows but Y has {}",
                y.nrows()
            )));
        }
        let k = self.n_components;
        if k == 0 || k > dx.min(dy).min(n) {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components must be in 1..={}",
                dx.min(dy).min(n)
            )));
        }
        if n < 2 {
            return Err(RustMlError::EmptyInput("need at least 2 samples".into()));
        }
        let (xc, x_mean, x_std) = center_scale(x, self.scale);
        let (yc, y_mean, y_std) = center_scale(y, self.scale);
        let c = xc.t().dot(&yc);
        let m = Mat::<f64>::from_fn(c.nrows(), c.ncols(), |i, j| c[[i, j]]);
        let svd = Svd::new_thin(m.as_ref())
            .map_err(|e| RustMlError::InvalidParameter(format!("PLSSVD SVD failed: {e:?}")))?;
        let (u, v) = (svd.U(), svd.V());
        let mut x_weights = Array2::from_shape_fn((dx, k), |(i, j)| u[(i, j)]);
        let mut y_weights = Array2::from_shape_fn((dy, k), |(i, j)| v[(i, j)]);
        for j in 0..k {
            let col = x_weights.column(j);
            let best = (0..dx).fold(0, |b, i| if col[i].abs() > col[b].abs() { i } else { b });
            if col[best] < 0.0 {
                x_weights.column_mut(j).mapv_inplace(|a| -a);
                y_weights.column_mut(j).mapv_inplace(|a| -a);
            }
        }
        Ok(FittedPlsSvd {
            x_mean,
            y_mean,
            x_std,
            y_std,
            x_weights,
            y_weights,
        })
    }
}

impl FittedPlsSvd {
    /// X scores of new data: `((X − mean) / std) x_weights`.
    pub fn transform_x<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
        if x.ncols() != self.x_mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} X features, got {}",
                self.x_mean.len(),
                x.ncols()
            )));
        }
        Ok(((&x - &self.x_mean) / &self.x_std).dot(&self.x_weights))
    }

    /// Y scores of new data: `((Y − mean) / std) y_weights`.
    pub fn transform_y<'a>(&self, y: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let y: ArrayView2<f64> = y.into();
        if y.ncols() != self.y_mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} Y features, got {}",
                self.y_mean.len(),
                y.ncols()
            )));
        }
        Ok(((&y - &self.y_mean) / &self.y_std).dot(&self.y_weights))
    }
}
