//! FastICA — fixed-point independent component analysis.
//!
//! A port of `sklearn.decomposition.FastICA` (sklearn 1.8):
//!
//! 1. Centre and whiten: `K = (U / d)ᵀ[:k]` from the SVD (or the
//!    eigendecomposition of `XᵀX`) of the centred data, eigenvector signs
//!    fixed by the first row (`u *= sign(u[0])`), whitened data
//!    `X1 = K Xᵀ · sqrt(n)`.
//! 2. From `w_init` (default `N(0, 1)` entries), iterate the fixed-point
//!    update `W ← E[g(W X1) X1ᵀ] − diag(E[g'(W X1)]) W` with symmetric
//!    decorrelation `W ← (W Wᵀ)^{-1/2} W` (`Parallel`, the default) or one
//!    component at a time with Gram–Schmidt deflation (`Deflation`), until
//!    `max |abs(diag(W_new Wᵀ)) − 1| < tol`.
//! 3. With `Whiten::UnitVariance` (default) the sources are rescaled to unit
//!    variance; `components_ = W K`, `mixing_ = pinv(components_)`.
//!
//! Given the same `w_init` the result equals sklearn's; the default random
//! `w_init` comes from this crate's generator.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use faer::linalg::solvers::{SelfAdjointEigen, Svd};
use faer::{Mat, Side};
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Fixed-point scheme (sklearn `algorithm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum IcaAlgorithm {
    /// All components at once with symmetric decorrelation.
    #[default]
    Parallel,
    /// One component at a time with Gram–Schmidt deflation.
    Deflation,
}

/// Contrast function (sklearn `fun`).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum IcaFun {
    /// `g(u) = tanh(alpha u)` (default `alpha = 1`).
    Logcosh { alpha: f64 },
    /// `g(u) = u exp(−u²/2)`.
    Exp,
    /// `g(u) = u³`.
    Cube,
}

impl Default for IcaFun {
    fn default() -> Self {
        IcaFun::Logcosh { alpha: 1.0 }
    }
}

/// Whitening (sklearn `whiten`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum IcaWhiten {
    /// Whiten and rescale the sources to unit variance.
    #[default]
    UnitVariance,
    /// Whiten without the final rescaling.
    ArbitraryVariance,
    /// The data are already white: no centring, no whitening.
    None,
}

/// Whitening solver (sklearn `whiten_solver`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum IcaWhitenSolver {
    #[default]
    Svd,
    /// Eigendecomposition of `XᵀX` (faster for `n_samples ≫ n_features`).
    Eigh,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FastIca {
    /// Number of components; `None` = `min(n_samples, n_features)`.
    pub n_components: Option<usize>,
    #[serde(default)]
    pub algorithm: IcaAlgorithm,
    #[serde(default)]
    pub whiten: IcaWhiten,
    #[serde(default)]
    pub fun: IcaFun,
    pub max_iter: usize,
    pub tol: f64,
    /// Initial unmixing matrix, shape `(k, k)`; `None` draws `N(0, 1)`.
    #[serde(default)]
    pub w_init: Option<Array2<f64>>,
    #[serde(default)]
    pub whiten_solver: IcaWhitenSolver,
    pub seed: u64,
}

impl FastIca {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: Some(n_components),
            algorithm: IcaAlgorithm::Parallel,
            whiten: IcaWhiten::UnitVariance,
            fun: IcaFun::default(),
            max_iter: 200,
            tol: 1e-4,
            w_init: None,
            whiten_solver: IcaWhitenSolver::Svd,
            seed: 0,
        }
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn with_algorithm(mut self, a: IcaAlgorithm) -> Self {
        self.algorithm = a;
        self
    }
    pub fn with_fun(mut self, f: IcaFun) -> Self {
        self.fun = f;
        self
    }
    pub fn with_whiten(mut self, w: IcaWhiten) -> Self {
        self.whiten = w;
        self
    }
    pub fn with_whiten_solver(mut self, s: IcaWhitenSolver) -> Self {
        self.whiten_solver = s;
        self
    }
    pub fn with_w_init(mut self, w: Array2<f64>) -> Self {
        self.w_init = Some(w);
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedFastIca {
    /// Unmixing matrix in data space, `(n_components, n_features)`
    /// (`components_`).
    pub components: Array2<f64>,
    /// Pseudo-inverse of `components`, `(n_features, n_components)`
    /// (`mixing_`).
    pub mixing: Array2<f64>,
    /// Per-feature mean (zeros without whitening) (`mean_`).
    pub mean: Array1<f64>,
    /// Whitening matrix `K`, `(n_components, n_features)` (`whitening_`);
    /// `None` without whitening.
    pub whitening: Option<Array2<f64>>,
    /// Iterations run (`n_iter_`).
    pub n_iter: usize,
    pub n_features: usize,
}

/// `g(x)` row-wise and the row means of `g'(x)` (sklearn `_logcosh` /
/// `_exp` / `_cube`).
fn contrast(fun: IcaFun, x: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    match fun {
        IcaFun::Logcosh { alpha } => {
            let gx = x.mapv(|v| (alpha * v).tanh());
            let g_x = gx.map_axis(Axis(1), |r| {
                r.mapv(|t| alpha * (1.0 - t * t)).mean().unwrap()
            });
            (gx, g_x)
        }
        IcaFun::Exp => {
            let e = x.mapv(|v| (-(v * v) / 2.0).exp());
            let gx = x * &e;
            let g_x = (x.mapv(|v| 1.0 - v * v) * &e).mean_axis(Axis(1)).unwrap();
            (gx, g_x)
        }
        IcaFun::Cube => (
            x.mapv(|v| v * v * v),
            x.mapv(|v| 3.0 * v * v).mean_axis(Axis(1)).unwrap(),
        ),
    }
}

fn to_mat(a: &Array2<f64>) -> Mat<f64> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]])
}

/// sklearn `_sym_decorrelation`: `(W Wᵀ)^{-1/2} W`.
fn sym_decorrelation(w: &Array2<f64>) -> Result<Array2<f64>> {
    let wwt = w.dot(&w.t());
    let eig = SelfAdjointEigen::new(to_mat(&wwt).as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("FastICA eigh failed: {e:?}")))?;
    let s = eig.S().column_vector();
    let u = eig.U();
    let k = wwt.nrows();
    let tiny = f64::MIN_POSITIVE;
    let mut inv_sqrt = Array2::<f64>::zeros((k, k));
    for a in 0..k {
        for b in 0..k {
            let mut acc = 0.0;
            for c in 0..k {
                acc += u[(a, c)] * u[(b, c)] / s[c].max(tiny).sqrt();
            }
            inv_sqrt[[a, b]] = acc;
        }
    }
    Ok(inv_sqrt.dot(w))
}

/// sklearn `_ica_par`.
fn ica_par(
    x1: &Array2<f64>,
    tol: f64,
    fun: IcaFun,
    max_iter: usize,
    w_init: &Array2<f64>,
) -> Result<(Array2<f64>, usize)> {
    let mut w = sym_decorrelation(w_init)?;
    let p = x1.ncols() as f64;
    let mut n_iter = max_iter;
    for it in 0..max_iter {
        let (gwtx, g_wtx) = contrast(fun, &w.dot(x1));
        let mut upd = gwtx.dot(&x1.t()) / p;
        for (mut row, (&g, wr)) in upd.rows_mut().into_iter().zip(g_wtx.iter().zip(w.rows())) {
            row.scaled_add(-g, &wr);
        }
        let w1 = sym_decorrelation(&upd)?;
        let lim = w1
            .rows()
            .into_iter()
            .zip(w.rows())
            .map(|(a, b)| (a.dot(&b).abs() - 1.0).abs())
            .fold(0.0, f64::max);
        w = w1;
        if lim < tol {
            n_iter = it + 1;
            break;
        }
    }
    Ok((w, n_iter))
}

/// sklearn `_ica_def`.
fn ica_def(
    x1: &Array2<f64>,
    tol: f64,
    fun: IcaFun,
    max_iter: usize,
    w_init: &Array2<f64>,
) -> (Array2<f64>, usize) {
    let k = w_init.nrows();
    let mut w_all = Array2::<f64>::zeros((k, k));
    let mut max_n_iter = 0;
    for j in 0..k {
        let mut w = w_init.row(j).to_owned();
        w /= w.dot(&w).sqrt();
        let mut n_iter = max_iter;
        for it in 0..max_iter {
            let wx = w.dot(x1).insert_axis(Axis(0));
            let (gwtx, g_wtx) = contrast(fun, &wx);
            let gwtx = gwtx.row(0);
            let mut w1 = (x1 * &gwtx).mean_axis(Axis(1)).unwrap() - &(&w * g_wtx[0]);
            // Gram–Schmidt against the components found so far.
            let prev = w_all.slice(ndarray::s![..j, ..]);
            let proj = prev.t().dot(&prev.dot(&w1));
            w1 -= &proj;
            w1 /= w1.dot(&w1).sqrt();
            let lim = (w1.dot(&w).abs() - 1.0).abs();
            w = w1;
            if lim < tol {
                n_iter = it + 1;
                break;
            }
        }
        max_n_iter = max_n_iter.max(n_iter);
        w_all.row_mut(j).assign(&w);
    }
    (w_all, max_n_iter)
}

/// Moore–Penrose pseudo-inverse via the SVD.
fn pinv(a: &Array2<f64>) -> Result<Array2<f64>> {
    let svd = Svd::new_thin(to_mat(a).as_ref())
        .map_err(|e| RustMlError::InvalidParameter(format!("FastICA pinv failed: {e:?}")))?;
    let (u, s, v) = (svd.U(), svd.S().column_vector(), svd.V());
    let smax = (0..s.nrows()).map(|i| s[i]).fold(0.0, f64::max);
    let cutoff = smax * f64::EPSILON * a.nrows().max(a.ncols()) as f64;
    Ok(Array2::from_shape_fn((a.ncols(), a.nrows()), |(i, j)| {
        (0..s.nrows())
            .filter(|&c| s[c] > cutoff)
            .map(|c| v[(i, c)] * u[(j, c)] / s[c])
            .sum()
    }))
}

/// `N(0, 1)` (Box–Muller).
fn normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

impl FitUnsupervised<f64> for FastIca {
    type Fitted = FittedFastIca;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let (n, d) = x.dim();
        if n < 2 || d == 0 {
            return Err(RustMlError::EmptyInput("need at least 2 samples".into()));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "FastICA input contains NaN or infinite values".into(),
            ));
        }
        let whiten = self.whiten != IcaWhiten::None;
        let k = match self.n_components {
            Some(0) => {
                return Err(RustMlError::InvalidParameter("n_components >= 1".into()));
            }
            // sklearn ignores n_components without whitening.
            _ if !whiten => d,
            Some(k) => k.min(n.min(d)),
            None => n.min(d),
        };

        let (mean, x1, k_white, xc) = if whiten {
            let mean = x.mean_axis(Axis(0)).unwrap();
            let xc = x - &mean;
            // u (d × r) and d (r) of XT = (d × n), sorted descending.
            let (u, dvals): (Array2<f64>, Vec<f64>) = match self.whiten_solver {
                IcaWhitenSolver::Svd => {
                    let svd = Svd::new_thin(to_mat(&xc.t().to_owned()).as_ref()).map_err(|e| {
                        RustMlError::InvalidParameter(format!("FastICA SVD failed: {e:?}"))
                    })?;
                    let s = svd.S().column_vector();
                    let r = s.nrows();
                    let mut order: Vec<usize> = (0..r).collect();
                    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
                    let uu = svd.U();
                    (
                        Array2::from_shape_fn((d, r), |(i, c)| uu[(i, order[c])]),
                        order.iter().map(|&c| s[c]).collect(),
                    )
                }
                IcaWhitenSolver::Eigh => {
                    let cov = xc.t().dot(&xc);
                    let eig =
                        SelfAdjointEigen::new(to_mat(&cov).as_ref(), Side::Lower).map_err(|e| {
                            RustMlError::InvalidParameter(format!("FastICA eigh failed: {e:?}"))
                        })?;
                    let s = eig.S().column_vector();
                    let uu = eig.U();
                    let eps = f64::EPSILON * 10.0;
                    let mut order: Vec<usize> = (0..d).collect();
                    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
                    (
                        Array2::from_shape_fn((d, d), |(i, c)| uu[(i, order[c])]),
                        order.iter().map(|&c| s[c].max(eps).sqrt()).collect(),
                    )
                }
            };
            // u *= sign(u[0]); K = (u / d).T[:k]
            let mut kw = Array2::<f64>::zeros((k, d));
            for c in 0..k {
                let sign = if u[[0, c]] < 0.0 { -1.0 } else { 1.0 };
                for j in 0..d {
                    kw[[c, j]] = sign * u[[j, c]] / dvals[c];
                }
            }
            let x1 = kw.dot(&xc.t()) * (n as f64).sqrt();
            (mean, x1, Some(kw), xc)
        } else {
            (Array1::zeros(d), x.t().to_owned(), None, x.to_owned())
        };

        let w_init = match &self.w_init {
            Some(w) if w.dim() != (k, k) => {
                return Err(RustMlError::ShapeMismatch(format!(
                    "w_init has shape {:?}, expected ({k}, {k})",
                    w.dim()
                )))
            }
            Some(w) => w.clone(),
            None => {
                let mut rng = StdRng::seed_from_u64(self.seed);
                Array2::from_shape_fn((k, k), |_| normal(&mut rng))
            }
        };
        let (mut w, n_iter) = match self.algorithm {
            IcaAlgorithm::Parallel => ica_par(&x1, self.tol, self.fun, self.max_iter, &w_init)?,
            IcaAlgorithm::Deflation => ica_def(&x1, self.tol, self.fun, self.max_iter, &w_init),
        };

        let components = match &k_white {
            Some(kw) => {
                if self.whiten == IcaWhiten::UnitVariance {
                    // S = (W K XcT)ᵀ; W /= std(S, axis=0)ᵀ
                    let s = xc.dot(&kw.t()).dot(&w.t());
                    let std = s.std_axis(Axis(0), 0.0);
                    for (mut row, &sd) in w.rows_mut().into_iter().zip(std.iter()) {
                        row /= sd;
                    }
                }
                w.dot(kw)
            }
            None => w,
        };
        let mixing = pinv(&components)?;
        Ok(FittedFastIca {
            components,
            mixing,
            mean,
            whitening: k_white,
            n_iter,
            n_features: d,
        })
    }
}

impl Transform<f64> for FittedFastIca {
    /// Recovered sources `S = (X − mean) componentsᵀ`.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok((x - &self.mean).dot(&self.components.t()))
    }
}

impl anofox_ml_core::InverseTransform<f64> for FittedFastIca {
    /// `S mixingᵀ + mean` (sklearn `inverse_transform`).
    fn inverse_transform(&self, s: &Array2<f64>) -> Result<Array2<f64>> {
        if s.ncols() != self.mixing.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} components, got {}",
                self.mixing.ncols(),
                s.ncols()
            )));
        }
        Ok(s.dot(&self.mixing.t()) + &self.mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fast_ica_runs() {
        // Build a mixture of two simple signals; FastICA should separate them.
        let n = 100;
        let mut s = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 * 0.1;
            s[[i, 0]] = t.sin(); // sine
            s[[i, 1]] = (t * 3.0).sin().signum(); // square
        }
        // Mixing matrix.
        let a = array![[1.0_f64, 0.5], [0.5, 1.0]];
        let x = s.dot(&a);
        let fitted = FastIca::new(2).with_seed(1).fit(&x).unwrap();
        let recovered = fitted.transform(&x).unwrap();
        assert_eq!(recovered.shape(), &[n, 2]);
        // Unit-variance sources.
        for c in 0..2 {
            let sd = recovered.column(c).std(0.0);
            assert!((sd - 1.0).abs() < 1e-10, "{sd}");
        }
    }

    #[test]
    fn test_matches_sklearn_given_w_init() {
        // sklearn 1.8 FastICA(w_init=W0, tol=1e-10, max_iter=1000, ...)
        //   .fit(X).components_ for each variant.
        let s = Array2::from_shape_fn((200, 3), |(i, j)| {
            let t = i as f64;
            match j {
                0 => (t * 0.21).sin(),
                // numpy's sign: 0 at 0.
                1 => {
                    let v = (t * 0.057).sin();
                    if v == 0.0 {
                        0.0
                    } else {
                        v.signum()
                    }
                }
                _ => ((t * 0.37) % 2.0) - 1.0,
            }
        });
        let a = array![[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]];
        let x = s.dot(&a.t());
        let w0 = array![[0.3, -1.2, 0.5], [1.1, 0.4, -0.7], [-0.2, 0.9, 1.3]];
        let base = || {
            FastIca::new(3)
                .with_tol(1e-10)
                .with_max_iter(1000)
                .with_w_init(w0.clone())
        };
        let cases: Vec<(FastIca, Vec<f64>)> = vec![
            (
                base(),
                vec![
                    0.405_741_534_297_044_57,
                    0.534_387_429_458_471_7,
                    -0.466_358_788_344_762_9,
                    -4.245_520_178_389_894,
                    0.832_715_873_894_491_8,
                    2.572_716_875_602_366,
                    4.255_992_056_405_312,
                    -1.417_855_667_778_510_1,
                    -1.414_564_035_979_856,
                ],
            ),
            (
                base().with_algorithm(IcaAlgorithm::Deflation),
                vec![
                    0.497_289_152_233_148_2,
                    0.505_353_774_690_250_2,
                    -0.499_893_555_588_592_4,
                    -4.261_556_778_150_989_5,
                    0.839_942_106_446_280_5,
                    2.577_103_717_713_523,
                    4.230_174_167_527_997,
                    -1.424_224_408_678_507_6,
                    -1.394_989_933_508_964_6,
                ],
            ),
            (
                base().with_fun(IcaFun::Exp),
                vec![
                    0.406_068_572_217_043_9,
                    0.534_286_967_818_386_6,
                    -0.466_484_441_404_887_6,
                    -4.244_736_172_588_157,
                    0.832_464_463_100_765_7,
                    2.572_451_440_962_769_5,
                    4.256_742_804_273_456,
                    -1.418_041_148_819_219_7,
                    -1.415_005_266_131_174_3,
                ],
            ),
            (
                base().with_fun(IcaFun::Cube),
                vec![
                    0.394_330_868_383_117_65,
                    0.537_380_996_886_119_4,
                    -0.460_954_715_727_884_16,
                    -4.241_913_058_622_797_5,
                    0.830_587_353_196_197_9,
                    2.571_975_601_808_323_7,
                    4.260_658_743_818_869,
                    -1.417_972_728_562_492_1,
                    -1.417_679_902_909_731_8,
                ],
            ),
            (
                base().with_whiten_solver(IcaWhitenSolver::Eigh),
                vec![
                    0.405_741_534_297_019,
                    0.534_387_429_458_478_6,
                    -0.466_358_788_344_751,
                    -4.245_520_178_389_931,
                    0.832_715_873_894_507_5,
                    2.572_716_875_602_377,
                    4.255_992_056_405_35,
                    -1.417_855_667_778_519,
                    -1.414_564_035_979_877_7,
                ],
            ),
            (
                base().with_whiten(IcaWhiten::ArbitraryVariance),
                vec![
                    0.028_690_259_031_047_435,
                    0.037_786_897_515_093_315,
                    -0.032_976_546_170_452_36,
                    -0.300_203_610_780_380_8,
                    0.058_881_904_123_247_585,
                    0.181_918_554_881_149_64,
                    0.300_944_084_376_028_2,
                    -0.100_257_535_742_996_74,
                    -0.100_024_782_226_396_96,
                ],
            ),
            (
                FastIca::new(2)
                    .with_tol(1e-10)
                    .with_max_iter(1000)
                    .with_w_init(w0.slice(ndarray::s![..2, ..2]).to_owned()),
                vec![
                    0.027_433_055_798_458_856,
                    0.628_793_190_348_455_8,
                    -0.279_504_793_106_626_1,
                    0.221_790_136_411_343_2,
                    -0.475_686_476_183_631_36,
                    0.720_168_684_827_901_1,
                ],
            ),
        ];
        for (i, (ica, want)) in cases.into_iter().enumerate() {
            let f = ica.fit(&x).unwrap();
            for (got, w) in f.components.iter().zip(&want) {
                assert!((got - w).abs() < 1e-8, "case {i}: {got} vs {w}");
            }
            // mixing is the pseudo-inverse of components.
            let eye = f.components.dot(&f.mixing);
            for r in 0..eye.nrows() {
                for c in 0..eye.ncols() {
                    let e = if r == c { 1.0 } else { 0.0 };
                    assert!((eye[[r, c]] - e).abs() < 1e-9);
                }
            }
        }
    }
}
