//! Partial Least Squares Regression (PLS1).
//!
//! Mirrors `sklearn.cross_decomposition.PLSRegression` for a 1-D target.
//! The fit is delegated to `anofox_regression::solvers::PlsRegressor`
//! (SIMPLS, which for a single response spans exactly the same score space
//! as sklearn's NIPALS and gives identical coefficients). Upstream normalises
//! the scores to unit length; this wrapper rescales them (and the rotations)
//! to sklearn's convention — unit-norm `x_weights_` with the largest entry
//! positive (`_svd_flip_1d`) and `x_scores_ = X_k w_k` — so `transform`,
//! `x_scores`, `x_weights`, `x_loadings` and `x_rotations` match sklearn.
//! 2-D `y` (PLS2) is not supported.

use anofox_ml_core::{Fit, Predict, Result, RustMlError, Transform};
use anofox_regression::solvers::{FittedRegressor as _, PlsRegressor, Regressor as _};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlsRegression {
    pub n_components: usize,
    /// Kept for sklearn API compatibility; PLS1 needs no inner iterations.
    pub max_iter: usize,
    /// Kept for sklearn API compatibility; PLS1 needs no inner iterations.
    pub tol: f64,
    /// Scale `X` (and `y`) to unit variance before fitting (sklearn
    /// `scale`, default `true`).
    pub scale: bool,
}

impl PlsRegression {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 500,
            tol: 1e-6,
            scale: true,
        }
    }

    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedPlsRegression {
    pub x_mean: Array1<f64>,
    pub y_mean: f64,
    /// Column scales of `X` (sample std, `ddof = 1`; 1 when `scale = false`).
    pub x_std: Array1<f64>,
    /// Scale of `y` (sample std, `ddof = 1`; 1 when `scale = false`).
    pub y_std: f64,
    /// Regression coefficients on the original feature scale (sklearn
    /// `coef_`).
    pub coef: Array1<f64>,
    /// Intercept for raw `X`: `ȳ − x̄ᵀcoef` (sklearn's `intercept_` is `ȳ`
    /// applied to centred `X`; predictions are identical).
    #[serde(default)]
    pub intercept: f64,
    /// Unit-norm X weights `W`, `n_features × n_components`.
    #[serde(default)]
    pub x_weights: Array2<f64>,
    /// X loadings `P`, `n_features × n_components`.
    #[serde(default)]
    pub x_loadings: Array2<f64>,
    /// Rotations `R = W (PᵀW)⁻¹` mapping scaled X to scores.
    #[serde(default)]
    pub x_rotations: Array2<f64>,
    /// Training scores `T`, `n_samples × n_components`.
    #[serde(default)]
    pub x_scores: Array2<f64>,
    /// y loadings `q` (on the scaled target), one per component.
    #[serde(default)]
    pub y_loadings: Array1<f64>,
    n_features: usize,
}

fn sample_std(v: ndarray::ArrayView1<f64>, mean: f64) -> f64 {
    let n = v.len();
    if n < 2 {
        return 1.0;
    }
    let s = (v.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    if s > 0.0 {
        s
    } else {
        1.0
    }
}

impl Fit<f64> for PlsRegression {
    type Fitted = FittedPlsRegression;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        let (n, d) = x.dim();
        if self.n_components == 0 || self.n_components > d.min(n) {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components must be in 1..={}",
                d.min(n)
            )));
        }
        let k = self.n_components;
        let xm = faer::Mat::from_fn(n, d, |i, j| x[[i, j]]);
        let ym = faer::Col::from_fn(n, |i| y[i]);
        let fitted = PlsRegressor::builder()
            .n_components(k)
            .scale(self.scale)
            .build()
            .fit(&xm, &ym)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;
        let res = fitted.result();
        let coef = Array1::from_shape_fn(d, |j| res.coefficients[j]);
        let intercept = res.intercept.unwrap_or(0.0);

        // sklearn's preprocessing: centre, scale by the sample std.
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean().unwrap();
        let (x_std, y_std) = if self.scale {
            (
                Array1::from_shape_fn(d, |j| sample_std(x.column(j), x_mean[j])),
                sample_std(y.view(), y_mean),
            )
        } else {
            (Array1::ones(d), 1.0)
        };
        let xs = (&x - &x_mean) / &x_std;
        let ys = (&y - y_mean) / y_std;

        // Upstream's unit-norm scores T̃ = Xs R̃.
        let t_up = fitted.scores();
        let r_up = fitted.weights();
        let ncomp = t_up.ncols().min(k);
        let mut x_weights = Array2::<f64>::zeros((d, k));
        let mut x_scores = Array2::<f64>::zeros((n, k));
        let mut x_rotations = Array2::<f64>::zeros((d, k));
        let mut x_loadings = Array2::<f64>::zeros((d, k));
        let mut y_loadings = Array1::<f64>::zeros(k);
        let mut y_res = ys.clone();
        for a in 0..ncomp {
            let t_a = Array1::from_shape_fn(n, |i| t_up[(i, a)]);
            // NIPALS weight on the deflated data: Xᵀ P⊥ y (P⊥ projects out the
            // previous scores), normalised, largest entry positive.
            let mut w = xs.t().dot(&y_res);
            let norm = w.dot(&w).sqrt();
            if norm > 0.0 {
                w /= norm;
            }
            let imax = (0..d).fold(0, |b, j| if w[j].abs() > w[b].abs() { j } else { b });
            if w[imax] < 0.0 {
                w.mapv_inplace(|v| -v);
            }
            // sklearn score t_a = c · t̃_a with c = t̃_aᵀ Xs w.
            let c = t_a.dot(&xs.dot(&w));
            let t = &t_a * c;
            let tt = t.dot(&t);
            x_weights.column_mut(a).assign(&w);
            x_scores.column_mut(a).assign(&t);
            for j in 0..d {
                x_rotations[[j, a]] = r_up[(j, a)] * c;
            }
            if tt > 0.0 {
                x_loadings.column_mut(a).assign(&(xs.t().dot(&t) / tt));
                y_loadings[a] = ys.dot(&t) / tt;
            }
            let proj = t_a.dot(&y_res);
            y_res.scaled_add(-proj, &t_a);
        }

        Ok(FittedPlsRegression {
            x_mean,
            y_mean,
            x_std,
            y_std,
            coef,
            intercept,
            x_weights,
            x_loadings,
            x_rotations,
            x_scores,
            y_loadings,
            n_features: d,
        })
    }
}

impl FittedPlsRegression {
    fn check(&self, x: ArrayView2<f64>) -> Result<()> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(())
    }
}

impl Predict<f64> for FittedPlsRegression {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.check(x)?;
        Ok(x.dot(&self.coef) + self.intercept)
    }
}

impl Transform<f64> for FittedPlsRegression {
    /// X scores of new rows (sklearn `transform`): `((X − mean) / std) · R`.
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.check(x)?;
        if self.x_rotations.nrows() != self.n_features {
            return Err(RustMlError::NotFitted(
                "rotations not available (model deserialized from an older version)".into(),
            ));
        }
        Ok(((&x - &self.x_mean) / &self.x_std).dot(&self.x_rotations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pls1_recovers_linear() {
        let rng_x: Vec<f64> = (0..40)
            .flat_map(|i| {
                let i = i as f64;
                vec![i, 0.5 * i, -0.3 * i + 1.0]
            })
            .collect();
        let x = Array2::from_shape_vec((40, 3), rng_x).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v) + x.column(1).mapv(|v| 1.5 * v);
        let fitted = PlsRegression::new(2).fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let rss: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (t - p).powi(2))
            .sum();
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let tss: f64 = y.iter().map(|t| (t - mean).powi(2)).sum();
        let r2 = 1.0 - rss / tss;
        assert!(r2 > 0.99, "R² too low: {r2}");
    }

    /// `data.make(20, 4, m=7, coefs=[1.5, -2, 0.5, 0])` with column 3
    /// rescaled to `3·x + 1`, from the sklearn reference script.
    fn make() -> (Array2<f64>, Array1<f64>) {
        let mut x = Array2::from_shape_fn((20, 4), |(i, j)| {
            ((i * (j + 3) * 7 + j * 5 + (i * i) % 11) % 17) as f64 / 4.0 - 2.0
        });
        let coefs = [1.5, -2.0, 0.5, 0.0];
        let y = Array1::from_shape_fn(20, |i| {
            (0..4).map(|j| x[[i, j]] * coefs[j]).sum::<f64>()
                + 1.0
                + 0.3 * (((i * 13) % 7) as f64 - 3.0)
        });
        x.column_mut(3).mapv_inplace(|v| 3.0 * v + 1.0);
        (x, y)
    }

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (u, v)) in a.iter().zip(b).enumerate() {
            assert!((u - v).abs() < tol, "[{i}] {u} vs {v}");
        }
    }

    fn flat(a: &Array2<f64>) -> Vec<f64> {
        a.iter().copied().collect()
    }

    #[test]
    fn test_pls_matches_sklearn_coef_scores_and_transform() {
        // sklearn PLSRegression(n_components=2).fit(X, y)
        let (x, y) = make();
        let f = PlsRegression::new(2).fit(&x, &y).unwrap();
        close(
            f.coef.as_slice().unwrap(),
            &[
                1.4599946821642962,
                -2.0822793117914102,
                0.6040695435258747,
                -0.03896190698218954,
            ],
            1e-10,
        );
        // sklearn's `intercept_` (1.43) is ȳ and applies to centred X; ours
        // applies to raw X.
        assert!((f.y_mean - 1.4300000000000002).abs() < 1e-10);
        assert!((f.intercept - (f.y_mean - f.x_mean.dot(&f.coef))).abs() < 1e-10);
        let x3 = x.slice(ndarray::s![..3, ..]).to_owned();
        close(
            f.predict(&x3).unwrap().as_slice().unwrap(),
            &[-0.2872696905827816, 4.676379015517926, 0.7227902429812367],
            1e-10,
        );
        let sk_scores = [
            0.6253896504493411,
            1.4565071093203832,
            -0.9133442978848508,
            0.45495023776846394,
            0.014619405509260108,
            -2.298569662265971,
        ];
        close(
            &flat(&f.x_scores.slice(ndarray::s![..3, ..]).to_owned()),
            &sk_scores,
            1e-10,
        );
        close(&flat(&f.transform(&x3).unwrap()), &sk_scores, 1e-10);
        close(
            &flat(&f.x_weights),
            &[
                -0.5521329389287771,
                -0.199984879623659,
                0.7845607226585466,
                -0.41008597386404966,
                -0.26349439046346124,
                -0.5275193217065876,
                0.10091777051470686,
                0.7166302443995133,
            ],
            1e-10,
        );
        close(
            &flat(&f.x_loadings),
            &[
                -0.577963698568099,
                -0.271366832484963,
                0.7315925570538593,
                -0.4650590718932853,
                -0.3316306657145875,
                -0.5745700249724985,
                0.19348028638184328,
                0.6306177390018194,
            ],
            1e-10,
        );
        close(
            &flat(&f.x_rotations),
            &[
                -0.552132938928777,
                -0.12866942186875008,
                0.7845607226585465,
                -0.511422632353345,
                -0.2634943904634612,
                -0.49348544734978805,
                0.10091777051470684,
                0.7035953455691257,
            ],
            1e-10,
        );
    }

    #[test]
    fn test_pls_without_scaling_matches_sklearn() {
        let (x, y) = make();
        let f = PlsRegression::new(2).with_scale(false).fit(&x, &y).unwrap();
        close(
            f.coef.as_slice().unwrap(),
            &[
                1.382531603762664,
                -2.130422659121052,
                0.5922827195957387,
                -0.035077155193356635,
            ],
            1e-10,
        );
        let x3 = x.slice(ndarray::s![..3, ..]).to_owned();
        close(
            &flat(&f.transform(&x3).unwrap()),
            &[
                2.153876970115691,
                2.963248702878137,
                -1.2078175848389325,
                2.088617364808865,
                -0.8404541661541988,
                -3.3575806960462664,
            ],
            1e-10,
        );
    }
}
