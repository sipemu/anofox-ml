//! Incremental principal component analysis.
//!
//! Mirrors `sklearn.decomposition.IncrementalPCA` (sklearn 1.8): each batch is
//! centred on its own mean and stacked below the current components (scaled
//! by their singular values) and a mean-correction row; the thin SVD of that
//! stack gives the updated components (Ross et al., 2008). `fit` feeds the
//! data in batches of `batch_size` rows (default `5 · n_features`), so memory
//! stays `O(batch_size · n_features)`.

use anofox_ml_core::{
    FitUnsupervised, Float, InverseTransform, PartialFitUnsupervised, Result, RustMlError,
    Transform,
};
use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::{s, Array1, Array2, Axis};

use crate::pca::{svd_flip_rows, FittedPca};

/// Parameters for incremental PCA (unfitted state).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IncrementalPca {
    /// Number of components to keep. `None` keeps `min(n_samples, n_features)`
    /// of the first batch.
    pub n_components: Option<usize>,
    /// Divide the projections by `sqrt(explained_variance)` (sklearn's
    /// `whiten`).
    pub whiten: bool,
    /// Rows per batch in `fit`; `None` uses `5 · n_features`.
    pub batch_size: Option<usize>,
}

impl IncrementalPca {
    /// Keep `n_components` components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: Some(n_components),
            ..Self::default()
        }
    }

    /// Enable / disable whitening.
    pub fn with_whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    /// Rows per batch used by `fit`.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }
}

/// Fitted incremental PCA: the components plus the running statistics that
/// later `partial_fit` calls update.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedIncrementalPca<F: Float> {
    pca: FittedPca<F>,
    /// Per-feature population variance (`var_`).
    var: Array1<F>,
    /// Number of samples seen so far (`n_samples_seen_`).
    n_samples_seen: usize,
}

/// `sklearn.utils.gen_batches(n, batch_size, min_batch_size)`: consecutive
/// slices of `batch_size` rows; a trailing remainder shorter than
/// `min_batch_size` is merged into the last full batch.
fn gen_batches(n: usize, batch_size: usize, min_batch_size: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    for _ in 0..n / batch_size {
        let end = start + batch_size;
        if end + min_batch_size > n {
            continue;
        }
        out.push((start, end));
        start = end;
    }
    if start < n {
        out.push((start, n));
    }
    out
}

impl<F: Float> FitUnsupervised<F> for IncrementalPca {
    type Fitted = FittedIncrementalPca<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let (n, d) = x.dim();
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        let batch_size = self.batch_size.unwrap_or(5 * d);
        if batch_size == 0 {
            return Err(RustMlError::InvalidParameter(
                "batch_size must be at least 1".into(),
            ));
        }
        let mut state = None;
        for (a, b) in gen_batches(n, batch_size, self.n_components.unwrap_or(0)) {
            state = Some(self.partial_fit(state, &x.slice(s![a..b, ..]).to_owned())?);
        }
        state.ok_or_else(|| RustMlError::EmptyInput("input array is empty".into()))
    }
}

impl<F: Float> PartialFitUnsupervised<F> for IncrementalPca {
    type Fitted = FittedIncrementalPca<F>;

    fn partial_fit(&self, state: Option<Self::Fitted>, x: &Array2<F>) -> Result<Self::Fitted> {
        let (n, d) = x.dim();
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if let Some(prev) = &state {
            if prev.pca.mean.len() != d {
                return Err(RustMlError::ShapeMismatch(format!(
                    "expected {} features, got {d}",
                    prev.pca.mean.len()
                )));
            }
        }
        let k = match (self.n_components, &state) {
            (Some(0), _) => {
                return Err(RustMlError::InvalidParameter(
                    "n_components must be at least 1".into(),
                ))
            }
            (Some(k), _) if k > d => {
                return Err(RustMlError::InvalidParameter(format!(
                    "n_components={k} must be <= n_features={d}"
                )))
            }
            (Some(k), _) if k > n => {
                return Err(RustMlError::InvalidParameter(format!(
                    "n_components={k} must be <= the number of samples in the batch ({n})"
                )))
            }
            (Some(k), _) => k,
            (None, Some(prev)) => prev.pca.components.nrows(),
            (None, None) => n.min(d),
        };

        let xf: Array2<f64> = x.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
        if xf.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "IncrementalPca input contains NaN or infinite values".into(),
            ));
        }
        let to64 = |a: &Array1<F>| a.mapv(|v| v.to_f64().unwrap());

        // Running mean / variance (`_incremental_mean_and_var`).
        let n_seen = state.as_ref().map_or(0, |p| p.n_samples_seen);
        let batch_mean = xf.mean_axis(Axis(0)).expect("n > 0");
        let xc = &xf - &batch_mean;
        let batch_unnorm = {
            // Two-pass variance with sklearn's correction term.
            let corr = xc.sum_axis(Axis(0));
            (&xc * &xc).sum_axis(Axis(0)) - &(&corr * &corr / n as f64)
        };
        let n_total = n_seen + n;
        let (col_mean, col_var) = match &state {
            None => (batch_mean.clone(), batch_unnorm.clone() / n as f64),
            Some(prev) => {
                let last_mean = to64(&prev.pca.mean);
                let last_var = to64(&prev.var);
                let (ls, nn, nt) = (n_seen as f64, n as f64, n_total as f64);
                let last_sum = &last_mean * ls;
                let new_sum = &batch_mean * nn;
                let last_over_new = ls / nn;
                let d2 = (&last_sum / last_over_new - &new_sum).mapv(|v| v * v);
                let unnorm = &last_var * ls + &batch_unnorm + &(d2 * (last_over_new / nt));
                ((last_sum + new_sum) / nt, unnorm / nt)
            }
        };

        // Matrix to decompose.
        let stacked = match &state {
            None => &xf - &col_mean,
            Some(prev) => {
                let sv = to64(&prev.pca.singular_values);
                let comps = prev.pca.components.mapv(|v| v.to_f64().unwrap());
                let kp = comps.nrows();
                let corr_scale = ((n_seen as f64 / n_total as f64) * n as f64).sqrt();
                let mean_corr = (to64(&prev.pca.mean) - &batch_mean) * corr_scale;
                let mut m = Array2::<f64>::zeros((kp + n + 1, d));
                for c in 0..kp {
                    m.row_mut(c).assign(&(&comps.row(c) * sv[c]));
                }
                m.slice_mut(s![kp..kp + n, ..]).assign(&xc);
                m.row_mut(kp + n).assign(&mean_corr);
                m
            }
        };

        let (rows, cols) = stacked.dim();
        let mat = Mat::<f64>::from_fn(rows, cols, |i, j| stacked[[i, j]]);
        let svd = Svd::new_thin(mat.as_ref()).map_err(|e| {
            RustMlError::InvalidParameter(format!("IncrementalPca SVD failed: {e:?}"))
        })?;
        let sv = svd.S().column_vector();
        let v = svd.V();
        let r = sv.nrows();
        let mut order: Vec<usize> = (0..r).collect();
        order.sort_by(|&a, &b| sv[b].total_cmp(&sv[a]));
        let mut vt = Array2::from_shape_fn((r, d), |(c, j)| v[(j, order[c])]);
        svd_flip_rows(&mut vt);
        let svals: Vec<f64> = order.iter().map(|&i| sv[i]).collect();

        let dof = (n_total as f64 - 1.0).max(0.0);
        let total = col_var.sum() * n_total as f64;
        let explained: Vec<f64> = svals.iter().map(|&s| s * s / dof).collect();
        let noise_variance = if k != n && k != d && explained.len() > k {
            explained[k..].iter().sum::<f64>() / (explained.len() - k) as f64
        } else {
            0.0
        };

        let from = |v: f64| F::from_f64(v).unwrap();
        Ok(FittedIncrementalPca {
            pca: FittedPca {
                components: vt.slice(s![..k, ..]).mapv(from),
                explained_variance: explained[..k].iter().map(|&e| from(e)).collect(),
                explained_variance_ratio: svals[..k]
                    .iter()
                    .map(|&s| from(if total > 0.0 { s * s / total } else { 0.0 }))
                    .collect(),
                singular_values: svals[..k].iter().map(|&s| from(s)).collect(),
                noise_variance: from(noise_variance),
                mean: col_mean.mapv(from),
                whiten: self.whiten,
            },
            var: col_var.mapv(from),
            n_samples_seen: n_total,
        })
    }
}

impl<F: Float> Transform<F> for FittedIncrementalPca<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.pca.transform(x)
    }
}

impl<F: Float> InverseTransform<F> for FittedIncrementalPca<F> {
    fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.pca.inverse_transform(x)
    }
}

impl<F: Float> FittedIncrementalPca<F> {
    /// Principal axes, shape `(n_components, n_features)` (`components_`).
    pub fn components(&self) -> &Array2<F> {
        self.pca.components()
    }

    /// Variance explained by each component (`explained_variance_`).
    pub fn explained_variance(&self) -> &Array1<F> {
        self.pca.explained_variance()
    }

    /// Fraction of the total variance explained by each component.
    pub fn explained_variance_ratio(&self) -> &Array1<F> {
        self.pca.explained_variance_ratio()
    }

    /// Singular values of the (implicitly) centred data seen so far.
    pub fn singular_values(&self) -> &Array1<F> {
        self.pca.singular_values()
    }

    /// Probabilistic-PCA noise variance (`noise_variance_`).
    pub fn noise_variance(&self) -> F {
        self.pca.noise_variance()
    }

    /// Running per-feature mean (`mean_`).
    pub fn mean(&self) -> &Array1<F> {
        self.pca.mean()
    }

    /// Running per-feature population variance (`var_`).
    pub fn var(&self) -> &Array1<F> {
        &self.var
    }

    /// Number of samples seen so far (`n_samples_seen_`).
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn data() -> Array2<f64> {
        // Deterministic correlated data, 23 × 4.
        Array2::from_shape_fn((23, 4), |(i, j)| {
            let t = i as f64;
            match j {
                0 => (t * 0.7).sin() * 3.0 + t * 0.1,
                1 => (t * 0.3).cos() * 2.0 - t * 0.05,
                2 => (t * 1.3).sin() + 0.5 * (t * 0.7).sin(),
                _ => ((t * 0.37).sin() * 5.0).round() * 0.2,
            }
        })
    }

    #[test]
    fn test_gen_batches_matches_sklearn() {
        // sklearn: list(gen_batches(7, 3, min_batch_size=2)) -> [0:3, 3:7]
        assert_eq!(gen_batches(7, 3, 2), vec![(0, 3), (3, 7)]);
        assert_eq!(gen_batches(7, 3, 0), vec![(0, 3), (3, 6), (6, 7)]);
        assert_eq!(gen_batches(6, 3, 0), vec![(0, 3), (3, 6)]);
    }

    #[test]
    fn test_matches_sklearn() {
        // sklearn 1.8: IncrementalPCA(n_components=2, batch_size=6).fit(X)
        let x = data();
        let f =
            FitUnsupervised::<f64>::fit(&IncrementalPca::new(2).with_batch_size(6), &x).unwrap();
        let comps = [
            [
                0.952_335_139_949_796_1,
                0.243_871_075_430_882_32,
                0.176_052_471_726_250_94,
                0.050_894_076_119_572_37,
            ],
            [
                -0.255_000_330_545_878_6,
                0.899_952_349_077_414_6,
                0.030_923_953_077_662_49,
                0.352_284_416_257_071_63,
            ],
        ];
        for c in 0..2 {
            for j in 0..4 {
                assert_abs_diff_eq!(f.components()[[c, j]], comps[c][j], epsilon = 1e-10);
            }
        }
        let ev = [5.266_697_731_393_716, 2.342_737_849_670_281];
        let evr = [0.632_275_925_353_316_1, 0.281_249_621_167_936_84];
        let sv = [10.764_169_735_314_551, 7.179_152_644_480_14];
        for c in 0..2 {
            assert_abs_diff_eq!(f.explained_variance()[c], ev[c], epsilon = 1e-10);
            assert_abs_diff_eq!(f.explained_variance_ratio()[c], evr[c], epsilon = 1e-10);
            assert_abs_diff_eq!(f.singular_values()[c], sv[c], epsilon = 1e-10);
        }
        assert_abs_diff_eq!(
            f.noise_variance(),
            0.087_756_666_904_366_65,
            epsilon = 1e-10
        );
        assert_eq!(f.n_samples_seen(), 23);
        let t = f.transform(&x).unwrap();
        assert_abs_diff_eq!(t[[0, 0]], -0.847_156_133_505_803_9, epsilon = 1e-9);
        assert_abs_diff_eq!(t[[22, 1]], 0.925_484_611_343_160_2, epsilon = 1e-9);
        // whiten=True
        let w = IncrementalPca::new(2).with_batch_size(6).with_whiten(true);
        let t = FitUnsupervised::<f64>::fit(&w, &x)
            .unwrap()
            .transform(&x)
            .unwrap();
        assert_abs_diff_eq!(t[[3, 0]], 0.731_708_230_927_097_5, epsilon = 1e-9);
        assert_abs_diff_eq!(t[[3, 1]], 0.765_587_747_879_665_8, epsilon = 1e-9);
    }

    #[test]
    fn test_one_batch_equals_pca() {
        // With a single batch the result is exact PCA.
        let x = data();
        let inc =
            FitUnsupervised::<f64>::fit(&IncrementalPca::new(3).with_batch_size(100), &x).unwrap();
        let pca = FitUnsupervised::<f64>::fit(&crate::Pca::new(3), &x).unwrap();
        for (a, b) in inc.components().iter().zip(pca.components().iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-10);
        }
        for (a, b) in inc
            .explained_variance()
            .iter()
            .zip(pca.explained_variance())
        {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_partial_fit_equals_fit_and_keeps_all_components() {
        // Without truncation the incremental update is exact, so the
        // running statistics must match PCA on all data.
        let x = data();
        let ipca = IncrementalPca::default();
        let mut st = None;
        for (a, b) in [(0, 8), (8, 15), (15, 23)] {
            st = Some(
                ipca.partial_fit(st, &x.slice(s![a..b, ..]).to_owned())
                    .unwrap(),
            );
        }
        let f: FittedIncrementalPca<f64> = st.unwrap();
        assert_eq!(f.components().nrows(), 4);
        let pca = FitUnsupervised::<f64>::fit(&crate::Pca::new(4), &x).unwrap();
        for (a, b) in f.explained_variance().iter().zip(pca.explained_variance()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-9);
        }
        for (a, b) in f.mean().iter().zip(pca.mean()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
        let var = x.var_axis(Axis(0), 0.0);
        for (a, b) in f.var().iter().zip(var.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_errors() {
        let x = data();
        let r = IncrementalPca::new(5).partial_fit(None, &x);
        assert!(matches!(r, Err(RustMlError::InvalidParameter(_))));
        // n_components larger than a batch is an error, as in sklearn.
        let r: Result<FittedIncrementalPca<f64>> =
            IncrementalPca::new(3).partial_fit(None, &x.slice(s![..2, ..]).to_owned());
        assert!(r.is_err());
        let f: FittedIncrementalPca<f64> = IncrementalPca::new(2).partial_fit(None, &x).unwrap();
        assert!(IncrementalPca::new(2)
            .partial_fit(Some(f), &Array2::zeros((3, 3)))
            .is_err());
    }
}
