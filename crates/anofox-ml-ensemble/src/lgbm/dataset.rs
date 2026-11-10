//! Binned training data (#214): LightGBM's `Dataset`, built once from `f64`
//! or `f32` features and reusable across fits.

use super::*;

/// Feature values the engine bins: `f64` or `f32` (LightGBM's float32
/// input; values bin as their exact `f64`).
pub trait LgbmFloat: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
}

impl LgbmFloat for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

impl LgbmFloat for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

/// Rows LightGBM bins from (`bin_construct_sample_cnt`): above it, the bin
/// edges and each feature's most frequent bin come from a sample.
pub(super) const LGBM_BIN_SAMPLE: usize = 200_000;

/// The settings a dataset was binned with; a fit checks that they are its
/// own.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct BinningSettings {
    pub max_bins: usize,
    pub min_data_in_bin: usize,
    pub categorical_features: Vec<usize>,
    pub method: BinningMethod,
    pub seed: Option<u64>,
}

/// Features binned once (LightGBM's `Dataset`), to fit on with
/// [`LgbmRegressor::fit_dataset`], [`LgbmClassifier::fit_dataset`] or
/// [`LgbmRegressor::fit_custom_dataset`].
///
/// The raw matrix is no longer needed once the dataset exists, so a caller
/// can drop it before training: the dataset holds one byte per value. Build
/// it with the estimator that will fit on it ([`LgbmRegressor::dataset`],
/// [`LgbmClassifier::dataset`]), which fixes the binning settings
/// (`max_bins`, `min_data_in_bin`, categorical features, binning method,
/// seed). Above 200 000 rows the bin edges come from a sample of 200 000
/// rows drawn with LightGBM's generator, as LightGBM does
/// (`bin_construct_sample_cnt`).
///
/// ```
/// use anofox_ml_ensemble::LgbmRegressor;
/// use ndarray::{Array1, Array2};
///
/// let x = Array2::<f32>::from_shape_fn((200, 3), |(i, j)| ((i * (j + 2)) % 17) as f32);
/// let y = Array1::from_shape_fn(200, |i| (i % 5) as f64);
/// let model = LgbmRegressor::new().with_n_estimators(10);
/// let data = model.dataset(x.view()).unwrap();
/// drop(x); // only the binned values remain
/// let fitted = model.fit_dataset(&data, y.view(), &Default::default()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LgbmDataset {
    /// Column-major bin indices.
    pub(super) binned: Array2<u8>,
    pub(super) bins: Vec<FeatureBins>,
    /// Per feature: whether it has a missing value.
    pub(super) has_missing: Vec<bool>,
    /// LightGBM's `most_freq_bin` per feature (see `lgbm_fixed_bins`).
    pub(super) fixed_bins: Vec<Option<u8>>,
    pub(super) settings: BinningSettings,
}

impl LgbmDataset {
    pub fn n_rows(&self) -> usize {
        self.binned.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.binned.ncols()
    }

    /// The rows `rows` (in that order) with the same bins, e.g. a
    /// cross-validation fold.
    pub fn subset(&self, rows: &[usize]) -> Result<LgbmDataset> {
        let n = self.n_rows();
        if let Some(&bad) = rows.iter().find(|&&i| i >= n) {
            return Err(RustMlError::InvalidParameter(format!(
                "row {bad} out of range for {n} rows"
            )));
        }
        let mut binned = Array2::zeros((rows.len(), self.n_features()).f());
        for (j, mut col) in binned.columns_mut().into_iter().enumerate() {
            let src = self.binned.column(j);
            for (o, &i) in col.iter_mut().zip(rows) {
                *o = src[i];
            }
        }
        let has_missing = (0..self.n_features())
            .map(|j| binned.column(j).iter().any(|&b| b == MISSING_BIN) && self.has_missing[j])
            .collect();
        Ok(LgbmDataset {
            binned,
            bins: self.bins.clone(),
            has_missing,
            fixed_bins: self.fixed_bins.clone(),
            settings: self.settings.clone(),
        })
    }

    /// An error unless the dataset was binned with `settings`.
    pub(super) fn check_settings(&self, settings: &BinningSettings) -> Result<()> {
        if &self.settings != settings {
            return Err(RustMlError::InvalidParameter(format!(
                "the dataset was binned with other settings ({:?}) than this estimator's ({settings:?}); \
                 build it with this estimator's `dataset`",
                self.settings
            )));
        }
        Ok(())
    }

    /// Bytes held by the bin indices.
    pub fn binned_bytes(&self) -> usize {
        self.binned.len()
    }

    /// Bin `x` with `settings` (new bin edges) or with existing `bins`.
    pub(super) fn build<T: LgbmFloat>(
        x: ArrayView2<T>,
        settings: BinningSettings,
        given: Option<&[FeatureBins]>,
    ) -> LgbmDataset {
        let (n, p) = x.dim();
        let method = settings.method;
        // The rows that place the bin edges.
        let sample = if given.is_some() {
            None
        } else {
            sample_rows(n, &settings)
        };
        let cat_set: std::collections::HashSet<usize> =
            settings.categorical_features.iter().copied().collect();

        use rayon::prelude::*;
        let mut data = vec![0u8; n * p];
        let per_column: Vec<(FeatureBins, bool)> = data
            .par_chunks_mut(n.max(1))
            .take(p)
            .enumerate()
            .map(|(j, out)| {
                let col = x.column(j);
                let bins = match given {
                    Some(b) => b[j].clone(),
                    None => {
                        let is_categorical = cat_set.contains(&j);
                        // LightGBM bins every kind of feature from the
                        // sample; sklearn only its numeric thresholds.
                        let sampled = match &sample {
                            Some(rows) if method == BinningMethod::LightGbm || !is_categorical => {
                                rows.iter().map(|&i| col[i].to_f64()).collect()
                            }
                            _ => col.iter().map(|v| v.to_f64()).collect::<Vec<f64>>(),
                        };
                        column_bins(&sampled, is_categorical, &settings)
                    }
                };
                let mut has_missing = false;
                for (o, &v) in out.iter_mut().zip(col) {
                    let v = v.to_f64();
                    has_missing |= v.is_nan();
                    *o = bin_value(v, &bins);
                }
                (bins, has_missing)
            })
            .collect();
        let binned = Array2::from_shape_vec((n, p).f(), data).expect("column-major bins");
        let (bins, has_missing): (Vec<FeatureBins>, Vec<bool>) = per_column.into_iter().unzip();
        // LightGBM counts the most frequent bin on its sample.
        let fixed_bins = match &sample {
            Some(rows) if method == BinningMethod::LightGbm => {
                lgbm_fixed_bins_of(&binned, &bins, &has_missing, Some(rows))
            }
            _ => lgbm_fixed_bins_of(&binned, &bins, &has_missing, None),
        };
        LgbmDataset {
            binned,
            bins,
            has_missing,
            fixed_bins,
            settings,
        }
    }
}

/// The rows a dataset of `n` rows takes its bin edges from, when not all
/// (LightGBM's `CreateSampleIndices`, sklearn's `subsample`).
pub(super) fn sample_rows(n: usize, settings: &BinningSettings) -> Option<Vec<usize>> {
    match settings.method {
        BinningMethod::LightGbm if n > LGBM_BIN_SAMPLE => {
            let seed = LgbmSeeds::new(settings.seed, 0).data;
            Some(LgbmRandom(seed).sample(n, LGBM_BIN_SAMPLE))
        }
        BinningMethod::Sklearn if n > SKLEARN_BIN_SUBSAMPLE => {
            let mut rng = StdRng::seed_from_u64(settings.seed.unwrap_or(0));
            Some(rand::seq::index::sample(&mut rng, n, SKLEARN_BIN_SUBSAMPLE).into_vec())
        }
        _ => None,
    }
}

/// Builds an [`LgbmDataset`] from rows pushed in blocks, so the full raw
/// matrix never exists (LightGBM's `LGBM_DatasetCreateFromSampledColumn` +
/// `LGBM_DatasetPushRows`). Get the rows the bins come from with
/// `bin_sample_rows`, pass them to `dataset_builder`, then push every row
/// in order.
///
/// ```
/// use anofox_ml_ensemble::LgbmRegressor;
/// use ndarray::Array2;
///
/// let row = |i: usize| [(i % 13) as f32, (i % 7) as f32];
/// let n = 1000;
/// let model = LgbmRegressor::new();
/// let sample_rows = model.bin_sample_rows(n);
/// let sample = Array2::from_shape_fn((sample_rows.len(), 2), |(k, j)| row(sample_rows[k])[j]);
/// let mut builder = model.dataset_builder(&sample, n).unwrap();
/// for start in (0..n).step_by(256) {
///     let end = (start + 256).min(n);
///     let block = Array2::from_shape_fn((end - start, 2), |(k, j)| row(start + k)[j]);
///     builder.push_rows(&block).unwrap();
/// }
/// let data = builder.finish().unwrap();
/// assert_eq!(data.n_rows(), n);
/// ```
#[derive(Debug)]
pub struct LgbmDatasetBuilder {
    bins: Vec<FeatureBins>,
    settings: BinningSettings,
    /// The sample's bins (for the most frequent bin of each feature).
    sample_binned: Array2<u8>,
    /// Column-major bins of all rows.
    binned: Vec<u8>,
    n_rows: usize,
    filled: usize,
    has_missing: Vec<bool>,
}

impl LgbmDatasetBuilder {
    pub(super) fn new<T: LgbmFloat>(
        sample: ArrayView2<T>,
        n_rows: usize,
        settings: BinningSettings,
    ) -> Result<Self> {
        if settings.method != BinningMethod::LightGbm {
            return Err(RustMlError::InvalidParameter(
                "streamed datasets need LightGBM's binning".into(),
            ));
        }
        let expected = sample_rows(n_rows, &settings).map_or(n_rows, |r| r.len());
        if sample.nrows() != expected || n_rows == 0 || sample.ncols() == 0 {
            return Err(RustMlError::ShapeMismatch(format!(
                "the sample has {} rows; {n_rows} rows need the {expected} of `bin_sample_rows`",
                sample.nrows()
            )));
        }
        let sampled = LgbmDataset::build(sample, settings.clone(), None);
        // `build` would have sampled the sample itself.
        debug_assert!(expected <= LGBM_BIN_SAMPLE);
        let p = sample.ncols();
        Ok(Self {
            bins: sampled.bins,
            settings,
            sample_binned: sampled.binned,
            binned: vec![0u8; n_rows * p],
            n_rows,
            filled: 0,
            has_missing: vec![false; p],
        })
    }

    /// Bin the next rows (all features each).
    pub fn push_rows<'a, T: LgbmFloat>(&mut self, rows: impl AsArray<'a, T, Ix2>) -> Result<()> {
        let rows: ArrayView2<T> = rows.into();
        let p = self.bins.len();
        if rows.ncols() != p {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {p} features, got {}",
                rows.ncols()
            )));
        }
        if self.filled + rows.nrows() > self.n_rows {
            return Err(RustMlError::ShapeMismatch(format!(
                "more than the {} rows of the dataset",
                self.n_rows
            )));
        }
        let (n, start) = (self.n_rows, self.filled);
        use rayon::prelude::*;
        let missing: Vec<bool> = self
            .binned
            .par_chunks_mut(n)
            .zip(&self.bins)
            .enumerate()
            .map(|(j, (col, bins))| {
                let mut missing = false;
                for (o, &v) in col[start..start + rows.nrows()]
                    .iter_mut()
                    .zip(rows.column(j))
                {
                    let v = v.to_f64();
                    missing |= v.is_nan();
                    *o = bin_value(v, bins);
                }
                missing
            })
            .collect();
        for (h, m) in self.has_missing.iter_mut().zip(missing) {
            *h |= m;
        }
        self.filled += rows.nrows();
        Ok(())
    }

    /// The dataset, once every row was pushed.
    pub fn finish(self) -> Result<LgbmDataset> {
        if self.filled != self.n_rows {
            return Err(RustMlError::ShapeMismatch(format!(
                "{} of {} rows pushed",
                self.filled, self.n_rows
            )));
        }
        let p = self.bins.len();
        let binned =
            Array2::from_shape_vec((self.n_rows, p).f(), self.binned).expect("column-major bins");
        let fixed_bins =
            lgbm_fixed_bins_of(&self.sample_binned, &self.bins, &self.has_missing, None);
        Ok(LgbmDataset {
            binned,
            bins: self.bins,
            has_missing: self.has_missing,
            fixed_bins,
            settings: self.settings,
        })
    }
}

/// The bin edges of one column from its (sampled) values.
fn column_bins(values: &[f64], is_categorical: bool, s: &BinningSettings) -> FeatureBins {
    let col = ArrayView1::from(values);
    let edges = if is_categorical && s.method == BinningMethod::Sklearn {
        // sklearn ordinal-encodes every category (its estimators reject
        // more than `max_bins` of them).
        let mut edges: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
        edges.sort_by(f64::total_cmp);
        edges.dedup();
        edges.truncate(s.max_bins);
        edges
    } else if is_categorical {
        categorical_edges(col, s.max_bins, s.min_data_in_bin)
    } else if s.method == BinningMethod::Sklearn {
        let mut raw: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
        raw.sort_by(f64::total_cmp);
        if raw.is_empty() {
            Vec::new()
        } else {
            sklearn_thresholds(&raw, s.max_bins)
        }
    } else {
        lightgbm_bin_edges(col, s.max_bins, s.min_data_in_bin)
    };
    FeatureBins {
        edges,
        is_categorical,
    }
}

/// Bin every row of `x` with existing bin edges, row-major, for any float
/// input.
pub(super) fn bin_rows<T: LgbmFloat>(x: ArrayView2<T>, all_bins: &[FeatureBins]) -> Array2<u8> {
    use rayon::prelude::*;
    let (n, p) = x.dim();
    let mut out = Array2::<u8>::zeros((n, p));
    if p > 0 {
        out.as_slice_mut()
            .expect("row-major")
            .par_chunks_mut(PREDICT_BLOCK_ROWS * p)
            .enumerate()
            .for_each(|(block, out)| {
                let first = block * PREDICT_BLOCK_ROWS;
                for (k, row) in out.chunks_exact_mut(p).enumerate() {
                    for (j, (o, bins)) in row.iter_mut().zip(all_bins).enumerate() {
                        *o = bin_value(x[[first + k, j]].to_f64(), bins);
                    }
                }
            });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xy() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((300, 3), |(i, j)| ((i * (2 * j + 3)) % 23) as f64 / 4.0);
        let y = Array1::from_shape_fn(300, |i| x[[i, 0]] - 0.5 * x[[i, 1]] + (i % 3) as f64);
        (x, y)
    }

    #[test]
    fn dataset_fit_equals_fit_and_subset_reuses_the_bins() {
        let (x, y) = xy();
        let m = LgbmRegressor::new().with_n_estimators(8);
        let direct = m.fit(&x, &y).unwrap();
        let data = m.dataset(&x).unwrap();
        let reused = m.fit_dataset(&data, &y, &Default::default()).unwrap();
        assert_eq!(reused.predict(&x).unwrap(), direct.predict(&x).unwrap());
        assert_eq!(data.binned_bytes(), 900);

        // A fold: its rows binned with the full data's bins.
        let rows: Vec<usize> = (0..300).filter(|i| i % 4 != 0).collect();
        let fold = data.subset(&rows).unwrap();
        let ys = y.select(ndarray::Axis(0), &rows);
        let on_fold = m.fit_dataset(&fold, &ys, &Default::default()).unwrap();
        assert_eq!(on_fold.n_estimators(), 8);
        assert!(data.subset(&[300]).is_err());
    }

    #[test]
    fn dataset_settings_must_match() {
        let (x, y) = xy();
        let data = LgbmRegressor::new().with_max_bins(31).dataset(&x).unwrap();
        let err = LgbmRegressor::new().fit_dataset(&data, &y, &Default::default());
        assert!(err.is_err());
        let clf = LgbmClassifier::new().with_max_bins(31);
        let labels = y.mapv(|v| (v > 2.0) as u8 as f64);
        let cdata = clf.dataset(x.mapv(|v| v as f32).view()).unwrap();
        let fitted = clf
            .fit_dataset(&cdata, &labels, &Default::default())
            .unwrap();
        assert_eq!(
            fitted.predict_proba(x.mapv(|v| v as f32).view()).unwrap(),
            clf.fit(&x, &labels).unwrap().predict_proba(&x).unwrap()
        );
    }

    #[test]
    fn fits_inside_a_rayon_pool_equal_standalone_fits() {
        // Per-fold fits run in parallel by the caller share its pool (no
        // pool of their own) and give the same models.
        use rayon::prelude::*;
        let (x, y) = xy();
        let models: Vec<LgbmRegressor> = (0..4)
            .map(|k| {
                LgbmRegressor::new()
                    .with_n_estimators(10)
                    .with_subsample(0.8)
                    .with_subsample_freq(1)
                    .with_seed(k)
            })
            .collect();
        let alone: Vec<Array1<f64>> = models
            .iter()
            .map(|m| m.fit(&x, &y).unwrap().predict(&x).unwrap())
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let pooled: Vec<Array1<f64>> = pool.install(|| {
            models
                .par_iter()
                .map(|m| {
                    assert_eq!(rayon::current_num_threads(), 3);
                    m.fit(&x, &y).unwrap().predict(&x).unwrap()
                })
                .collect()
        });
        assert_eq!(pooled, alone);
    }

    #[test]
    fn builder_checks_its_rows() {
        let (x, _) = xy();
        let m = LgbmRegressor::new();
        assert!(m
            .dataset_builder(x.slice(ndarray::s![..10, ..]), 300)
            .is_err());
        let mut b = m.dataset_builder(&x, 300).unwrap();
        b.push_rows(x.slice(ndarray::s![..100, ..])).unwrap();
        assert!(b.push_rows(x.slice(ndarray::s![.., ..2])).is_err());
        assert!(b.finish().is_err());
    }
}
