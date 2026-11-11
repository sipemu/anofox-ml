//! CatBoost's embedding features: the online LDA calcer
//! (`embedding_features/lda.cpp`), in float32 with CBLAS's operation order.

use super::lapack::sygst_syev;
use anofox_ml_core::error::{Result, RustMlError};

/// How the features of one embedding column are calculated (CatBoost's
/// `embedding_calcers`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CatBoostEmbeddingCalcer {
    /// Linear discriminant analysis: the projection on the leading
    /// generalized eigenvectors of the total vs within-class scatter
    /// (`LDA`). `components` defaults to min(classes - 1, dim - 1) (1 for
    /// regression); `reg` (0.00005) is added to the within-class scatter's
    /// diagonal.
    Lda { components: Option<usize>, reg: f32 },
}

impl CatBoostEmbeddingCalcer {
    /// CatBoost's `LDA` with its defaults.
    pub fn lda() -> Self {
        Self::Lda {
            components: None,
            reg: 0.00005,
        }
    }
}

/// The processing of the embedding columns: the calcers of every column,
/// in order. CatBoost's default is `[LDA, KNN]`; KNN is not available yet.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CatBoostEmbeddingOptions {
    pub calcers: Vec<CatBoostEmbeddingCalcer>,
}

impl CatBoostEmbeddingOptions {
    pub fn lda() -> Self {
        Self {
            calcers: vec![CatBoostEmbeddingCalcer::lda()],
        }
    }
}

/// The LDA calcer of one column, as CatBoost builds it for a fold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LdaSpec {
    pub dim: usize,
    /// Classes, or `None` for regression (one cloud).
    pub n_classes: Option<usize>,
    pub components: usize,
    pub reg: f32,
}

impl LdaSpec {
    pub(crate) fn new(
        calcer: &CatBoostEmbeddingCalcer,
        dim: usize,
        n_classes: Option<usize>,
    ) -> Result<Self> {
        let CatBoostEmbeddingCalcer::Lda { components, reg } = *calcer;
        let components = components.unwrap_or_else(|| {
            n_classes
                .map_or(1, |k| k.saturating_sub(1))
                .min(dim.saturating_sub(1))
        });
        if components == 0 {
            return Err(RustMlError::InvalidParameter(
                "LDA: the projection needs at least one component".into(),
            ));
        }
        if components >= dim {
            return Err(RustMlError::InvalidParameter(format!(
                "LDA: {components} components need an embedding of more dimensions than {dim}"
            )));
        }
        if reg.is_nan() || reg < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "LDA: reg must be non-negative".into(),
            ));
        }
        Ok(Self {
            dim,
            n_classes,
            components,
            reg,
        })
    }
}

/// CatBoost's `IncrementalCloud`: a class's running center and scatter,
/// updated in blocks (every row for the first 128, then every 32).
#[derive(Debug, Clone)]
struct Cloud {
    base_size: i32,
    additional: i32,
    center: Vec<f32>,
    shift: Vec<f32>,
    scatter: Vec<f32>,
    buffer: Vec<f32>,
}

impl Cloud {
    fn new(dim: usize) -> Self {
        Self {
            base_size: 0,
            additional: 0,
            center: vec![0.0; dim],
            shift: vec![0.0; dim],
            scatter: vec![0.0; dim * dim],
            buffer: Vec::new(),
        }
    }

    fn total_size(&self) -> f32 {
        (self.base_size + self.additional) as f32
    }

    fn add(&mut self, x: &[f32]) {
        self.additional += 1;
        for (i, &v) in x.iter().enumerate() {
            let d = v - self.center[i];
            self.buffer.push(d);
            self.shift[i] += d;
        }
        if self.base_size < 128 || self.additional >= 32 {
            self.update();
        }
    }

    fn update(&mut self) {
        if self.additional == 0 {
            return;
        }
        let dim = self.center.len();
        let total = self.total_size();
        for i in 0..dim {
            self.shift[i] /= total;
            self.center[i] += self.shift[i];
        }
        // cblas_sgemm(RowMajor, Trans, NoTrans): S = B'B / total + S * base
        // / total, as the reference sgemm("N", "T") runs it.
        let alpha = (1.0 / f64::from(total)) as f32;
        let beta = self.base_size as f32 / total;
        let k = self.additional as usize;
        for j in 0..dim {
            let col = &mut self.scatter[j * dim..(j + 1) * dim];
            if beta == 0.0 {
                col.fill(0.0);
            } else if beta != 1.0 {
                col.iter_mut().for_each(|c| *c = beta * *c);
            }
            for l in 0..k {
                let row = &self.buffer[l * dim..(l + 1) * dim];
                if row[j] != 0.0 {
                    let temp = alpha * row[j];
                    for i in 0..dim {
                        col[i] += temp * row[i];
                    }
                }
            }
        }
        self.buffer.clear();
        // S -= shift shift'.
        rank_one(&mut self.scatter, -1.0, &self.shift, &self.shift);
        self.base_size += self.additional;
        self.additional = 0;
        self.shift.fill(0.0);
    }
}

/// cblas_sgemm(RowMajor, NoTrans, Trans, n, n, 1, alpha, u, 1, v, 1, 1, C):
/// C = alpha (u v') + C, as the reference sgemm("T", "N") runs it.
fn rank_one(c: &mut [f32], alpha: f32, u: &[f32], v: &[f32]) {
    if alpha == 0.0 {
        return;
    }
    let n = u.len();
    for j in 0..n {
        for i in 0..n {
            let temp = 0.0 + v[i] * u[j];
            let k = i + j * n;
            c[k] = alpha * temp + 1.0 * c[k];
        }
    }
}

/// CatBoost's `TLinearDACalcer` with its visitor: the clouds, and the
/// projection recomputed whenever the row count doubles.
#[derive(Debug, Clone)]
pub(crate) struct Lda {
    spec: LdaSpec,
    size: i32,
    last_flush: i32,
    clouds: Vec<Cloud>,
    /// `components` x `dim`, row-major.
    projection: Vec<f32>,
}

impl Lda {
    pub(crate) fn new(spec: LdaSpec) -> Self {
        Self {
            spec,
            size: 0,
            last_flush: 0,
            clouds: vec![Cloud::new(spec.dim); spec.n_classes.unwrap_or(1)],
            projection: vec![0.0; spec.components * spec.dim],
        }
    }

    pub(crate) fn projection(&self) -> LdaProjection {
        LdaProjection {
            dim: self.spec.dim,
            projection: self.projection.clone(),
        }
    }

    pub(crate) fn compute(&self, x: &[f32]) -> Vec<f32> {
        project(&self.projection, self.spec.dim, x)
    }

    /// `TLinearDACalcerVisitor::Update` (`target` is the class, or ignored
    /// for regression).
    pub(crate) fn update(&mut self, target: usize, x: &[f32]) {
        let cloud = if self.spec.n_classes.is_some() {
            target
        } else {
            0
        };
        self.clouds[cloud].add(x);
        self.size += 1;
        if 2 * self.last_flush <= self.size {
            self.flush();
            self.last_flush = self.size;
        }
    }

    fn flush(&mut self) {
        let dim = self.spec.dim;
        let size = self.size as f32;
        // TotalScatterCalculation.
        let mut total = vec![0.0f32; dim * dim];
        let mut mean = vec![0.0f32; dim];
        for cloud in &self.clouds {
            let weight = cloud.total_size() / size;
            rank_one(&mut total, weight, &cloud.center, &cloud.center);
            for (m, &c) in mean.iter_mut().zip(&cloud.center) {
                *m += weight * c;
            }
        }
        rank_one(&mut total, -1.0, &mean, &mean);
        let mut within = if self.spec.n_classes.is_some() {
            let mut w = vec![0.0f32; dim * dim];
            for cloud in &self.clouds {
                let weight = cloud.total_size() / size;
                for (b, &s) in w.iter_mut().zip(&cloud.scatter) {
                    *b += weight * s;
                }
            }
            w
        } else {
            self.clouds[0].scatter.clone()
        };
        for i in 0..dim {
            within[i * (dim + 1)] += self.spec.reg;
        }
        sygst_syev(dim, &mut total, &within);
        let n = self.projection.len();
        self.projection.copy_from_slice(&total[dim * dim - n..]);
    }
}

/// The fitted projection of a model's LDA calcer.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct LdaProjection {
    dim: usize,
    projection: Vec<f32>,
}

impl LdaProjection {
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn compute(&self, x: &[f32]) -> Vec<f32> {
        project(&self.projection, self.dim, x)
    }
}

/// cblas_sgemv(RowMajor, NoTrans): the projection of `x`, as the reference
/// sgemv("T") runs it.
fn project(projection: &[f32], dim: usize, x: &[f32]) -> Vec<f32> {
    projection
        .chunks(dim)
        .map(|row| {
            let mut temp = 0.0f32;
            for (&p, &v) in row.iter().zip(x) {
                temp += p * v;
            }
            0.0 + 1.0 * temp
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lda_spec_defaults_and_limits() {
        let lda = CatBoostEmbeddingCalcer::lda();
        assert_eq!(LdaSpec::new(&lda, 5, Some(3)).unwrap().components, 2);
        assert_eq!(LdaSpec::new(&lda, 2, Some(4)).unwrap().components, 1);
        assert_eq!(LdaSpec::new(&lda, 5, None).unwrap().components, 1);
        // One dimension leaves no component; components must stay below dim.
        assert!(LdaSpec::new(&lda, 1, Some(2)).is_err());
        let three = CatBoostEmbeddingCalcer::Lda {
            components: Some(3),
            reg: 0.1,
        };
        assert!(LdaSpec::new(&three, 3, Some(2)).is_err());
        let negative = CatBoostEmbeddingCalcer::Lda {
            components: None,
            reg: -1.0,
        };
        assert!(LdaSpec::new(&negative, 3, Some(2)).is_err());
    }

    /// Regression has one cloud and a zero total scatter, so (as in
    /// CatBoost) the projection is the last coordinate.
    #[test]
    fn regression_lda_projects_on_the_last_coordinate() {
        let spec = LdaSpec::new(&CatBoostEmbeddingCalcer::lda(), 3, None).unwrap();
        let mut lda = Lda::new(spec);
        assert_eq!(lda.compute(&[1.0, 2.0, 3.0]), vec![0.0]);
        for i in 0..50 {
            let v = i as f32;
            lda.update(0, &[v.sin(), v.cos(), 0.1 * v]);
        }
        assert_eq!(lda.compute(&[1.0, 2.0, 3.0]), vec![3.0]);
    }
}
