//! Nearest centroid classifier (sklearn's `NearestCentroid`), with
//! Tibshirani's shrunken centroids (`shrink_threshold`) and class priors.

use anofox_ml_core::{DecisionFunction, Fit, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// Distance to the centroids. Manhattan centroids are per-class medians.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CentroidMetric {
    #[default]
    Euclidean,
    Manhattan,
}

/// Class priors of a [`NearestCentroid`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum CentroidPriors {
    /// Equal priors: predict the nearest centroid.
    #[default]
    Uniform,
    /// Class frequencies in the training data.
    Empirical,
    /// Given priors, one per class in sorted label order (normalised to
    /// sum to 1).
    Given(Vec<f64>),
}

/// Nearest centroid classifier.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NearestCentroid {
    pub metric: CentroidMetric,
    /// Shrink each centroid towards the overall centroid by this many
    /// within-class standard deviations (soft thresholding).
    pub shrink_threshold: Option<f64>,
    pub priors: CentroidPriors,
}

impl NearestCentroid {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_metric(mut self, metric: CentroidMetric) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_shrink_threshold(mut self, threshold: f64) -> Self {
        self.shrink_threshold = Some(threshold);
        self
    }

    pub fn with_priors(mut self, priors: CentroidPriors) -> Self {
        self.priors = priors;
        self
    }
}

/// Fitted [`NearestCentroid`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedNearestCentroid {
    metric: CentroidMetric,
    classes: Vec<f64>,
    centroids: Array2<f64>,
    class_prior: Vec<f64>,
    within_class_std_dev: Array1<f64>,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

impl Fit<f64> for NearestCentroid {
    type Fitted = FittedNearestCentroid;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedNearestCentroid> {
        let (n, d) = x.dim();
        if y.len() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "x has {n} rows but y has {}",
                y.len()
            )));
        }
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("x has no rows or columns".into()));
        }
        if x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "x and y must be finite".into(),
            ));
        }
        let mut classes = y.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let k = classes.len();
        if k < 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "NearestCentroid needs at least 2 classes, got {k}"
            )));
        }
        let y_ind: Vec<usize> = y
            .iter()
            .map(|&v| classes.partition_point(|&c| c < v))
            .collect();
        let mut nk = vec![0.0; k];
        for &c in &y_ind {
            nk[c] += 1.0;
        }

        let mut class_prior = match &self.priors {
            CentroidPriors::Uniform => vec![1.0 / k as f64; k],
            CentroidPriors::Empirical => nk.iter().map(|c| c / n as f64).collect(),
            CentroidPriors::Given(p) => {
                if p.len() != k {
                    return Err(RustMlError::InvalidParameter(format!(
                        "{} priors for {k} classes",
                        p.len()
                    )));
                }
                p.clone()
            }
        };
        if class_prior.iter().any(|&p| p < 0.0) {
            return Err(RustMlError::InvalidParameter(
                "priors must be non-negative".into(),
            ));
        }
        let total: f64 = class_prior.iter().sum();
        class_prior.iter_mut().for_each(|p| *p /= total);

        let mut centroids = Array2::<f64>::zeros((k, d));
        for c in 0..k {
            let rows: Vec<usize> = (0..n).filter(|&i| y_ind[i] == c).collect();
            let sub = x.select(Axis(0), &rows);
            for j in 0..d {
                centroids[[c, j]] = match self.metric {
                    CentroidMetric::Euclidean => sub.column(j).mean().expect("non-empty"),
                    CentroidMetric::Manhattan => median(sub.column(j).to_vec()),
                };
            }
        }

        // Pooled within-class standard deviation per feature.
        let mut within = Array1::<f64>::zeros(d);
        for i in 0..n {
            for j in 0..d {
                within[j] += (x[[i, j]] - centroids[[y_ind[i], j]]).powi(2);
            }
        }
        within.mapv_inplace(|v| (v / (n - k) as f64).sqrt());

        let constant = (0..d).all(|j| {
            let col = x.column(j);
            col.iter().all(|&v| v == col[0])
        });
        if constant {
            return Err(RustMlError::InvalidParameter(
                "all features have zero variance".into(),
            ));
        }

        if let Some(threshold) = self.shrink_threshold.filter(|&t| t != 0.0) {
            let overall = x.mean_axis(Axis(0)).expect("non-empty");
            let s_med = median(within.to_vec());
            for c in 0..k {
                let m = (1.0 / nk[c] - 1.0 / n as f64).sqrt();
                for j in 0..d {
                    let ms = m * (within[j] + s_med);
                    let dev = (centroids[[c, j]] - overall[j]) / ms;
                    let shrunk = dev.signum() * (dev.abs() - threshold).max(0.0);
                    centroids[[c, j]] = overall[j] + ms * shrunk;
                }
            }
        }

        Ok(FittedNearestCentroid {
            metric: self.metric,
            classes,
            centroids,
            class_prior,
            within_class_std_dev: within,
        })
    }
}

impl FittedNearestCentroid {
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }

    /// One centroid per class (rows in [`classes`](Self::classes) order).
    pub fn centroids(&self) -> &Array2<f64> {
        &self.centroids
    }

    pub fn class_prior(&self) -> &[f64] {
        &self.class_prior
    }

    pub fn within_class_std_dev(&self) -> &Array1<f64> {
        &self.within_class_std_dev
    }

    fn check(&self, x: ArrayView2<f64>) -> Result<()> {
        if x.ncols() != self.centroids.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.centroids.ncols(),
                x.ncols()
            )));
        }
        Ok(())
    }

    fn distance(&self, a: impl Iterator<Item = f64>) -> f64 {
        match self.metric {
            CentroidMetric::Euclidean => a.map(|v| v * v).sum::<f64>().sqrt(),
            CentroidMetric::Manhattan => a.map(f64::abs).sum(),
        }
    }

    /// ESL eq. 18.2: `−‖(x − c_k) / s‖² + 2 log π_k`, with features scaled
    /// by the within-class standard deviation (where it is non-zero).
    fn scores(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let s = &self.within_class_std_dev;
        let scale = |j: usize| if s[j] != 0.0 { s[j] } else { 1.0 };
        Array2::from_shape_fn((x.nrows(), self.classes.len()), |(i, c)| {
            let dist = self
                .distance((0..x.ncols()).map(|j| (x[[i, j]] - self.centroids[[c, j]]) / scale(j)));
            -dist * dist + 2.0 * self.class_prior[c].ln()
        })
    }

    fn uniform_priors(&self) -> bool {
        let k = self.classes.len() as f64;
        self.class_prior
            .iter()
            .all(|&p| (p - 1.0 / k).abs() <= 1e-8 + 1e-5 * (1.0 / k))
    }

    fn euclidean_only(&self, method: &str) -> Result<()> {
        if self.metric != CentroidMetric::Euclidean {
            return Err(RustMlError::InvalidParameter(format!(
                "NearestCentroid.{method} needs the euclidean metric"
            )));
        }
        Ok(())
    }
}

fn argmax(row: ArrayView1<f64>) -> usize {
    row.iter()
        .enumerate()
        .fold(0, |b, (j, &v)| if v > row[b] { j } else { b })
}

impl Predict<f64> for FittedNearestCentroid {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.check(x)?;
        if self.uniform_priors() {
            // Plain nearest centroid on unscaled distances; first on ties.
            return Ok(x
                .rows()
                .into_iter()
                .map(|r| {
                    let mut best = (f64::INFINITY, 0);
                    for c in 0..self.classes.len() {
                        let dist =
                            self.distance(r.iter().zip(self.centroids.row(c)).map(|(a, b)| a - b));
                        if dist < best.0 {
                            best = (dist, c);
                        }
                    }
                    self.classes[best.1]
                })
                .collect());
        }
        let scores = self.scores(x);
        Ok(scores
            .rows()
            .into_iter()
            .map(|r| self.classes[argmax(r)])
            .collect())
    }
}

impl DecisionFunction<f64> for FittedNearestCentroid {
    /// Discriminant scores (`n × classes`); for two classes the difference
    /// `score[1] − score[0]` as one column, as sklearn.
    fn decision_function_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.check(x)?;
        self.euclidean_only("decision_function")?;
        let s = self.scores(x);
        if self.classes.len() == 2 {
            return Ok((&s.column(1) - &s.column(0)).insert_axis(Axis(1)));
        }
        Ok(s)
    }
}

impl PredictProba<f64> for FittedNearestCentroid {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.check(x)?;
        self.euclidean_only("predict_proba")?;
        let mut s = self.scores(x);
        for mut row in s.rows_mut() {
            let max = row.fold(f64::NEG_INFINITY, |m, &v| m.max(v));
            row.mapv_inplace(|v| v - max);
            let log_norm = row.mapv(f64::exp).sum().ln();
            row.mapv_inplace(|v| (v - log_norm).exp());
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn classifies_by_the_nearest_centroid() {
        let x = array![
            [0.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [10.0, 0.0],
            [10.0, 1.0]
        ];
        let y = array![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let m = NearestCentroid::new().fit(&x, &y).unwrap();
        assert_eq!(m.centroids().row(1).to_vec(), vec![5.0, 5.5]);
        assert_eq!(
            m.predict(&array![[4.0, 4.0], [9.0, 0.0]]).unwrap(),
            array![2.0, 3.0]
        );
        let p = m.predict_proba(&x).unwrap();
        for r in p.rows() {
            assert!((r.sum() - 1.0).abs() < 1e-12);
        }
        assert!(NearestCentroid::new()
            .fit(&x, &array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .is_err());
        let manhattan = NearestCentroid::new()
            .with_metric(CentroidMetric::Manhattan)
            .fit(&x, &y)
            .unwrap();
        assert!(manhattan.predict_proba(&x).is_err());
    }
}
