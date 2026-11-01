//! Mutual-information feature scoring and selection.
//!
//! Implements the k-nearest-neighbour estimators used by scikit-learn's
//! `mutual_info_regression` / `mutual_info_classif`:
//!
//! * continuous feature, continuous target — Kraskov, Stögbauer &
//!   Grassberger (2004), estimator 1 (max-norm in the joint space);
//! * continuous feature, discrete target (or vice versa) — Ross (2014);
//! * discrete feature, discrete target — the plug-in estimate from the
//!   contingency table (`sklearn.metrics.mutual_info_score`).
//!
//! As in sklearn, continuous features (and a continuous target) are scaled
//! to unit variance (without centring) and perturbed with tiny seeded
//! Gaussian noise (`1e-10 * max(1, mean|x|)`) to break ties. All values are
//! in **nats** (natural logarithm).

use anofox_ml_core::{looks_like_class_labels, Fit, Float, Result, RustMlError, Transform};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

use crate::special::digamma;

/// Which features are discrete (sklearn `discrete_features`).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DiscreteFeatures {
    /// sklearn `"auto"` for dense input: all features continuous.
    #[default]
    Auto,
    /// All features are discrete.
    All,
    /// Boolean mask, `true` = discrete (length `n_features`).
    Mask(Vec<bool>),
    /// Indices of the discrete features.
    Indices(Vec<usize>),
}

impl DiscreteFeatures {
    fn mask(&self, n_features: usize) -> Result<Vec<bool>> {
        match self {
            DiscreteFeatures::Auto => Ok(vec![false; n_features]),
            DiscreteFeatures::All => Ok(vec![true; n_features]),
            DiscreteFeatures::Mask(m) => {
                if m.len() != n_features {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "discrete_features mask has length {}, expected {}",
                        m.len(),
                        n_features
                    )));
                }
                Ok(m.clone())
            }
            DiscreteFeatures::Indices(idx) => {
                let mut m = vec![false; n_features];
                for &i in idx {
                    if i >= n_features {
                        return Err(RustMlError::InvalidParameter(format!(
                            "discrete feature index {i} out of range ({n_features} features)"
                        )));
                    }
                    m[i] = true;
                }
                Ok(m)
            }
        }
    }
}

/// How the target is interpreted by [`MutualInformationSelector`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MiTarget {
    /// Discrete if `y` looks like class labels
    /// ([`looks_like_class_labels`]), continuous otherwise.
    #[default]
    Auto,
    /// Class labels — `mutual_info_classif`.
    Discrete,
    /// Continuous target — `mutual_info_regression`.
    Continuous,
}

/// Next float towards zero (`np.nextafter(r, 0)` for `r >= 0`).
fn next_toward_zero(r: f64) -> f64 {
    if r > 0.0 {
        f64::from_bits(r.to_bits() - 1)
    } else {
        r
    }
}

/// Number of values `v` in the ascending slice with `|v - x| <= r`
/// (exact predicate, like `KDTree.query_radius(..., count_only=True)`).
fn count_within(sorted: &[f64], x: f64, r: f64) -> usize {
    let lo = sorted.partition_point(|&v| v < x && x - v > r);
    let hi = sorted.partition_point(|&v| v <= x || v - x <= r);
    hi - lo
}

/// Distance from `sorted[p]` to its k-th nearest neighbour (excluding
/// itself) in a sorted 1-D array.
fn kth_neighbour_1d(sorted: &[f64], p: usize, k: usize) -> f64 {
    let n = sorted.len();
    let x = sorted[p];
    let (mut l, mut r) = (p as isize - 1, p + 1);
    let mut d = 0.0;
    for _ in 0..k {
        let dl = if l >= 0 {
            x - sorted[l as usize]
        } else {
            f64::INFINITY
        };
        let dr = if r < n { sorted[r] - x } else { f64::INFINITY };
        if dl <= dr {
            d = dl;
            l -= 1;
        } else {
            d = dr;
            r += 1;
        }
    }
    d
}

/// KSG estimator for two continuous variables (sklearn `_compute_mi_cc`).
fn mi_cc(x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len();
    if n <= k {
        return 0.0;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let mut ys: Vec<f64> = y.to_vec();
    ys.sort_by(f64::total_cmp);

    let mut sum_nx = 0.0;
    let mut sum_ny = 0.0;
    let mut heap: Vec<f64> = Vec::with_capacity(k + 1);
    for p in 0..n {
        let i = order[p];
        // k nearest neighbours in the joint space under the max-norm,
        // scanning outwards in x order.
        heap.clear();
        let push = |heap: &mut Vec<f64>, d: f64| {
            if heap.len() < k {
                heap.push(d);
                heap.sort_by(f64::total_cmp);
            } else if d < heap[k - 1] {
                heap[k - 1] = d;
                heap.sort_by(f64::total_cmp);
            }
        };
        let (mut l, mut r) = (p as isize - 1, p + 1);
        loop {
            let kth = if heap.len() == k {
                heap[k - 1]
            } else {
                f64::INFINITY
            };
            let dl = if l >= 0 {
                xs[p] - xs[l as usize]
            } else {
                f64::INFINITY
            };
            let dr = if r < n { xs[r] - xs[p] } else { f64::INFINITY };
            if dl.min(dr) > kth || (dl.is_infinite() && dr.is_infinite()) {
                break;
            }
            let j = if dl <= dr {
                let j = order[l as usize];
                l -= 1;
                j
            } else {
                let j = order[r];
                r += 1;
                j
            };
            let d = (x[i] - x[j]).abs().max((y[i] - y[j]).abs());
            push(&mut heap, d);
        }
        let radius = next_toward_zero(heap[k - 1]);
        let nx = count_within(&xs, x[i], radius) - 1;
        let ny = count_within(&ys, y[i], radius) - 1;
        sum_nx += digamma(nx as f64 + 1.0);
        sum_ny += digamma(ny as f64 + 1.0);
    }
    let nf = n as f64;
    let mi = digamma(nf) + digamma(k as f64) - sum_nx / nf - sum_ny / nf;
    mi.max(0.0)
}

/// Group sample indices by exact label value (sorted by label).
fn group_labels(d: &[f64]) -> Vec<Vec<usize>> {
    let mut map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &v) in d.iter().enumerate() {
        // Normalise -0.0 to 0.0 so both land in the same class.
        let key = if v == 0.0 { 0.0_f64 } else { v }.to_bits();
        map.entry(key).or_default().push(i);
    }
    let mut groups: Vec<(f64, Vec<usize>)> = map.into_values().map(|g| (d[g[0]], g)).collect();
    groups.sort_by(|a, b| a.0.total_cmp(&b.0));
    groups.into_iter().map(|(_, g)| g).collect()
}

/// Ross (2014) estimator for a continuous `c` and a discrete `d`
/// (sklearn `_compute_mi_cd`).
fn mi_cd(c: &[f64], d: &[f64], n_neighbors: usize) -> f64 {
    let n = c.len();
    let mut radius = vec![0.0; n];
    let mut label_counts = vec![0usize; n];
    let mut k_all = vec![0usize; n];
    for group in group_labels(d) {
        let count = group.len();
        if count > 1 {
            let k = n_neighbors.min(count - 1);
            let mut order = group.clone();
            order.sort_by(|&a, &b| c[a].total_cmp(&c[b]));
            let vals: Vec<f64> = order.iter().map(|&i| c[i]).collect();
            for (p, &i) in order.iter().enumerate() {
                radius[i] = next_toward_zero(kth_neighbour_1d(&vals, p, k));
                k_all[i] = k;
            }
        }
        for &i in &group {
            label_counts[i] = count;
        }
    }
    // Ignore points with unique labels.
    let keep: Vec<usize> = (0..n).filter(|&i| label_counts[i] > 1).collect();
    let m = keep.len();
    if m == 0 {
        return 0.0;
    }
    let mut cs: Vec<f64> = keep.iter().map(|&i| c[i]).collect();
    cs.sort_by(f64::total_cmp);
    let mut sum_k = 0.0;
    let mut sum_label = 0.0;
    let mut sum_m = 0.0;
    for &i in &keep {
        let m_i = count_within(&cs, c[i], radius[i]);
        sum_k += digamma(k_all[i] as f64);
        sum_label += digamma(label_counts[i] as f64);
        sum_m += digamma(m_i as f64);
    }
    let mf = m as f64;
    let mi = digamma(mf) + sum_k / mf - sum_label / mf - sum_m / mf;
    mi.max(0.0)
}

/// Plug-in MI of two discrete variables (`sklearn.metrics.mutual_info_score`).
fn mi_dd(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let key = |v: f64| if v == 0.0 { 0.0_f64 } else { v }.to_bits();
    let mut joint: HashMap<(u64, u64), usize> = HashMap::new();
    let mut px: HashMap<u64, usize> = HashMap::new();
    let mut py: HashMap<u64, usize> = HashMap::new();
    for (&a, &b) in x.iter().zip(y) {
        *joint.entry((key(a), key(b))).or_insert(0) += 1;
        *px.entry(key(a)).or_insert(0) += 1;
        *py.entry(key(b)).or_insert(0) += 1;
    }
    let mut mi = 0.0;
    for (&(a, b), &c) in &joint {
        let pxy = c as f64 / n;
        mi += pxy * (c as f64 * n / (px[&a] as f64 * py[&b] as f64)).ln();
    }
    mi.max(0.0)
}

/// Draw a standard normal (Box-Muller).
fn standard_normal(rng: &mut StdRng) -> f64 {
    loop {
        let u1: f64 = rng.gen();
        if u1 > 0.0 {
            let u2: f64 = rng.gen();
            return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
}

/// `scale(v, with_mean=False)` followed by sklearn's tie-breaking noise.
fn scale_and_jitter(v: &mut [f64], rng: &mut StdRng) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    let scale = if std < 10.0 * f64::EPSILON { 1.0 } else { std };
    v.iter_mut().for_each(|t| *t /= scale);
    let mean_abs = v.iter().map(|t| t.abs()).sum::<f64>() / n;
    let amp = 1e-10 * mean_abs.max(1.0);
    v.iter_mut().for_each(|t| *t += amp * standard_normal(rng));
}

/// sklearn `_estimate_mi`.
fn estimate_mi<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    discrete_target: bool,
    discrete_features: &DiscreteFeatures,
    n_neighbors: usize,
    seed: u64,
) -> Result<Array1<F>> {
    let (n_samples, n_features) = x.dim();
    if n_samples == 0 || n_features == 0 {
        return Err(RustMlError::EmptyInput("input array is empty".into()));
    }
    if y.len() != n_samples {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} samples but y has {} elements",
            n_samples,
            y.len()
        )));
    }
    if n_neighbors == 0 {
        return Err(RustMlError::InvalidParameter(
            "n_neighbors must be at least 1".into(),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(RustMlError::InvalidParameter(
            "mutual information input must be finite (no NaN / inf)".into(),
        ));
    }
    let mask = discrete_features.mask(n_features)?;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut cols: Vec<Vec<f64>> = (0..n_features)
        .map(|j| x.column(j).iter().map(|v| v.to_f64().unwrap()).collect())
        .collect();
    for (j, col) in cols.iter_mut().enumerate() {
        if !mask[j] {
            scale_and_jitter(col, &mut rng);
        }
    }
    let mut yv: Vec<f64> = y.iter().map(|v| v.to_f64().unwrap()).collect();
    if !discrete_target {
        scale_and_jitter(&mut yv, &mut rng);
    }

    let mi = cols
        .iter()
        .zip(&mask)
        .map(|(col, &disc)| {
            let v = match (disc, discrete_target) {
                (true, true) => mi_dd(col, &yv),
                (true, false) => mi_cd(&yv, col, n_neighbors),
                (false, true) => mi_cd(col, &yv, n_neighbors),
                (false, false) => mi_cc(col, &yv, n_neighbors),
            };
            F::from_f64(v).unwrap()
        })
        .collect();
    Ok(Array1::from_vec(mi))
}

/// Mutual information between each feature and a **continuous** target
/// (sklearn `mutual_info_regression`), in nats.
pub fn mutual_info_regression<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    discrete_features: &DiscreteFeatures,
    n_neighbors: usize,
    seed: u64,
) -> Result<Array1<F>> {
    estimate_mi(x, y, false, discrete_features, n_neighbors, seed)
}

/// Mutual information between each feature and a **discrete** target
/// (sklearn `mutual_info_classif`), in nats.
pub fn mutual_info_classif<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    discrete_features: &DiscreteFeatures,
    n_neighbors: usize,
    seed: u64,
) -> Result<Array1<F>> {
    estimate_mi(x, y, true, discrete_features, n_neighbors, seed)
}

/// Parameters for `MutualInformationSelector` (unfitted state).
///
/// Selects the top-k features by mutual information with the target,
/// estimated exactly like sklearn's `mutual_info_classif` (discrete target)
/// or `mutual_info_regression` (continuous target) — see the
/// [module docs](self). Scores are in nats.
///
/// This is a **supervised** feature selector and requires target `y`.
///
/// # Example
///
/// ```
/// use anofox_ml_preprocessing::MutualInformationSelector;
/// use anofox_ml_core::{Fit, Transform};
/// use ndarray::array;
///
/// let x = array![
///     [1.0, 100.0],
///     [2.0, 200.0],
///     [1.0, 300.0],
///     [2.0, 400.0],
/// ];
/// let y = array![0.0, 1.0, 0.0, 1.0]; // perfectly correlated with col 0
///
/// let selector = MutualInformationSelector::new(1);
/// let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();
/// let x_selected = fitted.transform(&x).unwrap();
///
/// assert_eq!(x_selected.ncols(), 1);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MutualInformationSelector {
    /// Number of top features to select.
    pub n_features_to_select: usize,
    /// Number of neighbours of the k-NN estimators (sklearn default 3).
    #[serde(default = "default_n_neighbors")]
    pub n_neighbors: usize,
    /// Which features are discrete.
    #[serde(default)]
    pub discrete_features: DiscreteFeatures,
    /// Target interpretation (classification vs regression).
    #[serde(default)]
    pub target: MiTarget,
    /// Seed for the tie-breaking noise.
    #[serde(default)]
    pub random_state: u64,
}

fn default_n_neighbors() -> usize {
    3
}

impl MutualInformationSelector {
    /// Create a new selector that keeps the top `n_features_to_select` features.
    pub fn new(n_features_to_select: usize) -> Self {
        Self {
            n_features_to_select,
            n_neighbors: 3,
            discrete_features: DiscreteFeatures::Auto,
            target: MiTarget::Auto,
            random_state: 0,
        }
    }

    /// Set the number of neighbours of the k-NN estimators.
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Declare which features are discrete.
    pub fn with_discrete_features(mut self, discrete_features: DiscreteFeatures) -> Self {
        self.discrete_features = discrete_features;
        self
    }

    /// Set how the target is interpreted.
    pub fn with_target(mut self, target: MiTarget) -> Self {
        self.target = target;
        self
    }

    /// Set the seed for the tie-breaking noise.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
        self
    }
}

/// Fitted `MutualInformationSelector` — holds per-feature MI scores and
/// the indices of the selected top-k features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMutualInformationSelector<F: Float> {
    /// Mutual information score for each feature (nats).
    mi_scores: Array1<F>,
    /// Indices of the top-k features (sorted by index for stable column ordering).
    selected_indices: Vec<usize>,
    /// Total number of input features (before selection).
    n_features_in: usize,
}

impl<F: Float> FittedMutualInformationSelector<F> {
    /// Per-feature mutual information scores (nats).
    pub fn mi_scores(&self) -> &Array1<F> {
        &self.mi_scores
    }

    /// Indices of the selected features, sorted in ascending order.
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }
}

impl<F: Float> Fit<F> for MutualInformationSelector {
    type Fitted = FittedMutualInformationSelector<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        if y.len() != n_samples {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} samples but y has {} elements",
                n_samples,
                y.len()
            )));
        }

        if self.n_features_to_select == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_features_to_select must be at least 1".into(),
            ));
        }

        if self.n_features_to_select > n_features {
            return Err(RustMlError::InvalidParameter(format!(
                "n_features_to_select ({}) exceeds number of features ({})",
                self.n_features_to_select, n_features
            )));
        }

        let discrete_target = match self.target {
            MiTarget::Auto => looks_like_class_labels(y),
            MiTarget::Discrete => true,
            MiTarget::Continuous => false,
        };
        let mi_scores = estimate_mi(
            x,
            y,
            discrete_target,
            &self.discrete_features,
            self.n_neighbors,
            self.random_state,
        )?;

        let selected_indices = top_k_indices(&mi_scores, self.n_features_to_select);

        Ok(FittedMutualInformationSelector {
            mi_scores,
            selected_indices,
            n_features_in: n_features,
        })
    }
}

/// Indices of the `k` largest scores (ties by index, NaN lowest), sorted
/// ascending.
pub(crate) fn top_k_indices<F: Float>(scores: &Array1<F>, k: usize) -> Vec<usize> {
    let mut feature_scores: Vec<(usize, F)> = scores.iter().copied().enumerate().collect();
    feature_scores
        .sort_by(|a, b| crate::quantile::total_cmp_nan_low(&b.1, &a.1).then(a.0.cmp(&b.0)));
    let mut selected: Vec<usize> = feature_scores.iter().take(k).map(|&(i, _)| i).collect();
    selected.sort_unstable();
    selected
}

impl<F: Float> Transform<F> for FittedMutualInformationSelector<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features_in {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features_in,
                x.ncols()
            )));
        }

        let n_rows = x.nrows();
        let n_selected = self.selected_indices.len();
        let mut result = Array2::<F>::zeros((n_rows, n_selected));

        for (i, row) in x.rows().into_iter().enumerate() {
            for (out_j, &src_j) in self.selected_indices.iter().enumerate() {
                result[[i, out_j]] = row[src_j];
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Data + reference values generated with scikit-learn 1.8.0
    // (see the generator in the PR description): X ~ N(0,1) (3 continuous
    // columns) + one discrete column in {0,1,2}.
    const X: [[f64; 4]; 60] = [
        [
            0.4967141530112327,
            -0.13826430117118466,
            0.6476885381006925,
            0.0,
        ],
        [
            1.5230298564080254,
            -0.23415337472333597,
            -0.23413695694918055,
            0.0,
        ],
        [
            1.5792128155073915,
            0.7674347291529088,
            -0.4694743859349521,
            2.0,
        ],
        [
            0.5425600435859647,
            -0.46341769281246226,
            -0.46572975357025687,
            2.0,
        ],
        [
            0.24196227156603412,
            -1.913280244657798,
            -1.7249178325130328,
            0.0,
        ],
        [
            -0.5622875292409727,
            -1.0128311203344238,
            0.3142473325952739,
            0.0,
        ],
        [
            -0.9080240755212109,
            -1.4123037013352915,
            1.465648768921554,
            1.0,
        ],
        [
            -0.22577630048653566,
            0.06752820468792384,
            -1.4247481862134568,
            0.0,
        ],
        [
            -0.5443827245251827,
            0.11092258970986608,
            -1.1509935774223028,
            1.0,
        ],
        [
            0.37569801834567196,
            -0.600638689918805,
            -0.2916937497932768,
            0.0,
        ],
        [
            -0.6017066122293969,
            1.8522781845089378,
            -0.013497224737933921,
            1.0,
        ],
        [
            -1.0577109289559004,
            0.822544912103189,
            -1.2208436499710222,
            2.0,
        ],
        [
            0.2088635950047554,
            -1.9596701238797756,
            -1.3281860488984305,
            0.0,
        ],
        [
            0.19686123586912352,
            0.7384665799954104,
            0.1713682811899705,
            0.0,
        ],
        [
            -0.11564828238824053,
            -0.3011036955892888,
            -1.4785219903674274,
            0.0,
        ],
        [
            -0.7198442083947086,
            -0.4606387709597875,
            1.0571222262189157,
            0.0,
        ],
        [
            0.3436182895684614,
            -1.763040155362734,
            0.324083969394795,
            1.0,
        ],
        [
            -0.38508228041631654,
            -0.6769220003059587,
            0.6116762888408679,
            0.0,
        ],
        [
            1.030999522495951,
            0.9312801191161986,
            -0.8392175232226385,
            2.0,
        ],
        [
            -0.3092123758512146,
            0.33126343140356396,
            0.9755451271223592,
            2.0,
        ],
        [
            -0.47917423784528995,
            -0.18565897666381712,
            -1.1063349740060282,
            0.0,
        ],
        [
            -1.1962066240806708,
            0.812525822394198,
            1.356240028570823,
            2.0,
        ],
        [
            -0.07201012158033385,
            1.0035328978920242,
            0.36163602504763415,
            0.0,
        ],
        [
            -0.6451197546051243,
            0.36139560550841393,
            1.5380365664659692,
            0.0,
        ],
        [
            -0.03582603910995154,
            1.5646436558140062,
            -2.6197451040897444,
            2.0,
        ],
        [
            0.8219025043752238,
            0.08704706823817122,
            -0.29900735046586746,
            0.0,
        ],
        [
            0.0917607765355023,
            -1.9875689146008928,
            -0.21967188783751193,
            2.0,
        ],
        [
            0.3571125715117464,
            1.477894044741516,
            -0.5182702182736474,
            1.0,
        ],
        [
            -0.8084936028931876,
            -0.5017570435845365,
            0.9154021177020741,
            0.0,
        ],
        [
            0.32875110965968446,
            -0.5297602037670388,
            0.5132674331133561,
            1.0,
        ],
        [
            0.09707754934804039,
            0.9686449905328892,
            -0.7020530938773524,
            2.0,
        ],
        [
            -0.3276621465977682,
            -0.39210815313215763,
            -1.4635149481321186,
            1.0,
        ],
        [
            0.29612027706457605,
            0.26105527217988933,
            0.00511345664246089,
            1.0,
        ],
        [
            -0.23458713337514692,
            -1.4153707420504142,
            -0.42064532276535904,
            2.0,
        ],
        [
            -0.3427145165267695,
            -0.8022772692216189,
            -0.16128571166600914,
            1.0,
        ],
        [
            0.4040508568145384,
            1.8861859012105302,
            0.17457781283183896,
            1.0,
        ],
        [
            0.25755039072276437,
            -0.07444591576616721,
            -1.9187712152990415,
            2.0,
        ],
        [
            -0.026513875449216878,
            0.06023020994102644,
            2.463242112485286,
            1.0,
        ],
        [
            -0.19236096478112252,
            0.30154734233361247,
            -0.03471176970524331,
            2.0,
        ],
        [
            -1.168678037619532,
            1.1428228145150205,
            0.7519330326867741,
            1.0,
        ],
        [
            0.7910319470430469,
            -0.9093874547947389,
            1.4027943109360992,
            2.0,
        ],
        [
            -1.4018510627922809,
            0.5868570938002703,
            2.1904556258099785,
            2.0,
        ],
        [
            -0.9905363251306883,
            -0.5662977296027719,
            0.09965136508764122,
            1.0,
        ],
        [
            -0.5034756541161992,
            -1.5506634310661327,
            0.06856297480602733,
            1.0,
        ],
        [
            -1.0623037137261049,
            0.4735924306351816,
            -0.9194242342338032,
            1.0,
        ],
        [
            1.5499344050175394,
            -0.7832532923362371,
            -0.3220615162056756,
            0.0,
        ],
        [
            0.8135172173696698,
            -1.2308643164339552,
            0.22745993460412942,
            0.0,
        ],
        [
            1.307142754282428,
            -1.6074832345612275,
            0.1846338585323042,
            0.0,
        ],
        [
            0.25988279424842353,
            0.7818228717773104,
            -1.236950710878082,
            2.0,
        ],
        [
            -1.3204566130842763,
            0.5219415656168976,
            0.29698467323318606,
            1.0,
        ],
        [
            0.25049285034587654,
            0.3464482094969757,
            -0.6800247215784908,
            2.0,
        ],
        [
            0.23225369716100355,
            0.29307247329868125,
            -0.7143514180263678,
            0.0,
        ],
        [
            1.8657745111447566,
            0.4738329209117875,
            -1.1913034972026486,
            1.0,
        ],
        [
            0.6565536086338297,
            -0.9746816702273214,
            0.787084603742452,
            2.0,
        ],
        [
            1.158595579007404,
            -0.8206823183517105,
            0.9633761292443218,
            1.0,
        ],
        [
            0.4127809269364983,
            0.82206015999449,
            1.8967929826539474,
            2.0,
        ],
        [
            -0.2453881160028705,
            -0.7537361643574896,
            -0.8895144296255233,
            1.0,
        ],
        [
            -0.8158102849654383,
            -0.0771017094141042,
            0.3411519748166439,
            0.0,
        ],
        [
            0.27669079933001905,
            0.8271832490360238,
            0.01300189187790702,
            0.0,
        ],
        [
            1.4535340771573169,
            -0.2646568332379561,
            2.720169166589619,
            1.0,
        ],
    ];
    const Y_REG: [f64; 60] = [
        0.03811925886765076,
        0.8641590309404263,
        3.16732511736864,
        0.7588485624007497,
        0.6389508779687005,
        -0.545066226081783,
        0.49603336496116945,
        -0.010483805422123684,
        0.31771944465980456,
        -0.7857437489277116,
        -0.6692132877594155,
        0.515278040668028,
        0.7231637659420154,
        0.9810757707850637,
        -0.9384812547864019,
        -1.8311809582179908,
        1.6053167557468733,
        -1.2389168644790973,
        2.240993603995749,
        1.3298273335085566,
        -1.0935515894770635,
        0.5350773835794986,
        0.2367386373176101,
        0.17106211502828017,
        -0.1905385628078804,
        1.045772831012514,
        1.425167121860921,
        -0.046509719996791876,
        -1.5928834704320216,
        -0.14839317861939272,
        1.4961062521630426,
        -0.7971543541847015,
        1.5103033715258203,
        1.5993729049211827,
        -0.5811246745534691,
        0.2929150397391648,
        1.007999202679365,
        0.5927975350724798,
        1.538917734292161,
        -0.841955491465082,
        1.6005901964580418,
        0.5303549870114198,
        -1.5147826794290418,
        1.0471617996689453,
        0.7451678532997887,
        0.846708687274535,
        1.434320247973667,
        2.455539668176807,
        2.0089399080566124,
        0.2721542826775316,
        2.173348984106143,
        1.105744997572576,
        3.2512100802008743,
        1.4611461826075187,
        1.0782776451333713,
        2.084705694646516,
        -0.4026699644730901,
        -1.0003693590093332,
        0.7753466945191796,
        1.2995234054563156,
    ];
    const Y_CLS: [f64; 60] = [
        0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
        1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
        1.0, 2.0, 0.0, 0.0, 0.0, 2.0,
    ];

    fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((60, 4), |(i, j)| X[i][j]);
        (
            x,
            Array1::from_vec(Y_REG.to_vec()),
            Array1::from_vec(Y_CLS.to_vec()),
        )
    }

    fn assert_close(a: &Array1<f64>, b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-6, "got {a:?}, expected {b:?}");
        }
    }

    #[test]
    fn test_mutual_info_regression_matches_sklearn() {
        let (x, y, _) = data();
        let x3 = x.slice(ndarray::s![.., 0..3]).to_owned();
        // mutual_info_regression(X[:, :3], y, random_state=0)
        let mi = mutual_info_regression(&x3, &y, &DiscreteFeatures::Auto, 3, 0).unwrap();
        assert_close(&mi, &REG_CONT);
        // Independent of the noise seed for tie-free data.
        let mi = mutual_info_regression(&x3, &y, &DiscreteFeatures::Auto, 3, 99).unwrap();
        assert_close(&mi, &REG_CONT);
        // discrete_features=[F, F, F, T]
        let mask = DiscreteFeatures::Mask(vec![false, false, false, true]);
        let mi = mutual_info_regression(&x, &y, &mask, 3, 0).unwrap();
        assert_close(&mi, &REG_MASK);
        let idx = DiscreteFeatures::Indices(vec![3]);
        assert_close(
            &mutual_info_regression(&x, &y, &idx, 3, 0).unwrap(),
            &REG_MASK,
        );
    }

    #[test]
    fn test_mutual_info_classif_matches_sklearn() {
        let (x, _, y) = data();
        let x3 = x.slice(ndarray::s![.., 0..3]).to_owned();
        let mi = mutual_info_classif(&x3, &y, &DiscreteFeatures::Auto, 3, 0).unwrap();
        assert_close(&mi, &CLS_CONT);
        let mi = mutual_info_classif(&x3, &y, &DiscreteFeatures::Auto, 5, 0).unwrap();
        assert_close(&mi, &CLS_K5);
        // Discrete feature vs discrete target -> contingency-table MI.
        let mask = DiscreteFeatures::Mask(vec![false, false, false, true]);
        let mi = mutual_info_classif(&x, &y, &mask, 3, 0).unwrap();
        assert_close(&mi, &CLS_MASK);
    }

    #[test]
    fn test_selector_regression_target_ranks_like_sklearn() {
        // Regression for #53: a continuous target used to be treated as one
        // class per value, so MI ~ entropy of the binned feature and the
        // ranking was meaningless. sklearn ranks x0 > x1 > x2.
        let (x, y, _) = data();
        let x3 = x.slice(ndarray::s![.., 0..3]).to_owned();
        let fitted = Fit::<f64>::fit(&MutualInformationSelector::new(2), &x3, &y).unwrap();
        assert_close(fitted.mi_scores(), &REG_CONT);
        assert_eq!(fitted.selected_indices(), &[0, 1]);
        // Auto target on class labels -> mutual_info_classif.
        let (_, _, yc) = data();
        let fitted = Fit::<f64>::fit(&MutualInformationSelector::new(1), &x3, &yc).unwrap();
        assert_close(fitted.mi_scores(), &CLS_CONT);
        assert_eq!(fitted.selected_indices(), &[2]);
    }

    const REG_CONT: [f64; 3] = [0.3414745264233807, 0.2534131479242796, 0.017775651653259672];

    const REG_MASK: [f64; 4] = [
        0.3414745264233807,
        0.2534131479242796,
        0.017775651653259672,
        0.21512157325514303,
    ];

    const CLS_CONT: [f64; 3] = [0.14108264162545714, 0.0, 0.1852122915035328];

    const CLS_MASK: [f64; 4] = [
        0.14108264162545714,
        0.0,
        0.1852122915035328,
        0.17177493360282864,
    ];

    const CLS_K5: [f64; 3] = [0.12942231980646746, 0.0, 0.18717972222096346];

    #[test]
    fn test_selects_informative_feature_over_noise() {
        // Feature 0: perfectly predicts the class (0->class0, 1->class1).
        // Feature 1: random noise, uncorrelated with class.
        let x = array![
            [0.0, 0.5],
            [0.0, 0.8],
            [0.0, 0.2],
            [0.0, 0.9],
            [1.0, 0.3],
            [1.0, 0.7],
            [1.0, 0.1],
            [1.0, 0.6],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let selector = MutualInformationSelector::new(1);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();

        // Feature 0 should be selected (it perfectly separates the classes).
        assert_eq!(fitted.selected_indices(), &[0]);

        // MI of feature 0 should be substantially larger than feature 1.
        assert!(
            fitted.mi_scores()[0] > fitted.mi_scores()[1],
            "informative feature MI ({}) should be > noise MI ({})",
            fitted.mi_scores()[0],
            fitted.mi_scores()[1]
        );
    }

    #[test]
    fn test_scores_are_non_negative() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = array![0.0, 1.0, 0.0, 1.0];

        let selector = MutualInformationSelector::new(2);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();

        for (i, &score) in fitted.mi_scores().iter().enumerate() {
            assert!(
                score >= 0.0,
                "MI score for feature {} is negative: {}",
                i,
                score
            );
        }
    }

    #[test]
    fn test_transform_outputs_correct_shape() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ];
        let y = array![0.0, 1.0, 0.0];

        let selector = MutualInformationSelector::new(2);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();
        let result = fitted.transform(&x).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_selects_all_when_k_equals_n_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0.0, 1.0, 0.0];

        let selector = MutualInformationSelector::new(2);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();

        assert_eq!(fitted.selected_indices(), &[0, 1]);
    }

    #[test]
    fn test_shape_mismatch_x_y() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 2.0]; // 3 labels for 2 samples

        let selector = MutualInformationSelector::new(1);
        let result = Fit::<f64>::fit(&selector, &x, &y);

        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::ShapeMismatch(msg) => {
                assert!(msg.contains("samples"), "unexpected message: {}", msg);
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_error_on_empty_input() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<f64>::zeros(0);

        let selector = MutualInformationSelector::new(1);
        let result = Fit::<f64>::fit(&selector, &x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_error_n_features_to_select_exceeds_n_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let selector = MutualInformationSelector::new(5); // only 2 features
        let result = Fit::<f64>::fit(&selector, &x, &y);

        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("n_features_to_select"),
                    "unexpected message: {}",
                    msg
                );
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = array![0.0, 1.0];

        let selector = MutualInformationSelector::new(1);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();

        let wrong = array![[1.0, 2.0]]; // 2 cols instead of 3
        assert!(fitted.transform(&wrong).is_err());
    }

    #[test]
    fn test_works_with_f32() {
        let x: Array2<f32> = array![[0.0_f32, 0.5], [0.0, 0.8], [1.0, 0.3], [1.0, 0.7],];
        let y: Array1<f32> = array![0.0_f32, 0.0, 1.0, 1.0];

        let selector = MutualInformationSelector::new(1);
        let fitted = Fit::<f32>::fit(&selector, &x, &y).unwrap();

        assert_eq!(fitted.selected_indices().len(), 1);
        let result = fitted.transform(&x).unwrap();
        assert_eq!(result.ncols(), 1);
    }

    #[test]
    fn test_multiclass_labels() {
        // Feature 0 has 3 bins matching 3 classes; feature 1 is constant.
        let x = array![
            [0.0, 5.0],
            [0.0, 5.0],
            [0.5, 5.0],
            [0.5, 5.0],
            [1.0, 5.0],
            [1.0, 5.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let selector = MutualInformationSelector::new(1);
        let fitted = Fit::<f64>::fit(&selector, &x, &y).unwrap();

        // Feature 0 should be selected (feature 1 has zero MI since it's constant).
        assert_eq!(fitted.selected_indices(), &[0]);
    }
}
