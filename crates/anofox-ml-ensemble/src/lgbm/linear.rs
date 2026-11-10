//! Linear trees (#208): LightGBM's `linear_tree`, a ridge model per leaf on
//! the numeric features of its path (`LinearTreeLearner::CalculateLinear`).

use super::*;

/// LightGBM's `kZeroThreshold`: smaller coefficients are dropped.
const ZERO_THRESHOLD: f64 = 1e-35;

impl TreeNode {
    /// Output of one row: the leaf value times `lr`, or a linear leaf's
    /// (already shrunk) model on the row's raw values, `raw(f)`; a missing
    /// value in one of its features falls back to the leaf value.
    pub(super) fn predict_linear(&self, bins: &[u8], raw: impl Fn(usize) -> f64, lr: f64) -> f64 {
        let mut node = self;
        loop {
            match node {
                TreeNode::Leaf { value } => return lr * value,
                TreeNode::LinearLeaf {
                    value,
                    constant,
                    features,
                    coeffs,
                } => {
                    let mut out = *constant;
                    for (&f, &c) in features.iter().zip(coeffs) {
                        let v = raw(f);
                        if v.is_nan() {
                            return lr * value;
                        }
                        out += c * v;
                    }
                    return out;
                }
                TreeNode::Internal {
                    feature,
                    bin_threshold,
                    missing_left,
                    categorical_order,
                    categorical_left,
                    left,
                    right,
                } => {
                    let mask = categorical_order.as_ref().map(|order| {
                        categorical_left
                            .get_or_init(|| categorical_left_mask(order, *bin_threshold))
                            .as_slice()
                    });
                    node = if goes_left(bins[*feature], *bin_threshold, *missing_left, mask) {
                        left
                    } else {
                        right
                    };
                }
            }
        }
    }

    /// Whether any leaf is linear.
    pub(super) fn is_linear(&self) -> bool {
        match self {
            TreeNode::Leaf { .. } => false,
            TreeNode::LinearLeaf { .. } => true,
            TreeNode::Internal { left, right, .. } => left.is_linear() || right.is_linear(),
        }
    }

    /// The split features of the internal nodes.
    fn split_features(&self, out: &mut Vec<usize>) {
        if let TreeNode::Internal {
            feature,
            left,
            right,
            ..
        } = self
        {
            out.push(*feature);
            left.split_features(out);
            right.split_features(out);
        }
    }

    /// The split features on the path of `bins`.
    fn path_features(&self, bins: &[u8], path: &mut Vec<usize>) {
        if let TreeNode::Internal {
            feature,
            bin_threshold,
            missing_left,
            categorical_order,
            left,
            right,
            ..
        } = self
        {
            path.push(*feature);
            let mask = categorical_order
                .as_ref()
                .map(|order| categorical_left_mask(order, *bin_threshold));
            let next = if goes_left(
                bins[*feature],
                *bin_threshold,
                *missing_left,
                mask.as_deref(),
            ) {
                left
            } else {
                right
            };
            next.path_features(bins, path);
        }
    }

    /// The leaf `bins` reaches.
    fn leaf_mut(&mut self, bins: &[u8]) -> &mut TreeNode {
        match self {
            TreeNode::Internal {
                feature,
                bin_threshold,
                missing_left,
                categorical_order,
                left,
                right,
                ..
            } => {
                let mask = categorical_order
                    .as_ref()
                    .map(|order| categorical_left_mask(order, *bin_threshold));
                if goes_left(
                    bins[*feature],
                    *bin_threshold,
                    *missing_left,
                    mask.as_deref(),
                ) {
                    left.leaf_mut(bins)
                } else {
                    right.leaf_mut(bins)
                }
            }
            leaf => leaf,
        }
    }
}

/// The raw feature values linear leaves are fitted on (LightGBM keeps them
/// as float32), column-major, and which numeric columns have a NaN.
pub(super) struct LinearData<'a> {
    pub raw: &'a Array2<f32>,
    pub categorical: &'a [bool],
    pub has_nan: &'a [bool],
    pub lambda: f64,
}

/// Turn the leaves of `tree` into linear leaves (LightGBM's
/// `CalculateLinear`, not a refit): each fits
/// `-(XᵀHX + λ)⁻¹ Xᵀg` on its rows, `X` holding the numeric features of the
/// leaf's path and a constant. The model is stored shrunk by `lr`.
pub(super) fn fit_linear_leaves(
    tree: &mut TreeNode,
    leaf_rows: &[Vec<u32>],
    binned_rows: &Array2<u8>,
    data: &LinearData<'_>,
    gradients: &[f64],
    hessians: &[f64],
    lr: f64,
) {
    // LightGBM's HAS_NAN: a split feature of the tree has a missing value.
    let mut split = Vec::new();
    tree.split_features(&mut split);
    let has_nan = split
        .iter()
        .any(|&f| !data.categorical[f] && data.has_nan[f]);

    // Each leaf's sums run over its rows in order (as LightGBM's one-thread
    // pass); the leaves are independent.
    use rayon::prelude::*;
    let fits: Vec<Option<(Vec<usize>, Option<Vec<f64>>)>> = leaf_rows
        .par_iter()
        .map(|rows| {
            let &first = rows.first()?;
            let mut path = Vec::new();
            tree.path_features(row_of(binned_rows, first as usize), &mut path);
            path.sort_unstable();
            path.dedup();
            path.retain(|&f| !data.categorical[f]);
            let coeffs = leaf_coefficients(rows, &path, data, has_nan, gradients, hessians);
            Some((path, coeffs))
        })
        .collect();

    for (rows, fit) in leaf_rows.iter().zip(fits) {
        let Some((path, coeffs)) = fit else { continue };
        let leaf = tree.leaf_mut(row_of(binned_rows, rows[0] as usize));
        let TreeNode::Leaf { value } = *leaf else {
            continue;
        };
        // Too few values for the model: a constant leaf.
        let Some(coeffs) = coeffs else { continue };
        let k = path.len();
        let (mut features, mut kept) = (Vec::new(), Vec::new());
        for (&f, &c) in path.iter().zip(&coeffs[..k]) {
            if !(-ZERO_THRESHOLD..=ZERO_THRESHOLD).contains(&c) {
                features.push(f);
                kept.push(round_to_zero(c * lr));
            }
        }
        *leaf = TreeNode::LinearLeaf {
            value,
            constant: round_to_zero(coeffs[k] * lr),
            features,
            coeffs: kept,
        };
    }
}

/// `scores[i] +=` the output of `tree` on training row `i` (its raw
/// float32 values; LightGBM's `AddPredictionToScore` of linear trees).
pub(super) fn add_linear_scores(
    scores: &mut [f64],
    tree: &TreeNode,
    binned_rows: &Array2<u8>,
    raw: &Array2<f32>,
    lr: f64,
) {
    use rayon::prelude::*;
    scores
        .par_iter_mut()
        .with_min_len(4096)
        .enumerate()
        .for_each(|(i, s)| {
            *s += tree.predict_linear(row_of(binned_rows, i), |f| f64::from(raw[[i, f]]), lr);
        });
}

/// LightGBM's `MaybeRoundToZero`.
fn round_to_zero(v: f64) -> f64 {
    if v.abs() > ZERO_THRESHOLD {
        v
    } else {
        0.0
    }
}

/// The coefficients (features, then the constant) of one leaf, or `None`
/// when it has fewer values than coefficients.
fn leaf_coefficients(
    rows: &[u32],
    path: &[usize],
    data: &LinearData<'_>,
    has_nan: bool,
    gradients: &[f64],
    hessians: &[f64],
) -> Option<Vec<f64>> {
    let k = path.len();
    let m = k + 1;
    let mut xthx = vec![0.0f64; m * (m + 1) / 2];
    let mut xtg = vec![0.0f64; m];
    let cols: Vec<ArrayView1<f32>> = path.iter().map(|&f| data.raw.column(f)).collect();
    let mut row = vec![0.0f32; m];
    let mut nonzero = 0usize;
    'rows: for &i in rows {
        let i = i as usize;
        for (r, col) in row.iter_mut().zip(&cols) {
            let v = col[i];
            if has_nan {
                if v.is_nan() {
                    continue 'rows;
                }
                nonzero += 1;
            }
            *r = v;
        }
        row[k] = 1.0;
        let h = hessians[i] as f32;
        let g = gradients[i] as f32;
        let mut j = 0;
        for f1 in 0..m {
            let mut v1 = f64::from(row[f1]);
            xtg[f1] += v1 * f64::from(g);
            v1 *= f64::from(h);
            for &v2 in &row[f1..m] {
                xthx[j] += v1 * f64::from(v2);
                j += 1;
            }
        }
    }
    if !has_nan {
        nonzero = rows.len();
    }
    if nonzero < m {
        return None;
    }
    let mut a = vec![0.0f64; m * m];
    let mut j = 0;
    for f1 in 0..m {
        for f2 in f1..m {
            a[f1 * m + f2] = xthx[j];
            a[f2 * m + f1] = xthx[j];
            if f1 == f2 && f1 < k {
                a[f1 * m + f2] += data.lambda;
            }
            j += 1;
        }
    }
    let inv = full_piv_lu_inverse(&a, m);
    Some(
        (0..m)
            .map(|r| -(0..m).map(|c| inv[r * m + c] * xtg[c]).sum::<f64>())
            .collect(),
    )
}

/// The inverse of the row-major `m × m` matrix `a` through LU with full
/// pivoting (Eigen's `FullPivLU::inverse`, which LightGBM uses). As Eigen,
/// it solves on the numerical rank only: pivots up to
/// `max pivot · ε · m` are dropped, and their entries of the result are 0
/// (a feature constant in a leaf then gets no effect of its own).
fn full_piv_lu_inverse(a: &[f64], m: usize) -> Vec<f64> {
    let mut lu = a.to_vec();
    let at = |r: usize, c: usize| r * m + c;
    let mut row_t = vec![0usize; m];
    let mut col_t = vec![0usize; m];
    let mut pivots = m;
    let mut max_pivot = 0.0f64;
    for k in 0..m {
        // The largest |entry| of the remaining corner, first in column-major
        // order (Eigen's `maxCoeff`).
        let (mut best, mut br, mut bc) = (-1.0f64, k, k);
        for c in k..m {
            for r in k..m {
                let v = lu[at(r, c)].abs();
                if v > best {
                    (best, br, bc) = (v, r, c);
                }
            }
        }
        if best == 0.0 {
            pivots = k;
            for t in k..m {
                row_t[t] = t;
                col_t[t] = t;
            }
            break;
        }
        max_pivot = max_pivot.max(best);
        row_t[k] = br;
        col_t[k] = bc;
        if br != k {
            for c in 0..m {
                lu.swap(at(k, c), at(br, c));
            }
        }
        if bc != k {
            for r in 0..m {
                lu.swap(at(r, k), at(r, bc));
            }
        }
        let p = lu[at(k, k)];
        for r in k + 1..m {
            lu[at(r, k)] /= p;
        }
        for c in k + 1..m {
            let u = lu[at(k, c)];
            for r in k + 1..m {
                lu[at(r, c)] -= lu[at(r, k)] * u;
            }
        }
    }
    // Eigen's `rank()`.
    let threshold = max_pivot * f64::EPSILON * m as f64;
    let pivots = (0..pivots)
        .filter(|&k| lu[at(k, k)].abs() > threshold)
        .count();
    // P: the row transpositions applied to the identity's rows.
    let mut perm_p: Vec<usize> = (0..m).collect();
    for k in 0..m {
        perm_p.swap(k, row_t[k]);
    }
    let mut perm_q: Vec<usize> = (0..m).collect();
    for k in 0..m {
        perm_q.swap(k, col_t[k]);
    }
    let mut inv = vec![0.0f64; m * m];
    for col in 0..m {
        // c = P e_col
        let mut c: Vec<f64> = (0..m)
            .map(|r| f64::from(u8::from(perm_p[r] == col)))
            .collect();
        // Unit lower solve.
        for r in 0..m {
            for j in 0..r {
                c[r] -= lu[at(r, j)] * c[j];
            }
        }
        // Upper solve on the pivots.
        for r in (0..pivots).rev() {
            for j in r + 1..pivots {
                c[r] -= lu[at(r, j)] * c[j];
            }
            c[r] /= lu[at(r, r)];
        }
        for (i, &q) in perm_q.iter().enumerate() {
            inv[at(q, col)] = if i < pivots { c[i] } else { 0.0 };
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_pivot_inverse_inverts() {
        let a = [4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0];
        let inv = full_piv_lu_inverse(&a, 3);
        for r in 0..3 {
            for c in 0..3 {
                let v: f64 = (0..3).map(|k| a[r * 3 + k] * inv[k * 3 + c]).sum();
                assert!((v - f64::from(u8::from(r == c))).abs() < 1e-14);
            }
        }
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    fn trend() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((300, 2), |(i, j)| {
            if j == 0 {
                i as f64 / 100.0
            } else {
                ((i * 7) % 11) as f64
            }
        });
        let y = x.column(0).mapv(|v| 3.0 * v);
        (x, y)
    }

    #[test]
    fn linear_leaves_extrapolate_a_trend() {
        let (x, y) = trend();
        let m = LgbmRegressor::new()
            .with_n_estimators(50)
            .with_linear_tree(true, 0.0);
        let fitted = m.fit(&x, &y).unwrap();
        let beyond = ndarray::array![[4.0, 1.0], [5.0, 1.0]];
        let p = fitted.predict(&beyond).unwrap();
        // Constant leaves stop at the last value (~9); linear ones go on.
        assert!(p[1] > p[0] + 1.0 && p[0] > 9.5, "{p}");
        let plain = LgbmRegressor::new()
            .with_n_estimators(50)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(
            plain.predict(&beyond).unwrap()[0],
            plain.predict(&beyond).unwrap()[1]
        );

        // Saved and loaded, and from a streamed f32 dataset.
        let bytes = anofox_ml_core::persistence::to_binary_bytes(&fitted).unwrap();
        let back: FittedLgbmRegressor =
            anofox_ml_core::persistence::from_binary_bytes(&bytes).unwrap();
        assert_eq!(back.predict(&beyond).unwrap(), p);
        let x32 = x.mapv(|v| v as f32);
        let mut b = m.dataset_builder(&x32, 300).unwrap();
        b.push_rows(&x32).unwrap();
        let streamed = m
            .fit_dataset(&b.finish().unwrap(), &y, &Default::default())
            .unwrap();
        assert_eq!(streamed.predict(&x).unwrap(), fitted.predict(&x).unwrap());
    }

    #[test]
    fn linear_tree_rejects_dart_l1_and_foreign_datasets() {
        let (x, y) = trend();
        let lin = LgbmRegressor::new().with_linear_tree(true, 0.0);
        assert!(lin
            .clone()
            .with_boosting_type(BoostingType::Dart)
            .fit(&x, &y)
            .is_err());
        assert!(lin
            .clone()
            .with_objective(LgbmObjective::RegressionL1)
            .fit(&x, &y)
            .is_err());
        let data = LgbmRegressor::new().dataset(&x).unwrap();
        assert!(lin.fit_dataset(&data, &y, &Default::default()).is_err());
    }
}
