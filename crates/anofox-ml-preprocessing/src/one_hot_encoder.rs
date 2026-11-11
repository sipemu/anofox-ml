//! One-hot encoding of categorical features (`sklearn.preprocessing.OneHotEncoder`).

use std::cmp::Ordering;

use anofox_ml_core::{CsrMatrix, Result, RustMlError};
use ndarray::Array2;

/// A value that can be used as a category by [`OneHotEncoder`].
///
/// Categories are stored sorted (like sklearn's `categories_`) and looked
/// up by binary search, so only a total order is required. Implemented for
/// the integer types, `String`, `bool` and the floats (`f32`/`f64` use a
/// NaN-aware total order: NaN is a category of its own, sorted last, and
/// `-0.0 == 0.0`).
pub trait Category: Clone + std::fmt::Debug {
    /// Total order used to sort and look up categories.
    fn cmp_category(&self, other: &Self) -> Ordering;

    /// Whether the value is a missing value (NaN for the floats), which
    /// [`OrdinalEncoder`](crate::OrdinalEncoder) encodes apart.
    fn is_missing(&self) -> bool {
        false
    }
}

macro_rules! impl_category_ord {
    ($($t:ty),*) => {
        $(impl Category for $t {
            fn cmp_category(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        })*
    };
}
impl_category_ord!(usize, u8, u16, u32, u64, isize, i8, i16, i32, i64, bool, String);

impl Category for f64 {
    fn cmp_category(&self, other: &Self) -> Ordering {
        crate::quantile::total_cmp(self, other)
    }

    fn is_missing(&self) -> bool {
        self.is_nan()
    }
}

impl Category for f32 {
    fn cmp_category(&self, other: &Self) -> Ordering {
        crate::quantile::total_cmp(self, other)
    }

    fn is_missing(&self) -> bool {
        self.is_nan()
    }
}

/// What to do with a category at transform time that was not seen in fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum HandleUnknown {
    /// Return an error (sklearn default).
    #[default]
    Error,
    /// Encode the unknown category as an all-zero block.
    Ignore,
    /// Map unknown categories to the infrequent column if the feature has
    /// one; otherwise encode as an all-zero block.
    InfrequentIfExist,
}

/// Which category column (if any) to drop per feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum OneHotDrop {
    /// Keep all columns (sklearn default).
    #[default]
    None,
    /// Drop the first output column of every feature (after infrequent
    /// grouping), e.g. to avoid collinearity in unregularised linear models.
    First,
    /// Drop the first column only for features with exactly two output
    /// columns (after infrequent grouping).
    IfBinary,
}

/// Minimum frequency below which a category is grouped into the infrequent
/// bucket.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MinFrequency {
    /// Absolute count: categories seen fewer than this many times are
    /// infrequent.
    Count(usize),
    /// Fraction of the number of samples, in `(0, 1]`.
    Fraction(f64),
}

/// One-hot encoder for categorical features.
///
/// Mirrors `sklearn.preprocessing.OneHotEncoder` for dense input:
///
/// * the actual sorted categories of each column are stored
///   ([`FittedOneHotEncoder::categories`]), so a column `{0, 1000}` yields
///   two output columns, not 1001;
/// * `handle_unknown` = `Error` / `Ignore` / `InfrequentIfExist`;
/// * `drop` = `None` / `First` / `IfBinary`;
/// * `min_frequency` / `max_categories` group rare categories into a single
///   "infrequent" column (placed last within the feature's block);
/// * [`FittedOneHotEncoder::inverse_transform`] and a sparse
///   [`FittedOneHotEncoder::transform_sparse`] output.
///
/// The category type is generic ([`Category`]): `usize` codes, `String`s,
/// integers or floats all work.
///
/// ```text
/// [0]    -> [1, 0, 0]
/// [5]    -> [0, 1, 0]
/// [1000] -> [0, 0, 1]
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OneHotEncoder {
    /// Policy for categories not seen during fit.
    #[serde(default)]
    pub handle_unknown: HandleUnknown,
    /// Column-dropping policy.
    #[serde(default)]
    pub drop: OneHotDrop,
    /// Group categories rarer than this into the infrequent bucket.
    #[serde(default)]
    pub min_frequency: Option<MinFrequency>,
    /// Upper limit on the number of output columns per feature (including
    /// the infrequent column); the least frequent categories are grouped.
    #[serde(default)]
    pub max_categories: Option<usize>,
}

impl OneHotEncoder {
    /// Create a new `OneHotEncoder` (sklearn defaults: error on unknown, no
    /// drop, no infrequent grouping).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the unknown-category policy.
    pub fn with_handle_unknown(mut self, handle_unknown: HandleUnknown) -> Self {
        self.handle_unknown = handle_unknown;
        self
    }

    /// Set the column-dropping policy.
    pub fn with_drop(mut self, drop: OneHotDrop) -> Self {
        self.drop = drop;
        self
    }

    /// Group categories rarer than `min_frequency` into the infrequent bucket.
    pub fn with_min_frequency(mut self, min_frequency: MinFrequency) -> Self {
        self.min_frequency = Some(min_frequency);
        self
    }

    /// Limit the number of output columns per feature.
    pub fn with_max_categories(mut self, max_categories: usize) -> Self {
        self.max_categories = Some(max_categories);
        self
    }

    /// Fit the encoder: learns the sorted categories of each column, the
    /// infrequent categories and the dropped column.
    pub fn fit<T: Category>(&self, x: &Array2<T>) -> Result<FittedOneHotEncoder<T>> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if let Some(MinFrequency::Count(0)) = self.min_frequency {
            return Err(RustMlError::InvalidParameter(
                "min_frequency count must be at least 1".into(),
            ));
        }
        if let Some(MinFrequency::Fraction(f)) = self.min_frequency {
            if !(f > 0.0 && f <= 1.0) {
                return Err(RustMlError::InvalidParameter(
                    "min_frequency fraction must be in (0, 1]".into(),
                ));
            }
        }
        if self.max_categories == Some(0) {
            return Err(RustMlError::InvalidParameter(
                "max_categories must be at least 1".into(),
            ));
        }

        let n_samples = x.nrows();
        let ncols = x.ncols();
        let mut columns = Vec::with_capacity(ncols);
        for j in 0..ncols {
            let mut values: Vec<T> = x.column(j).iter().cloned().collect();
            values.sort_by(|a, b| a.cmp_category(b));
            // Unique sorted categories with counts.
            let mut cats: Vec<T> = Vec::new();
            let mut counts: Vec<usize> = Vec::new();
            for v in values {
                match cats.last() {
                    Some(last) if last.cmp_category(&v) == Ordering::Equal => {
                        *counts.last_mut().unwrap() += 1;
                    }
                    _ => {
                        cats.push(v);
                        counts.push(1);
                    }
                }
            }
            let infrequent = self.identify_infrequent(&counts, n_samples);
            columns.push(ColumnEncoding::new(cats, infrequent, self.drop));
        }

        Ok(FittedOneHotEncoder {
            columns,
            handle_unknown: self.handle_unknown,
        })
    }

    /// Port of sklearn's `_identify_infrequent`.
    fn identify_infrequent(&self, counts: &[usize], n_samples: usize) -> Vec<bool> {
        let mut mask: Vec<bool> = match self.min_frequency {
            Some(MinFrequency::Count(m)) => counts.iter().map(|&c| c < m).collect(),
            Some(MinFrequency::Fraction(f)) => {
                let abs = n_samples as f64 * f;
                counts.iter().map(|&c| (c as f64) < abs).collect()
            }
            None => vec![false; counts.len()],
        };
        let n_infrequent = mask.iter().filter(|&&b| b).count();
        let n_current = counts.len() - n_infrequent + 1;
        if let Some(max_cat) = self.max_categories {
            if max_cat < n_current {
                let frequent_count = max_cat - 1;
                if frequent_count == 0 {
                    mask.iter_mut().for_each(|b| *b = true);
                } else {
                    // np.argsort(counts, kind="mergesort")[:-frequent_count]
                    let mut order: Vec<usize> = (0..counts.len()).collect();
                    order.sort_by_key(|&i| counts[i]); // stable
                    for &i in &order[..counts.len() - frequent_count] {
                        mask[i] = true;
                    }
                }
            }
        }
        mask
    }
}

/// Per-feature encoding state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ColumnEncoding<T> {
    /// Sorted unique categories seen in fit.
    categories: Vec<T>,
    /// `infrequent[k]` — category `k` is grouped into the infrequent bucket.
    infrequent: Vec<bool>,
    /// Output slot of each category before dropping (frequent categories in
    /// sorted order, then the infrequent bucket).
    slot_of: Vec<usize>,
    /// Number of slots before dropping.
    n_slots: usize,
    /// Slot index of the infrequent bucket, if any.
    infrequent_slot: Option<usize>,
    /// Dropped slot, if any.
    drop_slot: Option<usize>,
}

impl<T: Category> ColumnEncoding<T> {
    fn new(categories: Vec<T>, infrequent: Vec<bool>, drop: OneHotDrop) -> Self {
        let n_frequent = infrequent.iter().filter(|&&b| !b).count();
        let has_infrequent = n_frequent < categories.len();
        let mut slot_of = Vec::with_capacity(categories.len());
        let mut next = 0;
        for &inf in &infrequent {
            if inf {
                slot_of.push(n_frequent);
            } else {
                slot_of.push(next);
                next += 1;
            }
        }
        let n_slots = n_frequent + usize::from(has_infrequent);
        let drop_slot = match drop {
            OneHotDrop::None => None,
            OneHotDrop::First => Some(0),
            OneHotDrop::IfBinary => (n_slots == 2).then_some(0),
        };
        Self {
            categories,
            infrequent,
            slot_of,
            n_slots,
            infrequent_slot: has_infrequent.then_some(n_frequent),
            drop_slot,
        }
    }

    fn n_outputs(&self) -> usize {
        self.n_slots - usize::from(self.drop_slot.is_some())
    }

    fn find(&self, v: &T) -> Option<usize> {
        self.categories.binary_search_by(|c| c.cmp_category(v)).ok()
    }

    /// Output column (within this feature's block) for a slot, or `None`
    /// for the dropped slot.
    fn output_of_slot(&self, slot: usize) -> Option<usize> {
        match self.drop_slot {
            Some(d) if slot == d => None,
            Some(d) if slot > d => Some(slot - 1),
            _ => Some(slot),
        }
    }

    fn slot_of_output(&self, out: usize) -> usize {
        match self.drop_slot {
            Some(d) if out >= d => out + 1,
            _ => out,
        }
    }

    /// The category represented by a slot (`None` for the infrequent bucket).
    fn category_of_slot(&self, slot: usize) -> Option<T> {
        if Some(slot) == self.infrequent_slot {
            return None;
        }
        self.slot_of
            .iter()
            .zip(&self.infrequent)
            .position(|(&s, &inf)| s == slot && !inf)
            .map(|k| self.categories[k].clone())
    }
}

/// Fitted OneHotEncoder — holds the sorted categories per column plus the
/// infrequent / drop bookkeeping.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedOneHotEncoder<T = usize> {
    columns: Vec<ColumnEncoding<T>>,
    handle_unknown: HandleUnknown,
}

impl<T: Category> FittedOneHotEncoder<T> {
    fn check_cols(&self, n: usize) -> Result<()> {
        if n != self.columns.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} columns, got {}",
                self.columns.len(),
                n
            )));
        }
        Ok(())
    }

    /// For every row, the list of active output columns (global indices).
    fn active_columns(&self, x: &Array2<T>) -> Result<Vec<Vec<usize>>> {
        self.check_cols(x.ncols())?;
        let offsets = self.offsets();
        let mut rows = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            let mut active = Vec::with_capacity(x.ncols());
            for (j, enc) in self.columns.iter().enumerate() {
                let v = &x[[i, j]];
                let slot = match enc.find(v) {
                    Some(k) => Some(enc.slot_of[k]),
                    None => match self.handle_unknown {
                        HandleUnknown::Error => {
                            return Err(RustMlError::InvalidParameter(format!(
                                "found unknown category {:?} in column {} during transform",
                                v, j
                            )));
                        }
                        HandleUnknown::Ignore => None,
                        HandleUnknown::InfrequentIfExist => enc.infrequent_slot,
                    },
                };
                if let Some(out) = slot.and_then(|s| enc.output_of_slot(s)) {
                    active.push(offsets[j] + out);
                }
            }
            rows.push(active);
        }
        Ok(rows)
    }

    fn offsets(&self) -> Vec<usize> {
        let mut acc = 0;
        self.columns
            .iter()
            .map(|c| {
                let o = acc;
                acc += c.n_outputs();
                o
            })
            .collect()
    }

    /// Transform categorical data into a dense one-hot matrix.
    pub fn transform(&self, x: &Array2<T>) -> Result<Array2<f64>> {
        let rows = self.active_columns(x)?;
        let mut result = Array2::<f64>::zeros((x.nrows(), self.n_output_features()));
        for (i, active) in rows.iter().enumerate() {
            for &c in active {
                result[[i, c]] = 1.0;
            }
        }
        Ok(result)
    }

    /// Transform categorical data into a sparse (CSR) one-hot matrix.
    pub fn transform_sparse(&self, x: &Array2<T>) -> Result<CsrMatrix<f64>> {
        let rows = self.active_columns(x)?;
        let triplets = rows
            .iter()
            .enumerate()
            .flat_map(|(i, a)| a.iter().map(move |&c| (i, c, 1.0)))
            .collect();
        Ok(CsrMatrix::from_triplets(
            x.nrows(),
            self.n_output_features(),
            triplets,
        ))
    }

    /// Map one-hot rows back to categories.
    ///
    /// Within each feature's block the arg-max column is decoded. An
    /// all-zero block decodes to the dropped category (if a column was
    /// dropped) or to `None` (an unknown category under
    /// `HandleUnknown::Ignore`). The infrequent column decodes to `None`
    /// (sklearn returns the placeholder `"infrequent_sklearn"`).
    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<Option<T>>> {
        if x.ncols() != self.n_output_features() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} one-hot columns, got {}",
                self.n_output_features(),
                x.ncols()
            )));
        }
        let offsets = self.offsets();
        let mut out = Array2::<Option<T>>::from_elem((x.nrows(), self.columns.len()), None);
        for i in 0..x.nrows() {
            for (j, enc) in self.columns.iter().enumerate() {
                let m = enc.n_outputs();
                let mut best: Option<usize> = None;
                for k in 0..m {
                    let v = x[[i, offsets[j] + k]];
                    if v != 0.0 && best.is_none_or(|b| v > x[[i, offsets[j] + b]]) {
                        best = Some(k);
                    }
                }
                let slot = match best {
                    Some(k) => Some(enc.slot_of_output(k)),
                    None => enc.drop_slot,
                };
                out[[i, j]] = slot.and_then(|s| enc.category_of_slot(s));
            }
        }
        Ok(out)
    }

    /// The sorted categories seen in fit, per column (sklearn `categories_`).
    pub fn categories(&self) -> Vec<&[T]> {
        self.columns
            .iter()
            .map(|c| c.categories.as_slice())
            .collect()
    }

    /// Number of categories seen in fit, per column.
    pub fn n_categories(&self) -> Vec<usize> {
        self.columns.iter().map(|c| c.categories.len()).collect()
    }

    /// The infrequent categories per column (sklearn
    /// `infrequent_categories_`); empty when the column has none.
    pub fn infrequent_categories(&self) -> Vec<Vec<T>> {
        self.columns
            .iter()
            .map(|c| {
                c.categories
                    .iter()
                    .zip(&c.infrequent)
                    .filter(|(_, &inf)| inf)
                    .map(|(v, _)| v.clone())
                    .collect()
            })
            .collect()
    }

    /// Index into [`categories`](Self::categories) of the dropped category
    /// per column (sklearn `drop_idx_`). For a dropped infrequent bucket
    /// this is the index of the first infrequent category.
    pub fn drop_idx(&self) -> Vec<Option<usize>> {
        self.columns
            .iter()
            .map(|c| {
                c.drop_slot
                    .and_then(|d| c.slot_of.iter().position(|&s| s == d))
            })
            .collect()
    }

    /// Number of output columns produced per input column.
    pub fn n_outputs_per_feature(&self) -> Vec<usize> {
        self.columns.iter().map(|c| c.n_outputs()).collect()
    }

    /// Return the total number of output columns after one-hot encoding.
    pub fn n_output_features(&self) -> usize {
        self.columns.iter().map(|c| c.n_outputs()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn s(v: &str) -> String {
        v.to_string()
    }

    #[test]
    fn test_single_column() {
        let x = array![[0usize], [1], [2]];
        let encoder = OneHotEncoder::new();
        let fitted = encoder.fit(&x).unwrap();
        let encoded = fitted.transform(&x).unwrap();

        assert_eq!(encoded.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(encoded[[i, j]], if i == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn test_sparse_codes_use_actual_categories() {
        // Regression for #54: categories used to be `max + 1`, so a column
        // {0, 5, 1000} produced 1001 output columns.
        // sklearn: OneHotEncoder().fit([[0],[1000],[0],[5]]).categories_
        //   -> [array([0, 5, 1000])], 3 output columns.
        let x = array![[0usize], [1000], [0], [5], [0], [0], [1000], [0]];
        let fitted = OneHotEncoder::new().fit(&x).unwrap();
        assert_eq!(fitted.categories(), vec![&[0usize, 5, 1000][..]]);
        let t = fitted.transform(&x).unwrap();
        assert_eq!(t.shape(), &[8, 3]);
        assert_eq!(t.row(1).to_vec(), vec![0.0, 0.0, 1.0]);
        assert_eq!(t.row(3).to_vec(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_multiple_columns() {
        let x = array![[0usize, 2], [1, 0], [0, 1]];
        let fitted = OneHotEncoder::new().fit(&x).unwrap();
        let encoded = fitted.transform(&x).unwrap();
        assert_eq!(encoded.shape(), &[3, 5]);
        assert_eq!(fitted.n_output_features(), 5);
        assert_eq!(encoded.row(0).to_vec(), vec![1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(encoded.row(1).to_vec(), vec![0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_handle_unknown() {
        let x = array![[0usize, 1], [1, 2], [1, 3]];
        // Error (default)
        let fitted = OneHotEncoder::new().fit(&x).unwrap();
        assert!(fitted.transform(&array![[5usize, 1]]).is_err());
        // Ignore -> all-zero block; sklearn:
        //   OneHotEncoder(handle_unknown='ignore').fit(X).transform([[2, 1]])
        //   -> [[0, 0, 1, 0, 0]], inverse -> [[None, 1]]
        let fitted = OneHotEncoder::new()
            .with_handle_unknown(HandleUnknown::Ignore)
            .fit(&x)
            .unwrap();
        let t = fitted.transform(&array![[2usize, 1]]).unwrap();
        assert_eq!(t.row(0).to_vec(), vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        let inv = fitted.inverse_transform(&t).unwrap();
        assert_eq!(inv[[0, 0]], None);
        assert_eq!(inv[[0, 1]], Some(1));
    }

    #[test]
    fn test_drop_if_binary_and_first() {
        // sklearn: OneHotEncoder(drop='if_binary').fit([[0,1],[1,2],[1,3]])
        //   feature names ['x0_1', 'x1_1', 'x1_2', 'x1_3']
        let x = array![[0usize, 1], [1, 2], [1, 3]];
        let fitted = OneHotEncoder::new()
            .with_drop(OneHotDrop::IfBinary)
            .fit(&x)
            .unwrap();
        assert_eq!(fitted.n_outputs_per_feature(), vec![1, 3]);
        let t = fitted.transform(&x).unwrap();
        assert_eq!(
            t,
            array![
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0]
            ]
        );
        assert_eq!(fitted.drop_idx(), vec![Some(0), None]);
        let inv = fitted.inverse_transform(&t).unwrap();
        assert_eq!(inv[[0, 0]], Some(0));
        assert_eq!(inv[[2, 1]], Some(3));

        let fitted = OneHotEncoder::new()
            .with_drop(OneHotDrop::First)
            .fit(&x)
            .unwrap();
        assert_eq!(fitted.n_output_features(), 3);
        let t = fitted.transform(&x).unwrap();
        assert_eq!(t.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
        let inv = fitted.inverse_transform(&t).unwrap();
        assert_eq!(inv[[0, 0]], Some(0));
        assert_eq!(inv[[0, 1]], Some(1));
    }

    fn strings() -> Array2<String> {
        array![
            [s("a")],
            [s("b")],
            [s("b")],
            [s("c")],
            [s("a")],
            [s("b")],
            [s("d")],
            [s("b")],
            [s("e")]
        ]
    }

    #[test]
    fn test_min_frequency_infrequent_if_exist_strings() {
        // sklearn: OneHotEncoder(min_frequency=2,
        //   handle_unknown='infrequent_if_exist').fit(X)
        //   infrequent_categories_ -> ['c', 'd', 'e'],
        //   names ['x0_a', 'x0_b', 'x0_infrequent_sklearn'],
        //   transform([['a'], ['z'], ['c']]) -> [[1,0,0],[0,0,1],[0,0,1]]
        let fitted = OneHotEncoder::new()
            .with_min_frequency(MinFrequency::Count(2))
            .with_handle_unknown(HandleUnknown::InfrequentIfExist)
            .fit(&strings())
            .unwrap();
        assert_eq!(
            fitted.infrequent_categories(),
            vec![vec![s("c"), s("d"), s("e")]]
        );
        assert_eq!(fitted.n_output_features(), 3);
        let t = fitted
            .transform(&array![[s("a")], [s("z")], [s("c")]])
            .unwrap();
        assert_eq!(t, array![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]);
        let inv = fitted.inverse_transform(&t).unwrap();
        assert_eq!(inv[[0, 0]], Some(s("a")));
        assert_eq!(inv[[1, 0]], None);
        // Fraction: 2/9 of the samples.
        let frac = OneHotEncoder::new()
            .with_min_frequency(MinFrequency::Fraction(0.2))
            .fit(&strings())
            .unwrap();
        assert_eq!(
            frac.infrequent_categories(),
            vec![vec![s("c"), s("d"), s("e")]]
        );
    }

    #[test]
    fn test_max_categories_with_drop_first() {
        // sklearn: OneHotEncoder(max_categories=3, drop='first').fit(X)
        //   names ['x0_b', 'x0_infrequent_sklearn'], drop_idx_ [0],
        //   transform([['a'],['b'],['d']]) -> [[0,0],[1,0],[0,1]],
        //   inverse([[0,0],[1,0],[0,1]]) -> [['a'],['b'],['infrequent_sklearn']]
        let fitted = OneHotEncoder::new()
            .with_max_categories(3)
            .with_drop(OneHotDrop::First)
            .fit(&strings())
            .unwrap();
        assert_eq!(fitted.drop_idx(), vec![Some(0)]);
        let t = fitted
            .transform(&array![[s("a")], [s("b")], [s("d")]])
            .unwrap();
        assert_eq!(t, array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let inv = fitted.inverse_transform(&t).unwrap();
        assert_eq!(inv[[0, 0]], Some(s("a")));
        assert_eq!(inv[[1, 0]], Some(s("b")));
        assert_eq!(inv[[2, 0]], None);
    }

    #[test]
    fn test_max_categories_ties_follow_sklearn_stable_argsort() {
        // sklearn: OneHotEncoder(max_categories=3).fit(
        //   [['a'],['a'],['b'],['b'],['c'],['c']]).infrequent_categories_
        //   -> [array(['a'])]  (stable argsort marks the first tie)
        let x = array![[s("a")], [s("a")], [s("b")], [s("b")], [s("c")], [s("c")]];
        let fitted = OneHotEncoder::new().with_max_categories(3).fit(&x).unwrap();
        assert_eq!(fitted.infrequent_categories(), vec![vec![s("a")]]);
    }

    #[test]
    fn test_float_categories_and_sparse_output() {
        let x = array![[1.5, f64::NAN], [-2.0, 3.0], [1.5, 3.0]];
        let fitted = OneHotEncoder::new().fit(&x).unwrap();
        assert_eq!(fitted.n_categories(), vec![2, 2]);
        let dense = fitted.transform(&x).unwrap();
        // NaN is its own category, sorted last.
        assert_eq!(dense.row(0).to_vec(), vec![0.0, 1.0, 0.0, 1.0]);
        let sparse = fitted.transform_sparse(&x).unwrap();
        assert_eq!(sparse.nnz(), 6);
        assert_eq!(sparse.to_dense(), dense);
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<usize> = Array2::zeros((0, 0));
        assert!(OneHotEncoder::new().fit(&x).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let fitted = OneHotEncoder::new()
            .fit(&array![[0usize, 1], [1, 0]])
            .unwrap();
        assert!(fitted.transform(&array![[0usize, 1, 2]]).is_err());
        assert!(fitted.inverse_transform(&array![[1.0]]).is_err());
    }

    #[test]
    fn test_all_zeros() {
        let x = array![[0usize, 0], [0, 0], [0, 0]];
        let fitted = OneHotEncoder::new().fit(&x).unwrap();
        let encoded = fitted.transform(&x).unwrap();
        assert_eq!(encoded.shape(), &[3, 2]);
        assert!(encoded.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_row_sums() {
        let x = array![[0usize, 2, 1], [2, 0, 0], [1, 1, 2]];
        let fitted = OneHotEncoder::new().fit(&x).unwrap();
        let encoded = fitted.transform(&x).unwrap();
        assert_eq!(encoded.shape(), &[3, 9]);
        for i in 0..3 {
            assert_abs_diff_eq!(encoded.row(i).sum(), 3.0, epsilon = 1e-10);
        }
        let inv = fitted.inverse_transform(&encoded).unwrap();
        assert_eq!(inv.mapv(|v| v.unwrap()), x);
    }

    #[test]
    fn test_default() {
        let fitted = OneHotEncoder::default()
            .fit(&array![[0usize], [1]])
            .unwrap();
        assert_eq!(fitted.n_categories(), vec![2]);
    }
}
