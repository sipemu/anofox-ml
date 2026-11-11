use std::cmp::Ordering;

use anofox_ml_core::{Result, RustMlError};
use ndarray::{Array2, ArrayView2, AsArray, Ix2};

use crate::one_hot_encoder::Category;

/// NaN as JSON `null` (JSON has no NaN), for the encoded values.
mod nan_as_null {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() && v.is_nan() {
            s.serialize_none()
        } else {
            s.serialize_f64(*v)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        if d.is_human_readable() {
            Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
        } else {
            f64::deserialize(d)
        }
    }
}

/// What [`OrdinalEncoder`] does with categories not seen during fit.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum OrdinalHandleUnknown {
    /// Return an error (sklearn default).
    #[default]
    Error,
    /// Encode unknown categories as this value (sklearn
    /// `handle_unknown="use_encoded_value", unknown_value=...`). May be NaN
    /// or negative (e.g. `-1.0`) when using
    /// [`FittedOrdinalEncoder::transform_f64`]; the `usize` API
    /// [`FittedOrdinalEncoder::transform`] requires a non-negative integer.
    UseEncodedValue(#[serde(with = "nan_as_null")] f64),
}

fn default_missing() -> f64 {
    f64::NAN
}

/// Encodes categories as ordinal values per column
/// (`sklearn.preprocessing.OrdinalEncoder`).
///
/// Takes a list of columns (each a `Vec<T>` of any [`Category`]: strings,
/// integers, floats) and maps each unique category to its index in the
/// sorted categories. A missing value (NaN) seen in fit is a category of
/// its own, sorted last, that [`transform_f64`](FittedOrdinalEncoder::transform_f64)
/// encodes as `encoded_missing_value`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrdinalEncoder {
    /// Unknown-category policy.
    #[serde(default)]
    pub handle_unknown: OrdinalHandleUnknown,
    /// Code of the missing values (sklearn's `encoded_missing_value`,
    /// default NaN).
    #[serde(default = "default_missing", with = "nan_as_null")]
    pub encoded_missing_value: f64,
}

impl Default for OrdinalEncoder {
    fn default() -> Self {
        Self {
            handle_unknown: OrdinalHandleUnknown::Error,
            encoded_missing_value: f64::NAN,
        }
    }
}

/// Whether `v` is one of the codes `0..cardinality`.
fn is_known_code(v: f64, cardinality: usize) -> bool {
    v >= 0.0 && v < cardinality as f64
}

impl OrdinalEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_handle_unknown(mut self, handle_unknown: OrdinalHandleUnknown) -> Self {
        self.handle_unknown = handle_unknown;
        self
    }

    pub fn with_encoded_missing_value(mut self, value: f64) -> Self {
        self.encoded_missing_value = value;
        self
    }

    pub fn fit<T: Category>(&self, columns: &[Vec<T>]) -> Result<FittedOrdinalEncoder<T>> {
        if columns.is_empty() {
            return Err(RustMlError::EmptyInput("columns slice is empty".into()));
        }

        let nrows = columns[0].len();
        if nrows == 0 {
            return Err(RustMlError::EmptyInput("columns contain no rows".into()));
        }

        for (j, col) in columns.iter().enumerate() {
            if col.len() != nrows {
                return Err(RustMlError::ShapeMismatch(format!(
                    "column {} has {} rows, expected {}",
                    j,
                    col.len(),
                    nrows
                )));
            }
        }

        let vocabularies: Vec<Vec<T>> = columns
            .iter()
            .map(|col| {
                let mut vocab = col.clone();
                vocab.sort_by(|a, b| sort_key(a, b));
                vocab.dedup_by(|a, b| sort_key(a, b) == Ordering::Equal);
                vocab
            })
            .collect();
        let fitted = FittedOrdinalEncoder {
            vocabularies,
            handle_unknown: self.handle_unknown,
            encoded_missing_value: self.encoded_missing_value,
        };

        // sklearn: the missing category is not part of the cardinality.
        if let OrdinalHandleUnknown::UseEncodedValue(v) = self.handle_unknown {
            if !v.is_nan() {
                if v.fract() != 0.0 {
                    return Err(RustMlError::InvalidParameter(format!(
                        "unknown_value must be an integer or NaN, got {v}"
                    )));
                }
                for j in 0..fitted.vocabularies.len() {
                    if is_known_code(v, fitted.cardinality(j)) {
                        return Err(RustMlError::InvalidParameter(format!(
                            "unknown_value {v} is already used to encode a known category \
                             in column {j}"
                        )));
                    }
                }
            }
        }
        let m = self.encoded_missing_value;
        if !m.is_nan() {
            for j in 0..fitted.vocabularies.len() {
                if fitted.has_missing(j) && is_known_code(m, fitted.cardinality(j)) {
                    return Err(RustMlError::InvalidParameter(format!(
                        "encoded_missing_value ({m}) is already used to encode a known \
                         category in column {j}"
                    )));
                }
            }
        }
        Ok(fitted)
    }
}

/// Category order with the missing values last.
fn sort_key<T: Category>(a: &T, b: &T) -> Ordering {
    match (a.is_missing(), b.is_missing()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.cmp_category(b),
    }
}

/// A category's place in one column of a fitted encoder.
enum Code {
    Known(usize),
    Missing,
    Unknown,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedOrdinalEncoder<T = String> {
    /// Sorted categories per column (sklearn's `categories_`), the missing
    /// value last when seen in fit.
    vocabularies: Vec<Vec<T>>,
    #[serde(default)]
    handle_unknown: OrdinalHandleUnknown,
    #[serde(default = "default_missing", with = "nan_as_null")]
    encoded_missing_value: f64,
}

impl<T: Category> FittedOrdinalEncoder<T> {
    fn has_missing(&self, j: usize) -> bool {
        self.vocabularies[j]
            .last()
            .is_some_and(Category::is_missing)
    }

    /// The categories of column `j` that get codes (all but the missing one).
    fn cardinality(&self, j: usize) -> usize {
        self.vocabularies[j].len() - usize::from(self.has_missing(j))
    }

    fn code(&self, j: usize, value: &T) -> Code {
        if value.is_missing() {
            return if self.has_missing(j) {
                Code::Missing
            } else {
                Code::Unknown
            };
        }
        let known = &self.vocabularies[j][..self.cardinality(j)];
        match known.binary_search_by(|c| c.cmp_category(value)) {
            Ok(i) => Code::Known(i),
            Err(_) => Code::Unknown,
        }
    }

    fn check_columns(&self, n: usize) -> Result<()> {
        if n != self.vocabularies.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} columns, got {}",
                self.vocabularies.len(),
                n
            )));
        }
        Ok(())
    }

    fn unknown_error(&self, val: &T, j: usize) -> RustMlError {
        RustMlError::InvalidParameter(format!("unknown category '{val:?}' in column {j}"))
    }

    /// Codes as `usize`; a missing value or an unknown value that is not a
    /// non-negative integer is an error (use
    /// [`transform_f64`](Self::transform_f64)).
    pub fn transform(&self, columns: &[Vec<T>]) -> Result<Vec<Vec<usize>>> {
        self.check_columns(columns.len())?;
        let mut result = Vec::with_capacity(columns.len());
        for (j, col) in columns.iter().enumerate() {
            let mut encoded = Vec::with_capacity(col.len());
            for val in col {
                match (self.code(j, val), self.handle_unknown) {
                    (Code::Known(idx), _) => encoded.push(idx),
                    (Code::Missing, _) => {
                        return Err(RustMlError::InvalidParameter(format!(
                            "missing value in column {j} is not representable as usize, \
                             use transform_f64"
                        )));
                    }
                    (Code::Unknown, OrdinalHandleUnknown::UseEncodedValue(u))
                        if u >= 0.0 && u.fract() == 0.0 =>
                    {
                        encoded.push(u as usize)
                    }
                    (Code::Unknown, OrdinalHandleUnknown::UseEncodedValue(u)) => {
                        return Err(RustMlError::InvalidParameter(format!(
                            "unknown category '{val:?}' in column {j}: unknown_value {u} is \
                             not representable as usize, use transform_f64"
                        )));
                    }
                    (Code::Unknown, OrdinalHandleUnknown::Error) => {
                        return Err(self.unknown_error(val, j));
                    }
                }
            }
            result.push(encoded);
        }
        Ok(result)
    }

    /// Codes as a `(n_rows, n_columns)` matrix: the missing values as
    /// `encoded_missing_value`, unknown ones as `unknown_value`.
    pub fn transform_f64(&self, columns: &[Vec<T>]) -> Result<Array2<f64>> {
        self.check_columns(columns.len())?;
        let nrows = columns.first().map_or(0, |c| c.len());
        let mut out = Array2::<f64>::zeros((nrows, columns.len()));
        for (j, col) in columns.iter().enumerate() {
            if col.len() != nrows {
                return Err(RustMlError::ShapeMismatch(format!(
                    "column {} has {} rows, expected {}",
                    j,
                    col.len(),
                    nrows
                )));
            }
            for (i, val) in col.iter().enumerate() {
                out[[i, j]] = match (self.code(j, val), self.handle_unknown) {
                    (Code::Known(idx), _) => idx as f64,
                    (Code::Missing, _) => self.encoded_missing_value,
                    (Code::Unknown, OrdinalHandleUnknown::UseEncodedValue(u)) => u,
                    (Code::Unknown, OrdinalHandleUnknown::Error) => {
                        return Err(self.unknown_error(val, j));
                    }
                };
            }
        }
        Ok(out)
    }

    /// Inverse of [`transform_f64`](Self::transform_f64): codes back to
    /// columns; the `unknown_value` code decodes to `None`, the
    /// `encoded_missing_value` code to the missing category.
    pub fn inverse_transform_f64<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<Vec<Vec<Option<T>>>> {
        let x: ArrayView2<f64> = x.into();
        self.check_columns(x.ncols())?;
        let unknown = match self.handle_unknown {
            OrdinalHandleUnknown::UseEncodedValue(u) => Some(u),
            OrdinalHandleUnknown::Error => None,
        };
        let same = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || a == b;
        let mut result = Vec::with_capacity(x.ncols());
        for (j, vocab) in self.vocabularies.iter().enumerate() {
            let cardinality = self.cardinality(j);
            let mut decoded = Vec::with_capacity(x.nrows());
            for i in 0..x.nrows() {
                let v = x[[i, j]];
                if self.has_missing(j) && same(v, self.encoded_missing_value) {
                    decoded.push(vocab.last().cloned());
                } else if unknown.is_some_and(|u| same(u, v)) {
                    decoded.push(None);
                } else if v.fract() == 0.0 && is_known_code(v, cardinality) {
                    decoded.push(Some(vocab[v as usize].clone()));
                } else {
                    return Err(RustMlError::InvalidParameter(format!(
                        "encoded value {} is out of range for column {} (vocabulary size {})",
                        v, j, cardinality
                    )));
                }
            }
            result.push(decoded);
        }
        Ok(result)
    }

    /// Inverse-transform ordinal-encoded columns back to category columns.
    pub fn inverse_transform(&self, columns: &[Vec<usize>]) -> Result<Vec<Vec<T>>> {
        self.check_columns(columns.len())?;
        let mut result = Vec::with_capacity(columns.len());
        for (j, col) in columns.iter().enumerate() {
            let vocab = &self.vocabularies[j];
            let cardinality = self.cardinality(j);
            let mut decoded = Vec::with_capacity(col.len());
            for &idx in col {
                if idx >= cardinality {
                    return Err(RustMlError::InvalidParameter(format!(
                        "encoded index {} is out of range for column {} (vocabulary size {})",
                        idx, j, cardinality
                    )));
                }
                decoded.push(vocab[idx].clone());
            }
            result.push(decoded);
        }
        Ok(result)
    }

    /// The sorted categories of a column (the missing value last, when
    /// seen in fit).
    pub fn vocabulary(&self, column: usize) -> Option<&[T]> {
        self.vocabularies.get(column).map(|v| v.as_slice())
    }

    /// Return the number of columns.
    pub fn n_columns(&self) -> usize {
        self.vocabularies.len()
    }

    /// Return the number of categories per column (the missing one included).
    pub fn n_categories(&self) -> Vec<usize> {
        self.vocabularies.iter().map(|v| v.len()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(val: &str) -> String {
        val.to_string()
    }

    #[test]
    fn test_fit_transform_single_column() {
        let columns = vec![vec![s("cat"), s("dog"), s("cat"), s("bird")]];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();
        let encoded = fitted.transform(&columns).unwrap();

        // Sorted vocab: ["bird", "cat", "dog"] -> [0, 1, 2]
        assert_eq!(encoded, vec![vec![1, 2, 1, 0]]);
    }

    #[test]
    fn test_fit_transform_multiple_columns() {
        let columns = vec![
            vec![s("red"), s("blue"), s("green")],
            vec![s("small"), s("large"), s("small")],
        ];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();
        let encoded = fitted.transform(&columns).unwrap();

        // Col 0 vocab: ["blue", "green", "red"] -> [0, 1, 2]
        // Col 1 vocab: ["large", "small"] -> [0, 1]
        assert_eq!(encoded[0], vec![2, 0, 1]);
        assert_eq!(encoded[1], vec![1, 0, 1]);
    }

    #[test]
    fn test_use_encoded_value_for_unknown() {
        // sklearn: OrdinalEncoder(handle_unknown='use_encoded_value',
        //   unknown_value=-1).fit([['a'],['b']]).transform([['b'],['z']])
        //   -> [[1.], [-1.]]
        let train = vec![vec![s("a"), s("b")]];
        let test = vec![vec![s("b"), s("z")]];
        let enc =
            OrdinalEncoder::new().with_handle_unknown(OrdinalHandleUnknown::UseEncodedValue(-1.0));
        let fitted = enc.fit(&train).unwrap();
        let t = fitted.transform_f64(&test).unwrap();
        assert_eq!(t, ndarray::array![[1.0], [-1.0]]);
        let inv = fitted.inverse_transform_f64(&t).unwrap();
        assert_eq!(inv, vec![vec![Some(s("b")), None]]);
        // -1 is not a usize code.
        assert!(fitted.transform(&test).is_err());

        // NaN unknown value.
        let fitted = OrdinalEncoder::new()
            .with_handle_unknown(OrdinalHandleUnknown::UseEncodedValue(f64::NAN))
            .fit(&train)
            .unwrap();
        assert!(fitted.transform_f64(&test).unwrap()[[1, 0]].is_nan());

        // Non-negative unknown value works with the usize API.
        let fitted = OrdinalEncoder::new()
            .with_handle_unknown(OrdinalHandleUnknown::UseEncodedValue(5.0))
            .fit(&train)
            .unwrap();
        assert_eq!(fitted.transform(&test).unwrap(), vec![vec![1, 5]]);

        // An unknown value colliding with a known code is rejected (sklearn).
        let bad =
            OrdinalEncoder::new().with_handle_unknown(OrdinalHandleUnknown::UseEncodedValue(1.0));
        assert!(bad.fit(&train).is_err());
    }

    #[test]
    fn test_numeric_with_missing_matches_sklearn() {
        // sklearn: OrdinalEncoder(handle_unknown='use_encoded_value',
        //   unknown_value=-1, encoded_missing_value=-2)
        //   .fit([[10,2],[2,nan],[10,3],[nan,2]])
        let nan = f64::NAN;
        let train = vec![vec![10.0, 2.0, 10.0, nan], vec![2.0, nan, 3.0, 2.0]];
        let fitted = OrdinalEncoder::new()
            .with_handle_unknown(OrdinalHandleUnknown::UseEncodedValue(-1.0))
            .with_encoded_missing_value(-2.0)
            .fit(&train)
            .unwrap();
        // categories_ == [[2, 10, nan], [2, 3, nan]]
        let v0 = fitted.vocabulary(0).unwrap();
        assert_eq!(&v0[..2], &[2.0, 10.0]);
        assert!(v0[2].is_nan());
        // transform([[2,3],[5,nan],[nan,7]]) -> [[0,1],[-1,-2],[-2,-1]]
        let test = vec![vec![2.0, 5.0, nan], vec![3.0, nan, 7.0]];
        let t = fitted.transform_f64(&test).unwrap();
        assert_eq!(t, ndarray::array![[0.0, 1.0], [-1.0, -2.0], [-2.0, -1.0]]);
        // inverse_transform([[0,1],[-1,-2],[-2,0]]) -> [[2,3],[None,nan],[nan,2]]
        let inv = fitted
            .inverse_transform_f64(&ndarray::array![[0.0, 1.0], [-1.0, -2.0], [-2.0, 0.0]])
            .unwrap();
        assert_eq!(inv[0][..2], [Some(2.0), None]);
        assert!(inv[0][2].unwrap().is_nan());
        assert!(inv[1][1].unwrap().is_nan());
        assert_eq!((inv[1][0], inv[1][2]), (Some(3.0), Some(2.0)));
        // NaN not seen in fit is unknown.
        let fitted = OrdinalEncoder::new().fit(&[vec![1.0, 2.0]]).unwrap();
        assert!(fitted.transform_f64(&[vec![nan]]).is_err());
        // A missing code that collides with a known one is rejected.
        let bad = OrdinalEncoder::new().with_encoded_missing_value(0.0);
        assert!(bad.fit(&[vec![1.0, nan]]).is_err());
        assert!(bad.fit(&[vec![1.0, 2.0]]).is_ok());
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let columns = vec![
            vec![s("apple"), s("banana"), s("cherry")],
            vec![s("x"), s("y"), s("z")],
        ];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();
        let encoded = fitted.transform(&columns).unwrap();
        let recovered = fitted.inverse_transform(&encoded).unwrap();

        assert_eq!(recovered, columns);
    }

    #[test]
    fn test_unknown_category() {
        let columns = vec![vec![s("cat"), s("dog")]];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();

        let unknown = vec![vec![s("fish")]];
        assert!(fitted.transform(&unknown).is_err());
    }

    #[test]
    fn test_out_of_range_index() {
        let columns = vec![vec![s("a"), s("b")]];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();

        let bad = vec![vec![99]];
        assert!(fitted.inverse_transform(&bad).is_err());
    }

    #[test]
    fn test_empty_columns() {
        let columns: Vec<Vec<String>> = vec![];
        let encoder = OrdinalEncoder::new();
        assert!(encoder.fit(&columns).is_err());
    }

    #[test]
    fn test_empty_rows() {
        let columns: Vec<Vec<String>> = vec![vec![]];
        let encoder = OrdinalEncoder::new();
        assert!(encoder.fit(&columns).is_err());
    }

    #[test]
    fn test_column_length_mismatch() {
        let columns = vec![vec![s("a"), s("b")], vec![s("x")]];
        let encoder = OrdinalEncoder::new();
        assert!(encoder.fit(&columns).is_err());
    }

    #[test]
    fn test_shape_mismatch_transform() {
        let columns = vec![vec![s("a"), s("b")]];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();

        // Wrong number of columns
        let wrong = vec![vec![s("a")], vec![s("b")]];
        assert!(fitted.transform(&wrong).is_err());
    }

    #[test]
    fn test_shape_mismatch_inverse() {
        let columns = vec![vec![s("a"), s("b")]];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();

        let wrong = vec![vec![0], vec![1]];
        assert!(fitted.inverse_transform(&wrong).is_err());
    }

    #[test]
    fn test_vocabulary_accessor() {
        let columns = vec![
            vec![s("z"), s("a"), s("m")],
            vec![s("big"), s("small"), s("big")],
        ];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();

        assert_eq!(fitted.vocabulary(0).unwrap(), &[s("a"), s("m"), s("z")]);
        assert_eq!(fitted.vocabulary(1).unwrap(), &[s("big"), s("small")]);
        assert!(fitted.vocabulary(5).is_none());
    }

    #[test]
    fn test_n_categories() {
        let columns = vec![vec![s("a"), s("b"), s("c")], vec![s("x"), s("y"), s("x")]];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();

        assert_eq!(fitted.n_columns(), 2);
        assert_eq!(fitted.n_categories(), vec![3, 2]);
    }

    #[test]
    fn test_default() {
        let encoder = OrdinalEncoder::default();
        let columns = vec![vec![s("a")]];
        let fitted = encoder.fit(&columns).unwrap();
        assert_eq!(fitted.n_columns(), 1);
    }

    #[test]
    fn test_sorted_vocabulary() {
        let columns = vec![vec![s("zebra"), s("apple"), s("mango")]];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();

        assert_eq!(
            fitted.vocabulary(0).unwrap(),
            &[s("apple"), s("mango"), s("zebra")]
        );
    }

    #[test]
    fn test_duplicate_values() {
        let columns = vec![vec![s("a"), s("a"), s("b"), s("b"), s("a")]];
        let encoder = OrdinalEncoder::new();
        let fitted = encoder.fit(&columns).unwrap();
        let encoded = fitted.transform(&columns).unwrap();

        assert_eq!(encoded[0], vec![0, 0, 1, 1, 0]);
    }
}
