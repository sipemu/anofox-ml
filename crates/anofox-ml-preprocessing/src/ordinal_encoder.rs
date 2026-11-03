use anofox_ml_core::{Result, RustMlError};
use ndarray::{Array2, ArrayView2, AsArray, Ix2};
use std::collections::HashMap;

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
    UseEncodedValue(f64),
}

/// Encodes string categories as ordinal (integer) values per column.
///
/// Takes a list of columns (each column is a `Vec<String>`) and maps each
/// unique category to a sorted integer index. This is useful as a
/// preprocessing step before one-hot encoding or for ordinal features.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OrdinalEncoder {
    /// Unknown-category policy.
    #[serde(default)]
    pub handle_unknown: OrdinalHandleUnknown,
}

impl OrdinalEncoder {
    /// Create a new `OrdinalEncoder` (errors on unknown categories).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the unknown-category policy.
    pub fn with_handle_unknown(mut self, handle_unknown: OrdinalHandleUnknown) -> Self {
        self.handle_unknown = handle_unknown;
        self
    }

    /// Fit the encoder, learning the vocabulary for each column.
    ///
    /// `columns` is a list of columns where each column is a `Vec<String>`.
    /// All columns must have the same length.
    pub fn fit(&self, columns: &[Vec<String>]) -> Result<FittedOrdinalEncoder> {
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

        let mut vocabularies: Vec<Vec<String>> = Vec::with_capacity(columns.len());
        let mut mappings = Vec::with_capacity(columns.len());

        for col in columns {
            let mut vocab: Vec<String> = col.iter().cloned().collect();
            vocab.sort();
            vocab.dedup();

            let mapping: HashMap<String, usize> = vocab
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone(), i))
                .collect();

            vocabularies.push(vocab);
            mappings.push(mapping);
        }

        if let OrdinalHandleUnknown::UseEncodedValue(v) = self.handle_unknown {
            // sklearn: unknown_value must not collide with a known code.
            if !v.is_nan() {
                if v.fract() != 0.0 {
                    return Err(RustMlError::InvalidParameter(format!(
                        "unknown_value must be an integer or NaN, got {v}"
                    )));
                }
                for (j, vocab) in vocabularies.iter().enumerate() {
                    if v >= 0.0 && v < vocab.len() as f64 {
                        return Err(RustMlError::InvalidParameter(format!(
                            "unknown_value {v} is already used to encode a known category \
                             in column {j}"
                        )));
                    }
                }
            }
        }

        Ok(FittedOrdinalEncoder {
            vocabularies,
            mappings,
            handle_unknown: self.handle_unknown,
        })
    }
}

/// Fitted OrdinalEncoder — holds per-column vocabularies and mappings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedOrdinalEncoder {
    vocabularies: Vec<Vec<String>>,
    mappings: Vec<HashMap<String, usize>>,
    #[serde(default)]
    handle_unknown: OrdinalHandleUnknown,
}

impl FittedOrdinalEncoder {
    /// Transform string columns into ordinal-encoded integer columns.
    ///
    /// Returns a `Vec<Vec<usize>>` where each inner vec is an encoded column.
    pub fn transform(&self, columns: &[Vec<String>]) -> Result<Vec<Vec<usize>>> {
        if columns.len() != self.vocabularies.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} columns, got {}",
                self.vocabularies.len(),
                columns.len()
            )));
        }

        let mut result = Vec::with_capacity(columns.len());

        for (j, col) in columns.iter().enumerate() {
            let mapping = &self.mappings[j];
            let mut encoded = Vec::with_capacity(col.len());
            for val in col {
                match (mapping.get(val), self.handle_unknown) {
                    (Some(&idx), _) => encoded.push(idx),
                    (None, OrdinalHandleUnknown::UseEncodedValue(u))
                        if u >= 0.0 && u.fract() == 0.0 =>
                    {
                        encoded.push(u as usize)
                    }
                    (None, OrdinalHandleUnknown::UseEncodedValue(u)) => {
                        return Err(RustMlError::InvalidParameter(format!(
                            "unknown category '{}' in column {}: unknown_value {} is not \
                             representable as usize, use transform_f64",
                            val, j, u
                        )));
                    }
                    (None, OrdinalHandleUnknown::Error) => {
                        return Err(RustMlError::InvalidParameter(format!(
                            "unknown category '{}' in column {}",
                            val, j
                        )));
                    }
                }
            }
            result.push(encoded);
        }

        Ok(result)
    }

    /// Transform string columns into an `(n_rows, n_columns)` float matrix
    /// (sklearn's output layout). Unknown categories are encoded as the
    /// configured `unknown_value` (which may be NaN or negative).
    pub fn transform_f64(&self, columns: &[Vec<String>]) -> Result<Array2<f64>> {
        if columns.len() != self.vocabularies.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} columns, got {}",
                self.vocabularies.len(),
                columns.len()
            )));
        }
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
                out[[i, j]] = match (self.mappings[j].get(val), self.handle_unknown) {
                    (Some(&idx), _) => idx as f64,
                    (None, OrdinalHandleUnknown::UseEncodedValue(u)) => u,
                    (None, OrdinalHandleUnknown::Error) => {
                        return Err(RustMlError::InvalidParameter(format!(
                            "unknown category '{}' in column {}",
                            val, j
                        )));
                    }
                };
            }
        }
        Ok(out)
    }

    /// Inverse of [`transform_f64`](Self::transform_f64): codes back to
    /// string columns; the `unknown_value` code decodes to `None`.
    pub fn inverse_transform_f64<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<Vec<Vec<Option<String>>>> {
        let x: ArrayView2<f64> = x.into();
        if x.ncols() != self.vocabularies.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} columns, got {}",
                self.vocabularies.len(),
                x.ncols()
            )));
        }
        let unknown = match self.handle_unknown {
            OrdinalHandleUnknown::UseEncodedValue(u) => Some(u),
            OrdinalHandleUnknown::Error => None,
        };
        let mut result = Vec::with_capacity(x.ncols());
        for (j, vocab) in self.vocabularies.iter().enumerate() {
            let mut decoded = Vec::with_capacity(x.nrows());
            for i in 0..x.nrows() {
                let v = x[[i, j]];
                let is_unknown = unknown.is_some_and(|u| (u.is_nan() && v.is_nan()) || u == v);
                if is_unknown {
                    decoded.push(None);
                } else if v >= 0.0 && v.fract() == 0.0 && (v as usize) < vocab.len() {
                    decoded.push(Some(vocab[v as usize].clone()));
                } else {
                    return Err(RustMlError::InvalidParameter(format!(
                        "encoded value {} is out of range for column {} (vocabulary size {})",
                        v,
                        j,
                        vocab.len()
                    )));
                }
            }
            result.push(decoded);
        }
        Ok(result)
    }

    /// Inverse-transform ordinal-encoded columns back to string columns.
    pub fn inverse_transform(&self, columns: &[Vec<usize>]) -> Result<Vec<Vec<String>>> {
        if columns.len() != self.vocabularies.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} columns, got {}",
                self.vocabularies.len(),
                columns.len()
            )));
        }

        let mut result = Vec::with_capacity(columns.len());

        for (j, col) in columns.iter().enumerate() {
            let vocab = &self.vocabularies[j];
            let mut decoded = Vec::with_capacity(col.len());
            for &idx in col {
                if idx >= vocab.len() {
                    return Err(RustMlError::InvalidParameter(format!(
                        "encoded index {} is out of range for column {} (vocabulary size {})",
                        idx,
                        j,
                        vocab.len()
                    )));
                }
                decoded.push(vocab[idx].clone());
            }
            result.push(decoded);
        }

        Ok(result)
    }

    /// Return the vocabulary for a specific column.
    pub fn vocabulary(&self, column: usize) -> Option<&[String]> {
        self.vocabularies.get(column).map(|v| v.as_slice())
    }

    /// Return the number of columns.
    pub fn n_columns(&self) -> usize {
        self.vocabularies.len()
    }

    /// Return the number of categories per column.
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
        let columns = vec![vec![]];
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
