use rustml_core::{Result, RustMlError};
use std::collections::HashMap;

/// Encodes string categories as ordinal (integer) values per column.
///
/// Takes a list of columns (each column is a `Vec<String>`) and maps each
/// unique category to a sorted integer index. This is useful as a
/// preprocessing step before one-hot encoding or for ordinal features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrdinalEncoder;

impl OrdinalEncoder {
    /// Create a new `OrdinalEncoder`.
    pub fn new() -> Self {
        Self
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

        let mut vocabularies = Vec::with_capacity(columns.len());
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

        Ok(FittedOrdinalEncoder {
            vocabularies,
            mappings,
        })
    }
}

impl Default for OrdinalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted OrdinalEncoder — holds per-column vocabularies and mappings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedOrdinalEncoder {
    vocabularies: Vec<Vec<String>>,
    mappings: Vec<HashMap<String, usize>>,
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
                match mapping.get(val) {
                    Some(&idx) => encoded.push(idx),
                    None => {
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
                        idx, j, vocab.len()
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
        let columns = vec![
            vec![s("a"), s("b"), s("c")],
            vec![s("x"), s("y"), s("x")],
        ];
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
