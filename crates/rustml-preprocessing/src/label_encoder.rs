use ndarray::Array1;
use rustml_core::{Result, RustMlError};
use std::collections::HashMap;

/// Encodes string labels as integer indices.
///
/// Maps each unique label to a unique integer in sorted order. This is useful
/// for converting categorical target labels to numeric form for model training.
///
/// Unlike the numeric transformers, `LabelEncoder` works on string slices
/// rather than float arrays, so it does not implement `FitUnsupervised`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LabelEncoder;

impl LabelEncoder {
    /// Create a new `LabelEncoder`.
    pub fn new() -> Self {
        Self
    }

    /// Fit the encoder on the given labels, learning the vocabulary.
    pub fn fit(&self, labels: &[String]) -> Result<FittedLabelEncoder> {
        if labels.is_empty() {
            return Err(RustMlError::EmptyInput("labels slice is empty".into()));
        }

        let mut vocab: Vec<String> = labels.iter().cloned().collect();
        vocab.sort();
        vocab.dedup();

        let label_to_index: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        Ok(FittedLabelEncoder {
            vocab,
            label_to_index,
        })
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted LabelEncoder — holds the learned vocabulary and mapping.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLabelEncoder {
    vocab: Vec<String>,
    label_to_index: HashMap<String, usize>,
}

impl FittedLabelEncoder {
    /// Transform string labels into integer-encoded values.
    pub fn transform(&self, labels: &[String]) -> Result<Array1<usize>> {
        let mut encoded = Vec::with_capacity(labels.len());
        for label in labels {
            match self.label_to_index.get(label) {
                Some(&idx) => encoded.push(idx),
                None => {
                    return Err(RustMlError::InvalidParameter(format!(
                        "unknown label: '{}'",
                        label
                    )));
                }
            }
        }
        Ok(Array1::from_vec(encoded))
    }

    /// Inverse-transform integer-encoded values back to string labels.
    pub fn inverse_transform(&self, encoded: &Array1<usize>) -> Result<Vec<String>> {
        let mut labels = Vec::with_capacity(encoded.len());
        for &idx in encoded.iter() {
            if idx >= self.vocab.len() {
                return Err(RustMlError::InvalidParameter(format!(
                    "encoded index {} is out of range (vocabulary size {})",
                    idx,
                    self.vocab.len()
                )));
            }
            labels.push(self.vocab[idx].clone());
        }
        Ok(labels)
    }

    /// Return the learned vocabulary (sorted).
    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    /// Return the number of classes.
    pub fn num_classes(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn s(val: &str) -> String {
        val.to_string()
    }

    #[test]
    fn test_fit_transform() {
        let labels = vec![s("cat"), s("dog"), s("cat"), s("bird")];
        let encoder = LabelEncoder::new();
        let fitted = encoder.fit(&labels).unwrap();
        let encoded = fitted.transform(&labels).unwrap();

        // Sorted vocab: ["bird", "cat", "dog"]
        assert_eq!(fitted.vocab(), &[s("bird"), s("cat"), s("dog")]);
        assert_eq!(encoded, array![1, 2, 1, 0]);
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let labels = vec![
            s("apple"),
            s("banana"),
            s("cherry"),
            s("banana"),
            s("apple"),
        ];
        let encoder = LabelEncoder::new();
        let fitted = encoder.fit(&labels).unwrap();
        let encoded = fitted.transform(&labels).unwrap();
        let recovered = fitted.inverse_transform(&encoded).unwrap();

        assert_eq!(recovered, labels);
    }

    #[test]
    fn test_unknown_label() {
        let labels = vec![s("cat"), s("dog")];
        let encoder = LabelEncoder::new();
        let fitted = encoder.fit(&labels).unwrap();

        let unknown = vec![s("fish")];
        assert!(fitted.transform(&unknown).is_err());
    }

    #[test]
    fn test_out_of_range_index() {
        let labels = vec![s("a"), s("b")];
        let encoder = LabelEncoder::new();
        let fitted = encoder.fit(&labels).unwrap();

        let bad_encoded = array![0, 5];
        assert!(fitted.inverse_transform(&bad_encoded).is_err());
    }

    #[test]
    fn test_empty_labels() {
        let labels: Vec<String> = vec![];
        let encoder = LabelEncoder::new();
        assert!(encoder.fit(&labels).is_err());
    }

    #[test]
    fn test_single_label() {
        let labels = vec![s("only")];
        let encoder = LabelEncoder::new();
        let fitted = encoder.fit(&labels).unwrap();
        let encoded = fitted.transform(&labels).unwrap();

        assert_eq!(encoded, array![0]);
        assert_eq!(fitted.num_classes(), 1);
    }

    #[test]
    fn test_duplicate_labels() {
        let labels = vec![s("x"), s("x"), s("x"), s("y"), s("y")];
        let encoder = LabelEncoder::new();
        let fitted = encoder.fit(&labels).unwrap();

        assert_eq!(fitted.num_classes(), 2);
        assert_eq!(fitted.vocab(), &[s("x"), s("y")]);

        let encoded = fitted.transform(&labels).unwrap();
        assert_eq!(encoded, array![0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_sorted_vocabulary() {
        let labels = vec![s("zebra"), s("apple"), s("mango"), s("banana")];
        let encoder = LabelEncoder::new();
        let fitted = encoder.fit(&labels).unwrap();

        assert_eq!(
            fitted.vocab(),
            &[s("apple"), s("banana"), s("mango"), s("zebra")]
        );
    }

    #[test]
    fn test_default() {
        let encoder = LabelEncoder::default();
        let labels = vec![s("a"), s("b")];
        let fitted = encoder.fit(&labels).unwrap();
        assert_eq!(fitted.num_classes(), 2);
    }

    #[test]
    fn test_many_classes() {
        let labels: Vec<String> = (0..100).map(|i| format!("class_{:03}", i)).collect();
        let encoder = LabelEncoder::new();
        let fitted = encoder.fit(&labels).unwrap();
        let encoded = fitted.transform(&labels).unwrap();
        let recovered = fitted.inverse_transform(&encoded).unwrap();

        assert_eq!(fitted.num_classes(), 100);
        assert_eq!(recovered, labels);
    }
}
