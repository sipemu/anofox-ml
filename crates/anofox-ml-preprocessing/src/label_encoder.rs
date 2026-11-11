use anofox_ml_core::{Result, RustMlError};
use ndarray::Array1;

use crate::one_hot_encoder::Category;

/// Encodes labels as `0..n_classes` in their sorted order
/// (`sklearn.preprocessing.LabelEncoder`). The labels are any
/// [`Category`]: strings, integers, floats.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LabelEncoder;

impl LabelEncoder {
    pub fn new() -> Self {
        Self
    }

    pub fn fit<T: Category>(&self, labels: &[T]) -> Result<FittedLabelEncoder<T>> {
        if labels.is_empty() {
            return Err(RustMlError::EmptyInput("labels slice is empty".into()));
        }
        let mut vocab = labels.to_vec();
        vocab.sort_by(|a, b| a.cmp_category(b));
        vocab.dedup_by(|a, b| a.cmp_category(b).is_eq());
        Ok(FittedLabelEncoder { vocab })
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLabelEncoder<T = String> {
    /// The sorted labels (sklearn's `classes_`).
    vocab: Vec<T>,
}

impl<T: Category> FittedLabelEncoder<T> {
    pub fn transform(&self, labels: &[T]) -> Result<Array1<usize>> {
        labels
            .iter()
            .map(|label| {
                self.vocab
                    .binary_search_by(|c| c.cmp_category(label))
                    .map_err(|_| {
                        RustMlError::InvalidParameter(format!("unknown label: '{label:?}'"))
                    })
            })
            .collect::<Result<Vec<_>>>()
            .map(Array1::from_vec)
    }

    pub fn inverse_transform(&self, encoded: &Array1<usize>) -> Result<Vec<T>> {
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

    pub fn vocab(&self) -> &[T] {
        &self.vocab
    }

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
    fn test_numeric_labels() {
        // sklearn: LabelEncoder().fit([10, 2, 2.5, 10]).classes_ == [2, 2.5, 10]
        let fitted = LabelEncoder::new().fit(&[10.0, 2.0, 2.5, 10.0]).unwrap();
        assert_eq!(fitted.vocab(), &[2.0, 2.5, 10.0]);
        assert_eq!(fitted.transform(&[2.5, 10.0]).unwrap(), array![1, 2]);
        assert!(fitted.transform(&[3.0]).is_err());
        assert_eq!(
            fitted.inverse_transform(&array![0, 2]).unwrap(),
            vec![2.0, 10.0]
        );
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
