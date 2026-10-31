//! Per-class and per-sample weights shared by every classifier.
//!
//! Mirrors sklearn's `class_weight` parameter and
//! `sklearn.utils.class_weight.compute_sample_weight`.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::error::{Result, RustMlError};
use crate::float::Float;
use crate::labels::encode_labels;

/// Per-class weights (sklearn's `class_weight`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClassWeight {
    /// `w_c = n_samples / (n_classes · count(c))`.
    Balanced,
    /// Explicit `(label, weight)` pairs; unlisted classes get weight 1.
    /// Labels are matched exactly.
    Manual(Vec<(f64, f64)>),
}

/// Per-sample weights implied by `class_weight` for the labels `y`
/// (sklearn's `compute_sample_weight(class_weight, y)`).
pub fn compute_sample_weight<F: Float>(
    class_weight: &ClassWeight,
    y: &Array1<F>,
) -> Result<Array1<F>> {
    let (classes, idx) = encode_labels(y)?;
    let per_class: Vec<F> = match class_weight {
        ClassWeight::Balanced => {
            let mut counts = vec![0usize; classes.len()];
            for &k in &idx {
                counts[k] += 1;
            }
            let n = F::from_usize(y.len()).unwrap();
            let k = F::from_usize(classes.len()).unwrap();
            counts
                .iter()
                .map(|&c| n / (k * F::from_usize(c).unwrap()))
                .collect()
        }
        ClassWeight::Manual(pairs) => {
            for &(label, w) in pairs {
                if !w.is_finite() || w < 0.0 {
                    return Err(RustMlError::InvalidParameter(format!(
                        "class_weight for label {label} must be finite and non-negative"
                    )));
                }
            }
            classes
                .iter()
                .map(|c| {
                    let c = c.to_f64().unwrap();
                    pairs
                        .iter()
                        .find(|(label, _)| *label + 0.0 == c)
                        .map_or(F::one(), |&(_, w)| F::from_f64(w).unwrap())
                })
                .collect()
        }
    };
    Ok(idx.into_iter().map(|k| per_class[k]).collect())
}

/// Validate optional sample weights: length `n`, finite, non-negative and
/// not all zero.
pub fn check_sample_weight<F: Float>(sample_weight: &Array1<F>, n: usize) -> Result<()> {
    if sample_weight.len() != n {
        return Err(RustMlError::ShapeMismatch(format!(
            "sample_weight has {} elements but X has {} rows",
            sample_weight.len(),
            n
        )));
    }
    if sample_weight
        .iter()
        .any(|w| !w.is_finite() || *w < F::zero())
    {
        return Err(RustMlError::InvalidParameter(
            "sample_weight must be finite and non-negative".into(),
        ));
    }
    if sample_weight.iter().fold(F::zero(), |a, &b| a + b) <= F::zero() {
        return Err(RustMlError::InvalidParameter(
            "sample_weight must not sum to zero".into(),
        ));
    }
    Ok(())
}

/// Effective per-sample weights: `sample_weight × class weights`, as sklearn
/// combines them. Returns `None` when neither is given.
pub fn effective_sample_weight<F: Float>(
    y: &Array1<F>,
    sample_weight: Option<&Array1<F>>,
    class_weight: Option<&ClassWeight>,
) -> Result<Option<Array1<F>>> {
    if let Some(w) = sample_weight {
        check_sample_weight(w, y.len())?;
    }
    Ok(match (sample_weight, class_weight) {
        (None, None) => None,
        (Some(w), None) => Some(w.clone()),
        (None, Some(cw)) => Some(compute_sample_weight(cw, y)?),
        (Some(w), Some(cw)) => Some(compute_sample_weight(cw, y)? * w),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn balanced_matches_sklearn() {
        // sklearn: compute_sample_weight('balanced', [0, 0, 0, 1, 2, 2])
        let w: Array1<f64> = compute_sample_weight(
            &ClassWeight::Balanced,
            &array![0.0, 0.0, 0.0, 1.0, 2.0, 2.0],
        )
        .unwrap();
        let e = [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0, 1.0, 1.0];
        for (a, b) in w.iter().zip(e) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn manual_defaults_to_one_and_matches_exactly() {
        let cw = ClassWeight::Manual(vec![(1.0, 3.0), (-0.0, 0.5)]);
        let w = compute_sample_weight(&cw, &array![0.0, 1.0, 2.0, 1.0 + 1e-12]).unwrap();
        assert_eq!(w.to_vec(), vec![0.5, 3.0, 1.0, 1.0]);
        assert!(
            compute_sample_weight(&ClassWeight::Manual(vec![(0.0, -1.0)]), &array![0.0]).is_err()
        );
    }

    #[test]
    fn effective_weight_multiplies() {
        let y = array![0.0, 1.0, 1.0, 1.0];
        let sw = array![2.0, 1.0, 0.0, 1.0];
        let w = effective_sample_weight(&y, Some(&sw), Some(&ClassWeight::Balanced))
            .unwrap()
            .unwrap();
        assert_eq!(w.to_vec(), vec![4.0, 2.0 / 3.0, 0.0, 2.0 / 3.0]);
        assert!(effective_sample_weight::<f64>(&y, None, None)
            .unwrap()
            .is_none());
        assert!(effective_sample_weight(&y, Some(&array![1.0]), None).is_err());
    }
}
