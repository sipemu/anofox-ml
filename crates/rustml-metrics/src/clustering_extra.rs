use ndarray::Array1;
use rustml_core::{Float, Result, RustMlError};

/// Adjusted Rand Index (ARI) between two clusterings.
///
/// The ARI is defined as:
///
/// `ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)`
///
/// where the Rand Index is computed from the contingency table of the two
/// label assignments. The ARI is adjusted for chance so that random
/// clusterings have an expected score of 0.
///
/// Returns a value in `[-1, 1]`, where 1 indicates perfect agreement,
/// 0 indicates agreement equal to chance, and negative values indicate
/// agreement worse than chance.
///
/// # Errors
///
/// Returns an error if:
/// - The inputs have different lengths.
/// - The inputs are empty.
pub fn adjusted_rand_score<F: Float>(
    labels_true: &Array1<F>,
    labels_pred: &Array1<F>,
) -> Result<F> {
    check_lengths(labels_true, labels_pred)?;

    let n = labels_true.len();
    let eps = F::from_f64(1e-9).unwrap();

    // Find unique labels for true and predicted
    let classes_true = unique_sorted(labels_true);
    let classes_pred = unique_sorted(labels_pred);

    let n_true = classes_true.len();
    let n_pred = classes_pred.len();

    // Build contingency table: n_ij = number of samples in true cluster i AND pred cluster j
    let mut contingency = vec![vec![0usize; n_pred]; n_true];
    for k in 0..n {
        let i = classes_true
            .iter()
            .position(|&c| (c - labels_true[k]).abs() < eps)
            .unwrap();
        let j = classes_pred
            .iter()
            .position(|&c| (c - labels_pred[k]).abs() < eps)
            .unwrap();
        contingency[i][j] += 1;
    }

    // Compute row sums (a_i) and column sums (b_j)
    let a: Vec<usize> = (0..n_true)
        .map(|i| contingency[i].iter().sum())
        .collect();
    let b: Vec<usize> = (0..n_pred)
        .map(|j| (0..n_true).map(|i| contingency[i][j]).sum())
        .collect();

    // Helper: C(x, 2) = x * (x - 1) / 2
    let comb2 = |x: usize| -> i64 {
        if x < 2 {
            0
        } else {
            (x as i64) * (x as i64 - 1) / 2
        }
    };

    // Sum of C(n_ij, 2)
    let mut sum_comb_nij: i64 = 0;
    for i in 0..n_true {
        for j in 0..n_pred {
            sum_comb_nij += comb2(contingency[i][j]);
        }
    }

    // Sum of C(a_i, 2)
    let sum_comb_a: i64 = a.iter().map(|&ai| comb2(ai)).sum();

    // Sum of C(b_j, 2)
    let sum_comb_b: i64 = b.iter().map(|&bj| comb2(bj)).sum();

    let comb_n = comb2(n);

    // ARI = (sum_comb_nij - expected) / (mean_ab - expected)
    // where expected = sum_comb_a * sum_comb_b / C(n, 2)
    // and mean_ab = (sum_comb_a + sum_comb_b) / 2

    if comb_n == 0 {
        // Only one sample: perfect agreement by convention
        return Ok(F::one());
    }

    // Use f64 arithmetic for the intermediate computations to avoid overflow
    let sum_nij_f = sum_comb_nij as f64;
    let sum_a_f = sum_comb_a as f64;
    let sum_b_f = sum_comb_b as f64;
    let comb_n_f = comb_n as f64;

    let expected = sum_a_f * sum_b_f / comb_n_f;
    let mean_ab = (sum_a_f + sum_b_f) / 2.0;

    let denom = mean_ab - expected;

    if denom.abs() < 1e-15 {
        // Both clusterings assign all samples to a single cluster, or all
        // singletons: ARI is 1 when the clusterings match, 0 otherwise.
        // If sum_comb_nij == expected (both 0 for singletons), return 1.
        if (sum_nij_f - expected).abs() < 1e-15 {
            return Ok(F::one());
        }
        return Ok(F::zero());
    }

    let ari = (sum_nij_f - expected) / denom;

    Ok(F::from_f64(ari).unwrap())
}

/// Normalized Mutual Information (NMI) between two clusterings.
///
/// Computed as:
///
/// `NMI = 2 * MI(true, pred) / (H(true) + H(pred))`
///
/// where MI is the mutual information and H is the Shannon entropy. Both
/// are computed from the contingency table of the two label assignments.
///
/// Returns a value in `[0, 1]`, where 1 indicates perfect agreement and
/// 0 indicates no mutual information.
///
/// # Errors
///
/// Returns an error if:
/// - The inputs have different lengths.
/// - The inputs are empty.
pub fn normalized_mutual_info_score<F: Float>(
    labels_true: &Array1<F>,
    labels_pred: &Array1<F>,
) -> Result<F> {
    check_lengths(labels_true, labels_pred)?;

    let n = labels_true.len();
    let n_f = n as f64;
    let eps_label = F::from_f64(1e-9).unwrap();

    // Find unique labels
    let classes_true = unique_sorted(labels_true);
    let classes_pred = unique_sorted(labels_pred);

    let n_true = classes_true.len();
    let n_pred = classes_pred.len();

    // If either clustering has only one cluster, entropy is 0 => NMI = 0
    // (unless both have one cluster and they are identical, then NMI = 1
    //  but H(true) + H(pred) = 0 so we handle it as a special case).
    if n_true == 1 && n_pred == 1 {
        // Both are single clusters: perfect agreement.
        return Ok(F::one());
    }
    if n_true == 1 || n_pred == 1 {
        // One is a single cluster, the other is not. MI = 0, one entropy = 0
        // so NMI = 0.
        return Ok(F::zero());
    }

    // Build contingency table
    let mut contingency = vec![vec![0usize; n_pred]; n_true];
    for k in 0..n {
        let i = classes_true
            .iter()
            .position(|&c| (c - labels_true[k]).abs() < eps_label)
            .unwrap();
        let j = classes_pred
            .iter()
            .position(|&c| (c - labels_pred[k]).abs() < eps_label)
            .unwrap();
        contingency[i][j] += 1;
    }

    // Row sums and column sums
    let a: Vec<usize> = (0..n_true)
        .map(|i| contingency[i].iter().sum())
        .collect();
    let b: Vec<usize> = (0..n_pred)
        .map(|j| (0..n_true).map(|i| contingency[i][j]).sum())
        .collect();

    // H(true) = -sum_i (a_i / n) * log(a_i / n)
    let h_true: f64 = a
        .iter()
        .filter(|&&ai| ai > 0)
        .map(|&ai| {
            let p = ai as f64 / n_f;
            -p * p.ln()
        })
        .sum();

    // H(pred) = -sum_j (b_j / n) * log(b_j / n)
    let h_pred: f64 = b
        .iter()
        .filter(|&&bj| bj > 0)
        .map(|&bj| {
            let p = bj as f64 / n_f;
            -p * p.ln()
        })
        .sum();

    // MI = sum_ij (n_ij / n) * log(n * n_ij / (a_i * b_j))
    let mut mi: f64 = 0.0;
    for i in 0..n_true {
        for j in 0..n_pred {
            let nij = contingency[i][j];
            if nij > 0 && a[i] > 0 && b[j] > 0 {
                let p = nij as f64 / n_f;
                mi += p * (n_f * nij as f64 / (a[i] as f64 * b[j] as f64)).ln();
            }
        }
    }

    let denom = h_true + h_pred;

    if denom.abs() < 1e-15 {
        return Ok(F::one());
    }

    let nmi = 2.0 * mi / denom;

    // Clamp to [0, 1] to handle floating point noise
    let nmi_clamped = nmi.max(0.0).min(1.0);

    Ok(F::from_f64(nmi_clamped).unwrap())
}

fn unique_sorted<F: Float>(a: &Array1<F>) -> Vec<F> {
    let mut vals: Vec<F> = a.iter().copied().collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    vals.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-9).unwrap());
    vals
}

fn check_lengths<F: Float>(a: &Array1<F>, b: &Array1<F>) -> Result<()> {
    if a.len() != b.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "labels_true length {} != labels_pred length {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(RustMlError::EmptyInput("input arrays are empty".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ---------------------------------------------------------------
    // Adjusted Rand Score tests
    // ---------------------------------------------------------------

    #[test]
    fn test_ari_perfect() {
        // Identical clusterings should yield ARI = 1.
        let labels_true = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let labels_pred = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let ari: f64 = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(ari, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ari_perfect_relabeled() {
        // Relabeled but same structure: ARI should still be 1.
        let labels_true = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let labels_pred = array![5.0, 5.0, 3.0, 3.0, 8.0, 8.0];
        let ari: f64 = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(ari, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ari_partial_agreement() {
        // true: [0, 0, 0, 1, 1, 1], pred: [0, 1, 0, 1, 0, 1]
        // Contingency: [[2, 1], [1, 2]]
        // C(n,2) = 15, sum_comb_nij = C(2,2)+C(1,2)+C(1,2)+C(2,2) = 1+0+0+1 = 2
        // sum_comb_a = C(3,2)*2 = 6, sum_comb_b = C(3,2)*2 = 6
        // expected = 6*6/15 = 2.4
        // mean_ab = 6, ARI = (2 - 2.4) / (6 - 2.4) = -0.4/3.6 = -1/9
        let labels_true = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let labels_pred = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let ari: f64 = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(ari, -1.0 / 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ari_single_cluster() {
        // All samples in one cluster for both: perfect agreement.
        let labels_true = array![0.0, 0.0, 0.0, 0.0];
        let labels_pred = array![0.0, 0.0, 0.0, 0.0];
        let ari: f64 = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(ari, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ari_all_singletons() {
        // Each sample in its own cluster for both: perfect agreement.
        let labels_true = array![0.0, 1.0, 2.0, 3.0];
        let labels_pred = array![4.0, 5.0, 6.0, 7.0];
        let ari: f64 = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(ari, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ari_length_mismatch_error() {
        let labels_true = array![0.0, 1.0];
        let labels_pred = array![0.0, 1.0, 2.0];
        assert!(adjusted_rand_score(&labels_true, &labels_pred).is_err());
    }

    #[test]
    fn test_ari_empty_error() {
        let labels_true: Array1<f64> = array![];
        let labels_pred: Array1<f64> = array![];
        assert!(adjusted_rand_score(&labels_true, &labels_pred).is_err());
    }

    #[test]
    fn test_ari_f32() {
        let labels_true: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let labels_pred: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let ari = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(ari, 1.0f32, epsilon = 1e-6);
    }

    // ---------------------------------------------------------------
    // Normalized Mutual Information Score tests
    // ---------------------------------------------------------------

    #[test]
    fn test_nmi_perfect() {
        // Identical clusterings should yield NMI = 1.
        let labels_true = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let labels_pred = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let nmi: f64 = normalized_mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_perfect_relabeled() {
        // Relabeled but same structure: NMI should still be 1.
        let labels_true = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let labels_pred = array![5.0, 5.0, 3.0, 3.0, 8.0, 8.0];
        let nmi: f64 = normalized_mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_independent() {
        // When one clustering is a single cluster, NMI should be 0.
        let labels_true = array![0.0, 0.0, 1.0, 1.0];
        let labels_pred = array![0.0, 0.0, 0.0, 0.0];
        let nmi: f64 = normalized_mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(nmi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_single_cluster_both() {
        // Both are single clusters: perfect agreement, NMI = 1.
        let labels_true = array![0.0, 0.0, 0.0, 0.0];
        let labels_pred = array![0.0, 0.0, 0.0, 0.0];
        let nmi: f64 = normalized_mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_known_value() {
        // true: [0, 0, 1, 1], pred: [0, 1, 0, 1]
        // Contingency: [[1, 1], [1, 1]]
        // H(true) = H(pred) = -2*(0.5*ln(0.5)) = ln(2)
        // MI = 4 * (0.25 * ln(4*1/(2*2))) = 0
        // NMI = 0
        let labels_true = array![0.0, 0.0, 1.0, 1.0];
        let labels_pred = array![0.0, 1.0, 0.0, 1.0];
        let nmi: f64 = normalized_mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(nmi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_length_mismatch_error() {
        let labels_true = array![0.0, 1.0];
        let labels_pred = array![0.0, 1.0, 2.0];
        assert!(normalized_mutual_info_score(&labels_true, &labels_pred).is_err());
    }

    #[test]
    fn test_nmi_empty_error() {
        let labels_true: Array1<f64> = array![];
        let labels_pred: Array1<f64> = array![];
        assert!(normalized_mutual_info_score(&labels_true, &labels_pred).is_err());
    }

    #[test]
    fn test_nmi_f32() {
        let labels_true: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let labels_pred: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let nmi = normalized_mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(nmi, 1.0f32, epsilon = 1e-6);
    }
}
