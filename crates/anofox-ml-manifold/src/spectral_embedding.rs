//! Spectral embedding of a graph (sklearn's `manifold.spectral_embedding`):
//! the bottom eigenvectors of the normalised Laplacian, scaled back by the
//! square-root degrees.

use anofox_ml_core::{Result, RustMlError};
use ndarray::Array2;

use crate::sparse_eigen::smallest_eigenpairs;

/// Embed the `n` nodes of the symmetric weighted graph `adjacency` (entries
/// `(i, j, w)` of both triangles; the diagonal is ignored) in
/// `n_components` dimensions.
///
/// As sklearn: `L = I − D^{-1/2} A D^{-1/2}` with unit diagonal (degrees
/// without self loops; isolated nodes keep degree 1), its `n_components`
/// smallest eigenvectors divided by `√degree`, each flipped so its largest
/// absolute entry is positive. With `drop_first` the first (trivial)
/// eigenvector is dropped and one more is computed.
pub fn spectral_embedding(
    n: usize,
    adjacency: &[(usize, usize, f64)],
    n_components: usize,
    drop_first: bool,
) -> Result<Array2<f64>> {
    let k = n_components + usize::from(drop_first);
    if n_components == 0 || k > n {
        return Err(RustMlError::InvalidParameter(format!(
            "n_components ({n_components}) must be in [1, {}]",
            n - usize::from(drop_first)
        )));
    }
    let mut deg = vec![0.0_f64; n];
    for &(i, j, w) in adjacency {
        if i != j {
            deg[i] += w;
        }
    }
    let dd: Vec<f64> = deg
        .iter()
        .map(|&d| if d == 0.0 { 1.0 } else { d.sqrt() })
        .collect();
    let entries: Vec<(usize, usize, f64)> = adjacency
        .iter()
        .filter(|e| e.0 != e.1)
        .map(|&(i, j, w)| (i, j, -w / (dd[i] * dd[j])))
        .chain((0..n).map(|i| (i, i, 1.0)))
        .collect();
    let (_, vecs) = smallest_eigenpairs(n, &entries, k)?;
    let first = usize::from(drop_first);
    let mut emb = Array2::from_shape_fn((n, n_components), |(i, c)| vecs[[i, c + first]] / dd[i]);
    // sklearn's `_deterministic_vector_sign_flip`.
    for mut col in emb.columns_mut() {
        let big = col
            .iter()
            .copied()
            .fold(0.0_f64, |b, v| if v.abs() > b.abs() { v } else { b });
        if big < 0.0 {
            col.mapv_inplace(|v| -v);
        }
    }
    Ok(emb)
}
