//! Image patches (sklearn's `feature_extraction.image.PatchExtractor`).
//!
//! The images are the rows of `x`, each a flattened `height × width ×
//! channels` array (row-major, as numpy's `reshape`); the patches are the
//! rows of the output, each a flattened `patch_h × patch_w × channels`
//! array. With `max_patches`, each image's patch corners are numpy
//! `RandomState(seed).randint` draws (rows, then columns), one stream over
//! all images, so the patches are sklearn's for the same `random_state`.

use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// How many patches per image (sklearn's `max_patches`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxPatches {
    /// At most this many (all when there are fewer).
    Count(usize),
    /// `int(fraction · all)`, the fraction in `(0, 1)`.
    Fraction(f64),
}

/// sklearn's `PatchExtractor`: every `patch_size` patch of every image
/// (all positions, row-major), or `max_patches` random ones per image.
/// `patch_size = None` is sklearn's `(height // 10, width // 10)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PatchExtractor {
    /// `(height, width, channels)` of each image.
    pub image_shape: (usize, usize, usize),
    pub patch_size: Option<(usize, usize)>,
    pub max_patches: Option<MaxPatches>,
    pub seed: u32,
}

impl Default for PatchExtractor {
    fn default() -> Self {
        Self {
            image_shape: (0, 0, 1),
            patch_size: None,
            max_patches: None,
            seed: 0,
        }
    }
}

/// A fitted (validated) [`PatchExtractor`]; it holds no state from the data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedPatchExtractor {
    config: PatchExtractor,
    patch_size: (usize, usize),
}

impl FittedPatchExtractor {
    /// The resolved `(patch_h, patch_w)`.
    pub fn patch_size(&self) -> (usize, usize) {
        self.patch_size
    }
}

impl PatchExtractor {
    fn n_patches(&self, all: usize) -> Result<usize> {
        Ok(match self.max_patches {
            None | Some(MaxPatches::Count(0)) => all,
            Some(MaxPatches::Count(k)) => k.min(all),
            Some(MaxPatches::Fraction(f)) if f > 0.0 && f < 1.0 => (f * all as f64) as usize,
            Some(MaxPatches::Fraction(f)) => {
                return Err(RustMlError::InvalidParameter(format!(
                    "Invalid value for max_patches: {f}"
                )))
            }
        })
    }
}

impl FitUnsupervised<f64> for PatchExtractor {
    type Fitted = FittedPatchExtractor;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedPatchExtractor> {
        let (h, w, c) = self.image_shape;
        if h == 0 || w == 0 || c == 0 || x.ncols() != h * w * c {
            return Err(RustMlError::ShapeMismatch(format!(
                "rows of {} values do not match image_shape {:?}",
                x.ncols(),
                self.image_shape
            )));
        }
        let (ph, pw) = self.patch_size.unwrap_or((h / 10, w / 10));
        if ph == 0 || pw == 0 {
            return Err(RustMlError::InvalidParameter(format!(
                "patch size {:?} must be positive (the default is a tenth of the image)",
                (ph, pw)
            )));
        }
        if ph > h {
            return Err(RustMlError::InvalidParameter(
                "Height of the patch should be less than the height of the image.".into(),
            ));
        }
        if pw > w {
            return Err(RustMlError::InvalidParameter(
                "Width of the patch should be less than the width of the image.".into(),
            ));
        }
        self.n_patches((h - ph + 1) * (w - pw + 1))?;
        Ok(FittedPatchExtractor {
            config: self.clone(),
            patch_size: (ph, pw),
        })
    }
}

impl Transform<f64> for FittedPatchExtractor {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let (h, w, c) = self.config.image_shape;
        if x.ncols() != h * w * c {
            return Err(RustMlError::ShapeMismatch(format!(
                "rows of {} values do not match image_shape {:?}",
                x.ncols(),
                self.config.image_shape
            )));
        }
        let (ph, pw) = self.patch_size;
        let (nh, nw) = (h - ph + 1, w - pw + 1);
        let per_image = self.config.n_patches(nh * nw)?;
        let random = self.config.max_patches.is_some()
            && !matches!(self.config.max_patches, Some(MaxPatches::Count(0)));
        let mut rng = RandomState::new(self.config.seed);
        let mut out = Array2::zeros((x.nrows() * per_image, ph * pw * c));
        let mut r = 0;
        for img in x.rows() {
            let corners: Vec<(usize, usize)> = if random {
                let is: Vec<usize> = (0..per_image)
                    .map(|_| rng.randint(nh as u64) as usize)
                    .collect();
                let js: Vec<usize> = (0..per_image)
                    .map(|_| rng.randint(nw as u64) as usize)
                    .collect();
                is.into_iter().zip(js).collect()
            } else {
                (0..nh).flat_map(|i| (0..nw).map(move |j| (i, j))).collect()
            };
            for (i0, j0) in corners {
                let mut k = 0;
                for i in i0..i0 + ph {
                    for j in j0..j0 + pw {
                        for ch in 0..c {
                            out[[r, k]] = img[(i * w + j) * c + ch];
                            k += 1;
                        }
                    }
                }
                r += 1;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn all_patches_in_row_major_order() {
        // One 3 × 3 grey image, 2 × 2 patches: 4 of them.
        let x = array![[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]];
        let est = PatchExtractor {
            image_shape: (3, 3, 1),
            patch_size: Some((2, 2)),
            ..Default::default()
        };
        let p = est.fit(&x).unwrap().transform(&x).unwrap();
        assert_eq!(
            p,
            array![
                [0.0, 1.0, 3.0, 4.0],
                [1.0, 2.0, 4.0, 5.0],
                [3.0, 4.0, 6.0, 7.0],
                [4.0, 5.0, 7.0, 8.0]
            ]
        );
    }
}
