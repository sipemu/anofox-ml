//! K-nearest neighbors classifiers and regressors with KD-tree acceleration.
//!
//! This crate provides [`KnnClassifier`] and [`KnnRegressor`], both of which
//! support configurable distance metrics ([`DistanceMetric`]) and weighting
//! strategies ([`WeightFunction`]), plus the radius-neighbour estimators and
//! the local outlier factor. All of them search through a shared
//! [`NeighborIndex`] (KD-tree or brute force, chosen like sklearn's
//! `algorithm="auto"`), which the clustering and manifold crates use too.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use anofox_ml_core::{Fit, Predict};
//! use anofox_ml_neighbors::KnnClassifier;
//!
//! let x_train = array![
//!     [0.0, 0.0],
//!     [1.0, 1.0],
//!     [2.0, 2.0],
//!     [10.0, 10.0],
//!     [11.0, 11.0],
//!     [12.0, 12.0]
//! ];
//! let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//!
//! let knn = KnnClassifier::new(3);
//! let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();
//!
//! let x_test = array![[0.5, 0.5], [10.5, 10.5]];
//! let preds = fitted.predict(&x_test).unwrap();
//! assert!((preds[0] - 0.0_f64).abs() < 1e-10);
//! assert!((preds[1] - 1.0_f64).abs() < 1e-10);
//! ```

pub mod distance;
pub mod index;
mod kdtree;
pub mod kernel_density;
pub mod knn_classifier;
pub mod knn_regressor;
pub mod lof;
pub mod nearest_centroid;
pub mod radius;
pub mod semi_supervised;

pub use distance::DistanceMetric;
pub use index::{NeighborAlgorithm, NeighborIndex, RadiusNeighbors};
pub use kernel_density::{Bandwidth, DensityKernel, FittedKernelDensity, KernelDensity};
pub use knn_classifier::{FittedKnnClassifier, KnnClassifier, WeightFunction};
pub use knn_regressor::{FittedKnnRegressor, KnnRegressor};
pub use lof::{FittedLocalOutlierFactor, LocalOutlierFactor, LofAlgorithm};
pub use nearest_centroid::{
    CentroidMetric, CentroidPriors, FittedNearestCentroid, NearestCentroid,
};
pub use radius::{
    FittedRadiusNeighborsClassifier, FittedRadiusNeighborsRegressor, RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
};
pub use semi_supervised::{FittedLabelPropagation, GraphKernel, LabelPropagation, LabelSpreading};

/// Return `x` in standard (C-contiguous) layout: borrowed when it already is,
/// otherwise a one-off row-major copy. Lets `row(i).as_slice()` fast paths
/// work on Fortran-order / transposed input (#42).
pub(crate) fn standard_layout<F: Clone>(
    x: &ndarray::Array2<F>,
) -> std::borrow::Cow<'_, ndarray::Array2<F>> {
    if x.is_standard_layout() {
        std::borrow::Cow::Borrowed(x)
    } else {
        std::borrow::Cow::Owned(x.as_standard_layout().into_owned())
    }
}

/// Total order on floats (NaN-safe, never panics) via `f64::total_cmp`.
#[inline]
pub(crate) fn total_cmp<F: anofox_ml_core::Float>(a: &F, b: &F) -> std::cmp::Ordering {
    a.to_f64()
        .unwrap_or(f64::NAN)
        .total_cmp(&b.to_f64().unwrap_or(f64::NAN))
}
