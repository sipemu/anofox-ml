//! Clustering algorithms: K-Means and DBSCAN.
//!
//! This crate provides unsupervised clustering methods for grouping data points:
//!
//! - [`KMeans`] -- Lloyd's algorithm with k-means++ initialization.
//! - [`Dbscan`] -- density-based spatial clustering of applications with noise.
//!
//! Both implement [`FitUnsupervised`](anofox_ml_core::FitUnsupervised) and
//! [`Predict`](anofox_ml_core::Predict), so the fitted model can assign cluster
//! labels to new data points.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use anofox_ml_core::{FitUnsupervised, Predict};
//! use anofox_ml_cluster::KMeans;
//!
//! let x = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [0.0, 1.0],
//!     [10.0, 10.0],
//!     [11.0, 10.0],
//!     [10.0, 11.0]
//! ];
//!
//! let kmeans = KMeans::new(2).with_seed(42);
//! let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();
//!
//! // Points in the same group get the same label
//! let labels = fitted.labels();
//! assert_eq!(labels[0] as usize, labels[1] as usize);
//! assert_eq!(labels[3] as usize, labels[4] as usize);
//! assert_ne!(labels[0] as usize, labels[3] as usize);
//! ```

pub mod affinity_propagation;
pub mod agglomerative;
pub mod bgmm;
pub mod birch;
pub mod dbscan;
pub mod gmm;
pub mod hdbscan;
pub mod kmeans;
pub mod mean_shift;
pub mod mini_batch_kmeans;
pub mod optics;
pub mod spectral;

pub use affinity_propagation::{AffinityPropagation, FittedAffinityPropagation};
pub use agglomerative::{AgglomerativeClustering, FittedAgglomerativeClustering, Linkage};
pub use bgmm::{BayesianGaussianMixture, FittedBayesianGaussianMixture};
pub use birch::{Birch, FittedBirch};
pub use dbscan::{Dbscan, FittedDbscan};
pub use gmm::{CovarianceType, FittedGaussianMixture, GaussianMixture};
pub use hdbscan::{FittedHdbscan, Hdbscan};
pub use kmeans::{FittedKMeans, KMeans};
pub use mean_shift::{FittedMeanShift, MeanShift};
pub use mini_batch_kmeans::{FittedMiniBatchKMeans, MiniBatchKMeans};
pub use optics::{FittedOptics, Optics};
pub use spectral::{Affinity, FittedSpectralClustering, SpectralClustering};
