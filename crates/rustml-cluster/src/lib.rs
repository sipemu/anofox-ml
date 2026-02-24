//! Clustering algorithms: K-Means and DBSCAN.
//!
//! This crate provides unsupervised clustering methods for grouping data points:
//!
//! - [`KMeans`] -- Lloyd's algorithm with k-means++ initialization.
//! - [`Dbscan`] -- density-based spatial clustering of applications with noise.
//!
//! Both implement [`FitUnsupervised`](rustml_core::FitUnsupervised) and
//! [`Predict`](rustml_core::Predict), so the fitted model can assign cluster
//! labels to new data points.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use rustml_core::{FitUnsupervised, Predict};
//! use rustml_cluster::KMeans;
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

pub mod dbscan;
pub mod kmeans;

pub use dbscan::{Dbscan, FittedDbscan};
pub use kmeans::{FittedKMeans, KMeans};
