//! Feature preprocessing, scaling, and dimensionality reduction.
//!
//! This crate provides transformers for preparing data before model training,
//! including [`StandardScaler`] (z-score normalization), [`MinMaxScaler`]
//! (min-max normalization), [`Pca`] (principal component analysis),
//! [`VarianceThreshold`] (low-variance feature removal), and
//! [`MutualInformationSelector`] (feature selection by mutual information).
//!
//! All transformers follow the type-state pattern: call
//! [`FitUnsupervised::fit`](rustml_core::FitUnsupervised::fit) to learn
//! parameters, then [`Transform::transform`](rustml_core::Transform::transform)
//! on the fitted result to apply the transformation.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use rustml_core::{FitUnsupervised, Transform};
//! use rustml_preprocessing::StandardScaler;
//!
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//!
//! let scaler = StandardScaler::new();
//! let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
//! let x_scaled = fitted.transform(&x).unwrap();
//!
//! // Each column now has mean ~0 and std ~1
//! let col0_mean: f64 = x_scaled.column(0).sum() / 3.0;
//! assert!(col0_mean.abs() < 1e-10);
//! ```

pub mod binarizer;
pub mod cca;
pub mod fast_ica;
pub mod kbins_discretizer;
pub mod kernel_pca;
pub mod label_encoder;
pub mod max_abs_scaler;
pub mod minmax_scaler;
pub mod mutual_information;
pub mod nmf;
pub mod normalizer;
pub mod one_hot_encoder;
pub mod ordinal_encoder;
pub mod pca;
pub mod pls;
pub mod polynomial_features;
pub mod power_transformer;
pub mod quantile_transformer;
pub mod rfe;
pub mod robust_scaler;
pub mod select_from_model;
pub mod select_k_best;
pub mod simple_imputer;
pub mod standard_scaler;
pub mod truncated_svd;
pub mod variance_threshold;

pub use binarizer::{Binarizer, FittedBinarizer};
pub use cca::{Cca, FittedCca};
pub use fast_ica::{FastIca, FittedFastIca};
pub use kbins_discretizer::{
    BinStrategy, EncodeStrategy, FittedKBinsDiscretizer, KBinsDiscretizer,
};
pub use kernel_pca::{FittedKernelPca, KernelPca, KpcaKernel};
pub use label_encoder::{FittedLabelEncoder, LabelEncoder};
pub use max_abs_scaler::{FittedMaxAbsScaler, MaxAbsScaler};
pub use minmax_scaler::{FittedMinMaxScaler, MinMaxScaler};
pub use mutual_information::{FittedMutualInformationSelector, MutualInformationSelector};
pub use nmf::{FittedNmf, Nmf};
pub use normalizer::{FittedNormalizer, NormType, Normalizer};
pub use one_hot_encoder::{FittedOneHotEncoder, OneHotEncoder};
pub use ordinal_encoder::{FittedOrdinalEncoder, OrdinalEncoder};
pub use pca::{FittedPca, Pca};
pub use pls::{FittedPlsRegression, PlsRegression};
pub use polynomial_features::{FittedPolynomialFeatures, PolynomialFeatures};
pub use power_transformer::{FittedPowerTransformer, PowerTransformer};
pub use quantile_transformer::{
    FittedQuantileTransformer, OutputDistribution, QuantileTransformer,
};
pub use rfe::{FittedRfe, FittedSequentialFeatureSelector, Rfe, SequentialFeatureSelector};
pub use robust_scaler::{FittedRobustScaler, RobustScaler};
pub use select_from_model::{FittedSelectFromModel, SelectFromModel};
pub use select_k_best::{FittedSelectKBest, SelectKBest};
pub use simple_imputer::{FittedSimpleImputer, ImputeStrategy, SimpleImputer};
pub use standard_scaler::{FittedStandardScaler, StandardScaler};
pub use truncated_svd::{FittedTruncatedSvd, TruncatedSvd};
pub use variance_threshold::{FittedVarianceThreshold, VarianceThreshold};
