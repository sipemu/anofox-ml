//! Feature preprocessing, scaling, and dimensionality reduction.
//!
//! This crate provides transformers for preparing data before model training,
//! including [`StandardScaler`] (z-score normalization), [`MinMaxScaler`]
//! (min-max normalization), [`Pca`] (principal component analysis),
//! [`VarianceThreshold`] (low-variance feature removal), and
//! [`MutualInformationSelector`] (feature selection by mutual information).
//!
//! All transformers follow the type-state pattern: call
//! [`FitUnsupervised::fit`](anofox_ml_core::FitUnsupervised::fit) to learn
//! parameters, then [`Transform::transform`](anofox_ml_core::Transform::transform)
//! on the fitted result to apply the transformation.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use anofox_ml_core::{FitUnsupervised, Transform};
//! use anofox_ml_preprocessing::StandardScaler;
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

pub mod areg_imputer;
pub mod binarizer;
pub mod cca;
pub mod covariance;
pub mod factor_analysis;
pub mod fast_ica;
pub mod incremental_pca;
pub mod iterative_imputer;
pub mod kbins_discretizer;
pub mod kernel_approximation;
pub mod kernel_pca;
pub mod knn_imputer;
pub mod label_encoder;
pub mod lda;
pub mod max_abs_scaler;
pub mod minmax_scaler;
pub mod missing_indicator;
pub mod mutual_information;
pub mod nmf;
pub mod normalizer;
pub mod one_hot_encoder;
pub mod ordinal_encoder;
pub mod pca;
pub mod pls;
pub mod polynomial_features;
pub mod power_transformer;
pub(crate) mod quantile;
pub mod quantile_transformer;
pub mod rcs_transformer;
pub mod rfe;
pub mod robust_scaler;
pub mod select_from_model;
pub mod select_k_best;
pub mod simple_imputer;
pub(crate) mod special;
pub mod spline_transformer;
pub mod standard_scaler;
pub mod target_encoder;
pub mod truncated_svd;
pub mod univariate_select;
pub mod variance_threshold;

pub use areg_imputer::{AregImputer, AregMatch, AregVariable, MultipleImputation};
pub use binarizer::{Binarizer, FittedBinarizer};
pub use cca::{Cca, FittedCca, FittedPlsSvd, PlsCanonical, PlsSvd};
pub use covariance::{
    EllipticEnvelope, EmpiricalCovariance, FittedCovariance, FittedEllipticEnvelope, LedoitWolf,
    MinCovDet, Oas, ShrunkCovariance,
};
pub use factor_analysis::{FactorAnalysis, FactorRotation, FittedFactorAnalysis};
pub use fast_ica::{FastIca, FittedFastIca, IcaAlgorithm, IcaFun, IcaWhiten, IcaWhitenSolver};
pub use incremental_pca::{FittedIncrementalPca, IncrementalPca};
pub use iterative_imputer::{
    AnyModel, AnyRegressor, FittedIterativeImputer, ImputationEstimator, ImputationModel,
    ImputationOrder, ImputationStep, IterativeImputer,
};
pub use kbins_discretizer::{
    BinStrategy, EncodeStrategy, FittedKBinsDiscretizer, KBinsDiscretizer,
};
pub use kernel_approximation::{
    johnson_lindenstrauss_min_dim, FittedKernelCenterer, FittedNystroem, FittedRandomProjection,
    FittedRbfSampler, GaussianRandomProjection, KernelCenterer, Nystroem, NystroemKernel,
    ProjectionComponents, RbfSampler, SparseRandomProjection,
};
pub use kernel_pca::{FittedKernelPca, KernelPca, KpcaKernel};
pub use knn_imputer::{FittedKnnImputer, KnnImputeWeights, KnnImputer};
pub use label_encoder::{FittedLabelEncoder, LabelEncoder};
pub use lda::{FittedLatentDirichletAllocation, LatentDirichletAllocation, LdaLearning};
pub use max_abs_scaler::{FittedMaxAbsScaler, MaxAbsScaler};
pub use minmax_scaler::{FittedMinMaxScaler, MinMaxScaler};
pub use missing_indicator::{FittedMissingIndicator, MissingFeatures, MissingIndicator};
pub use mutual_information::{
    mutual_info_classif, mutual_info_regression, DiscreteFeatures, FittedMutualInformationSelector,
    MiTarget, MutualInformationSelector,
};
pub use nmf::{FittedNmf, Nmf};
pub use normalizer::{FittedNormalizer, NormType, Normalizer};
pub use one_hot_encoder::{
    Category, FittedOneHotEncoder, HandleUnknown, MinFrequency, OneHotDrop, OneHotEncoder,
};
pub use ordinal_encoder::{FittedOrdinalEncoder, OrdinalEncoder, OrdinalHandleUnknown};
pub use pca::{FittedPca, Pca};
pub use pls::{FittedPlsRegression, PlsRegression};
pub use polynomial_features::{FittedPolynomialFeatures, PolynomialFeatures};
pub use power_transformer::{FittedPowerTransformer, PowerMethod, PowerTransformer};
pub use quantile_transformer::{
    FittedQuantileTransformer, OutputDistribution, QuantileTransformer,
};
pub use rcs_transformer::{FittedRcsTransformer, RcsTransformer, SplineNorm};
pub use rfe::{
    FittedRfe, FittedRfecv, FittedSequentialFeatureSelector, Rfe, Rfecv, SequentialFeatureSelector,
};
pub use robust_scaler::{FittedRobustScaler, RobustScaler};
pub use select_from_model::{FittedSelectFromModel, SelectFromModel, Threshold};
pub use select_k_best::{FittedSelectKBest, SelectKBest};
pub use simple_imputer::{FittedSimpleImputer, ImputeStrategy, SimpleImputer};
pub use spline_transformer::{
    Extrapolation, FittedSplineTransformer, SplineKnots, SplineTransformer,
};
pub use standard_scaler::{FittedStandardScaler, StandardScaler};
pub use target_encoder::{FittedTargetEncoder, TargetEncoder, TargetSmooth, TargetType};
pub use truncated_svd::{FittedTruncatedSvd, TruncatedSvd};
pub use univariate_select::{GenericUnivariateSelect, SelectionMode};
pub use variance_threshold::{FittedVarianceThreshold, VarianceThreshold};
