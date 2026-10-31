//! # anofox-ml SVM
//!
//! Support Vector Machine classifiers for the anofox-ml machine learning library.
//!
//! This crate provides two SVM classifiers:
//!
//! - [`LinearSvc`] -- Linear Support Vector Classifier with (squared) hinge
//!   loss + L2 regularization, solved by liblinear's dual coordinate descent
//!   (Hsieh et al., 2008), as sklearn's `LinearSVC(dual=True)`.
//! - [`Svc`] -- Support Vector Classifier with kernel support (linear, RBF,
//!   polynomial, sigmoid, precomputed), solved via a simplified SMO
//!   algorithm. Multi-class problems use one-vs-one (as libsvm / sklearn),
//!   with optional class weights and Platt-scaled `predict_proba`.
//!
//! `gamma` defaults to [`Gamma::Scale`] (`1 / (n_features · X.var())`), as in
//! sklearn; it is resolved at fit time.
//!
//! ## Example
//!
//! ```rust
//! use anofox_ml_core::{Fit, Predict};
//! use anofox_ml_svm::{LinearSvc, Svc, SvmKernel};
//! use ndarray::array;
//!
//! // Linear SVC
//! let x = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];
//! let y = array![0.0, 0.0, 1.0, 1.0];
//!
//! let svc = LinearSvc::new().with_c(1.0);
//! let model = svc.fit(&x, &y).unwrap();
//! let preds = model.predict(&x).unwrap();
//!
//! // Kernel SVC with RBF
//! let svc = Svc::new()
//!     .with_kernel(SvmKernel::rbf(0.5))
//!     .with_c(10.0);
//! let model = svc.fit(&x, &y).unwrap();
//! let preds = model.predict(&x).unwrap();
//! ```

mod kernel;
mod linear_svc;
mod linear_svr;
mod nu_svc;
mod nu_svr;
mod one_class_svm;
mod solver;
mod svc;
mod svr;

pub use kernel::{Gamma, SvmKernel};
pub use linear_svc::{FittedLinearSvc, LinearSvc, LinearSvcLoss};
pub use linear_svr::{FittedLinearSvr, LinearSvr, LinearSvrLoss};
pub use nu_svc::{FittedNuSvc, NuSvc};
pub use nu_svr::{FittedNuSvr, NuSvr};
pub use one_class_svm::{FittedOneClassSvm, OneClassSvm};
pub use svc::{ClassWeight, DecisionFunctionShape, FittedSvc, Svc};
pub use svr::{FittedSvr, Svr};
