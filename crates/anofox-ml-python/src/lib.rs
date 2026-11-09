//! Python bindings for anofox-ml.
//!
//! The extension module `anofox_ml_python._native` exposes one generic
//! estimator class (`Model`, backed by `anofox_ml_registry`) plus
//! metric functions. The sklearn-compatible classes live in the Python
//! package `anofox_ml_python` on top of it.

mod helpers;
mod metrics;
mod native;
mod text;
mod tune;

use pyo3::prelude::*;

macro_rules! register_functions {
    ($m:expr, $($func:path),* $(,)?) => {
        $( $m.add_function(wrap_pyfunction!($func, $m)?)?; )*
    };
}

#[pymodule]
#[pyo3(name = "_native")]
fn native_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<native::PyModel>()?;
    m.add_function(wrap_pyfunction!(tune::tune, m)?)?;
    m.add_function(wrap_pyfunction!(text::feature_hash, m)?)?;
    register_functions!(
        m,
        metrics::accuracy_score,
        metrics::mse,
        metrics::r2_score,
        metrics::mae,
        metrics::precision_score,
        metrics::recall_score,
        metrics::f1_score,
        metrics::balanced_accuracy_score,
        metrics::cohen_kappa_score,
        metrics::explained_variance_score,
        metrics::matthews_corrcoef,
        metrics::max_error,
        metrics::mean_absolute_percentage_error,
        metrics::mean_squared_log_error,
        metrics::median_absolute_error,
        metrics::adjusted_rand_score,
        metrics::normalized_mutual_info_score,
        metrics::log_loss,
        metrics::roc_auc_score,
        metrics::average_precision_score,
        metrics::brier_score_loss,
        metrics::silhouette_score,
        metrics::confusion_matrix,
    );
    Ok(())
}
