mod helpers;
mod preprocessing;
mod neighbors;
mod trees;
mod ensemble;
mod cluster;
mod naive_bayes;
mod svm;
mod neural_net;
mod metrics;

use pyo3::prelude::*;

use preprocessing::{StandardScaler, MinMaxScaler, Pca, VarianceThreshold, MutualInformationSelector};
use neighbors::{KnnClassifier, KnnRegressor};
use trees::{DecisionTreeClassifier, DecisionTreeRegressor};
use ensemble::{
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
};
use cluster::{KMeans, Dbscan};
use naive_bayes::GaussianNB;
use svm::{LinearSvc, Svc};
use neural_net::{MlpClassifier, MlpRegressor};

macro_rules! register_classes {
    ($m:expr, $($class:ty),* $(,)?) => {
        $( $m.add_class::<$class>()?; )*
    };
}

macro_rules! register_functions {
    ($m:expr, $($func:path),* $(,)?) => {
        $( $m.add_function(wrap_pyfunction!($func, $m)?)?; )*
    };
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
#[pymodule]
fn rustml_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_classes!(m,
        StandardScaler, MinMaxScaler, Pca, VarianceThreshold, MutualInformationSelector,
        KnnClassifier, KnnRegressor,
        DecisionTreeClassifier, DecisionTreeRegressor,
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        KMeans, Dbscan,
        GaussianNB,
        LinearSvc, Svc,
        MlpClassifier, MlpRegressor,
    );
    register_functions!(m,
        metrics::accuracy_score, metrics::mse, metrics::r2_score, metrics::mae,
        metrics::precision_score, metrics::recall_score, metrics::f1_score,
    );
    Ok(())
}
