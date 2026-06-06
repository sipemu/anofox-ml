mod cluster;
mod ensemble;
mod helpers;
mod manifold;
mod metrics;
mod naive_bayes;
mod neighbors;
mod neural_net;
mod preprocessing;
mod svm;
mod trees;

use pyo3::prelude::*;

use cluster::{AffinityPropagation, BayesianGaussianMixture, Dbscan, Hdbscan, KMeans, MeanShift};
use ensemble::{
    GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
    RandomForestRegressor,
};
use manifold::{Isomap, LocallyLinearEmbedding, TSne};
use naive_bayes::GaussianNB;
use neighbors::{KnnClassifier, KnnRegressor, LocalOutlierFactor};
use neural_net::{MlpClassifier, MlpRegressor};
use preprocessing::{
    MinMaxScaler, MutualInformationSelector, Pca, StandardScaler, VarianceThreshold,
};
use svm::{LinearSvc, Svc};
use trees::{DecisionTreeClassifier, DecisionTreeRegressor};

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
    register_classes!(
        m,
        StandardScaler,
        MinMaxScaler,
        Pca,
        VarianceThreshold,
        MutualInformationSelector,
        KnnClassifier,
        KnnRegressor,
        LocalOutlierFactor,
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        KMeans,
        Dbscan,
        Hdbscan,
        MeanShift,
        AffinityPropagation,
        BayesianGaussianMixture,
        GaussianNB,
        LinearSvc,
        Svc,
        MlpClassifier,
        MlpRegressor,
        TSne,
        Isomap,
        LocallyLinearEmbedding,
    );
    register_functions!(
        m,
        metrics::accuracy_score,
        metrics::mse,
        metrics::r2_score,
        metrics::mae,
        metrics::precision_score,
        metrics::recall_score,
        metrics::f1_score,
    );
    Ok(())
}
