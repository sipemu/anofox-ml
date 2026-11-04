//! Name-based estimator registry for anofox-ml.
//!
//! Every estimator is addressed by a `kind` string (e.g. `"RandomForestClassifier"`)
//! and configured by JSON parameters that use the Rust config's serde field
//! names; they are merged into the config's default, so callers only send
//! what they override. A model is used through the object-safe [`DynModel`]
//! trait (fit, predict, predict_proba, transform, ...) and persisted with
//! [`DynModel::to_bytes`] / [`load`] (the config and the fitted state, in
//! the binary persistence format of `anofox_ml_core::persistence`).
//!
//! This is the layer shared by the Python bindings (`anofox-ml-python`) and
//! the DuckDB extension: both only translate their own data types to
//! `ndarray` views and back.
//!
//! ```
//! use ndarray::array;
//!
//! let mut model = anofox_ml_registry::make(
//!     "DecisionTreeRegressor",
//!     &serde_json::json!({ "max_depth": 2 }),
//! )
//! .unwrap();
//! let x = array![[0.0], [1.0], [2.0], [3.0]];
//! let y = array![0.0, 0.0, 1.0, 1.0];
//! model.fit(x.view(), Some(y.view()), None).unwrap();
//! let bytes = model.to_bytes().unwrap();
//! let loaded = anofox_ml_registry::load("DecisionTreeRegressor", &bytes).unwrap();
//! assert!(loaded.is_fitted());
//! ```

mod model;
mod registry;

pub use model::{DynModel, Out};
pub use registry::{default_params, kinds, load, make};

#[cfg(test)]
mod tests {
    #[test]
    fn every_kind_builds_from_its_default_params() {
        let kinds = super::kinds();
        assert!(kinds.len() > 100, "{} kinds", kinds.len());
        let mut failures = Vec::new();
        for kind in kinds {
            let params = super::default_params(kind).unwrap();
            assert!(params.is_object() || params.is_null(), "{kind}: {params}");
            // The defaults must be accepted back (JSON round trip).
            match super::make(kind, &params) {
                Ok(model) => assert!(!model.is_fitted(), "{kind}"),
                Err(e) => failures.push(format!("{kind}: {e}")),
            }
            assert!(super::make(kind, &serde_json::json!({})).is_ok(), "{kind}");
        }
        assert!(failures.is_empty(), "{failures:#?}");
        assert!(super::make("NoSuchModel", &serde_json::json!({})).is_err());
    }

    #[test]
    fn fitted_model_round_trips_through_bytes() {
        let x = ndarray::array![[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]];
        let y = ndarray::array![0.0, 0.0, 1.0, 1.0, 1.0];
        let mut model = super::make("RandomForestClassifier", &serde_json::json!({})).unwrap();
        model.fit(x.view(), Some(y.view()), None).unwrap();
        let loaded = super::load("RandomForestClassifier", &model.to_bytes().unwrap()).unwrap();
        let (a, b) = (
            model.predict(x.view()).unwrap(),
            loaded.predict(x.view()).unwrap(),
        );
        match (a, b) {
            (super::Out::A1(a), super::Out::A1(b)) => assert_eq!(a, b),
            _ => panic!("expected 1-D predictions"),
        }
    }
}

#[cfg(all(test, feature = "perpetual"))]
mod perpetual_tests {
    #[test]
    fn perpetual_kinds_fit_predict_and_round_trip() {
        let x = ndarray::Array2::from_shape_fn((120, 2), |(i, j)| ((i * (j + 2)) % 23) as f64);
        let y_reg = x.column(0).mapv(|v| 0.5 * v) - &x.column(1);
        let y_clf = x.column(0).mapv(|v| if v > 11.0 { 1.0 } else { 0.0 });
        for (kind, y) in [
            ("PerpetualRegressor", &y_reg),
            ("PerpetualClassifier", &y_clf),
        ] {
            assert!(super::kinds().contains(&kind));
            let params = serde_json::json!({ "budget": 1.0, "num_threads": 1 });
            let mut model = super::make(kind, &params).unwrap();
            model.fit(x.view(), Some(y.view()), None).unwrap();
            let loaded = super::load(kind, &model.to_bytes().unwrap()).unwrap();
            assert_eq!(loaded.params().unwrap()["budget"], 1.0);
            match (
                model.predict(x.view()).unwrap(),
                loaded.predict(x.view()).unwrap(),
            ) {
                (super::Out::A1(a), super::Out::A1(b)) => assert_eq!(a, b),
                _ => panic!("expected 1-D predictions"),
            }
        }
        assert!(super::make(
            "PerpetualRegressor",
            &serde_json::json!({ "learning_rate": 0.1 })
        )
        .is_err());
    }
}
