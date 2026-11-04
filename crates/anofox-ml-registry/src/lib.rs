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

mod estimator;
mod model;
mod registry;

pub use estimator::{RegistryEstimator, RegistryFitted};
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
    fn perpetual_inside_a_conformal_kind() {
        let x = ndarray::Array2::from_shape_fn((160, 2), |(i, j)| ((i * (j + 2)) % 23) as f64);
        let y = x.column(0).mapv(|v| 0.5 * v) - &x.column(1);
        let params = serde_json::json!({
            "estimator": "PerpetualRegressor",
            "estimator_params": { "budget": 0.5, "num_threads": 1 },
        });
        let mut model = super::make("SplitConformalRegressor", &params).unwrap();
        model.fit(x.view(), Some(y.view()), None).unwrap();
        let loaded = super::load("SplitConformalRegressor", &model.to_bytes().unwrap()).unwrap();
        match (
            model.predict_interval(x.view(), 0.1).unwrap(),
            loaded.predict_interval(x.view(), 0.1).unwrap(),
        ) {
            (super::Out::A2(a), super::Out::A2(b)) => {
                assert_eq!(a.dim(), (160, 2));
                assert_eq!(a, b);
            }
            _ => panic!("expected n × 2 intervals"),
        }
    }

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

#[cfg(test)]
mod conformal_tests {
    use super::Out;
    use anofox_ml_core::Fit;
    use ndarray::{Array1, Array2};
    use serde_json::json;

    fn regression() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((200, 2), |(i, j)| ((i * (j + 3)) % 37) as f64 / 37.0);
        let y = x
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, r)| 2.0 * r[0] - r[1] + ((i * 17) % 11) as f64 / 11.0)
            .collect();
        (x, y)
    }

    fn a2(out: Out) -> Array2<f64> {
        match out {
            Out::A2(a) => a,
            Out::A1(_) => panic!("expected a 2-D output"),
        }
    }

    #[test]
    fn conformal_regressor_kinds_give_intervals_and_round_trip() {
        let (x, y) = regression();
        for (kind, params) in [
            (
                "SplitConformalRegressor",
                json!({ "estimator": "DecisionTreeRegressor", "estimator_params": { "max_depth": 3 } }),
            ),
            (
                "CrossConformalRegressor",
                json!({ "estimator": "RidgeRegressor", "cv": 4, "method": "MinMax" }),
            ),
            (
                "NormalizedConformalRegressor",
                json!({ "residual_estimator": "KnnRegressor" }),
            ),
        ] {
            let mut model = super::make(kind, &params).unwrap();
            assert!(model.capabilities().contains(&"predict_interval"), "{kind}");
            model.fit(x.view(), Some(y.view()), None).unwrap();
            let iv = a2(model.predict_interval(x.view(), 0.1).unwrap());
            assert_eq!(iv.dim(), (200, 2), "{kind}");
            assert!(iv.rows().into_iter().all(|r| r[0] <= r[1]), "{kind}");
            let loaded = super::load(kind, &model.to_bytes().unwrap()).unwrap();
            assert_eq!(
                iv,
                a2(loaded.predict_interval(x.view(), 0.1).unwrap()),
                "{kind}"
            );
            // Narrower at a larger alpha.
            let iv5 = a2(model.predict_interval(x.view(), 0.5).unwrap());
            let width = |a: &Array2<f64>| (&a.column(1) - &a.column(0)).sum();
            assert!(width(&iv5) <= width(&iv), "{kind}");
            assert!(model.predict_set(x.view(), 0.1).is_err());
        }
    }

    #[test]
    fn registry_conformal_matches_the_typed_wrapper() {
        let (x, y) = regression();
        let params = json!({ "estimator": "KnnRegressor", "calibration_size": 0.3, "seed": 7 });
        let mut model = super::make("SplitConformalRegressor", &params).unwrap();
        model.fit(x.view(), Some(y.view()), None).unwrap();
        let typed = anofox_ml_conformal::SplitConformalRegressor::new(
            anofox_ml_neighbors::KnnRegressor::default(),
        )
        .with_calibration_size(0.3)
        .with_seed(7)
        .fit(&x, &y)
        .unwrap();
        let (lo, hi) = typed.predict_interval(x.view(), 0.2).unwrap();
        let iv = a2(model.predict_interval(x.view(), 0.2).unwrap());
        assert_eq!(iv.column(0), lo);
        assert_eq!(iv.column(1), hi);
    }

    #[test]
    fn conformal_classifier_kind_gives_sets() {
        let x = Array2::from_shape_fn((240, 2), |(i, j)| ((i * (j + 5)) % 41) as f64 / 41.0);
        let y: Array1<f64> = x.column(0).mapv(|v| {
            if v < 0.33 {
                0.0
            } else if v < 0.66 {
                1.0
            } else {
                2.0
            }
        });
        for score in [json!("Lac"), json!({ "Aps": "Include" })] {
            let params = json!({ "estimator": "RandomForestClassifier", "score": score });
            let mut model = super::make("SplitConformalClassifier", &params).unwrap();
            model.fit(x.view(), Some(y.view()), None).unwrap();
            let sets = a2(model.predict_set(x.view(), 0.1).unwrap());
            assert_eq!(sets.dim(), (240, 3));
            assert!(sets.iter().all(|&v| v == 0.0 || v == 1.0));
            let covered = y
                .iter()
                .enumerate()
                .filter(|&(i, &c)| sets[[i, c as usize]] == 1.0)
                .count();
            assert!(covered as f64 / 240.0 >= 0.85, "{score}: {covered}");
            assert_eq!(a2(model.predict_proba(x.view()).unwrap()).dim(), (240, 3));
            let loaded =
                super::load("SplitConformalClassifier", &model.to_bytes().unwrap()).unwrap();
            assert_eq!(sets, a2(loaded.predict_set(x.view(), 0.1).unwrap()));
        }
    }

    #[test]
    fn conformal_errors_name_the_problem() {
        let (x, y) = regression();
        let mut bad = super::make(
            "SplitConformalRegressor",
            &json!({ "estimator": "NoSuchModel" }),
        )
        .unwrap();
        let err = bad
            .fit(x.view(), Some(y.view()), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("NoSuchModel"), "{err}");
        let mut plain = super::make("OlsRegressor", &json!({})).unwrap();
        plain.fit(x.view(), Some(y.view()), None).unwrap();
        let err = plain
            .predict_interval(x.view(), 0.1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("predict_interval"), "{err}");
    }
}
