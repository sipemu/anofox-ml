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

mod compose;
mod encoders;
mod estimator;
mod meta;
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
    fn encoders_and_select_from_model() {
        use ndarray::array;
        let a1 = |o| match o {
            super::Out::A1(a) => a,
            super::Out::A2(_) => panic!("expected 1-D output"),
        };
        let a2 = |o| match o {
            super::Out::A2(a) => a,
            super::Out::A1(_) => panic!("expected 2-D output"),
        };
        let mut le = super::make("LabelEncoder", &serde_json::json!({})).unwrap();
        le.fit(array![[10.0], [2.0], [10.0]].view(), None, None)
            .unwrap();
        assert_eq!(
            a1(le.transform(array![[2.0], [10.0]].view()).unwrap()),
            array![0.0, 1.0]
        );
        assert_eq!(
            a1(le.inverse_transform(array![[1.0]].view()).unwrap()),
            array![10.0]
        );
        assert!(le.transform(array![[3.0]].view()).is_err());

        let nan = f64::NAN;
        let mut oe = super::make(
            "OrdinalEncoder",
            &serde_json::json!({"handle_unknown": {"UseEncodedValue": null}}),
        )
        .unwrap();
        oe.fit(
            array![[3.0, 1.0], [1.0, nan], [3.0, 1.0]].view(),
            None,
            None,
        )
        .unwrap();
        let t = a2(oe.transform(array![[1.0, nan], [7.0, 1.0]].view()).unwrap());
        assert_eq!(t.row(0)[0], 0.0);
        assert!(t[[0, 1]].is_nan() && t[[1, 0]].is_nan());
        assert_eq!(t[[1, 1]], 0.0);
        let loaded = super::load("OrdinalEncoder", &oe.to_bytes().unwrap()).unwrap();
        assert_eq!(
            a2(loaded.inverse_transform(array![[1.0, 0.0]].view()).unwrap()),
            array![[3.0, 1.0]]
        );

        // y depends on x0 and x2 only: Ridge's |coef| above the mean.
        let x =
            ndarray::Array2::from_shape_fn((30, 4), |(i, j)| (i as f64 * (j as f64 + 1.3)).sin());
        let y = x.column(0).mapv(|v| 3.0 * v) - x.column(2).mapv(|v| 2.0 * v);
        let params = serde_json::json!({"estimator": "RidgeRegressor",
                                        "estimator_params": {"lambda": 1e-6}});
        let mut sfm = super::make("SelectFromModel", &params).unwrap();
        sfm.fit(x.view(), Some(y.view()), None).unwrap();
        let state: serde_json::Value = serde_json::from_str(&sfm.fitted_json().unwrap()).unwrap();
        assert_eq!(
            state["support"],
            serde_json::json!([true, false, true, false])
        );
        assert_eq!(a2(sfm.transform(x.view()).unwrap()).ncols(), 2);
    }

    #[test]
    fn multi_output_classifiers_and_output_codes() {
        let x = ndarray::Array2::from_shape_fn((40, 2), |(i, j)| {
            (i as f64 * (j as f64 + 0.7)).sin() * 3.0
        });
        // Two targets: labels {2, 5} and {0, 1}, both from x0 / x1.
        let y0 = x.column(0).mapv(|v| if v > 0.0 { 5.0 } else { 2.0 });
        let y1 = x.column(1).mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
        let xy = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                x.view(),
                y0.view().insert_axis(ndarray::Axis(1)),
                y1.view().insert_axis(ndarray::Axis(1)),
            ],
        )
        .unwrap();
        let tree = serde_json::json!({"estimator": "DecisionTreeClassifier",
                                      "estimator_params": {}, "y_columns": 2});
        for kind in ["MultiOutputClassifier", "ClassifierChain"] {
            let mut m = super::make(kind, &tree).unwrap();
            m.fit(xy.view(), None, None).unwrap();
            let super::Out::A2(pred) = m.predict(x.view()).unwrap() else {
                panic!("{kind}: 2-D predictions")
            };
            // A full tree reproduces its training labels.
            assert_eq!(pred.column(0), y0, "{kind}");
            assert_eq!(pred.column(1), y1, "{kind}");
            let loaded = super::load(kind, &m.to_bytes().unwrap()).unwrap();
            assert!(loaded.predict_proba(x.view()).is_ok(), "{kind}");
        }

        // Three classes, a code book with a constant column.
        let y = x
            .column(0)
            .mapv(|v| (v > -1.0) as u8 as f64 + (v > 1.0) as u8 as f64);
        let params = serde_json::json!({
            "estimator": "DecisionTreeClassifier", "estimator_params": {},
            "code_book": [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]});
        let mut ecoc = super::make("OutputCodeClassifier", &params).unwrap();
        ecoc.fit(x.view(), Some(y.view()), None).unwrap();
        let super::Out::A1(pred) = ecoc.predict(x.view()).unwrap() else {
            panic!("1-D predictions")
        };
        assert_eq!(pred, y);
        let bad = serde_json::json!({"code_book": [[1.0], [0.0]]});
        let mut short = super::make("OutputCodeClassifier", &bad).unwrap();
        assert!(short.fit(x.view(), Some(y.view()), None).is_err());
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
    fn perpetual_as_a_cqr_estimator() {
        let x = ndarray::Array2::from_shape_fn((200, 2), |(i, j)| ((i * (j + 2)) % 23) as f64);
        let y = x.column(0).mapv(|v| 0.5 * v) - &x.column(1);
        let params = serde_json::json!({
            "estimator": "PerpetualRegressor",
            "estimator_params": { "num_threads": 1 },
            "alpha": 0.2,
        });
        let mut model = super::make("ConformalizedQuantileRegressor", &params).unwrap();
        model.fit(x.view(), Some(y.view()), None).unwrap();
        match model.predict_interval(x.view(), 0.2).unwrap() {
            super::Out::A2(iv) => {
                assert_eq!(iv.dim(), (200, 2));
                assert!(iv.rows().into_iter().all(|r| r[0] <= r[1]));
            }
            _ => panic!("expected n × 2 intervals"),
        }
    }

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
    fn cqr_kind_with_each_quantile_estimator() {
        let (x, y) = regression();
        for (est, params) in [
            ("QuantileRegressor", json!({})),
            (
                "GradientBoostingRegressor",
                json!({ "n_estimators": 30, "max_depth": 2 }),
            ),
            (
                "HistGradientBoostingRegressor",
                json!({ "n_estimators": 30 }),
            ),
        ] {
            let mut model = super::make(
                "ConformalizedQuantileRegressor",
                &json!({ "estimator": est, "estimator_params": params, "alpha": 0.2 }),
            )
            .unwrap();
            model.fit(x.view(), Some(y.view()), None).unwrap();
            let iv = a2(model.predict_interval(x.view(), 0.2).unwrap());
            assert_eq!(iv.dim(), (200, 2), "{est}");
            assert!(iv.rows().into_iter().all(|r| r[0] <= r[1]), "{est}");
            let covered = (0..200)
                .filter(|&i| iv[[i, 0]] <= y[i] && y[i] <= iv[[i, 1]])
                .count();
            assert!(covered >= 150, "{est}: {covered} of 200");
            let err = model
                .predict_interval(x.view(), 0.1)
                .unwrap_err()
                .to_string();
            assert!(err.contains("refit"), "{err}");
            let loaded =
                super::load("ConformalizedQuantileRegressor", &model.to_bytes().unwrap()).unwrap();
            assert_eq!(iv, a2(loaded.predict_interval(x.view(), 0.2).unwrap()));
        }
        let mut bad = super::make(
            "ConformalizedQuantileRegressor",
            &json!({ "estimator": "RidgeRegressor" }),
        )
        .unwrap();
        let err = bad
            .fit(x.view(), Some(y.view()), None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("RidgeRegressor") && err.contains("supported"),
            "{err}"
        );
    }

    #[test]
    fn cqr_matches_the_typed_wrapper() {
        let (x, y) = regression();
        let mut model = super::make(
            "ConformalizedQuantileRegressor",
            &json!({ "alpha": 0.2, "seed": 3 }),
        )
        .unwrap();
        model.fit(x.view(), Some(y.view()), None).unwrap();
        let typed = anofox_ml_conformal::ConformalizedQuantileRegressor::new(0.2, |q| {
            anofox_ml_regression::QuantileRegressor::new(q)
        })
        .with_seed(3)
        .fit(&x, &y)
        .unwrap();
        let (lo, hi) = typed.predict_interval(x.view()).unwrap();
        let iv = a2(model.predict_interval(x.view(), 0.2).unwrap());
        assert_eq!(iv.column(0), lo);
        assert_eq!(iv.column(1), hi);
    }

    /// Column 0 is the group (0 or 1); group 1 is much noisier.
    fn grouped_regression() -> (Array2<f64>, Array1<f64>) {
        let n = 400;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => (i % 2) as f64,
            _ => ((i * (j + 3)) % 37) as f64 / 37.0,
        });
        let y = (0..n)
            .map(|i| {
                let noise = ((i * 7919) % 101) as f64 / 101.0 - 0.5;
                x[[i, 1]] - x[[i, 2]] + noise * if i % 2 == 1 { 4.0 } else { 0.2 }
            })
            .collect();
        (x, y)
    }

    #[test]
    fn mondrian_regressor_kind_widens_the_noisy_group() {
        let (x, y) = grouped_regression();
        let params =
            json!({ "estimator": "OlsRegressor", "group_column": 0, "calibration_size": 0.5 });
        let mut model = super::make("MondrianConformalRegressor", &params).unwrap();
        model.fit(x.view(), Some(y.view()), None).unwrap();
        let iv = a2(model.predict_interval(x.view(), 0.1).unwrap());
        let width = |i: usize| iv[[i, 1]] - iv[[i, 0]];
        assert!(width(1) > 5.0 * width(0), "{} vs {}", width(1), width(0));
        let loaded = super::load("MondrianConformalRegressor", &model.to_bytes().unwrap()).unwrap();
        assert_eq!(iv, a2(loaded.predict_interval(x.view(), 0.1).unwrap()));
        // Group ids must be non-negative integers, and the column must exist.
        let mut bad = x.clone();
        bad[[0, 0]] = 0.5;
        assert!(model.predict_interval(bad.view(), 0.1).is_err());
        let mut out_of_range =
            super::make("MondrianConformalRegressor", &json!({ "group_column": 9 })).unwrap();
        let err = out_of_range
            .fit(x.view(), Some(y.view()), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("group_column"), "{err}");
    }

    #[test]
    fn mondrian_classifier_kind_gives_sets_per_group() {
        let n = 400;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            2 => (i % 3) as f64,
            _ => ((i * (j + 5)) % 41) as f64 / 41.0,
        });
        let y: Array1<f64> = x.column(0).mapv(|v| {
            if v < 0.33 {
                0.0
            } else if v < 0.66 {
                1.0
            } else {
                2.0
            }
        });
        let params = json!({ "estimator": "RandomForestClassifier", "group_column": 2 });
        let mut model = super::make("MondrianConformalClassifier", &params).unwrap();
        model.fit(x.view(), Some(y.view()), None).unwrap();
        let sets = a2(model.predict_set(x.view(), 0.2).unwrap());
        assert_eq!(sets.dim(), (n, 3));
        let covered = (0..n).filter(|&i| sets[[i, y[i] as usize]] == 1.0).count();
        assert!(covered as f64 / n as f64 >= 0.75, "{covered}");
        assert_eq!(a2(model.predict_proba(x.view()).unwrap()).dim(), (n, 3));
        match model.predict(x.view()).unwrap() {
            super::Out::A1(p) => assert_eq!(p.len(), n),
            super::Out::A2(_) => panic!("expected 1-D predictions"),
        }
        let loaded =
            super::load("MondrianConformalClassifier", &model.to_bytes().unwrap()).unwrap();
        assert_eq!(sets, a2(loaded.predict_set(x.view(), 0.2).unwrap()));
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

#[cfg(test)]
mod boosting_surface_tests {
    use super::Out;
    use ndarray::{Array1, Array2};
    use serde_json::json;

    fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((120, 4), |(i, j)| ((i * (j + 2)) as f64 * 0.37).sin());
        let y = Array1::from_shape_fn(120, |i| 2.0 * x[[i, 0]] - x[[i, 2]]);
        let labels = y.mapv(|v| f64::from(v > 0.0) + f64::from(v > 1.0));
        (x, y, labels)
    }

    fn a1(out: Out) -> Array1<f64> {
        match out {
            Out::A1(a) => a,
            Out::A2(_) => panic!("expected 1-D"),
        }
    }

    fn a2(out: Out) -> Array2<f64> {
        match out {
            Out::A2(a) => a,
            Out::A1(_) => panic!("expected 2-D"),
        }
    }

    const SMALL: &[(&str, &str)] = &[
        ("DecisionTreeClassifier", "{}"),
        ("DecisionTreeRegressor", "{}"),
        ("RandomForestClassifier", r#"{"n_estimators": 5}"#),
        ("RandomForestRegressor", r#"{"n_estimators": 5}"#),
        ("ExtraTreesClassifier", r#"{"n_estimators": 5}"#),
        ("ExtraTreesRegressor", r#"{"n_estimators": 5}"#),
        ("GradientBoostingClassifier", r#"{"n_estimators": 6}"#),
        ("GradientBoostingRegressor", r#"{"n_estimators": 6}"#),
        ("BaggingClassifier", r#"{"n_estimators": 4}"#),
        ("BaggingRegressor", r#"{"n_estimators": 4}"#),
        ("HistGradientBoostingClassifier", r#"{"n_estimators": 6}"#),
        ("HistGradientBoostingRegressor", r#"{"n_estimators": 6}"#),
        (
            "LgbmClassifier",
            r#"{"n_estimators": 6, "min_child_samples": 5}"#,
        ),
        (
            "LgbmRegressor",
            r#"{"n_estimators": 6, "min_child_samples": 5}"#,
        ),
    ];

    fn fitted(kind: &str, params: &str) -> Box<dyn super::DynModel> {
        let (x, y, labels) = data();
        let target = if kind.ends_with("Classifier") {
            labels
        } else {
            y
        };
        let params: serde_json::Value = serde_json::from_str(params).unwrap();
        let mut m = super::make(kind, &params).unwrap_or_else(|e| panic!("{kind}: {e}"));
        m.fit(x.view(), Some(target.view()), None).unwrap();
        m
    }

    #[test]
    fn tree_and_boosting_kinds_give_feature_importances() {
        let (x, ..) = data();
        for &(kind, params) in SMALL {
            let m = fitted(kind, params);
            assert!(m.capabilities().contains(&"feature_importances"), "{kind}");
            let imp = a1(m.feature_importances().unwrap());
            assert_eq!(imp.len(), x.ncols(), "{kind}");
            assert!(imp.iter().all(|&v| v >= 0.0), "{kind}: {imp}");
            assert!((imp.sum() - 1.0).abs() < 1e-9, "{kind}: {imp}");
            // A restored model gives the same importances.
            let loaded = super::load(kind, &m.to_bytes().unwrap()).unwrap();
            assert_eq!(a1(loaded.feature_importances().unwrap()), imp, "{kind}");
        }
        let unfitted = super::make("LgbmRegressor", &json!({})).unwrap();
        assert!(unfitted.feature_importances().is_err());
        let linear = super::make("OlsRegressor", &json!({})).unwrap();
        assert!(linear.feature_importances().is_err());
    }

    #[test]
    fn boosting_kinds_give_staged_predictions() {
        let (x, ..) = data();
        for &(kind, params) in SMALL {
            let m = fitted(kind, params);
            let staged = m.capabilities().contains(&"staged_predict");
            let boosting = kind.contains("Boosting") || kind.starts_with("Lgbm");
            assert_eq!(staged, boosting, "{kind}");
            if !staged {
                assert!(m.staged_predict(x.view()).is_err(), "{kind}");
                continue;
            }
            let stages = a2(m.staged_predict(x.view()).unwrap());
            assert_eq!(stages.dim(), (x.nrows(), 6), "{kind}");
            let last = stages.column(5).to_owned();
            assert_eq!(last, a1(m.predict(x.view()).unwrap()), "{kind}");
        }
    }

    #[test]
    fn gradient_boosting_warm_start_continues_the_fit() {
        let (x, y, labels) = data();
        for (kind, target) in [
            ("GradientBoostingRegressor", &y),
            ("GradientBoostingClassifier", &labels),
        ] {
            let params = json!({ "n_estimators": 4, "subsample": 0.7 });
            let mut m = super::make(kind, &params).unwrap();
            assert!(m.capabilities().contains(&"warm_start"), "{kind}");
            assert!(m.warm_start(x.view(), Some(target.view()), None).is_err());
            m.fit(x.view(), Some(target.view()), None).unwrap();
            m.set_params(&json!({ "n_estimators": 9 })).unwrap();
            m.warm_start(x.view(), Some(target.view()), None).unwrap();
            let mut once =
                super::make(kind, &json!({ "n_estimators": 9, "subsample": 0.7 })).unwrap();
            once.fit(x.view(), Some(target.view()), None).unwrap();
            assert_eq!(
                a1(m.predict(x.view()).unwrap()),
                a1(once.predict(x.view()).unwrap()),
                "{kind}"
            );
            // Fewer stages than fitted is an error and keeps the model.
            m.set_params(&json!({ "n_estimators": 3 })).unwrap();
            assert!(m.warm_start(x.view(), Some(target.view()), None).is_err());
            assert_eq!(a2(m.staged_predict(x.view()).unwrap()).ncols(), 9, "{kind}");
        }
    }

    #[test]
    fn lgbm_and_hist_gradient_boosting_warm_start() {
        let (x, y, labels) = data();
        for (kind, target, base) in [
            (
                "LgbmRegressor",
                &y,
                json!({ "min_child_samples": 5, "subsample": 0.8, "subsample_freq": 2,
                        "colsample_bytree": 0.7 }),
            ),
            (
                "LgbmClassifier",
                &labels,
                json!({ "min_child_samples": 5, "subsample": 0.8, "subsample_freq": 1 }),
            ),
            (
                "HistGradientBoostingRegressor",
                &y,
                json!({ "early_stopping": "Disabled" }),
            ),
            (
                "HistGradientBoostingClassifier",
                &labels,
                json!({ "early_stopping": "Disabled" }),
            ),
        ] {
            let mut m = super::make(kind, &base).unwrap();
            m.set_params(&json!({ "n_estimators": 3 })).unwrap();
            m.fit(x.view(), Some(target.view()), None).unwrap();
            m.set_params(&json!({ "n_estimators": 8 })).unwrap();
            m.warm_start(x.view(), Some(target.view()), None).unwrap();
            let mut once = super::make(kind, &base).unwrap();
            once.set_params(&json!({ "n_estimators": 8 })).unwrap();
            once.fit(x.view(), Some(target.view()), None).unwrap();
            assert_eq!(
                a2(m.staged_predict(x.view()).unwrap()),
                a2(once.staged_predict(x.view()).unwrap()),
                "{kind}"
            );
        }
    }

    #[test]
    fn set_params_merges_and_validates() {
        let mut m = super::make("LgbmRegressor", &json!({ "n_estimators": 7 })).unwrap();
        m.set_params(&json!({ "learning_rate": 0.3 })).unwrap();
        let p = m.params().unwrap();
        assert_eq!(
            (p["n_estimators"].as_u64(), p["learning_rate"].as_f64()),
            (Some(7), Some(0.3))
        );
        assert!(m.set_params(&json!({ "no_such": 1 })).is_err());
        assert_eq!(m.params().unwrap()["learning_rate"].as_f64(), Some(0.3));
    }

    #[test]
    fn hist_gradient_boosting_fits_with_an_explicit_validation_set() {
        use anofox_ml_ensemble as ens;
        let (x, y, _) = data();
        let (xt, yt) = (x.slice(ndarray::s![..80, ..]), y.slice(ndarray::s![..80]));
        let (xv, yv) = (x.slice(ndarray::s![80.., ..]), y.slice(ndarray::s![80..]));
        let params =
            json!({ "n_estimators": 300, "early_stopping": "Enabled", "n_iter_no_change": 3 });
        let mut m = super::make("HistGradientBoostingRegressor", &params).unwrap();
        assert!(m.capabilities().contains(&"fit_with_validation"));
        m.fit_with_validation(xt, Some(yt), None, xv, Some(yv), None)
            .unwrap();
        let typed = ens::HistGradientBoostingRegressor::new()
            .with_n_estimators(300)
            .with_early_stopping(ens::EarlyStopping::Enabled)
            .with_n_iter_no_change(3)
            .fit_with_validation(xt, yt, None, xv, yv, None)
            .unwrap();
        let stages = a2(m.staged_predict(x.view()).unwrap()).ncols();
        assert!(stages < 300, "early stopping did not stop: {stages}");
        assert_eq!(stages, typed.n_estimators());
        assert!(m
            .fit_with_validation(xt, None, None, xv, Some(yv), None)
            .is_err());
        let mut lgbm = super::make("LgbmRegressor", &json!({})).unwrap();
        assert!(lgbm
            .fit_with_validation(xt, Some(yt), None, xv, Some(yv), None)
            .is_err());
        // CatBoost: its eval set, overfitting detector and use_best_model.
        let params = json!({ "iterations": 200, "learning_rate": 0.5, "od_type": "Iter",
                             "od_wait": 3 });
        let mut cb = super::make("CatBoostRegressor", &params).unwrap();
        assert!(cb.capabilities().contains(&"fit_with_validation"));
        cb.fit_with_validation(xt, Some(yt), None, xv, Some(yv), None)
            .unwrap();
        let typed = ens::CatBoostRegressor::new()
            .with_iterations(200)
            .with_learning_rate(0.5)
            .with_early_stopping_rounds(3)
            .fit_with_eval(
                xt,
                yt,
                None,
                ens::CatBoostEvalSet {
                    x: xv,
                    y: yv,
                    sample_weight: None,
                },
            )
            .unwrap();
        assert!(typed.n_estimators() < 200);
        use anofox_ml_core::Predict;
        assert_eq!(
            a1(cb.predict(x.view()).unwrap()),
            typed.predict(&x).unwrap()
        );
    }

    #[test]
    fn extra_trees_classifier_gives_probabilities() {
        let m = fitted("ExtraTreesClassifier", r#"{"n_estimators": 5}"#);
        let (x, ..) = data();
        assert!(m.capabilities().contains(&"predict_proba"));
        let p = a2(m.predict_proba(x.view()).unwrap());
        assert_eq!(p.dim(), (x.nrows(), 3));
        assert!(p.rows().into_iter().all(|r| (r.sum() - 1.0).abs() < 1e-12));
    }
}
