//! Golden tests for the covariance estimators (#113, batch 4) against
//! scikit-learn 1.8.0 (`test_harness/generators/gen_covariance.py`).

mod common;

use anofox_ml_core::FitUnsupervised;
use anofox_ml_preprocessing::{
    EllipticEnvelope, EmpiricalCovariance, FittedCovariance, LedoitWolf, MinCovDet, Oas,
    ShrunkCovariance,
};
use common::{
    assert_array1_close, assert_array2_close, assert_close, json_to_array1, json_to_array2,
};
use serde_json::Value;

fn golden() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden_data/covariance.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn indices(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_u64().unwrap() as usize)
        .collect()
}

fn mask_indices(m: &[bool]) -> Vec<usize> {
    (0..m.len()).filter(|&i| m[i]).collect()
}

#[test]
fn plain_estimators_match_sklearn() {
    for c in golden()["plain"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let centered = c["params"]["assume_centered"].as_bool().unwrap_or(false);
        let x = json_to_array2(&c["X"]);
        let f: FittedCovariance = match name.split('_').next().unwrap() {
            "empirical" => EmpiricalCovariance {
                assume_centered: centered,
            }
            .fit(&x),
            "shrunk" => ShrunkCovariance {
                assume_centered: centered,
                shrinkage: c["params"]["shrinkage"].as_f64().unwrap(),
            }
            .fit(&x),
            "ledoit" => LedoitWolf {
                assume_centered: centered,
            }
            .fit(&x),
            _ => Oas {
                assume_centered: centered,
            }
            .fit(&x),
        }
        .unwrap();
        let xt = json_to_array2(&c["X_test"]);
        assert_array1_close(f.location(), &json_to_array1(&c["location"]), 1e-12, name);
        assert_array2_close(
            f.covariance(),
            &json_to_array2(&c["covariance"]),
            1e-12,
            name,
        );
        assert_array2_close(f.precision(), &json_to_array2(&c["precision"]), 1e-9, name);
        if let Some(s) = f.shrinkage() {
            assert_close(s, c["shrinkage"].as_f64().unwrap(), 1e-12, name);
        }
        assert_array1_close(
            &f.mahalanobis(xt.view()).unwrap(),
            &json_to_array1(&c["mahalanobis"]),
            1e-8,
            name,
        );
        assert_close(
            f.score(xt.view()).unwrap(),
            c["score"].as_f64().unwrap(),
            1e-9,
            name,
        );
    }
}

#[test]
fn min_cov_det_and_envelope_match_sklearn() {
    for c in golden()["mcd"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let frac = c["params"]["support_fraction"].as_f64();
        let x = json_to_array2(&c["X"]);
        let m = MinCovDet {
            support_fraction: frac,
            ..Default::default()
        }
        .fit(&x)
        .unwrap();
        if name == "p1" {
            // Univariate: the two ends of the shortest half are exactly as
            // far from its midpoint, and which one enters the support is
            // up to the order numpy's (unstable, SIMD) argsort gives ties.
            // Check the location and that the supports differ in at most
            // that one swapped end.
            assert_array1_close(
                m.raw_location().unwrap(),
                &json_to_array1(&c["raw_location"]),
                1e-12,
                name,
            );
            let ours = mask_indices(m.raw_support().unwrap());
            let theirs = indices(&c["raw_support"]);
            let diff = ours.iter().filter(|i| !theirs.contains(i)).count();
            assert!(
                ours.len() == theirs.len() && diff <= 1,
                "{name}: {diff} rows differ"
            );
            continue;
        }
        assert_eq!(
            mask_indices(m.raw_support().unwrap()),
            indices(&c["raw_support"]),
            "{name}: raw support"
        );
        assert_array1_close(
            m.raw_location().unwrap(),
            &json_to_array1(&c["raw_location"]),
            1e-10,
            name,
        );
        assert_array2_close(
            m.raw_covariance().unwrap(),
            &json_to_array2(&c["raw_covariance"]),
            1e-10,
            name,
        );
        assert_eq!(
            mask_indices(m.support().unwrap()),
            indices(&c["support"]),
            "{name}: support"
        );
        assert_array1_close(m.location(), &json_to_array1(&c["location"]), 1e-10, name);
        assert_array2_close(
            m.covariance(),
            &json_to_array2(&c["covariance"]),
            1e-9,
            name,
        );
        assert_array1_close(m.dist().unwrap(), &json_to_array1(&c["dist"]), 1e-7, name);
        let env = EllipticEnvelope {
            support_fraction: frac,
            ..Default::default()
        }
        .fit(&x)
        .unwrap();
        assert_close(
            env.offset(),
            c["envelope_offset"].as_f64().unwrap(),
            1e-7,
            name,
        );
        assert_eq!(
            env.predict(x.view()).unwrap(),
            json_to_array1(&c["envelope_predict"]),
            "{name}"
        );
        assert_array1_close(
            &env.decision_function(x.view()).unwrap(),
            &json_to_array1(&c["envelope_decision"]),
            1e-7,
            name,
        );
    }
}
