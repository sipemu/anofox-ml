//! MiniBatchNMF against scikit-learn 1.8.0 (`gen_minibatch_nmf.py`).

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_preprocessing::nmf::BetaLoss;
use anofox_ml_preprocessing::MiniBatchNmf;
use common::{json_to_array2, load_golden_file};

#[test]
fn minibatch_nmf_matches_scikit_learn() {
    let data = load_golden_file("minibatch_nmf.json");
    let x = json_to_array2(&data["X"]);
    for case in data["cases"].as_array().unwrap() {
        let p = &case["params"];
        let m = MiniBatchNmf {
            n_components: p["n_components"].as_u64().unwrap() as usize,
            batch_size: p["batch_size"].as_u64().unwrap() as usize,
            max_iter: p["max_iter"].as_u64().unwrap_or(200) as usize,
            tol: p["tol"].as_f64().unwrap_or(1e-4),
            max_no_improvement: match p.get("max_no_improvement") {
                Some(v) if v.is_null() => None,
                Some(v) => Some(v.as_u64().unwrap() as usize),
                None => Some(10),
            },
            beta_loss: if p["beta_loss"] == "kullback-leibler" {
                BetaLoss::KullbackLeibler
            } else {
                BetaLoss::Frobenius
            },
            alpha_w: p["alpha_W"].as_f64().unwrap_or(0.0),
            l1_ratio: p["l1_ratio"].as_f64().unwrap_or(0.0),
            forget_factor: p["forget_factor"].as_f64().unwrap_or(0.7),
            ..Default::default()
        };
        let f = m.fit(&x).unwrap();
        assert_eq!(
            f.n_steps as u64,
            case["n_steps"].as_u64().unwrap(),
            "{p}: n_steps"
        );
        assert_eq!(
            f.n_iter as u64,
            case["n_iter"].as_u64().unwrap(),
            "{p}: n_iter"
        );
        let close = |o: f64, e: f64, what: &str| {
            assert!(
                (o - e).abs() <= 1e-9 * (1.0 + e.abs()),
                "{p}: {what} {o} vs {e}"
            )
        };
        for (o, e) in f
            .components
            .iter()
            .zip(&json_to_array2(&case["components"]))
        {
            close(*o, *e, "H");
        }
        for (o, e) in f.w.iter().zip(&json_to_array2(&case["W"])) {
            close(*o, *e, "W");
        }
        close(
            f.reconstruction_err,
            case["reconstruction_err"].as_f64().unwrap(),
            "err",
        );
        let t = f
            .transform(&x.slice(ndarray::s![..5, ..]).to_owned())
            .unwrap();
        for (o, e) in t.iter().zip(&json_to_array2(&case["transform"])) {
            close(*o, *e, "transform");
        }
    }
}
