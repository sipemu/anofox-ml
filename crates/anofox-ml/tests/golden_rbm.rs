//! BernoulliRBM against scikit-learn 1.8.0 (`gen_rbm.py`): seeded fits
//! follow numpy's RandomState draw for draw.

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_neural_networks::BernoulliRbm;
use common::{json_to_array1, json_to_array2, load_golden_file};

fn close<'a>(
    o: impl IntoIterator<Item = &'a f64>,
    e: impl IntoIterator<Item = &'a f64>,
    what: &str,
) {
    for (o, e) in o.into_iter().zip(e) {
        assert!(
            (o - e).abs() <= 1e-12 * (1.0 + e.abs()),
            "{what}: {o} vs {e}"
        );
    }
}

#[test]
fn bernoulli_rbm_matches_scikit_learn() {
    let data = load_golden_file("rbm.json");
    let x = json_to_array2(&data["X"]);
    for case in data["cases"].as_array().unwrap() {
        let p = &case["params"];
        let m = BernoulliRbm {
            n_components: p["n_components"].as_u64().unwrap() as usize,
            batch_size: p["batch_size"].as_u64().unwrap_or(10) as usize,
            n_iter: p["n_iter"].as_u64().unwrap_or(10) as usize,
            learning_rate: p["learning_rate"].as_f64().unwrap_or(0.1),
            seed: p["random_state"].as_u64().unwrap() as u32,
        };
        let mut f = m.fit(&x).unwrap();
        close(
            &f.components,
            &json_to_array2(&case["components"]),
            "components",
        );
        close(
            &f.intercept_hidden,
            &json_to_array1(&case["intercept_hidden"]),
            "b_h",
        );
        close(
            &f.intercept_visible,
            &json_to_array1(&case["intercept_visible"]),
            "b_v",
        );
        let head = x.slice(ndarray::s![..5, ..]);
        close(
            &f.transform(&head.to_owned()).unwrap(),
            &json_to_array2(&case["transform"]),
            "transform",
        );
        close(
            &f.score_samples(head).unwrap(),
            &json_to_array1(&case["score_samples"]),
            "score",
        );
        let g = f.gibbs(x.slice(ndarray::s![..4, ..])).unwrap();
        close(&g, &json_to_array2(&case["gibbs"]), "gibbs");
    }
    let m = BernoulliRbm {
        n_components: 3,
        batch_size: 5,
        seed: 4,
        ..Default::default()
    };
    let mut f = m.partial_fit_first(x.slice(ndarray::s![0..5, ..])).unwrap();
    f.partial_fit(x.slice(ndarray::s![5..12, ..])).unwrap();
    f.partial_fit(x.slice(ndarray::s![12.., ..])).unwrap();
    let pf = &data["partial_fit"];
    close(
        &f.components,
        &json_to_array2(&pf["components"]),
        "partial components",
    );
    close(
        &f.intercept_visible,
        &json_to_array1(&pf["intercept_visible"]),
        "partial b_v",
    );
}
