//! LatentDirichletAllocation against scikit-learn 1.8.0
//! (`gen_lda_topics.py`): seeded batch and online fits.

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_preprocessing::{LatentDirichletAllocation, LdaLearning};
use common::{json_to_array2, load_golden_file};

fn close(o: f64, e: f64, tol: f64, what: &str) {
    assert!((o - e).abs() <= tol * (1.0 + e.abs()), "{what}: {o} vs {e}");
}

#[test]
fn lda_topics_match_scikit_learn() {
    let data = load_golden_file("lda_topics.json");
    let x = json_to_array2(&data["X"]);
    for case in data["cases"].as_array().unwrap() {
        let p = &case["params"];
        let m = LatentDirichletAllocation {
            n_components: p["n_components"].as_u64().unwrap() as usize,
            max_iter: p["max_iter"].as_u64().unwrap_or(10) as usize,
            doc_topic_prior: p["doc_topic_prior"].as_f64(),
            topic_word_prior: p["topic_word_prior"].as_f64(),
            learning_method: if p["learning_method"] == "online" {
                LdaLearning::Online
            } else {
                LdaLearning::Batch
            },
            batch_size: p["batch_size"].as_u64().unwrap_or(128) as usize,
            evaluate_every: p["evaluate_every"].as_i64().unwrap_or(-1),
            perp_tol: p["perp_tol"].as_f64().unwrap_or(0.1),
            seed: p["random_state"].as_u64().unwrap() as u32,
            ..Default::default()
        };
        let f = m.fit(&x).unwrap();
        assert_eq!(
            f.n_iter as u64,
            case["n_iter"].as_u64().unwrap(),
            "{p}: n_iter"
        );
        for (o, e) in f
            .components
            .iter()
            .zip(&json_to_array2(&case["components"]))
        {
            close(*o, *e, 1e-10, &format!("{p}: components"));
        }
        let t = f
            .transform(&x.slice(ndarray::s![..6, ..]).to_owned())
            .unwrap();
        for (o, e) in t.iter().zip(&json_to_array2(&case["transform"])) {
            close(*o, *e, 1e-10, &format!("{p}: transform"));
        }
        close(
            f.bound,
            case["bound"].as_f64().unwrap(),
            1e-10,
            &format!("{p}: bound"),
        );
        // gammaln is a Lanczos approximation here (~1e-13 relative).
        close(
            f.score(x.view()).unwrap(),
            case["score"].as_f64().unwrap(),
            1e-10,
            "score",
        );
        close(
            f.perplexity(x.view()).unwrap(),
            case["perplexity"].as_f64().unwrap(),
            1e-10,
            "ppl",
        );
    }
}
