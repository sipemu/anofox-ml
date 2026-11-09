//! The text vectorizer kinds (#183): `fit_text` / `transform_text` on
//! documents, dense output, byte round trip.

use anofox_ml_registry::{load, make, Out};
use serde_json::json;

const DOCS: [&str; 4] = [
    "The quick brown fox",
    "the lazy dog, the lazy cat",
    "Quick quick fox!",
    "a dog and a fox",
];

fn dense(o: Out) -> ndarray::Array2<f64> {
    match o {
        Out::A2(a) => a,
        Out::A1(_) => panic!("expected a 2-D output"),
    }
}

#[test]
fn count_vectorizer_matches_the_text_crate() {
    let mut m = make("CountVectorizer", &json!({})).unwrap();
    assert_eq!(m.capabilities(), vec!["fit_text", "transform_text"]);
    m.fit_text(&DOCS).unwrap();
    let m = load("CountVectorizer", &m.to_bytes().unwrap()).unwrap();
    let got = dense(
        m.transform_text(&["fox and the dog", "unknown words"])
            .unwrap(),
    );
    let direct = anofox_ml_text::CountVectorizer::new().fit(&DOCS).unwrap();
    assert_eq!(
        got,
        direct
            .transform(&["fox and the dog", "unknown words"])
            .unwrap()
    );
    let vocab: serde_json::Value = serde_json::from_str(&m.fitted_json().unwrap()).unwrap();
    assert_eq!(vocab["vocabulary"][0], "and");
    assert_eq!(got.row(1).sum(), 0.0);
}

#[test]
fn tfidf_and_hashing_vectorizers() {
    let mut m = make("TfidfVectorizer", &json!({"min_df": 2})).unwrap();
    m.fit_text(&DOCS).unwrap();
    let m = load("TfidfVectorizer", &m.to_bytes().unwrap()).unwrap();
    let got = dense(m.transform_text(&DOCS).unwrap());
    let direct = anofox_ml_text::TfidfVectorizer::new()
        .with_min_df(2)
        .fit(&DOCS)
        .unwrap();
    assert_eq!(got, direct.transform(&DOCS).unwrap());

    let mut h = make("HashingVectorizer", &json!({"n_features": 16})).unwrap();
    h.fit_text(&DOCS).unwrap();
    let got = dense(h.transform_text(&DOCS).unwrap());
    assert_eq!(got.dim(), (4, 16));
    assert_eq!(
        got,
        anofox_ml_text::HashingVectorizer::new(16).transform(&DOCS)
    );
}

#[test]
fn numeric_fit_is_rejected() {
    let mut m = make("CountVectorizer", &json!({})).unwrap();
    let x = ndarray::Array2::<f64>::zeros((2, 2));
    let err = m.fit(x.view(), None, None).unwrap_err().to_string();
    assert!(err.contains("fit_text"), "{err}");
    assert!(m.transform_text(&DOCS).is_err());
}
