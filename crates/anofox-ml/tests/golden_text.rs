//! Golden test for CountVectorizer / TfidfVectorizer against sklearn 1.8.0.

mod common;

use anofox_ml_text::{CountVectorizer, TfidfVectorizer};
use common::{json_to_array2, load_golden_data};

#[test]
fn test_count_vectorizer_matches_sklearn() {
    let cases = load_golden_data("text.json");
    let case = &cases[0];
    let docs: Vec<String> = case["docs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let sk_vocab: Vec<String> = case["vocab"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let sk_counts = json_to_array2(&case["count_matrix"]);

    let cv = CountVectorizer::new();
    let docs_ref: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let (vocab, x) = cv.fit_transform(&docs_ref).unwrap();
    assert_eq!(vocab, sk_vocab);
    assert_eq!(x.shape(), sk_counts.shape());
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert_eq!(x[[i, j]], sk_counts[[i, j]], "[{},{}]", i, j);
        }
    }
}

#[test]
fn test_tfidf_vectorizer_matches_sklearn() {
    let cases = load_golden_data("text.json");
    let case = &cases[0];
    let docs: Vec<String> = case["docs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let sk_tfidf = json_to_array2(&case["tfidf_matrix"]);

    let tv = TfidfVectorizer::new();
    let docs_ref: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let (_, x) = tv.fit_transform(&docs_ref).unwrap();
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert!(
                (x[[i, j]] - sk_tfidf[[i, j]]).abs() < 1e-6,
                "[{},{}]: {} vs {}",
                i,
                j,
                x[[i, j]],
                sk_tfidf[[i, j]]
            );
        }
    }
}
