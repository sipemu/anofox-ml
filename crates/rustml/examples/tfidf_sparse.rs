//! Vectorise a small corpus with TfidfVectorizer's sparse output path and
//! print the density of the resulting CSR matrix.

use rustml::prelude::*;

fn main() {
    let docs = [
        "the quick brown fox jumps over the lazy dog",
        "rust is a systems programming language",
        "machine learning with rust and python",
        "the fox and the dog are friends",
        "python and rust both have strong type systems",
    ];
    let tv = TfidfVectorizer::new();
    let (vocab, csr) = tv.fit_transform_sparse(&docs).unwrap();
    println!("Vocab size: {}", vocab.len());
    println!("Sparse matrix shape: {} × {}", csr.n_rows, csr.n_cols);
    println!("Non-zeros: {}", csr.nnz());
    println!("Density: {:.3}%", csr.density() * 100.0);
    println!("First doc's non-zero entries:");
    for (col, val) in csr.row_iter(0) {
        println!("  {:20} → {:.4}", vocab[col], val);
    }
}
