//! Cluster 3 well-separated Gaussian blobs with KMeans and print per-cluster
//! centroids + inertia.

use anofox_ml::prelude::*;
use ndarray::{Array1, Array2};

fn main() {
    // Generate 90 points: 3 blobs of 30 each.
    let centers = [(0.0_f64, 0.0), (8.0, 8.0), (16.0, 0.0)];
    let mut data = Vec::with_capacity(90 * 2);
    let mut labels = Vec::with_capacity(90);
    for (c, &(cx, cy)) in centers.iter().enumerate() {
        for i in 0..30 {
            let t = i as f64 * 0.05;
            data.push(cx + t.sin() * 0.5);
            data.push(cy + t.cos() * 0.5);
            labels.push(c as f64);
        }
    }
    let x = Array2::from_shape_vec((90, 2), data).unwrap();
    let _y_true: Array1<f64> = labels.into();

    let km = KMeans::new(3).with_seed(0);
    let fitted = FitUnsupervised::<f64>::fit(&km, &x).unwrap();
    println!("Inertia: {:.4}", fitted.inertia());
    println!("Centroids:");
    for (i, row) in fitted.centroids().rows().into_iter().enumerate() {
        println!("  cluster {i}: ({:.3}, {:.3})", row[0], row[1]);
    }
    println!("Iterations: {}", fitted.n_iter());
}
