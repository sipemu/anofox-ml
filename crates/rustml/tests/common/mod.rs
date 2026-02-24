use ndarray::{Array1, Array2};
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Load a golden data JSON file and return the parsed cases.
pub fn load_golden_data(filename: &str) -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden_data")
        .join(filename);
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&contents).expect("Failed to parse JSON")
}

/// Parse a JSON array of numbers into an ndarray Array1<f64>.
pub fn json_to_array1(val: &Value) -> Array1<f64> {
    let vec: Vec<f64> = val
        .as_array()
        .expect("expected JSON array")
        .iter()
        .map(|v| v.as_f64().expect("expected number"))
        .collect();
    Array1::from_vec(vec)
}

/// Parse a JSON 2D array into an ndarray Array2<f64>.
pub fn json_to_array2(val: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = val
        .as_array()
        .expect("expected JSON array of arrays")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("expected inner array")
                .iter()
                .map(|v| v.as_f64().expect("expected number"))
                .collect()
        })
        .collect();

    let nrows = rows.len();
    let ncols = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((nrows, ncols), flat).expect("shape mismatch")
}

/// Assert two f64 values are within tolerance.
pub fn assert_close(actual: f64, expected: f64, tol: f64, context: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < tol,
        "{}: expected {}, got {}, diff {} > tol {}",
        context,
        expected,
        actual,
        diff,
        tol
    );
}

/// Assert two Array1<f64> are element-wise within tolerance.
pub fn assert_array1_close(actual: &Array1<f64>, expected: &Array1<f64>, tol: f64, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch: {} vs {}",
        context,
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, tol, &format!("{}[{}]", context, i));
    }
}

/// Assert two Array2<f64> are element-wise within tolerance.
#[allow(dead_code)]
pub fn assert_array2_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64, context: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{}: shape mismatch", context);
    for ((r, c), &a) in actual.indexed_iter() {
        let e = expected[[r, c]];
        assert_close(a, e, tol, &format!("{}[{},{}]", context, r, c));
    }
}
