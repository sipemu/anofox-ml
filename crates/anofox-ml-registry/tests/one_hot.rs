//! The OneHotEncoder kind on numeric category codes (#125).

use anofox_ml_registry::{load, make, Out};
use ndarray::array;
use serde_json::json;

#[test]
fn one_hot_encoder_round_trips() {
    let x = array![[1.0, 5.0], [2.0, 5.0], [1.0, 7.0]];
    let mut m = make("OneHotEncoder", &json!({"drop": "IfBinary"})).unwrap();
    m.fit(x.view(), None, None).unwrap();
    let m = load("OneHotEncoder", &m.to_bytes().unwrap()).unwrap();
    let Out::A2(t) = m.transform(x.view()).unwrap() else {
        panic!()
    };
    assert_eq!(t, array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    let Out::A2(back) = m.inverse_transform(t.view()).unwrap() else {
        panic!()
    };
    assert_eq!(back, x);
    assert!(
        m.transform(array![[3.0, 5.0]].view()).is_err(),
        "unknown category"
    );
    let mut w = make("OneHotEncoder", &json!({})).unwrap();
    let wt = ndarray::Array1::ones(3);
    assert!(w.fit(x.view(), None, Some(wt.view())).is_err());
}
