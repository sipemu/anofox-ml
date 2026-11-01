use ndarray::NdFloat;
use num_traits::{Float as NumFloat, FromPrimitive};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::iter::Sum;

/// Trait bound for floating-point types used throughout anofox-ml.
///
/// Combines ndarray's `NdFloat` (for array operations), `num_traits::Float`
/// (for math functions), `FromPrimitive` (for numeric conversions), and
/// serde's `Serialize + Deserialize` (for model serialization).
pub trait Float:
    NdFloat
    + NumFloat
    + FromPrimitive
    + Debug
    + Display
    + Sum
    + Default
    + Serialize
    + DeserializeOwned
    + 'static
{
}

impl Float for f32 {}
impl Float for f64 {}

/// View `x` as `f64`: borrowed when `F` is `f64`, converted otherwise. For
/// estimators that compute in `f64` whatever the input type.
pub fn as_f64<F: Float, D: ndarray::Dimension + 'static>(
    x: &ndarray::Array<F, D>,
) -> std::borrow::Cow<'_, ndarray::Array<f64, D>> {
    match (x as &dyn std::any::Any).downcast_ref::<ndarray::Array<f64, D>>() {
        Some(v) => std::borrow::Cow::Borrowed(v),
        None => std::borrow::Cow::Owned(x.mapv(|v| v.to_f64().unwrap_or(f64::NAN))),
    }
}

/// Convert an `f64` result back to `F` (a move when `F` is `f64`).
pub fn from_f64<F: Float, D: ndarray::Dimension + 'static>(
    x: ndarray::Array<f64, D>,
) -> ndarray::Array<F, D> {
    let any: Box<dyn std::any::Any> = Box::new(x);
    match any.downcast::<ndarray::Array<F, D>>() {
        Ok(v) => *v,
        Err(any) => any
            .downcast::<ndarray::Array<f64, D>>()
            .expect("the boxed value is an f64 array")
            .mapv(|v| F::from_f64(v).unwrap()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_f64_round_trip() {
        let a = array![[1.0f64, 2.5]];
        assert!(matches!(as_f64(&a), std::borrow::Cow::Borrowed(_)));
        let b = array![1.5f32, -2.0];
        assert_eq!(as_f64(&b).into_owned(), array![1.5f64, -2.0]);
        let c: ndarray::Array1<f32> = from_f64(array![0.25, 4.0]);
        assert_eq!(c, array![0.25f32, 4.0]);
        let d: ndarray::Array1<f64> = from_f64(array![0.25, 4.0]);
        assert_eq!(d, array![0.25, 4.0]);
    }
}
