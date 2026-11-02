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
    /// `x` as an `f64` view when `Self` is `f64`, `None` otherwise. Backs
    /// [`as_f64`].
    fn view_as_f64<'a, D: ndarray::Dimension>(
        _x: ndarray::ArrayView<'a, Self, D>,
    ) -> Option<ndarray::ArrayView<'a, f64, D>> {
        None
    }
}

impl Float for f32 {}
impl Float for f64 {
    fn view_as_f64<'a, D: ndarray::Dimension>(
        x: ndarray::ArrayView<'a, f64, D>,
    ) -> Option<ndarray::ArrayView<'a, f64, D>> {
        Some(x)
    }
}

/// View `x` as `f64`: borrowed when `F` is `f64`, converted otherwise. For
/// estimators that compute in `f64` whatever the input type.
pub fn as_f64<'a, F: Float, D: ndarray::Dimension + 'static>(
    x: impl ndarray::AsArray<'a, F, D>,
) -> ndarray::CowArray<'a, f64, D> {
    let x: ndarray::ArrayView<'a, F, D> = x.into();
    match F::view_as_f64(x.clone()) {
        Some(v) => v.into(),
        None => x.mapv(|v| v.to_f64().unwrap_or(f64::NAN)).into(),
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
        assert!(as_f64(&a).is_view());
        let b = array![1.5f32, -2.0];
        assert_eq!(as_f64(&b).into_owned(), array![1.5f64, -2.0]);
        let c: ndarray::Array1<f32> = from_f64(array![0.25, 4.0]);
        assert_eq!(c, array![0.25f32, 4.0]);
        let d: ndarray::Array1<f64> = from_f64(array![0.25, 4.0]);
        assert_eq!(d, array![0.25, 4.0]);
    }
}
