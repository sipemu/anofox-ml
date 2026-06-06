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
