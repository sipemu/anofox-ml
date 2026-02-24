use ndarray::NdFloat;
use num_traits::{Float as NumFloat, FromPrimitive};
use std::fmt::{Debug, Display};
use std::iter::Sum;

/// Trait bound for floating-point types used throughout rustml.
///
/// Combines ndarray's `NdFloat` (for array operations), `num_traits::Float`
/// (for math functions), and `FromPrimitive` (for numeric conversions).
pub trait Float:
    NdFloat + NumFloat + FromPrimitive + Debug + Display + Sum + Default + 'static
{
}

impl Float for f32 {}
impl Float for f64 {}
