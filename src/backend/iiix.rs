use crate::MinOptMax;

/// A struct for input composed of the i-th input, the ii-th dimension, and the value.
#[derive(Clone, Debug, Default)]
pub struct Iiix {
    /// Input index.
    pub i: usize,
    /// Dimension index.
    pub ii: usize,
    /// Min-Opt-Max value specification.
    pub x: MinOptMax,
}

impl From<(usize, usize, MinOptMax)> for Iiix {
    fn from((i, ii, x): (usize, usize, MinOptMax)) -> Self {
        Self { i, ii, x }
    }
}
