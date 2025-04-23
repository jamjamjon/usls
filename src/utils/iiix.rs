use crate::MinOptMax;

/// A struct for input composed of the i-th input, the ii-th dimension, and the value.
#[derive(Clone, Debug, Default)]
pub struct Iiix {
    pub i: usize,
    pub ii: usize,
    pub x: MinOptMax,
}

impl From<(usize, usize, MinOptMax)> for Iiix {
    fn from((i, ii, x): (usize, usize, MinOptMax)) -> Self {
        Self { i, ii, x }
    }
}
