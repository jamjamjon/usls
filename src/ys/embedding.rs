use aksr::Builder;
use anyhow::Result;
use ndarray::{Array, Axis, Ix2, IxDyn};

use crate::X;

/// Embedding for image or text.
#[derive(Builder, Clone, PartialEq, Default)]
pub struct Embedding(#[args(alias = "embedding")] Array<f32, IxDyn>);

impl std::fmt::Debug for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("").field("Shape", &self.0.shape()).finish()
    }
}

impl From<X> for Embedding {
    fn from(x: X) -> Self {
        Self(x.0)
    }
}

impl Embedding {
    pub fn new(x: Array<f32, IxDyn>) -> Self {
        Self(x)
    }

    pub fn data(&self) -> &Array<f32, IxDyn> {
        &self.0
    }

    pub fn norm(mut self) -> Self {
        let std_ = self.0.mapv(|x| x * x).sum_axis(Axis(0)).mapv(f32::sqrt);
        self.0 = self.0 / std_;
        self
    }

    pub fn dot2(&self, other: &Embedding) -> Result<Vec<Vec<f32>>> {
        // (m, ndim) * (n, ndim).t => (m, n)
        let query = self.0.to_owned().into_dimensionality::<Ix2>()?;
        let gallery = other.0.to_owned().into_dimensionality::<Ix2>()?;
        let matrix = query.dot(&gallery.t());
        let exps = matrix.mapv(|x| x.exp());
        let stds = exps.sum_axis(Axis(1));
        let matrix = exps / stds.insert_axis(Axis(1));
        let matrix: Vec<Vec<f32>> = matrix.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();
        Ok(matrix)
    }
}
