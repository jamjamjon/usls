use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, Dim, IxDyn, IxDynImpl};

use crate::Ops;

#[derive(Debug, Clone, Default)]
pub struct X(pub Array<f32, IxDyn>);

impl From<Array<f32, IxDyn>> for X {
    fn from(x: Array<f32, IxDyn>) -> Self {
        Self(x)
    }
}

impl std::ops::Deref for X {
    type Target = Array<f32, IxDyn>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl X {
    pub fn zeros(shape: &[usize]) -> Self {
        Self(Array::zeros(Dim(IxDynImpl::from(shape.to_vec()))))
    }

    pub fn apply(ops: &[Ops]) -> Result<Self> {
        Ops::apply(ops)
    }

    pub fn permute(mut self, shape: &[usize]) -> Result<Self> {
        self.0 = Ops::permute(self.0, shape)?;
        Ok(self)
    }

    pub fn nhwc2nchw(mut self) -> Result<Self> {
        self.0 = Ops::nhwc2nchw(self.0)?;
        Ok(self)
    }

    pub fn nchw2nhwc(mut self) -> Result<Self> {
        self.0 = Ops::nchw2nhwc(self.0)?;
        Ok(self)
    }

    pub fn insert_axis(mut self, d: usize) -> Result<Self> {
        self.0 = Ops::insert_axis(self.0, d)?;
        Ok(self)
    }

    pub fn dims(&self) -> &[usize] {
        self.0.shape()
    }

    pub fn normalize(mut self, min_: f32, max_: f32) -> Result<Self> {
        self.0 = Ops::normalize(self.0, min_, max_)?;
        Ok(self)
    }

    pub fn standardize(mut self, mean: &[f32], std: &[f32], dim: usize) -> Result<Self> {
        self.0 = Ops::standardize(self.0, mean, std, dim)?;
        Ok(self)
    }

    pub fn norm(mut self, d: usize) -> Result<Self> {
        self.0 = Ops::norm(self.0, d)?;
        Ok(self)
    }

    pub fn resize(xs: &[DynamicImage], height: u32, width: u32, filter: &str) -> Result<Self> {
        Ok(Self(Ops::resize(xs, height, width, filter)?))
    }

    pub fn letterbox(
        xs: &[DynamicImage],
        height: u32,
        width: u32,
        filter: &str,
        bg: u8,
    ) -> Result<Self> {
        Ok(Self(Ops::letterbox(xs, height, width, filter, bg)?))
    }

    pub fn resize_with_fixed_height(
        xs: &[DynamicImage],
        height: u32,
        width: u32,
        filter: &str,
        bg: u8,
    ) -> Result<Self> {
        Ok(Self(Ops::resize_with_fixed_height(
            xs, height, width, filter, bg,
        )?))
    }
}
