use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, Dim, IxDyn, IxDynImpl};

use crate::Ops;

/// Model input, alias for [`Array<f32, IxDyn>`]
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
        let mut y = Self::default();
        for op in ops {
            y = match op {
                Ops::Resize(xs, h, w, filter) => Self::resize(xs, *h, *w, filter)?,
                Ops::Letterbox(xs, h, w, filter, bg, resize_by, center) => {
                    Self::letterbox(xs, *h, *w, filter, *bg, resize_by, *center)?
                }
                Ops::Normalize(min_, max_) => y.normalize(*min_, *max_)?,
                Ops::Standardize(mean, std, d) => y.standardize(mean, std, *d)?,
                Ops::Permute(shape) => y.permute(shape)?,
                Ops::InsertAxis(d) => y.insert_axis(*d)?,
                Ops::Nhwc2nchw => y.nhwc2nchw()?,
                Ops::Nchw2nhwc => y.nchw2nhwc()?,
                _ => todo!(),
            }
        }
        Ok(y)
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
        resize_by: &str,
        center: bool,
    ) -> Result<Self> {
        Ok(Self(Ops::letterbox(
            xs, height, width, filter, bg, resize_by, center,
        )?))
    }
}
