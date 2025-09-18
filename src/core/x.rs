use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, Dim, IntoDimension, Ix2, IxDyn, IxDynImpl};

use crate::{Ops, ResizeMode};

/// Tensor: wrapper over [`Array<f32, IxDyn>`]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct X(pub Array<f32, IxDyn>);

impl From<Array<f32, IxDyn>> for X {
    fn from(x: Array<f32, IxDyn>) -> Self {
        Self(x)
    }
}

impl From<Vec<f32>> for X {
    fn from(x: Vec<f32>) -> Self {
        Self(Array::from_vec(x).into_dyn().into_owned())
    }
}

impl TryFrom<Vec<(u32, u32)>> for X {
    type Error = anyhow::Error;

    fn try_from(values: Vec<(u32, u32)>) -> Result<Self, Self::Error> {
        if values.is_empty() {
            Ok(Self::default())
        } else {
            let mut flattened: Vec<u32> = Vec::new();
            for &(a, b) in values.iter() {
                flattened.push(a);
                flattened.push(b);
            }
            let shape = (values.len(), 2);
            let x = Array::from_shape_vec(shape, flattened)?
                .map(|x| *x as f32)
                .into_dyn();
            Ok(Self(x))
        }
    }
}

impl TryFrom<Vec<Vec<f32>>> for X {
    type Error = anyhow::Error;

    fn try_from(xs: Vec<Vec<f32>>) -> Result<Self, Self::Error> {
        if xs.is_empty() {
            Ok(Self::default())
        } else {
            let shape = (xs.len(), xs[0].len());
            let flattened: Vec<f32> = xs.iter().flatten().cloned().collect();
            let x = Array::from_shape_vec(shape, flattened)?.into_dyn();
            Ok(Self(x))
        }
    }
}

impl std::ops::Deref for X {
    type Target = Array<f32, IxDyn>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for X {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::ops::Mul<f32> for X {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        Self(self.0 * other)
    }
}

impl std::ops::Div<f32> for X {
    type Output = Self;

    fn div(self, other: f32) -> Self::Output {
        Self(self.0 / other)
    }
}

impl std::ops::Add<f32> for X {
    type Output = Self;

    fn add(self, other: f32) -> Self::Output {
        Self(self.0 + other)
    }
}

impl std::ops::Sub<f32> for X {
    type Output = Self;

    fn sub(self, other: f32) -> Self::Output {
        Self(self.0 - other)
    }
}

impl X {
    pub fn zeros(shape: &[usize]) -> Self {
        Self::from(Array::zeros(Dim(IxDynImpl::from(shape.to_vec()))))
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::from(Array::ones(Dim(IxDynImpl::from(shape.to_vec()))))
    }

    pub fn zeros_like(x: &Self) -> Self {
        Self::from(Array::zeros(x.raw_dim()))
    }

    pub fn ones_like(x: &Self) -> Self {
        Self::from(Array::ones(x.raw_dim()))
    }

    pub fn full(shape: &[usize], x: f32) -> Self {
        Self::from(Array::from_elem(shape, x))
    }

    pub fn from_shape_vec(shape: &[usize], xs: Vec<f32>) -> Result<Self> {
        Ok(Self::from(Array::from_shape_vec(shape, xs)?))
    }

    pub fn apply(ops: &[Ops]) -> Result<Self> {
        let mut y = Self::default();
        for op in ops {
            y = match op {
                Ops::FitExact(xs, h, w, filter) => Self::resize(xs, *h, *w, filter)?,
                Ops::Letterbox(xs, h, w, filter, bg, resize_by, center) => {
                    Self::letterbox(xs, *h, *w, filter, *bg, resize_by, *center)?
                }
                Ops::Normalize(min_, max_) => y.normalize(*min_, *max_)?,
                Ops::Standardize(mean, std, d) => y.standardize(mean, std, *d)?,
                Ops::Permute(shape) => y.permute(shape)?,
                Ops::InsertAxis(d) => y.insert_axis(*d)?,
                Ops::Nhwc2nchw => y.nhwc2nchw()?,
                Ops::Nchw2nhwc => y.nchw2nhwc()?,
                Ops::Sigmoid => y.sigmoid()?,
                _ => todo!(),
            }
        }
        Ok(y)
    }

    pub fn sigmoid(mut self) -> Result<Self> {
        self.0 = Ops::sigmoid(self.0);
        Ok(self)
    }

    pub fn broadcast<D: IntoDimension + std::fmt::Debug + Copy>(mut self, dim: D) -> Result<Self> {
        self.0 = Ops::broadcast(self.0, dim)?;
        Ok(self)
    }

    pub fn to_shape<D: ndarray::ShapeArg>(mut self, dim: D) -> Result<Self> {
        self.0 = Ops::to_shape(self.0, dim)?;
        Ok(self)
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

    pub fn repeat(mut self, d: usize, n: usize) -> Result<Self> {
        self.0 = Ops::repeat(self.0, d, n)?;
        Ok(self)
    }

    pub fn concatenate(mut self, other: &Self, d: usize) -> Result<Self> {
        self.0 = Ops::concatenate(&self.0, other, d)?;
        Ok(self)
    }

    pub fn concat(xs: &[Self], d: usize) -> Result<Self> {
        let xs = xs.iter().cloned().map(|x| x.0).collect::<Vec<_>>();
        let x = Ops::concat(&xs, d)?;
        Ok(x.into())
    }

    pub fn dims(&self) -> &[usize] {
        self.0.shape()
    }

    pub fn ndim(&self) -> usize {
        self.0.ndim()
    }

    pub fn normalize(mut self, min_: f32, max_: f32) -> Result<Self> {
        Ops::normalize(&mut self.0, min_, max_)?;

        Ok(self)
    }

    pub fn standardize(mut self, mean: &[f32], std: &[f32], dim: usize) -> Result<Self> {
        Ops::standardize(&mut self.0, mean.into(), std.into(), dim)?;
        Ok(self)
    }

    pub fn norm(mut self, d: usize) -> Result<Self> {
        self.0 = Ops::norm(self.0, d)?;
        Ok(self)
    }

    pub fn dot2(&self, other: &Self) -> Result<Self> {
        // Check dimensions
        if self.ndim() != 2 || other.ndim() != 2 {
            anyhow::bail!(
                "dot2 requires 2D matrices, got {}D and {}D",
                self.ndim(),
                other.ndim()
            );
        }

        let a = self.0.as_standard_layout().into_dimensionality::<Ix2>()?;
        let b = other.0.as_standard_layout().into_dimensionality::<Ix2>()?;

        // Check compatibility
        if a.shape()[1] != b.shape()[1] {
            anyhow::bail!(
                "Incompatible dimensions for dot2: {:?} and {:?}",
                a.shape(),
                b.shape()
            );
        }

        Ok(a.dot(&b.t()).into_dyn().into())
    }

    pub fn softmax(mut self, d: usize) -> Result<Self> {
        self.0 = Ops::softmax(self.0, d)?;
        Ok(self)
    }

    pub fn unsigned(mut self) -> Self {
        self.0.par_mapv_inplace(|x| if x < 0.0 { 0.0 } else { x });
        self
    }

    pub fn resize(xs: &[DynamicImage], height: u32, width: u32, filter: &str) -> Result<Self> {
        Ok(Self::from(Ops::resize(xs, height, width, filter)?))
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
        Ok(Self::from(Ops::letterbox(
            xs, height, width, filter, bg, resize_by, center,
        )?))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn preprocess(
        xs: &[image::DynamicImage],
        image_width: u32,
        image_height: u32,
        resize_mode: &ResizeMode,
        resizer_filter: &str,
        padding_value: u8,
        letterbox_center: bool,
        normalize: bool,
        image_std: &[f32],
        image_mean: &[f32],
        nchw: bool,
    ) -> Result<Self> {
        let mut x = match resize_mode {
            ResizeMode::FitExact => X::resize(xs, image_height, image_width, resizer_filter)?,
            ResizeMode::Letterbox => X::letterbox(
                xs,
                image_height,
                image_width,
                resizer_filter,
                padding_value,
                "auto",
                letterbox_center,
            )?,
            _ => unimplemented!(),
        };

        if normalize {
            x = x.normalize(0., 255.)?;
        }

        if !image_std.is_empty() && !image_mean.is_empty() {
            x = x.standardize(image_mean, image_std, 3)?;
        }

        if nchw {
            x = x.nhwc2nchw()?;
        }

        Ok(x)
    }
}
