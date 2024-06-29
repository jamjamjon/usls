use anyhow::Result;
use fast_image_resize::{
    images::{CroppedImageMut, Image},
    pixels::PixelType,
    FilterType, ResizeAlg, ResizeOptions, Resizer,
};
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array, Axis, IxDyn};

use crate::X;

pub enum Ops<'a> {
    Resize(&'a [DynamicImage], u32, u32, &'a str),
    Letterbox(&'a [DynamicImage], u32, u32, &'a str, u8, &'a str, bool),
    Normalize(f32, f32),
    Standardize(&'a [f32], &'a [f32], usize),
    Permute(&'a [usize]),
    InsertAxis(usize),
    Nhwc2nchw,
    Nchw2nhwc,
    Norm,
}

impl Ops<'_> {
    pub fn apply(ops: &[Self]) -> Result<X> {
        let mut y = X::default();

        for op in ops {
            y = match op {
                Self::Resize(xs, h, w, filter) => X::resize(xs, *h, *w, filter)?,
                Self::Letterbox(xs, h, w, filter, bg, resize_by, center) => {
                    X::letterbox(xs, *h, *w, filter, *bg, resize_by, *center)?
                }
                Self::Normalize(min_, max_) => y.normalize(*min_, *max_)?,
                Self::Standardize(mean, std, d) => y.standardize(mean, std, *d)?,
                Self::Permute(shape) => y.permute(shape)?,
                Self::InsertAxis(d) => y.insert_axis(*d)?,
                Self::Nhwc2nchw => y.nhwc2nchw()?,
                Self::Nchw2nhwc => y.nchw2nhwc()?,
                _ => todo!(),
            }
        }
        Ok(y)
    }

    pub fn normalize(x: Array<f32, IxDyn>, min: f32, max: f32) -> Result<Array<f32, IxDyn>> {
        if min > max {
            anyhow::bail!("Input `min` is greater than `max`");
        }
        Ok((x - min) / (max - min))
    }

    pub fn standardize(
        x: Array<f32, IxDyn>,
        mean: &[f32],
        std: &[f32],
        dim: usize,
    ) -> Result<Array<f32, IxDyn>> {
        if mean.len() != std.len() {
            anyhow::bail!("The lengths of mean and std are not equal.");
        }
        let shape = x.shape();
        if dim >= shape.len() || shape[dim] != mean.len() {
            anyhow::bail!("The specified dimension or mean/std length is inconsistent with the input dimensions.");
        }
        let mut shape = vec![1; shape.len()];
        shape[dim] = mean.len();
        let mean = Array::from_shape_vec(shape.clone(), mean.to_vec()).unwrap();
        let std = Array::from_shape_vec(shape, std.to_vec()).unwrap();
        Ok((x - mean) / std)
    }

    pub fn permute(x: Array<f32, IxDyn>, shape: &[usize]) -> Result<Array<f32, IxDyn>> {
        if shape.len() != x.shape().len() {
            anyhow::bail!(
                "Shape inconsistent. Target: {:?}, {}, got: {:?}, {}",
                x.shape(),
                x.shape().len(),
                shape,
                shape.len()
            );
        }
        Ok(x.permuted_axes(shape.to_vec()).into_dyn())
    }

    pub fn nhwc2nchw(x: Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        Self::permute(x, &[0, 3, 1, 2])
    }

    pub fn nchw2nhwc(x: Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        Self::permute(x, &[0, 2, 3, 1])
    }

    pub fn insert_axis(x: Array<f32, IxDyn>, d: usize) -> Result<Array<f32, IxDyn>> {
        if x.shape().len() < d {
            anyhow::bail!(
                "The specified axis insertion position {} exceeds the shape's maximum limit of {}.",
                d,
                x.shape().len()
            );
        }
        Ok(x.insert_axis(Axis(d)))
    }

    pub fn norm(xs: Array<f32, IxDyn>, d: usize) -> Result<Array<f32, IxDyn>> {
        if xs.shape().len() < d {
            anyhow::bail!(
                "The specified axis {} exceeds the shape's maximum limit of {}.",
                d,
                xs.shape().len()
            );
        }
        let std_ = xs
            .mapv(|x| x * x)
            .sum_axis(Axis(d))
            .mapv(f32::sqrt)
            .insert_axis(Axis(d));
        Ok(xs / std_)
    }

    pub fn scale_wh(w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }

    pub fn make_divisible(x: usize, divisor: usize) -> usize {
        (x + divisor - 1) / divisor * divisor
    }

    pub fn descale_mask(mask: DynamicImage, w0: f32, h0: f32, w1: f32, h1: f32) -> DynamicImage {
        // 0 -> 1
        let (_, w, h) = Ops::scale_wh(w1, h1, w0, h0);
        let mut mask = mask.to_owned();
        let mask = mask.crop(0, 0, w as u32, h as u32);
        mask.resize_exact(w1 as u32, h1 as u32, image::imageops::FilterType::Triangle)
    }

    pub fn build_resizer_filter(ty: &str) -> Result<(Resizer, ResizeOptions)> {
        let ty = match ty {
            "Box" => FilterType::Box,
            "Bilinear" => FilterType::Bilinear,
            "Hamming" => FilterType::Hamming,
            "CatmullRom" => FilterType::CatmullRom,
            "Mitchell" => FilterType::Mitchell,
            "Gaussian" => FilterType::Gaussian,
            "Lanczos3" => FilterType::Lanczos3,
            _ => anyhow::bail!("Unsupported resize filter type: {ty}"),
        };
        Ok((
            Resizer::new(),
            ResizeOptions::new().resize_alg(ResizeAlg::Convolution(ty)),
        ))
    }

    pub fn resize(
        xs: &[DynamicImage],
        height: u32,
        width: u32,
        filter: &str,
    ) -> Result<Array<f32, IxDyn>> {
        let mut ys = Array::ones((xs.len(), height as usize, width as usize, 3)).into_dyn();
        let (mut resizer, options) = Self::build_resizer_filter(filter)?;
        for (idx, x) in xs.iter().enumerate() {
            let buffer = if x.dimensions() == (width, height) {
                x.to_rgba8().into_raw()
            } else {
                let mut dst_image = Image::new(width, height, PixelType::U8x3);
                resizer.resize(x, &mut dst_image, &options).unwrap();
                dst_image.into_vec()
            };
            let y_ = Array::from_shape_vec((height as usize, width as usize, 3), buffer)
                .unwrap()
                .mapv(|x| x as f32);
            ys.slice_mut(s![idx, .., .., ..]).assign(&y_);
        }
        Ok(ys)
    }

    pub fn letterbox(
        xs: &[DynamicImage],
        height: u32,
        width: u32,
        filter: &str,
        bg: u8,
        resize_by: &str,
        center: bool,
    ) -> Result<Array<f32, IxDyn>> {
        let mut ys = Array::ones((xs.len(), height as usize, width as usize, 3)).into_dyn();
        let (mut resizer, options) = Self::build_resizer_filter(filter)?;

        for (idx, x) in xs.iter().enumerate() {
            let (w0, h0) = x.dimensions();
            let buffer = if w0 == width && h0 == height {
                x.to_rgba8().into_raw()
            } else {
                let (w, h) = match resize_by {
                    "auto" => {
                        let r = (width as f32 / w0 as f32).min(height as f32 / h0 as f32);
                        (
                            (w0 as f32 * r).round() as u32,
                            (h0 as f32 * r).round() as u32,
                        )
                    }
                    "height" => (height * w0 / h0, height),
                    "width" => (width, width * h0 / w0),
                    _ => anyhow::bail!("Option: width, height, auto"),
                };

                let mut dst_image = Image::from_vec_u8(
                    width,
                    height,
                    vec![bg; 3 * height as usize * width as usize],
                    PixelType::U8x3,
                )
                .unwrap();
                let (l, t) = if center {
                    if w == width {
                        (0, (height - h) / 2)
                    } else {
                        ((width - w) / 2, 0)
                    }
                } else {
                    (0, 0)
                };
                let mut cropped_dst_image =
                    CroppedImageMut::new(&mut dst_image, l, t, w, h).unwrap();
                resizer.resize(x, &mut cropped_dst_image, &options).unwrap();
                dst_image.into_vec()
            };
            let y_ = Array::from_shape_vec((height as usize, width as usize, 3), buffer)
                .unwrap()
                .mapv(|x| x as f32);
            ys.slice_mut(s![idx, .., .., ..]).assign(&y_);
        }
        Ok(ys)
    }
}
