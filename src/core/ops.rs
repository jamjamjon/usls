//! Some processing functions to image and ndarray.

use anyhow::Result;
use fast_image_resize::{
    images::{CroppedImageMut, Image},
    pixels::PixelType,
    FilterType, ResizeAlg, ResizeOptions, Resizer,
};
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array, Axis, IntoDimension, IxDyn};
use rayon::prelude::*;

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
    Sigmoid,
    Broadcast,
    ToShape,
    Repeat,
}

impl Ops<'_> {
    pub fn normalize(x: Array<f32, IxDyn>, min: f32, max: f32) -> Result<Array<f32, IxDyn>> {
        if min >= max {
            anyhow::bail!(
                "Invalid range in `normalize`: `min` ({}) must be less than `max` ({}).",
                min,
                max
            );
        }
        Ok((x - min) / (max - min))
    }

    pub fn sigmoid(x: Array<f32, IxDyn>) -> Array<f32, IxDyn> {
        x.mapv(|x| 1. / ((-x).exp() + 1.))
    }

    pub fn broadcast<D: IntoDimension + std::fmt::Debug + Copy>(
        x: Array<f32, IxDyn>,
        dim: D,
    ) -> Result<Array<f32, IxDyn>> {
        match x.broadcast(dim) {
            Some(x) => Ok(x.to_owned().into_dyn()),
            None => anyhow::bail!(
                "Failed to broadcast. Shape: {:?}, dim: {:?}",
                x.shape(),
                dim
            ),
        }
    }

    pub fn repeat(x: Array<f32, IxDyn>, d: usize, n: usize) -> Result<Array<f32, IxDyn>> {
        if d >= x.ndim() {
            anyhow::bail!("Index {d} is out of bounds with size {}.", x.ndim());
        } else {
            let mut dim = x.shape().to_vec();
            dim[d] = n;
            Self::broadcast(x, dim.as_slice())
        }
    }

    pub fn to_shape<D: ndarray::ShapeArg>(
        x: Array<f32, IxDyn>,
        dim: D,
    ) -> Result<Array<f32, IxDyn>> {
        Ok(x.to_shape(dim).map(|x| x.to_owned().into_dyn())?)
    }

    pub fn standardize(
        x: Array<f32, IxDyn>,
        mean: &[f32],
        std: &[f32],
        dim: usize,
    ) -> Result<Array<f32, IxDyn>> {
        if mean.len() != std.len() {
            anyhow::bail!("`standardize`: `mean` and `std` lengths are not equal. Mean length: {}, Std length: {}.", mean.len(), std.len());
        }
        let shape = x.shape();
        if dim >= shape.len() || shape[dim] != mean.len() {
            anyhow::bail!("`standardize`: Dimension mismatch. `dim` is {} but shape length is {} or `mean` length is {}.", dim, shape.len(), mean.len());
        }
        let mut shape = vec![1; shape.len()];
        shape[dim] = mean.len();
        let mean = Array::from_shape_vec(shape.clone(), mean.to_vec())?;
        let std = Array::from_shape_vec(shape, std.to_vec())?;
        Ok((x - mean) / std)
    }

    pub fn permute(x: Array<f32, IxDyn>, shape: &[usize]) -> Result<Array<f32, IxDyn>> {
        if shape.len() != x.shape().len() {
            anyhow::bail!(
                "`permute`: Shape length mismatch. Expected: {}, got: {}. Target shape: {:?}, provided shape: {:?}.",
                x.shape().len(),
                shape.len(),
                x.shape(),
                shape
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
                "`insert_axis`: The specified axis position {} exceeds the maximum shape length {}.",
                d,
                x.shape().len()
            );
        }
        Ok(x.insert_axis(Axis(d)))
    }

    pub fn norm(xs: Array<f32, IxDyn>, d: usize) -> Result<Array<f32, IxDyn>> {
        if xs.shape().len() < d {
            anyhow::bail!(
                "`norm`: Specified axis {} exceeds the maximum dimension length {}.",
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

    // deprecated
    pub fn descale_mask(mask: DynamicImage, w0: f32, h0: f32, w1: f32, h1: f32) -> DynamicImage {
        // 0 -> 1
        let (_, w, h) = Ops::scale_wh(w1, h1, w0, h0);
        let mut mask = mask.to_owned();
        let mask = mask.crop(0, 0, w as u32, h as u32);
        mask.resize_exact(w1 as u32, h1 as u32, image::imageops::FilterType::Triangle)
    }

    pub fn resize_lumaf32_u8(
        v: &[f32],
        w0: f32,
        h0: f32,
        w1: f32,
        h1: f32,
        crop_src: bool,
        filter: &str,
    ) -> Result<Vec<u8>> {
        let mask_f32 = Self::resize_lumaf32_f32(v, w0, h0, w1, h1, crop_src, filter)?;
        let v: Vec<u8> = mask_f32.par_iter().map(|&x| (x * 255.0) as u8).collect();
        Ok(v)
    }

    pub fn resize_lumaf32_f32(
        v: &[f32],
        w0: f32,
        h0: f32,
        w1: f32,
        h1: f32,
        crop_src: bool,
        filter: &str,
    ) -> Result<Vec<f32>> {
        let src = Image::from_vec_u8(
            w0 as _,
            h0 as _,
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            PixelType::F32,
        )?;
        let mut dst = Image::new(w1 as _, h1 as _, src.pixel_type());
        let (mut resizer, mut options) = Self::build_resizer_filter(filter)?;
        if crop_src {
            let (_, w, h) = Self::scale_wh(w1 as _, h1 as _, w0 as _, h0 as _);
            options = options.crop(0., 0., w.into(), h.into());
        };
        resizer.resize(&src, &mut dst, &options)?;

        // u8*2 -> f32
        let mask_f32: Vec<f32> = dst
            .into_vec()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(mask_f32)
    }

    pub fn resize_luma8_vec(
        v: &[u8],
        w0: f32,
        h0: f32,
        w1: f32,
        h1: f32,
        crop_src: bool,
        filter: &str,
    ) -> Result<Vec<u8>> {
        let src = Image::from_vec_u8(w0 as _, h0 as _, v.to_vec(), PixelType::U8)?;
        let mut dst = Image::new(w1 as _, h1 as _, src.pixel_type());
        let (mut resizer, mut options) = Self::build_resizer_filter(filter)?;
        if crop_src {
            let (_, w, h) = Self::scale_wh(w1 as _, h1 as _, w0 as _, h0 as _);
            options = options.crop(0., 0., w.into(), h.into());
        };
        resizer.resize(&src, &mut dst, &options)?;
        Ok(dst.into_vec())
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
            _ => anyhow::bail!("Unsupported resizer's filter type: {ty}"),
        };
        Ok((
            Resizer::new(),
            ResizeOptions::new().resize_alg(ResizeAlg::Convolution(ty)),
        ))
    }

    pub fn resize(
        xs: &[DynamicImage],
        th: u32,
        tw: u32,
        filter: &str,
    ) -> Result<Array<f32, IxDyn>> {
        let mut ys = Array::ones((xs.len(), th as usize, tw as usize, 3)).into_dyn();
        let (mut resizer, options) = Self::build_resizer_filter(filter)?;
        for (idx, x) in xs.iter().enumerate() {
            let buffer = if x.dimensions() == (tw, th) {
                x.to_rgb8().into_raw()
            } else {
                let mut dst = Image::new(tw, th, PixelType::U8x3);
                resizer.resize(x, &mut dst, &options)?;
                dst.into_vec()
            };
            let y_ =
                Array::from_shape_vec((th as usize, tw as usize, 3), buffer)?.mapv(|x| x as f32);
            ys.slice_mut(s![idx, .., .., ..]).assign(&y_);
        }
        Ok(ys)
    }

    pub fn letterbox(
        xs: &[DynamicImage],
        th: u32,
        tw: u32,
        filter: &str,
        bg: u8,
        resize_by: &str,
        center: bool,
    ) -> Result<Array<f32, IxDyn>> {
        let mut ys = Array::ones((xs.len(), th as usize, tw as usize, 3)).into_dyn();
        let (mut resizer, options) = Self::build_resizer_filter(filter)?;

        for (idx, x) in xs.iter().enumerate() {
            let (w0, h0) = x.dimensions();
            let buffer = if w0 == tw && h0 == th {
                x.to_rgb8().into_raw()
            } else {
                let (w, h) = match resize_by {
                    "auto" => {
                        let r = (tw as f32 / w0 as f32).min(th as f32 / h0 as f32);
                        (
                            (w0 as f32 * r).round() as u32,
                            (h0 as f32 * r).round() as u32,
                        )
                    }
                    "height" => (th * w0 / h0, th),
                    "width" => (tw, tw * h0 / w0),
                    _ => anyhow::bail!("Options for `letterbox`: width, height, auto"),
                };

                let mut dst = Image::from_vec_u8(
                    tw,
                    th,
                    vec![bg; 3 * th as usize * tw as usize],
                    PixelType::U8x3,
                )?;
                let (l, t) = if center {
                    if w == tw {
                        (0, (th - h) / 2)
                    } else {
                        ((tw - w) / 2, 0)
                    }
                } else {
                    (0, 0)
                };
                let mut dst_cropped = CroppedImageMut::new(&mut dst, l, t, w, h)?;
                resizer.resize(x, &mut dst_cropped, &options)?;
                dst.into_vec()
            };
            let y_ =
                Array::from_shape_vec((th as usize, tw as usize, 3), buffer)?.mapv(|x| x as f32);
            ys.slice_mut(s![idx, .., .., ..]).assign(&y_);
        }
        Ok(ys)
    }
}
