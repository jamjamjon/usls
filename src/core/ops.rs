//! Some processing functions.

use anyhow::Result;
use fast_image_resize::{
    images::{CroppedImageMut, Image},
    pixels::PixelType,
    FilterType, ResizeAlg, ResizeOptions, Resizer,
};
use image::{DynamicImage, GenericImageView};
use ndarray::{concatenate, s, Array, Array3, ArrayView1, Axis, IntoDimension, Ix2, IxDyn, Zip};

use rayon::prelude::*;

/// Image and tensor operations for preprocessing and postprocessing.
pub enum Ops<'a> {
    /// Resize images to exact dimensions.
    FitExact(&'a [DynamicImage], u32, u32, &'a str),
    /// Apply letterbox padding to maintain aspect ratio.
    Letterbox(&'a [DynamicImage], u32, u32, &'a str, u8, &'a str, bool),
    /// Normalize values to a specific range.
    Normalize(f32, f32),
    /// Standardize using mean and standard deviation.
    Standardize(&'a [f32], &'a [f32], usize),
    /// Permute tensor dimensions.
    Permute(&'a [usize]),
    /// Insert a new axis at specified position.
    InsertAxis(usize),
    /// Convert from NHWC to NCHW format.
    Nhwc2nchw,
    /// Convert from NCHW to NHWC format.
    Nchw2nhwc,
    /// Apply L2 normalization.
    Norm,
    /// Apply sigmoid activation function.
    Sigmoid,
    /// Broadcast tensor to larger dimensions.
    Broadcast,
    /// Reshape tensor to specified shape.
    ToShape,
    /// Repeat tensor elements.
    Repeat,
}

impl Ops<'_> {
    pub fn normalize(x: &mut Array<f32, IxDyn>, min: f32, max: f32) -> Result<()> {
        if min >= max {
            anyhow::bail!(
                "Invalid range in `normalize`: `min` ({}) must be less than `max` ({}).",
                min,
                max
            );
        }
        let range = max - min;
        x.par_mapv_inplace(|x| (x - min) / range);

        Ok(())
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
        x: &mut Array<f32, IxDyn>,
        mean: ArrayView1<f32>,
        std: ArrayView1<f32>,
        dim: usize,
    ) -> Result<()> {
        if mean.len() != std.len() {
            anyhow::bail!(
                "`standardize`: `mean` and `std` lengths are not equal. Mean length: {}, Std length: {}.",
                mean.len(),
                std.len()
            );
        }

        let shape = x.shape();
        if dim >= shape.len() || shape[dim] != mean.len() {
            anyhow::bail!(
                "`standardize`: Dimension mismatch. `dim` is {} but shape length is {} or `mean` length is {}.",
                dim,
                shape.len(),
                mean.len()
            );
        }
        let mean_broadcast = mean.broadcast(shape).ok_or_else(|| {
            anyhow::anyhow!("Failed to broadcast `mean` to the shape of the input array.")
        })?;
        let std_broadcast = std.broadcast(shape).ok_or_else(|| {
            anyhow::anyhow!("Failed to broadcast `std` to the shape of the input array.")
        })?;
        Zip::from(x)
            .and(mean_broadcast)
            .and(std_broadcast)
            .par_for_each(|x_val, &mean_val, &std_val| {
                *x_val = (*x_val - mean_val) / std_val;
            });

        Ok(())
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

    pub fn concatenate(
        x: &Array<f32, IxDyn>,
        y: &Array<f32, IxDyn>,
        d: usize,
    ) -> Result<Array<f32, IxDyn>> {
        Ok(concatenate(Axis(d), &[x.view(), y.view()])?)
    }

    pub fn concat(xs: &[Array<f32, IxDyn>], d: usize) -> Result<Array<f32, IxDyn>> {
        let xs = xs.iter().map(|x| x.view()).collect::<Vec<_>>();
        Ok(concatenate(Axis(d), &xs)?)
    }

    pub fn dot2(x: &Array<f32, IxDyn>, other: &Array<f32, IxDyn>) -> Result<Vec<Vec<f32>>> {
        // (m, ndim) * (n, ndim).t => (m, n)
        let query = x.to_owned().into_dimensionality::<Ix2>()?;
        let gallery = other.to_owned().into_dimensionality::<Ix2>()?;
        let matrix = query.dot(&gallery.t());
        let exps = matrix.mapv(|x| x.exp());
        let stds = exps.sum_axis(Axis(1));
        let matrix = exps / stds.insert_axis(Axis(1));
        let matrix: Vec<Vec<f32>> = matrix.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();
        Ok(matrix)
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

    pub fn softmax(xs: Array<f32, IxDyn>, d: usize) -> Result<Array<f32, IxDyn>> {
        if xs.shape().len() <= d {
            anyhow::bail!(
                "`softmax`: Specified axis {} exceeds the maximum dimension length {}.",
                d,
                xs.shape().len()
            );
        }
        let max_vals = xs
            .map_axis(Axis(d), |view| {
                view.fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            })
            .insert_axis(Axis(d));
        let exps = (&xs - &max_vals).mapv(f32::exp);
        let sums = exps.sum_axis(Axis(d)).insert_axis(Axis(d));

        Ok(exps / sums)
    }

    pub fn scale_wh(w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }

    pub fn make_divisible(x: usize, divisor: usize) -> usize {
        // (x + divisor - 1) / divisor * divisor
        x.div_ceil(divisor) * divisor
    }

    // deprecated
    pub fn descale_mask(mask: DynamicImage, w0: f32, h0: f32, w1: f32, h1: f32) -> DynamicImage {
        // 0 -> 1
        let (_, w, h) = Ops::scale_wh(w1, h1, w0, h0);
        let mut mask = mask.to_owned();
        let mask = mask.crop(0, 0, w as u32, h as u32);
        mask.resize_exact(w1 as u32, h1 as u32, image::imageops::FilterType::Triangle)
    }

    pub fn interpolate_3d(
        xs: Array<f32, IxDyn>,
        tw: f32,
        th: f32,
        filter: &str,
    ) -> Result<Array<f32, IxDyn>> {
        let d_max = xs.ndim();
        if d_max != 3 {
            anyhow::bail!("`interpolate_3d`: The input's ndim: {} is not 3.", d_max);
        }
        let (n, h, w) = (xs.shape()[0], xs.shape()[1], xs.shape()[2]);
        let mut ys = Array3::zeros((n, th as usize, tw as usize));
        for (i, luma) in xs.axis_iter(Axis(0)).enumerate() {
            let v = Ops::resize_lumaf32_f32(
                &luma.to_owned().into_raw_vec_and_offset().0,
                w as _,
                h as _,
                tw as _,
                th as _,
                false,
                filter,
            )?;
            let y_ = Array::from_shape_vec((th as usize, tw as usize), v)?;
            ys.slice_mut(s![i, .., ..]).assign(&y_);
        }

        Ok(ys.into_dyn())
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
        let (mut resizer, mut config) = Self::build_resizer_filter(filter)?;
        if crop_src {
            let (_, w, h) = Self::scale_wh(w1 as _, h1 as _, w0 as _, h0 as _);
            config = config.crop(0., 0., w.into(), h.into());
        };
        resizer.resize(&src, &mut dst, &config)?;

        // u8 -> f32
        Self::u8_slice_to_f32(&dst.into_vec())
    }

    pub fn u8_slice_to_f32(data: &[u8]) -> Result<Vec<f32>> {
        let size_in_bytes = 4;
        let elem_count = data.len() / size_in_bytes;
        if (data.as_ptr() as usize) % size_in_bytes == 0 {
            let data: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, elem_count) };

            Ok(data.to_vec())
        } else {
            let mut c: Vec<f32> = Vec::with_capacity(elem_count);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
                c.set_len(elem_count)
            }

            Ok(c)
        }
    }

    pub fn f32_slice_to_u8(mut vs: Vec<f32>) -> Vec<u8> {
        let size_in_bytes = 4;
        let length = vs.len() * size_in_bytes;
        let capacity = vs.capacity() * size_in_bytes;
        let ptr = vs.as_mut_ptr() as *mut u8;
        std::mem::forget(vs);
        unsafe { Vec::from_raw_parts(ptr, length, capacity) }
    }

    pub fn resize_luma8_u8(
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
        let (mut resizer, mut config) = Self::build_resizer_filter(filter)?;
        if crop_src {
            let (_, w, h) = Self::scale_wh(w1 as _, h1 as _, w0 as _, h0 as _);
            config = config.crop(0., 0., w.into(), h.into());
        };
        resizer.resize(&src, &mut dst, &config)?;
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

    pub fn resize_rgb(
        x: &DynamicImage,
        th: u32,
        tw: u32,
        resizer: &mut Resizer,
        config: &ResizeOptions,
    ) -> Result<Array<f32, IxDyn>> {
        let buffer = if x.dimensions() == (tw, th) {
            x.to_rgb8().into_raw()
        } else {
            let mut dst = Image::new(tw, th, PixelType::U8x3);
            resizer.resize(x, &mut dst, config)?;
            dst.into_vec()
        };
        let y = Array::from_shape_vec((th as usize, tw as usize, 3), buffer)?
            .mapv(|x| x as f32)
            .into_dyn();
        Ok(y)
    }

    pub fn resize(
        xs: &[DynamicImage],
        th: u32,
        tw: u32,
        filter: &str,
    ) -> Result<Array<f32, IxDyn>> {
        let mut ys = Array::ones((xs.len(), th as usize, tw as usize, 3)).into_dyn();
        let (mut resizer, config) = Self::build_resizer_filter(filter)?;
        for (idx, x) in xs.iter().enumerate() {
            let y = Self::resize_rgb(x, th, tw, &mut resizer, &config)?;
            ys.slice_mut(s![idx, .., .., ..]).assign(&y);
        }
        Ok(ys)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn letterbox_rgb(
        x: &DynamicImage,
        th: u32,
        tw: u32,
        bg: u8,
        resize_by: &str,
        center: bool,
        resizer: &mut Resizer,
        config: &ResizeOptions,
    ) -> Result<Array<f32, IxDyn>> {
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
                _ => anyhow::bail!("ORTConfig for `letterbox`: width, height, auto"),
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
            resizer.resize(x, &mut dst_cropped, config)?;
            dst.into_vec()
        };
        let y = Array::from_shape_vec((th as usize, tw as usize, 3), buffer)?
            .mapv(|x| x as f32)
            .into_dyn();
        Ok(y)
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
        let (mut resizer, config) = Self::build_resizer_filter(filter)?;
        for (idx, x) in xs.iter().enumerate() {
            let y = Self::letterbox_rgb(x, th, tw, bg, resize_by, center, &mut resizer, &config)?;
            ys.slice_mut(s![idx, .., .., ..]).assign(&y);
        }
        Ok(ys)
    }
}
