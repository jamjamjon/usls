// TODO: clean

use anyhow::Result;
use fast_image_resize::{images::Image, pixels::PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::DynamicImage;
use ndarray::{concatenate, Array, ArrayView1, Axis, IntoDimension, Ix2, IxDyn, Zip};

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
                "Invalid range in `normalize`: `min` ({min}) must be less than `max` ({max})."
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

    /// Resize mask with automatic handling of different resize modes.
    #[allow(clippy::too_many_arguments)]
    pub fn resize_mask_with_mode(
        mask_proto: Vec<f32>,
        proto_w: usize,
        proto_h: usize,
        image_w: u32,
        image_h: u32,
        model_w: usize,
        model_h: usize,
        resize_mode: crate::ResizeModeType,
        info: &crate::ImageTransformInfo,
        filter: crate::ResizeFilter,
    ) -> Result<Vec<u8>> {
        if proto_w == 0 || proto_h == 0 {
            anyhow::bail!("Invalid proto size: {proto_w}x{proto_h}.");
        }
        if model_w == 0 || model_h == 0 {
            anyhow::bail!("Invalid model size: {model_w}x{model_h}.");
        }
        if image_w == 0 || image_h == 0 {
            anyhow::bail!("Invalid image size: {image_w}x{image_h}.");
        }

        // Calculate crop region in model space based on resize mode
        let (crop_x_model, crop_y_model, crop_w_model, crop_h_model) = match resize_mode {
            crate::ResizeModeType::FitExact => (0.0f32, 0.0f32, model_w as f32, model_h as f32),
            crate::ResizeModeType::Letterbox => {
                let r = info.height_scale;
                if r <= 0.0 {
                    anyhow::bail!("Invalid scale ratio: {r}.");
                }
                let cw = (image_w as f32 * r).round().max(1.0).min(model_w as f32);
                let ch = (image_h as f32 * r).round().max(1.0).min(model_h as f32);
                (info.width_pad, info.height_pad, cw, ch)
            }
            crate::ResizeModeType::FitAdaptive => {
                let r = info.height_scale;
                if r <= 0.0 {
                    anyhow::bail!("Invalid scale ratio: {r}.");
                }
                let cw = (image_w as f32 * r).round().max(1.0).min(model_w as f32);
                let ch = (image_h as f32 * r).round().max(1.0).min(model_h as f32);
                (0.0f32, 0.0f32, cw, ch)
            }
            _ => anyhow::bail!("Unsupported ResizeModeType for mask: {resize_mode:?}"),
        };

        // Scale crop region from model space to proto space
        let sx = proto_w as f32 / model_w as f32;
        let sy = proto_h as f32 / model_h as f32;

        let crop_x = (crop_x_model * sx).round().clamp(0.0, (proto_w - 1) as f32);
        let crop_y = (crop_y_model * sy).round().clamp(0.0, (proto_h - 1) as f32);
        let crop_w = (crop_w_model * sx)
            .round()
            .clamp(1.0, proto_w as f32 - crop_x);
        let crop_h = (crop_h_model * sy)
            .round()
            .clamp(1.0, proto_h as f32 - crop_y);

        // Use bytemuck for zero-copy f32 -> u8 slice conversion
        let src_buf = bytemuck::cast_slice::<f32, u8>(&mask_proto).to_vec();
        let src = Image::from_vec_u8(proto_w as _, proto_h as _, src_buf, PixelType::F32)?;
        let mut dst = Image::new(image_w, image_h, PixelType::F32);

        let config = ResizeOptions::new()
            .resize_alg(ResizeAlg::Interpolation(filter.into()))
            .crop(crop_x as _, crop_y as _, crop_w as _, crop_h as _);

        Resizer::new().resize(&src, &mut dst, &config)?;

        // Convert f32 output to u8 with parallel processing
        let raw = dst.into_vec();
        let out: Vec<u8> = bytemuck::try_cast_slice::<u8, f32>(&raw)
            .map_err(|_| anyhow::anyhow!("Failed to cast output to f32"))?
            .par_iter()
            .map(|&x| (x * 255.0) as u8)
            .collect();

        Ok(out)
    }

    /// Interpolate a single f32 mask to target dimensions (Bilinear filter).
    #[inline]
    pub fn interpolate_1d(
        src: &[f32],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        crop_src: bool,
    ) -> Result<Vec<f32>> {
        Self::interpolate(
            src,
            1,
            src_w,
            src_h,
            dst_w,
            dst_h,
            crop_src,
            crate::ResizeFilter::Bilinear,
        )
    }

    /// Interpolate a single f32 mask with custom filter.
    #[inline]
    pub fn interpolate_1d_with_filter(
        src: &[f32],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        crop_src: bool,
        filter: crate::ResizeFilter,
    ) -> Result<Vec<f32>> {
        Self::interpolate(src, 1, src_w, src_h, dst_w, dst_h, crop_src, filter)
    }

    /// Interpolate a single f32 mask and convert to u8 (Bilinear filter).
    #[inline]
    pub fn interpolate_1d_u8(
        src: &[f32],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        crop_src: bool,
    ) -> Result<Vec<u8>> {
        Self::interpolate(
            src,
            1,
            src_w,
            src_h,
            dst_w,
            dst_h,
            crop_src,
            crate::ResizeFilter::Bilinear,
        )
        .map(|v| v.into_par_iter().map(|x| (x * 255.0) as u8).collect())
    }

    /// Interpolate a single f32 mask and convert to u8 with custom filter.
    #[inline]
    pub fn interpolate_1d_u8_with_filter(
        src: &[f32],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        crop_src: bool,
        filter: crate::ResizeFilter,
    ) -> Result<Vec<u8>> {
        Self::interpolate(src, 1, src_w, src_h, dst_w, dst_h, crop_src, filter)
            .map(|v| v.into_par_iter().map(|x| (x * 255.0) as u8).collect())
    }

    /// Interpolate N f32 masks to target dimensions (Bilinear filter).
    #[inline]
    pub fn interpolate_nd(
        src: &[f32],
        n: usize,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        crop_src: bool,
    ) -> Result<Vec<f32>> {
        Self::interpolate(
            src,
            n,
            src_w,
            src_h,
            dst_w,
            dst_h,
            crop_src,
            crate::ResizeFilter::Bilinear,
        )
    }

    /// Interpolate N f32 masks with custom filter.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn interpolate_nd_with_filter(
        src: &[f32],
        n: usize,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        crop_src: bool,
        filter: crate::ResizeFilter,
    ) -> Result<Vec<f32>> {
        Self::interpolate(src, n, src_w, src_h, dst_w, dst_h, crop_src, filter)
    }

    /// Core interpolation function (internal).
    #[allow(clippy::too_many_arguments)]
    fn interpolate(
        src: &[f32],
        n: usize,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        crop_src: bool,
        filter: crate::ResizeFilter,
    ) -> Result<Vec<f32>> {
        let mask_size = (src_w as usize) * (src_h as usize);
        let expected_len = n * mask_size;
        if src.len() != expected_len {
            anyhow::bail!(
                "`interpolate`: Input length {} != expected {} (n={}, {}x{})",
                src.len(),
                expected_len,
                n,
                src_w,
                src_h
            );
        }

        let dst_size = (dst_w as usize) * (dst_h as usize);

        // Build resize config
        let base_config = ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(filter.into()));
        let config = if crop_src {
            let (_, w, h) = Self::scale_wh(dst_w as _, dst_h as _, src_w as _, src_h as _);
            base_config.crop(0., 0., w.into(), h.into())
        } else {
            base_config
        };

        // Parallel processing for multiple channels
        if n > 1 {
            let results: Vec<Result<Vec<f32>>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let offset = i * mask_size;
                    let mask_slice = &src[offset..offset + mask_size];
                    Self::interpolate_single(mask_slice, src_w, src_h, dst_w, dst_h, &config)
                })
                .collect();

            // Flatten results
            let mut output = Vec::with_capacity(n * dst_size);
            for r in results {
                output.extend(r?);
            }
            Ok(output)
        } else {
            Self::interpolate_single(src, src_w, src_h, dst_w, dst_h, &config)
        }
    }

    /// Single channel interpolation (internal).
    #[inline]
    fn interpolate_single(
        src: &[f32],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        config: &ResizeOptions,
    ) -> Result<Vec<f32>> {
        let src_img = Image::from_vec_u8(
            src_w,
            src_h,
            bytemuck::cast_slice(src).to_vec(),
            PixelType::F32,
        )?;
        let mut dst_img = Image::new(dst_w, dst_h, PixelType::F32);

        let mut resizer = Resizer::new();
        resizer.resize(&src_img, &mut dst_img, config)?;

        let raw = dst_img.into_vec();
        bytemuck::try_cast_slice::<u8, f32>(&raw)
            .map(|data| data.to_vec())
            .map_err(|_| anyhow::anyhow!("`interpolate`: Failed to convert u8 slice to f32 slice"))
    }
}
