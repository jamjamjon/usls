use anyhow::Result;
use fast_image_resize as fr;
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array, Axis, IxDyn};

use crate::X;

pub enum Ops<'a> {
    Resize(&'a [DynamicImage], u32, u32, &'a str),
    Letterbox(&'a [DynamicImage], u32, u32, &'a str, u8),
    ResizeWithFixedHeight(&'a [DynamicImage], u32, u32, &'a str, u8),
    Normalize(f32, f32),
    Standardize(&'a [f32], &'a [f32], usize),
    Permute(&'a [usize]),
    InsertAxis(usize),
    Nhwc2nchw,
    Nchw2nhwc,
    Norm,
}

impl<'a> Ops<'_> {
    pub fn apply(ops: &[Self]) -> Result<X> {
        let mut y = X::default();

        for op in ops {
            y = match op {
                Self::Resize(xs, h, w, filter) => X::resize(xs, *h, *w, filter)?,
                Self::Letterbox(xs, h, w, filter, bg) => X::letterbox(xs, *h, *w, filter, *bg)?,
                Self::ResizeWithFixedHeight(xs, h, w, filter, bg) => {
                    X::resize_with_fixed_height(xs, *h, *w, filter, *bg)?
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

    pub fn build_resizer(ty: &str) -> fr::Resizer {
        let ty = match ty {
            "box" => fr::FilterType::Box,
            "bilinear" => fr::FilterType::Bilinear,
            "hamming" => fr::FilterType::Hamming,
            "catmullRom" => fr::FilterType::CatmullRom,
            "mitchell" => fr::FilterType::Mitchell,
            "lanczos3" => fr::FilterType::Lanczos3,
            _ => todo!(),
        };
        fr::Resizer::new(fr::ResizeAlg::Convolution(ty))
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

    fn create_src_image(x: &DynamicImage) -> Result<fr::Image> {
        Ok(fr::Image::from_vec_u8(
            std::num::NonZeroU32::new(x.width()).unwrap(),
            std::num::NonZeroU32::new(x.height()).unwrap(),
            x.to_rgb8().into_raw(),
            fr::PixelType::U8x3,
        )?)
    }

    pub fn resize(
        xs: &[DynamicImage],
        height: u32,
        width: u32,
        filter: &str,
    ) -> Result<Array<f32, IxDyn>> {
        let mut ys = Array::ones((xs.len(), height as usize, width as usize, 3)).into_dyn();
        let mut resizer = Self::build_resizer(filter);
        for (idx, x) in xs.iter().enumerate() {
            let src_image = Self::create_src_image(x)?;
            let mut dst_image = fr::Image::new(
                std::num::NonZeroU32::new(width).unwrap(),
                std::num::NonZeroU32::new(height).unwrap(),
                src_image.pixel_type(),
            );

            // resize
            resizer
                .resize(&src_image.view(), &mut dst_image.view_mut())
                .unwrap();
            let buffer = dst_image.into_vec();

            // to ndarray
            let y_ = Array::from_shape_vec((height as usize, width as usize, 3), buffer)
                .unwrap()
                .mapv(|x| x as f32);
            // .permuted_axes([2, 0, 1]);
            let mut data = ys.slice_mut(s![idx, .., .., ..]);
            data.assign(&y_);
        }
        Ok(ys)
    }

    pub fn letterbox(
        xs: &[DynamicImage],
        height: u32,
        width: u32,
        filter: &'a str,
        bg: u8,
    ) -> Result<Array<f32, IxDyn>> {
        let mut ys = Array::ones((xs.len(), height as usize, width as usize, 3)).into_dyn();
        let mut resizer = Self::build_resizer(filter);
        for (idx, x) in xs.iter().enumerate() {
            let (w0, h0) = x.dimensions();
            let (_, w_new, h_new) =
                Self::scale_wh(w0 as f32, h0 as f32, width as f32, height as f32);

            // src
            let src_image = Self::create_src_image(x)?;

            // dst
            let mut dst_image = fr::Image::from_vec_u8(
                std::num::NonZeroU32::new(width).unwrap(),
                std::num::NonZeroU32::new(height).unwrap(),
                vec![bg; 3 * height as usize * width as usize],
                fr::PixelType::U8x3,
            )
            .unwrap(); // 57.118µs

            // mutable view
            let mut dst_view = dst_image
                .view_mut()
                .crop(
                    0,
                    0,
                    std::num::NonZeroU32::new(w_new as u32).unwrap(),
                    std::num::NonZeroU32::new(h_new as u32).unwrap(),
                )
                .unwrap();

            // resize
            resizer.resize(&src_image.view(), &mut dst_view).unwrap();
            let buffer = dst_image.into_vec();

            // to ndarray
            let y_ = Array::from_shape_vec((height as usize, width as usize, 3), buffer)
                .unwrap()
                .mapv(|x| x as f32);
            // .permuted_axes([2, 0, 1]);
            let mut data = ys.slice_mut(s![idx, .., .., ..]);
            data.assign(&y_);
        }
        Ok(ys)
    }

    pub fn resize_with_fixed_height(
        xs: &[DynamicImage],
        height: u32,
        width: u32,
        filter: &str,
        bg: u8,
    ) -> Result<Array<f32, IxDyn>> {
        let mut ys = Array::ones((xs.len(), height as usize, width as usize, 3)).into_dyn();
        let mut resizer = Self::build_resizer(filter);
        for (idx, x) in xs.iter().enumerate() {
            let (w0, h0) = x.dimensions();
            let h_new = height;
            let w_new = height * w0 / h0;

            // src
            let src_image = fr::Image::from_vec_u8(
                std::num::NonZeroU32::new(w0).unwrap(),
                std::num::NonZeroU32::new(h0).unwrap(),
                x.to_rgb8().into_raw(),
                fr::PixelType::U8x3,
            )
            .unwrap();

            // dst
            let mut dst_image = fr::Image::from_vec_u8(
                std::num::NonZeroU32::new(width).unwrap(),
                std::num::NonZeroU32::new(height).unwrap(),
                vec![bg; 3 * height as usize * width as usize],
                src_image.pixel_type(),
            )
            .unwrap();

            // mutable view
            let mut dst_view = dst_image
                .view_mut()
                .crop(
                    0,
                    0,
                    std::num::NonZeroU32::new(w_new).unwrap(),
                    std::num::NonZeroU32::new(h_new).unwrap(),
                )
                .unwrap();

            // resize
            resizer.resize(&src_image.view(), &mut dst_view).unwrap();
            let buffer = dst_image.into_vec();

            // to ndarray
            let y_ = Array::from_shape_vec((height as usize, width as usize, 3), buffer)
                .unwrap()
                .mapv(|x| x as f32);
            let mut data = ys.slice_mut(s![idx, .., .., ..]);
            data.assign(&y_);
        }
        Ok(ys)
    }
}
