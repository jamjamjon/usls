use anyhow::Result;
use fast_image_resize as fr;
use image::{DynamicImage, GenericImageView, ImageBuffer};
use ndarray::{s, Array, Axis, IxDyn};

pub fn standardize(xs: Array<f32, IxDyn>, mean: &[f32], std: &[f32]) -> Array<f32, IxDyn> {
    let mean = Array::from_shape_vec((1, mean.len(), 1, 1), mean.to_vec()).unwrap();
    let std = Array::from_shape_vec((1, std.len(), 1, 1), std.to_vec()).unwrap();
    (xs - mean) / std
}

pub fn normalize(xs: Array<f32, IxDyn>, min_: f32, max_: f32) -> Array<f32, IxDyn> {
    (xs - min_) / (max_ - min_)
}

pub fn norm2(xs: &Array<f32, IxDyn>) -> Array<f32, IxDyn> {
    let std_ = xs
        .mapv(|x| x * x)
        .sum_axis(Axis(1))
        .mapv(f32::sqrt)
        .insert_axis(Axis(1));
    xs / std_
}

pub fn scale_wh(w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
    let r = (w1 / w0).min(h1 / h0);
    (r, (w0 * r).round(), (h0 * r).round())
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

pub fn resize(
    xs: &[DynamicImage],
    height: u32,
    width: u32,
    filter: &str,
) -> Result<Array<f32, IxDyn>> {
    let mut ys = Array::ones((xs.len(), 3, height as usize, width as usize)).into_dyn();
    let mut resizer = build_resizer(filter);
    for (idx, x) in xs.iter().enumerate() {
        // src
        let src_image = fr::Image::from_vec_u8(
            std::num::NonZeroU32::new(x.width()).unwrap(),
            std::num::NonZeroU32::new(x.height()).unwrap(),
            x.to_rgb8().into_raw(),
            fr::PixelType::U8x3,
        )
        .unwrap();

        // dst
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
            .mapv(|x| x as f32)
            .permuted_axes([2, 0, 1]);
        let mut data = ys.slice_mut(s![idx, .., .., ..]);
        data.assign(&y_);
    }
    Ok(ys)
}

pub fn letterbox(
    xs: &[DynamicImage],
    height: u32,
    width: u32,
    filter: &str,
    bg: Option<u8>,
) -> Result<Array<f32, IxDyn>> {
    let mut ys = Array::ones((xs.len(), 3, height as usize, width as usize)).into_dyn();
    let mut resizer = build_resizer(filter);
    for (idx, x) in xs.iter().enumerate() {
        let (w0, h0) = x.dimensions();
        let (_, w_new, h_new) = scale_wh(w0 as f32, h0 as f32, width as f32, height as f32);

        // src
        let src_image = fr::Image::from_vec_u8(
            std::num::NonZeroU32::new(w0).unwrap(),
            std::num::NonZeroU32::new(h0).unwrap(),
            x.to_rgb8().into_raw(),
            fr::PixelType::U8x3,
        )
        .unwrap();

        // dst
        let mut dst_image = match bg {
            Some(bg) => fr::Image::from_vec_u8(
                std::num::NonZeroU32::new(width).unwrap(),
                std::num::NonZeroU32::new(height).unwrap(),
                vec![bg; 3 * height as usize * width as usize],
                src_image.pixel_type(),
            )
            .unwrap(),
            None => fr::Image::new(
                std::num::NonZeroU32::new(width).unwrap(),
                std::num::NonZeroU32::new(height).unwrap(),
                src_image.pixel_type(),
            ),
        };

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
            .mapv(|x| x as f32)
            .permuted_axes([2, 0, 1]);
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
    bg: Option<u8>,
) -> Result<Array<f32, IxDyn>> {
    let mut ys = Array::ones((xs.len(), 3, height as usize, width as usize)).into_dyn();
    let mut resizer = build_resizer(filter);
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
        let mut dst_image = match bg {
            Some(bg) => fr::Image::from_vec_u8(
                std::num::NonZeroU32::new(width).unwrap(),
                std::num::NonZeroU32::new(height).unwrap(),
                vec![bg; 3 * height as usize * width as usize],
                src_image.pixel_type(),
            )
            .unwrap(),
            None => fr::Image::new(
                std::num::NonZeroU32::new(width).unwrap(),
                std::num::NonZeroU32::new(height).unwrap(),
                src_image.pixel_type(),
            ),
        };

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
            .mapv(|x| x as f32)
            .permuted_axes([2, 0, 1]);
        let mut data = ys.slice_mut(s![idx, .., .., ..]);
        data.assign(&y_);
    }
    Ok(ys)
}

pub fn build_dyn_image_from_raw(v: Vec<f32>, height: u32, width: u32) -> DynamicImage {
    let v: ImageBuffer<image::Luma<_>, Vec<f32>> =
        ImageBuffer::from_raw(width, height, v).expect("Faild to create image from ndarray");
    image::DynamicImage::from(v)
}

pub fn descale_mask(mask: DynamicImage, w0: f32, h0: f32, w1: f32, h1: f32) -> DynamicImage {
    // 0 -> 1
    let (_, w, h) = scale_wh(w1, h1, w0, h0);
    let mut mask = mask.to_owned();
    let mask = mask.crop(0, 0, w as u32, h as u32);
    mask.resize_exact(w1 as u32, h1 as u32, image::imageops::FilterType::Triangle)
}

pub fn make_divisible(x: usize, divisor: usize) -> usize {
    (x - 1 + divisor) / divisor * divisor
}
