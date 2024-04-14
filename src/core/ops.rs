use crate::{Mask, Polygon};
use anyhow::Result;
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer};
use ndarray::{Array, Axis, Ix2, IxDyn};

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

pub fn dot2(query: &Array<f32, IxDyn>, gallery: &Array<f32, IxDyn>) -> Result<Vec<Vec<f32>>> {
    // (m, ndim) * (n, ndim).t => (m, n)
    let query = query.to_owned().into_dimensionality::<Ix2>()?;
    let gallery = gallery.to_owned().into_dimensionality::<Ix2>()?;
    let matrix = query.dot(&gallery.t());
    let exps = matrix.mapv(|x| x.exp());
    let stds = exps.sum_axis(Axis(1));
    let matrix = exps / stds.insert_axis(Axis(1));
    let matrix: Vec<Vec<f32>> = matrix.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();
    Ok(matrix)
}

pub fn scale_wh(w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
    let r = (w1 / w0).min(h1 / h0);
    (r, (w0 * r).round(), (h0 * r).round())
}

pub fn resize(xs: &[DynamicImage], height: u32, width: u32) -> Result<Array<f32, IxDyn>> {
    let mut ys = Array::ones((xs.len(), 3, height as usize, width as usize)).into_dyn();
    for (idx, x) in xs.iter().enumerate() {
        let img = x.resize_exact(width, height, image::imageops::FilterType::Triangle);
        for (x, y, rgb) in img.pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b, _] = rgb.0;
            ys[[idx, 0, y, x]] = r as f32;
            ys[[idx, 1, y, x]] = g as f32;
            ys[[idx, 2, y, x]] = b as f32;
        }
    }
    Ok(ys)
}

pub fn letterbox(
    xs: &[DynamicImage],
    height: u32,
    width: u32,
    bg: f32,
) -> Result<Array<f32, IxDyn>> {
    let mut ys = Array::ones((xs.len(), 3, height as usize, width as usize)).into_dyn();
    ys.fill(bg);
    for (idx, x) in xs.iter().enumerate() {
        let (w0, h0) = x.dimensions();
        let (_, w_new, h_new) = scale_wh(w0 as f32, h0 as f32, width as f32, height as f32);
        let img = x.resize_exact(
            w_new as u32,
            h_new as u32,
            image::imageops::FilterType::CatmullRom,
        );
        for (x, y, rgb) in img.pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b, _] = rgb.0;
            ys[[idx, 0, y, x]] = r as f32;
            ys[[idx, 1, y, x]] = g as f32;
            ys[[idx, 2, y, x]] = b as f32;
        }
    }
    Ok(ys)
}

pub fn resize_with_fixed_height(
    xs: &[DynamicImage],
    height: u32,
    width: u32,
    bg: f32,
) -> Result<Array<f32, IxDyn>> {
    let mut ys = Array::ones((xs.len(), 3, height as usize, width as usize)).into_dyn();
    ys.fill(bg);
    for (idx, x) in xs.iter().enumerate() {
        let (w0, h0) = x.dimensions();
        let h_new = height;
        let w_new = height * w0 / h0;
        let img = x.resize_exact(w_new, h_new, image::imageops::FilterType::CatmullRom);
        for (x, y, rgb) in img.pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b, _] = rgb.0;
            ys[[idx, 0, y, x]] = r as f32;
            ys[[idx, 1, y, x]] = g as f32;
            ys[[idx, 2, y, x]] = b as f32;
        }
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

pub fn get_masks_from_image(
    mask: GrayImage,
    thresh: u8,
    id: usize,
    name: Option<String>,
) -> Vec<Mask> {
    // let mask = mask.into_luma8();
    let contours: Vec<imageproc::contours::Contour<i32>> =
        imageproc::contours::find_contours_with_threshold(&mask, thresh);
    let mut masks: Vec<Mask> = Vec::new();
    contours.iter().for_each(|contour| {
        // contour.border_type == imageproc::contours::BorderType::Outer &&
        if contour.points.len() > 2 {
            masks.push(Mask {
                polygon: Polygon::from_contour(contour),
                id,
                name: name.to_owned(),
            });
        }
    });
    masks
}
