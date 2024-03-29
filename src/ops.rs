use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array, Axis, Ix2, IxDyn};

pub fn scale_wh(w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
    let r = (w1 / w0).min(h1 / h0);
    (r, (w0 * r).round(), (h0 * r).round())
}

pub fn resize(
    xs: &[DynamicImage],
    height: u32,
    width: u32,
    norm_imagenet: bool,
) -> Result<Array<f32, IxDyn>> {
    let norm = 255.0;
    let mut ys = Array::ones(vec![xs.len(), 3, height as usize, width as usize]).into_dyn();
    // let mut ys = Array::ones((xs.len(), 3, height as usize, width as usize)).into_dyn();
    for (idx, x) in xs.iter().enumerate() {
        let (w0, h0) = x.dimensions();
        let w0 = w0 as f32;
        let h0 = h0 as f32;
        let (_, w_new, h_new) = scale_wh(w0, h0, width as f32, height as f32); // f32 round
        let img = x.resize_exact(
            w_new as u32,
            h_new as u32,
            image::imageops::FilterType::Triangle,
        );
        for (x, y, rgb) in img.pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b, _] = rgb.0;
            ys[[idx, 0, y, x]] = (r as f32) / norm;
            ys[[idx, 1, y, x]] = (g as f32) / norm;
            ys[[idx, 2, y, x]] = (b as f32) / norm;
        }
    }

    if norm_imagenet {
        let mean =
            Array::from_shape_vec((1, 3, 1, 1), vec![0.48145466, 0.4578275, 0.40821073]).unwrap();
        let std = Array::from_shape_vec((1, 3, 1, 1), vec![0.26862954, 0.261_302_6, 0.275_777_1])
            .unwrap();
        ys = (ys - mean) / std;
    }
    Ok(ys)
}

pub fn letterbox(xs: &[DynamicImage], height: u32, width: u32) -> Result<Array<f32, IxDyn>> {
    let norm = 255.0;
    let bg = 144.0;
    let mut ys = Array::ones((xs.len(), 3, height as usize, width as usize)).into_dyn();
    ys.fill(bg / norm);
    for (idx, x) in xs.iter().enumerate() {
        let (w0, h0) = x.dimensions();
        let w0 = w0 as f32;
        let h0 = h0 as f32;
        let (_, w_new, h_new) = scale_wh(w0, h0, width as f32, height as f32); // f32 round
        let img = x.resize_exact(
            w_new as u32,
            h_new as u32,
            image::imageops::FilterType::Triangle,
        );
        for (x, y, rgb) in img.pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b, _] = rgb.0;
            ys[[idx, 0, y, x]] = (r as f32) / norm;
            ys[[idx, 1, y, x]] = (g as f32) / norm;
            ys[[idx, 2, y, x]] = (b as f32) / norm;
        }
    }
    Ok(ys)
}

pub fn norm(xs: &Array<f32, IxDyn>) -> Array<f32, IxDyn> {
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
