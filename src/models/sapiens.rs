use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Array3, Axis};
use rayon::prelude::*;

use crate::{Mask, MinOptMax, Ops, Options, OrtEngine, Polygon, Xs, X, Y};

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum SapiensTask {
    Depth,
    Seg,
    Normal,
    Pose,
}

#[derive(Debug)]
pub struct Sapiens {
    engine: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    task: SapiensTask,
    names: Option<Vec<String>>,
}

impl Sapiens {
    pub fn new(options: Options) -> Result<Self> {
        let mut engine = OrtEngine::new(&options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );
        let task = options
            .sapiens_task
            .expect("Error: No sapiens task specified.");
        let names = options.names;
        engine.dry_run()?;

        Ok(Self {
            engine,
            height,
            width,
            batch,
            task,
            names,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Y>> {
        let xs_ = X::apply(&[
            Ops::Resize(xs, self.height() as u32, self.width() as u32, "Bilinear"),
            Ops::Standardize(&[123.5, 116.5, 103.5], &[58.5, 57.0, 57.5], 3),
            Ops::Nhwc2nchw,
        ])?;

        let ys = self.engine.run(Xs::from(xs_))?;
        self.postprocess(ys, xs)
    }

    pub fn postprocess(&self, xs: Xs, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for (idx, b) in xs[0].axis_iter(Axis(0)).enumerate() {
            let (w1, h1) = (xs0[idx].width(), xs0[idx].height());
            match self.task {
                SapiensTask::Depth => {
                    let luma = b.slice(s![0, .., ..]);
                    let v = luma.to_owned().into_raw_vec_and_offset().0;
                    let max_ = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
                    let min_ = v.iter().min_by(|x, y| x.total_cmp(y)).unwrap();
                    let v = v
                        .iter()
                        .map(|x| (((*x - min_) / (max_ - min_)) * 255.) as u8)
                        .collect::<Vec<_>>();

                    let luma = Ops::resize_luma8_vec(
                        &v,
                        self.width() as _,
                        self.height() as _,
                        w1 as _,
                        h1 as _,
                        false,
                        "Bilinear",
                    )?;
                    println!("{:?}", luma);
                    let luma: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                        match image::ImageBuffer::from_raw(w1 as _, h1 as _, luma) {
                            None => continue,
                            Some(x) => x,
                        };
                    ys.push(Y::default().with_masks(&[Mask::default().with_mask(luma)]));
                }
                SapiensTask::Seg => {
                    let (n, h, w) = (b.shape()[0], b.shape()[1], b.shape()[2]);

                    // map 28 masks to original scale
                    let mut masks = Array::zeros((n, h1 as usize, w1 as usize)).into_dyn();
                    for (ida, luma) in b.axis_iter(Axis(0)).enumerate() {
                        let v = Ops::resize_lumaf32_f32(
                            &luma.to_owned().into_raw_vec_and_offset().0,
                            w as _,
                            h as _,
                            w1 as _,
                            h1 as _,
                            false,
                            "Bilinear",
                        )?;
                        let y_ = Array::from_shape_vec((h1 as usize, w1 as usize), v)?;
                        masks.slice_mut(s![ida, .., ..]).assign(&y_);
                    }

                    // generate one hxw matrix
                    let mut matrix = Array3::<usize>::zeros((1, h1 as _, w1 as _));
                    let mut ids = Vec::new(); // save index
                    for hh in 0..h1 {
                        for ww in 0..w1 {
                            let channel_slice = masks.slice(s![.., hh as usize, ww as usize]);

                            let (i, c) = match channel_slice
                                .to_owned()
                                .into_raw_vec_and_offset()
                                .0
                                .into_iter()
                                .enumerate()
                                .max_by(|a, b| a.1.total_cmp(&b.1))
                            {
                                Some((i, c)) => (i, c),
                                None => continue,
                            };

                            // TODO: filter background right here?
                            if c <= 0. || i == 0 {
                                continue;
                            }
                            matrix[[0, hh as _, ww as _]] = i;

                            if !ids.contains(&i) {
                                ids.push(i);
                            }
                        }
                    }

                    // iterate each class to generate each class mask
                    let mut y_masks: Vec<Mask> = Vec::new();
                    let mut y_polygons: Vec<Polygon> = Vec::new();
                    for i in ids.iter() {
                        let luma = matrix.mapv(|x| if x == *i { 255 } else { 0 });

                        // generate mask images and polygons
                        let luma: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                            match image::ImageBuffer::from_raw(
                                w1 as _,
                                h1 as _,
                                luma.into_raw_vec_and_offset().0,
                            ) {
                                None => continue,
                                Some(x) => x,
                            };

                        // find contours
                        let contours: Vec<imageproc::contours::Contour<i32>> =
                            imageproc::contours::find_contours_with_threshold(&luma, 0);
                        let polygon = contours
                            .into_par_iter()
                            .map(|x| {
                                let mut polygon = Polygon::default()
                                    .with_id(*i as _)
                                    .with_points_imageproc(&x.points);
                                if let Some(names) = &self.names {
                                    polygon = polygon.with_name(&names[*i]);
                                }
                                polygon
                            })
                            .max_by(|x, y| x.area().total_cmp(&y.area()))
                            .unwrap();

                        y_polygons.push(polygon);

                        let mut mask = Mask::default().with_mask(luma).with_id(*i as _);
                        if let Some(names) = &self.names {
                            mask = mask.with_name(&names[*i]);
                        }
                        y_masks.push(mask);
                    }
                    ys.push(Y::default().with_masks(&y_masks).with_polygons(&y_polygons));
                }

                _ => todo!(),
            }
        }
        Ok(ys)
    }

    pub fn batch(&self) -> isize {
        self.batch.opt
    }

    pub fn width(&self) -> isize {
        self.width.opt
    }

    pub fn height(&self) -> isize {
        self.height.opt
    }
}
