use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array2, Axis};

use crate::{Mask, MinOptMax, Ops, Options, OrtEngine, Polygon, Xs, X, Y};

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum SapiensTask {
    Seg,
    Depth,
    Normal,
    Pose,
}

#[derive(Debug)]
pub struct Sapiens {
    engine_seg: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    task: SapiensTask,
    names_body: Option<Vec<String>>,
}

impl Sapiens {
    pub fn new(options_seg: Options) -> Result<Self> {
        let mut engine_seg = OrtEngine::new(&options_seg)?;
        let (batch, height, width) = (
            engine_seg.batch().to_owned(),
            engine_seg.height().to_owned(),
            engine_seg.width().to_owned(),
        );
        let task = options_seg
            .sapiens_task
            .expect("Error: No sapiens task specified.");
        let names_body = options_seg.names;
        engine_seg.dry_run()?;

        Ok(Self {
            engine_seg,
            height,
            width,
            batch,
            task,
            names_body,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Y>> {
        let xs_ = X::apply(&[
            Ops::Resize(xs, self.height() as u32, self.width() as u32, "Bilinear"),
            Ops::Standardize(&[123.5, 116.5, 103.5], &[58.5, 57.0, 57.5], 3),
            Ops::Nhwc2nchw,
        ])?;

        match self.task {
            SapiensTask::Seg => {
                let ys = self.engine_seg.run(Xs::from(xs_))?;
                self.postprocess_seg(ys, xs)
            }
            _ => todo!(),
        }
    }

    pub fn postprocess_seg(&self, xs: Xs, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for (idx, b) in xs[0].axis_iter(Axis(0)).enumerate() {
            let (w1, h1) = (xs0[idx].width(), xs0[idx].height());

            // rescale
            let masks = Ops::interpolate_3d(b.to_owned(), w1 as _, h1 as _, "Bilinear")?;

            // generate mask
            let mut mask = Array2::<usize>::zeros((h1 as _, w1 as _));
            let mut ids = Vec::new();
            for hh in 0..h1 {
                for ww in 0..w1 {
                    let pt_slice = masks.slice(s![.., hh as usize, ww as usize]);
                    let (i, c) = match pt_slice
                        .into_iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                    {
                        Some((i, c)) => (i, c),
                        None => continue,
                    };

                    if *c <= 0. || i == 0 {
                        continue;
                    }
                    mask[[hh as _, ww as _]] = i;

                    if !ids.contains(&i) {
                        ids.push(i);
                    }
                }
            }

            // generate masks and polygons
            let mut y_masks: Vec<Mask> = Vec::new();
            let mut y_polygons: Vec<Polygon> = Vec::new();
            for i in ids.iter() {
                let luma = mask.mapv(|x| if x == *i { 255 } else { 0 });
                let luma: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                    match image::ImageBuffer::from_raw(
                        w1 as _,
                        h1 as _,
                        luma.into_raw_vec_and_offset().0,
                    ) {
                        None => continue,
                        Some(x) => x,
                    };

                // contours
                let contours: Vec<imageproc::contours::Contour<i32>> =
                    imageproc::contours::find_contours_with_threshold(&luma, 0);
                let polygon = match contours
                    .into_iter()
                    .map(|x| {
                        let mut polygon = Polygon::default()
                            .with_id(*i as _)
                            .with_points_imageproc(&x.points)
                            .verify();
                        if let Some(names_body) = &self.names_body {
                            polygon = polygon.with_name(&names_body[*i]);
                        }
                        polygon
                    })
                    .max_by(|x, y| x.area().total_cmp(&y.area()))
                {
                    Some(p) => p,
                    None => continue,
                };

                y_polygons.push(polygon);

                let mut mask = Mask::default().with_mask(luma).with_id(*i as _);
                if let Some(names_body) = &self.names_body {
                    mask = mask.with_name(&names_body[*i]);
                }
                y_masks.push(mask);
            }
            ys.push(Y::default().with_masks(&y_masks).with_polygons(&y_polygons));
        }
        Ok(ys)
    }

    pub fn batch(&self) -> isize {
        self.batch.opt() as _
    }

    pub fn width(&self) -> isize {
        self.width.opt() as _
    }

    pub fn height(&self) -> isize {
        self.height.opt() as _
    }
}
