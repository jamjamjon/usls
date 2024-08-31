use anyhow::Result;
use image::DynamicImage;
use ndarray::Axis;

use crate::{DynConf, Mbr, MinOptMax, Ops, Options, OrtEngine, Polygon, Xs, X, Y};

#[derive(Debug)]
pub struct DB {
    engine: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    confs: DynConf,
    unclip_ratio: f32,
    binary_thresh: f32,
    min_width: f32,
    min_height: f32,
}

impl DB {
    pub fn new(options: Options) -> Result<Self> {
        let mut engine = OrtEngine::new(&options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );
        let confs = DynConf::new(&options.confs, 1);
        let unclip_ratio = options.unclip_ratio;
        let binary_thresh = 0.2;
        let min_width = options.min_width.unwrap_or(0.);
        let min_height = options.min_height.unwrap_or(0.);
        engine.dry_run()?;

        Ok(Self {
            engine,
            confs,
            height,
            width,
            batch,
            min_width,
            min_height,
            unclip_ratio,
            binary_thresh,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Y>> {
        let xs_ = X::apply(&[
            Ops::Letterbox(
                xs,
                self.height() as u32,
                self.width() as u32,
                "Bilinear",
                114,
                "auto",
                false,
            ),
            Ops::Normalize(0., 255.),
            Ops::Standardize(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225], 3),
            Ops::Nhwc2nchw,
        ])?;
        let ys = self.engine.run(Xs::from(xs_))?;
        self.postprocess(ys, xs)
    }

    pub fn postprocess(&self, xs: Xs, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        let mut ys = Vec::new();
        for (idx, luma) in xs[0].axis_iter(Axis(0)).enumerate() {
            let mut y_bbox = Vec::new();
            let mut y_polygons: Vec<Polygon> = Vec::new();
            let mut y_mbrs: Vec<Mbr> = Vec::new();

            // input image
            let image_width = xs0[idx].width() as f32;
            let image_height = xs0[idx].height() as f32;

            // reshape
            let h = luma.dim()[1];
            let w = luma.dim()[2];
            let (ratio, _, _) = Ops::scale_wh(image_width, image_height, w as f32, h as f32);
            let v = luma
                .into_owned()
                .into_raw_vec_and_offset()
                .0
                .iter()
                .map(|x| {
                    if x <= &self.binary_thresh {
                        0u8
                    } else {
                        (*x * 255.0) as u8
                    }
                })
                .collect::<Vec<_>>();

            let luma = Ops::resize_luma8_u8(
                &v,
                self.width() as _,
                self.height() as _,
                image_width as _,
                image_height as _,
                true,
                "Bilinear",
            )?;
            let mask_im: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                match image::ImageBuffer::from_raw(image_width as _, image_height as _, luma) {
                    None => continue,
                    Some(x) => x,
                };

            // contours
            let contours: Vec<imageproc::contours::Contour<i32>> =
                imageproc::contours::find_contours_with_threshold(&mask_im, 1);

            // loop
            for contour in contours.iter() {
                if contour.border_type == imageproc::contours::BorderType::Hole
                    && contour.points.len() <= 2
                {
                    continue;
                }

                let polygon = Polygon::default().with_points_imageproc(&contour.points);
                let delta = polygon.area() * ratio.round() as f64 * self.unclip_ratio as f64
                    / polygon.perimeter();

                // TODO: optimize
                let polygon = polygon
                    .unclip(delta, image_width as f64, image_height as f64)
                    .resample(50)
                    // .simplify(6e-4)
                    .convex_hull();

                if let Some(bbox) = polygon.bbox() {
                    if bbox.height() < self.min_height || bbox.width() < self.min_width {
                        continue;
                    }
                    let confidence = polygon.area() as f32 / bbox.area();
                    if confidence < self.confs[0] {
                        continue;
                    }
                    y_bbox.push(bbox.with_confidence(confidence).with_id(0));

                    if let Some(mbr) = polygon.mbr() {
                        y_mbrs.push(mbr.with_confidence(confidence).with_id(0));
                    }
                    y_polygons.push(polygon.with_id(0));
                } else {
                    continue;
                }
            }

            ys.push(
                Y::default()
                    .with_bboxes(&y_bbox)
                    .with_polygons(&y_polygons)
                    .with_mbrs(&y_mbrs),
            );
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
