use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::Axis;

use crate::{elapsed, DynConf, Engine, Mbr, Ops, Options, Polygon, Processor, Ts, Xs, Ys, Y};

#[derive(Debug, Builder)]
pub struct DB {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    confs: DynConf,
    unclip_ratio: f32,
    binary_thresh: f32,
    min_width: f32,
    min_height: f32,
    spec: String,
    ts: Ts,
    processor: Processor,
}

impl DB {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (batch, height, width, ts, spec) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&960.into()).opt(),
            engine.try_width().unwrap_or(&960.into()).opt(),
            engine.ts.clone(),
            engine.spec().to_owned(),
        );
        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let confs = DynConf::new(options.class_confs(), 1);
        let binary_thresh = options.binary_thresh().unwrap_or(0.2);
        let unclip_ratio = options.unclip_ratio().unwrap_or(1.5);
        let min_width = options.min_width().unwrap_or(12.0);
        let min_height = options.min_height().unwrap_or(5.0);

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
            processor,
            spec,
            ts,
        })
    }

    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
        Ok(self.processor.process_images(xs)?.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[DynamicImage]) -> Result<Ys> {
        let ys = elapsed!("preprocess", self.ts, { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });

        Ok(ys)
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }

    pub fn postprocess(&mut self, xs: Xs) -> Result<Ys> {
        let mut ys = Vec::new();
        for (idx, luma) in xs[0].axis_iter(Axis(0)).enumerate() {
            let mut y_bbox = Vec::new();
            let mut y_polygons: Vec<Polygon> = Vec::new();
            let mut y_mbrs: Vec<Mbr> = Vec::new();

            // input image
            let (image_height, image_width) = self.processor.image0s_size[idx];

            // reshape
            let ratio = self.processor.scale_factors_hw[idx][0];
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
                self.width as _,
                self.height as _,
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
                    .convex_hull()
                    .verify();

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

        Ok(ys.into())
    }
}
