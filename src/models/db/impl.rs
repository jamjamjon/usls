use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    elapsed_module, Config, DynConf, Engine, Hbb, Image, Mask, Obb, Ops, Polygon, Processor, Xs, Y,
};

/// DB (Differentiable Binarization) model for text detection.
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
    processor: Processor,
}

impl DB {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, height, width, spec) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&960.into()).opt(),
            engine.try_width().unwrap_or(&960.into()).opt(),
            engine.spec().to_owned(),
        );
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let confs = DynConf::new_or_default(config.class_confs(), 1);
        let binary_thresh = config.db_binary_thresh().unwrap_or(0.2);
        let unclip_ratio = config.db_unclip_ratio().unwrap_or(1.5);
        let min_width = config.min_width().unwrap_or(12.0);
        let min_height = config.min_height().unwrap_or(5.0);

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
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        Ok(self.processor.process_images(xs)?.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("DB", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("DB", "inference", self.inference(ys)?);
        let ys = elapsed_module!("DB", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    pub fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, luma)| {
                // input image
                let (image_height, image_width) = (
                    self.processor.images_transform_info[idx].height_src,
                    self.processor.images_transform_info[idx].width_src,
                );

                // reshape
                let ratio = self.processor.images_transform_info[idx].height_scale;
                let v = luma
                    .as_slice()?
                    .par_iter()
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
                )
                .ok()?;
                let mask = Mask::new(&luma, image_width, image_height).ok()?;

                let (y_polygons, y_bbox, y_mbrs): (Vec<Polygon>, Vec<Hbb>, Vec<Obb>) = mask
                    .polygons()
                    .into_par_iter()
                    .filter_map(|polygon| {
                        let polygon = polygon.with_id(0);
                        let delta =
                            polygon.area() * ratio.round() as f64 * self.unclip_ratio as f64
                                / polygon.perimeter();

                        let polygon = polygon
                            .unclip(delta, image_width as f64, image_height as f64)
                            .resample(50)
                            // .simplify(6e-4)
                            .convex_hull()
                            .verify();

                        polygon.hbb().and_then(|bbox| {
                            if bbox.height() < self.min_height || bbox.width() < self.min_width {
                                return None;
                            }
                            let confidence = polygon.area() as f32 / bbox.area();
                            if confidence < self.confs[0] {
                                return None;
                            }
                            let bbox = bbox.with_confidence(confidence).with_id(0);
                            let mbr = polygon
                                .obb()
                                .map(|mbr| mbr.with_confidence(confidence).with_id(0));

                            Some((polygon.with_confidence(confidence), bbox, mbr))
                        })
                    })
                    .fold(
                        || (Vec::new(), Vec::new(), Vec::new()),
                        |mut acc, (polygon, bbox, mbr)| {
                            acc.0.push(polygon);
                            acc.1.push(bbox);
                            if let Some(mbr) = mbr {
                                acc.2.push(mbr);
                            }
                            acc
                        },
                    )
                    .reduce(
                        || (Vec::new(), Vec::new(), Vec::new()),
                        |mut acc, (polygons, bboxes, mbrs)| {
                            acc.0.extend(polygons);
                            acc.1.extend(bboxes);
                            acc.2.extend(mbrs);
                            acc
                        },
                    );

                Some(
                    Y::default()
                        .with_hbbs(&y_bbox)
                        .with_polygons(&y_polygons)
                        .with_obbs(&y_mbrs),
                )
            })
            .collect();

        Ok(ys)
    }
}
