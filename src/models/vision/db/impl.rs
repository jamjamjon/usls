use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    elapsed_module, inputs, Config, DynConf, Engine, Engines, FromConfig, Hbb, Image,
    ImageProcessor, Mask, Model, Module, Obb, Ops, Polygon, Xs, X, Y,
};

/// DB (Differentiable Binarization) model for text detection.
#[derive(Debug, Builder)]
pub struct DB {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub confs: DynConf,
    pub unclip_ratio: f32,
    pub binary_thresh: f32,
    pub min_width: f32,
    pub min_height: f32,
    pub spec: String,
    pub processor: ImageProcessor,
}

impl Model for DB {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width, spec) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&960.into()).opt(),
            engine.try_width().unwrap_or(&960.into()).opt(),
            engine.spec().to_owned(),
        );
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let confs = DynConf::new_or_default(&config.inference.class_confs, 1);
        let binary_thresh = config.inference.db_binary_thresh.unwrap_or(0.2);
        let unclip_ratio = config.inference.db_unclip_ratio.unwrap_or(1.5);
        let min_width = config.inference.min_width.unwrap_or(12.0);
        let min_height = config.inference.min_height.unwrap_or(5.0);

        let model = Self {
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
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x = elapsed_module!("DB", "preprocess", self.processor.process(images)?);
        let ys = elapsed_module!(
            "DB",
            "inference",
            engines.run(&Module::Model, inputs![&x]?)?
        );
        elapsed_module!("DB", "postprocess", self.postprocess(&ys))
    }
}

impl DB {
    fn postprocess(&mut self, outputs: &Xs) -> Result<Vec<Y>> {
        let xs = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output"))?;
        let xs = X::from(xs);
        let ys: Vec<Y> = xs
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, luma)| {
                let info = &self.processor.images_transform_info[idx];
                let (image_height, image_width) = (info.height_src, info.width_src);
                let ratio = self.processor.images_transform_info()[idx].height_scale;
                let v: Vec<f32> = luma
                    .as_slice()?
                    .par_iter()
                    .map(|x| if x <= &self.binary_thresh { 0.0f32 } else { *x })
                    .collect();

                let luma: Vec<u8> = Ops::interpolate_1d_u8(
                    &v,
                    self.width as _,
                    self.height as _,
                    image_width as _,
                    image_height as _,
                    true,
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
