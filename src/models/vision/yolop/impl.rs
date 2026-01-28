use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array, Axis, IxDyn};
use rayon::prelude::*;

use crate::{
    contour, elapsed_module, Config, Contour, DynConf, Engine, Engines, FromConfig, Hbb, Image,
    ImageProcessor, Model, Module, NmsOps, Ops, Polygon, Xs, X, Y,
};

/// YOLOP: Panoramic Driving Perception
#[derive(Builder, Debug)]
pub struct YOLOPv2 {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub spec: String,
    pub processor: ImageProcessor,
    pub confs: DynConf,
    pub iou: f32,
}

impl Model for YOLOPv2 {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&512.into()).opt(),
            engine.try_width().unwrap_or(&512.into()).opt(),
        );
        let confs = DynConf::new_or_default(&config.inference.class_confs, 80);
        let iou = config.inference.iou.unwrap_or(0.45);
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            confs,
            iou,
            processor,
            spec,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x = elapsed_module!("YOLOPv2", "preprocess", self.processor.process(images)?);
        let ys = elapsed_module!("YOLOPv2", "inference", engines.run(&Module::Model, &x)?);
        elapsed_module!("YOLOPv2", "postprocess", self.postprocess(&ys))
    }
}

impl YOLOPv2 {
    fn postprocess(&mut self, outputs: &Xs) -> Result<Vec<Y>> {
        let x0 = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 0"))?;
        let x1 = outputs
            .get::<f32>(1)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 1"))?;
        let x2 = outputs
            .get::<f32>(2)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 2"))?;
        let xs = (X::from(x0), X::from(x1), X::from(x2));
        let (xs_da, xs_ll, xs_det) = (&xs.0, &xs.1, &xs.2);

        let ys: Vec<Y> = xs_det
            .axis_iter(Axis(0))
            .zip(xs_ll.axis_iter(Axis(0)))
            .zip(xs_da.axis_iter(Axis(0)))
            .collect::<Vec<_>>()
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, ((x_det, x_ll), x_da))| {
                let info = &self.processor.images_transform_info[idx];
                let (image_height, image_width, ratio) =
                    (info.height_src, info.width_src, info.height_scale);

                // Vehicle
                let y_bboxes: Vec<Hbb> = x_det
                    .axis_iter(Axis(0))
                    .filter_map(|x| {
                        let bbox = x.slice(s![0..4]);
                        let clss = x.slice(s![5..]).to_owned();
                        let conf = x[4];
                        let clss = conf * clss;
                        let (id, conf) = clss
                            .into_iter()
                            .enumerate()
                            .reduce(|max, x| if x.1 > max.1 { x } else { max })
                            .unwrap();
                        if conf < self.confs[id] {
                            return None;
                        }
                        let cx = bbox[0] / ratio;
                        let cy = bbox[1] / ratio;
                        let w = bbox[2] / ratio;
                        let h = bbox[3] / ratio;
                        let x = cx - w / 2.;
                        let y = cy - h / 2.;
                        let x = x.max(0.0).min(image_width as _);
                        let y = y.max(0.0).min(image_height as _);
                        Some(
                            Hbb::default()
                                .with_xywh(x, y, w, h)
                                .with_confidence(conf)
                                .with_id(id),
                        )
                    })
                    .collect();

                let mut y_bboxes = y_bboxes;
                y_bboxes.apply_nms_inplace(self.iou);

                let mut y_polygons: Vec<Polygon> = Vec::new();

                // Drivable area
                let x_da = &x_da.slice(s![1, .., ..]) - &x_da.slice(s![0, .., ..]);
                let contours = match self.get_contours_from_mask(
                    x_da.into_dyn(),
                    0.0,
                    self.width as _,
                    self.height as _,
                    image_width as _,
                    image_height as _,
                ) {
                    Err(_) => return None,
                    Ok(x) => x,
                };
                if let Some(polygon) = contours
                    .iter()
                    .map(|x| {
                        let coords: Vec<[f32; 2]> =
                            x.points.iter().map(|p| [p.0 as f32, p.1 as f32]).collect();
                        Polygon::try_from(coords)
                            .unwrap_or_default()
                            .with_id(0)
                            .with_name("Drivable area")
                            .verify()
                    })
                    .max_by(|x, y| x.area().total_cmp(&y.area()))
                {
                    y_polygons.push(polygon);
                };

                // Lane line
                let contours = match self.get_contours_from_mask(
                    x_ll.to_owned(),
                    0.5,
                    self.width as _,
                    self.height as _,
                    image_width as _,
                    image_height as _,
                ) {
                    Err(_) => return None,
                    Ok(x) => x,
                };
                if let Some(polygon) = contours
                    .iter()
                    .map(|x| {
                        let coords: Vec<[f32; 2]> =
                            x.points.iter().map(|p| [p.0 as f32, p.1 as f32]).collect();
                        Polygon::try_from(coords)
                            .unwrap_or_default()
                            .with_id(1)
                            .with_name("Lane line")
                            .verify()
                    })
                    .max_by(|x, y| x.area().total_cmp(&y.area()))
                {
                    y_polygons.push(polygon);
                };

                Some(Y::default().with_hbbs(&y_bboxes).with_polygons(&y_polygons))
            })
            .collect();

        Ok(ys)
    }

    fn get_contours_from_mask(
        &self,
        mask: Array<f32, IxDyn>,
        thresh: f32,
        w0: f32,
        h0: f32,
        w1: f32,
        h1: f32,
    ) -> Result<Vec<Contour>> {
        let mask = mask.mapv(|x| if x < thresh { 0.0f32 } else { 1.0f32 });
        let mask: Vec<u8> = Ops::interpolate_1d_u8(
            &mask.into_raw_vec_and_offset().0,
            w0 as _,
            h0 as _,
            w1 as _,
            h1 as _,
            false,
        )?;
        let mask: image::ImageBuffer<image::Luma<_>, Vec<_>> =
            image::ImageBuffer::from_raw(w1 as _, h1 as _, mask)
                .ok_or(anyhow::anyhow!("Failed to build image"))?;
        let contours = contour::find_contours_with_threshold(&mask, 0);
        Ok(contours)
    }
}
