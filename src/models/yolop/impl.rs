use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array, Axis, IxDyn};

use crate::{
    elapsed_module, Config, DynConf, Engine, Hbb, Image, NmsOps, Ops, Polygon, Processor, Xs, Y,
};

#[derive(Builder, Debug)]
pub struct YOLOPv2 {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,

    spec: String,
    processor: Processor,
    confs: DynConf,
    iou: f32,
}

impl YOLOPv2 {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&512.into()).opt(),
            engine.try_width().unwrap_or(&512.into()).opt(),
        );
        let confs = DynConf::new_or_default(config.class_confs(), 80);
        let iou = config.iou.unwrap_or(0.45f32);
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            confs,
            iou,

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
        let ys = elapsed_module!("YOLOPv2", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("YOLOPv2", "inference", self.inference(ys)?);
        let ys = elapsed_module!("YOLOPv2", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }
    fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        let (xs_da, xs_ll, xs_det) = (&xs[0], &xs[1], &xs[2]);
        for (idx, ((x_det, x_ll), x_da)) in xs_det
            .axis_iter(Axis(0))
            .zip(xs_ll.axis_iter(Axis(0)))
            .zip(xs_da.axis_iter(Axis(0)))
            .enumerate()
        {
            let (image_height, image_width) = (
                self.processor.images_transform_info[idx].height_src,
                self.processor.images_transform_info[idx].width_src,
            );
            let ratio = self.processor.images_transform_info[idx].height_scale;

            // Vehicle
            let mut y_bboxes = Vec::new();
            for x in x_det.axis_iter(Axis(0)) {
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
                    continue;
                }
                let cx = bbox[0] / ratio;
                let cy = bbox[1] / ratio;
                let w = bbox[2] / ratio;
                let h = bbox[3] / ratio;
                let x = cx - w / 2.;
                let y = cy - h / 2.;
                let x = x.max(0.0).min(image_width as _);
                let y = y.max(0.0).min(image_height as _);
                y_bboxes.push(
                    Hbb::default()
                        .with_xywh(x, y, w, h)
                        .with_confidence(conf)
                        .with_id(id),
                );
            }
            y_bboxes.apply_nms_inplace(self.iou);

            let mut y_polygons: Vec<Polygon> = Vec::new();

            // Drivable area
            let x_da_0 = x_da.slice(s![0, .., ..]).to_owned();
            let x_da_1 = x_da.slice(s![1, .., ..]).to_owned();
            let x_da = x_da_1 - x_da_0;
            let contours = match self.get_contours_from_mask(
                x_da.into_dyn(),
                0.0,
                self.width as _,
                self.height as _,
                image_width as _,
                image_height as _,
            ) {
                Err(_) => continue,
                Ok(x) => x,
            };
            if let Some(polygon) = contours
                .iter()
                .map(|x| {
                    Polygon::default()
                        .with_id(0)
                        .with_points_imageproc(&x.points)
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
                Err(_) => continue,
                Ok(x) => x,
            };
            if let Some(polygon) = contours
                .iter()
                .map(|x| {
                    Polygon::default()
                        .with_id(1)
                        .with_points_imageproc(&x.points)
                        .with_name("Lane line")
                        .verify()
                })
                .max_by(|x, y| x.area().total_cmp(&y.area()))
            {
                y_polygons.push(polygon);
            };

            // save
            ys.push(
                Y::default().with_hbbs(&y_bboxes).with_polygons(&y_polygons), // .apply_nms(self.iou),
            );
        }

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
    ) -> Result<Vec<imageproc::contours::Contour<i32>>> {
        let mask = mask.mapv(|x| if x < thresh { 0u8 } else { 255u8 });
        let mask = Ops::resize_luma8_u8(
            &mask.into_raw_vec_and_offset().0,
            w0,
            h0,
            w1,
            h1,
            false,
            "Bilinear",
        )?;
        let mask: image::ImageBuffer<image::Luma<_>, Vec<_>> =
            image::ImageBuffer::from_raw(w1 as _, h1 as _, mask)
                .ok_or(anyhow::anyhow!("Failed to build image"))?;
        let contours: Vec<imageproc::contours::Contour<i32>> =
            imageproc::contours::find_contours_with_threshold(&mask, 0);
        Ok(contours)
    }
}
