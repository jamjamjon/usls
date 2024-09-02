use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis, IxDyn};

use crate::{Bbox, DynConf, MinOptMax, Ops, Options, OrtEngine, Polygon, Xs, X, Y};

#[derive(Debug)]
pub struct YOLOPv2 {
    engine: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    confs: DynConf,
    iou: f32,
}

impl YOLOPv2 {
    pub fn new(options: Options) -> Result<Self> {
        let mut engine = OrtEngine::new(&options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );
        let confs = DynConf::new(&options.kconfs, 80);
        let iou = options.iou.unwrap_or(0.45f32);
        engine.dry_run()?;

        Ok(Self {
            engine,
            confs,
            height,
            width,
            batch,
            iou,
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
            Ops::Nhwc2nchw,
        ])?;
        let ys = self.engine.run(Xs::from(xs_))?;
        self.postprocess(ys, xs)
    }

    pub fn postprocess(&self, xs: Xs, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        // pub fn postprocess(&self, xs: Vec<X>, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        let (xs_da, xs_ll, xs_det) = (&xs[0], &xs[1], &xs[2]);
        for (idx, ((x_det, x_ll), x_da)) in xs_det
            .axis_iter(Axis(0))
            .zip(xs_ll.axis_iter(Axis(0)))
            .zip(xs_da.axis_iter(Axis(0)))
            .enumerate()
        {
            let image_width = xs0[idx].width() as f32;
            let image_height = xs0[idx].height() as f32;
            let (ratio, _, _) = Ops::scale_wh(
                image_width,
                image_height,
                self.width() as f32,
                self.height() as f32,
            );

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
                let x = x.max(0.0).min(image_width);
                let y = y.max(0.0).min(image_height);
                y_bboxes.push(
                    Bbox::default()
                        .with_xywh(x, y, w, h)
                        .with_confidence(conf)
                        .with_id(id as isize),
                );
            }
            let mut y_polygons: Vec<Polygon> = Vec::new();

            // Drivable area
            let x_da_0 = x_da.slice(s![0, .., ..]).to_owned();
            let x_da_1 = x_da.slice(s![1, .., ..]).to_owned();
            let x_da = x_da_1 - x_da_0;
            let contours = match self.get_contours_from_mask(
                x_da.into_dyn(),
                0.0,
                self.width() as _,
                self.height() as _,
                image_width,
                image_height,
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
                self.width() as _,
                self.height() as _,
                image_width,
                image_height,
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
                Y::default()
                    .with_bboxes(&y_bboxes)
                    .with_polygons(&y_polygons)
                    .apply_nms(self.iou),
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
