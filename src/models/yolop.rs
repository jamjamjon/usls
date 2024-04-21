use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis, IxDyn};

use crate::{ops, Bbox, DynConf, Mask, MinOptMax, Options, OrtEngine, Y};

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
    pub fn new(options: &Options) -> Result<Self> {
        let engine = OrtEngine::new(options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );
        let nc = 80;
        let confs = DynConf::new(&options.kconfs, nc);
        engine.dry_run()?;

        Ok(Self {
            engine,
            confs,
            height,
            width,
            batch,
            iou: options.iou,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Y>> {
        let xs_ = ops::letterbox(xs, self.height() as u32, self.width() as u32, 114.0)?;
        let xs_ = ops::normalize(xs_, 0.0, 255.0);
        let ys = self.engine.run(&[xs_])?;
        self.postprocess(ys, xs)
    }

    pub fn postprocess(&self, xs: Vec<Array<f32, IxDyn>>, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
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
            let (ratio, _, _) = ops::scale_wh(
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

            // Drivable area
            let x_da_0 = x_da.slice(s![0, .., ..]).to_owned();
            let x_da_1 = x_da.slice(s![1, .., ..]).to_owned();
            let x_da = x_da_1 - x_da_0;
            let x_da = x_da
                .into_shape((self.height() as usize, self.width() as usize, 1))?
                .into_owned();
            let v = x_da
                .into_raw_vec()
                .iter()
                .map(|x| if x < &0.0 { 0.0 } else { 1.0 })
                .collect::<Vec<_>>();
            let mask_da =
                ops::build_dyn_image_from_raw(v, self.height() as u32, self.width() as u32);
            let mask_da = ops::descale_mask(
                mask_da,
                self.width() as f32,
                self.height() as f32,
                image_width,
                image_height,
            );
            let mask_da = mask_da.into_luma8();
            let mut y_masks: Vec<Mask> = Vec::new();
            let contours: Vec<imageproc::contours::Contour<i32>> =
                imageproc::contours::find_contours_with_threshold(&mask_da, 1);
            contours.iter().for_each(|contour| {
                if contour.border_type == imageproc::contours::BorderType::Outer
                    && contour.points.len() > 2
                {
                    y_masks.push(
                        Mask::default()
                            .with_id(0)
                            .with_points_imageproc(&contour.points)
                            .with_name(Some("Drivable area".to_string())),
                    );
                }
            });

            // Lane line
            let x_ll = x_ll
                .into_shape((self.height() as usize, self.width() as usize, 1))?
                .into_owned();
            let v = x_ll
                .into_raw_vec()
                .iter()
                .map(|x| if x < &0.5 { 0.0 } else { 1.0 })
                .collect::<Vec<_>>();
            let mask_ll =
                ops::build_dyn_image_from_raw(v, self.height() as u32, self.width() as u32);
            let mask_ll = ops::descale_mask(
                mask_ll,
                self.width() as f32,
                self.height() as f32,
                image_width,
                image_height,
            );
            let mask_ll = mask_ll.into_luma8();
            let contours: Vec<imageproc::contours::Contour<i32>> =
                imageproc::contours::find_contours_with_threshold(&mask_ll, 1);
            let mut masks: Vec<Mask> = Vec::new();
            contours.iter().for_each(|contour| {
                if contour.border_type == imageproc::contours::BorderType::Outer
                    && contour.points.len() > 2
                {
                    masks.push(
                        Mask::default()
                            .with_id(1)
                            .with_points_imageproc(&contour.points)
                            .with_name(Some("Lane line".to_string())),
                    );
                }
            });
            y_masks.extend(masks);

            // save
            ys.push(
                Y::default()
                    .with_bboxes(&y_bboxes)
                    .with_masks(&y_masks)
                    .apply_bboxes_nms(self.iou),
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
