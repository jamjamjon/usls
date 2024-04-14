use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis, IxDyn};

use crate::{ops, Bbox, DynConf, MinOptMax, Options, OrtEngine, Rect, Ys};

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

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Ys>> {
        let xs_ = ops::letterbox(xs, self.height() as u32, self.width() as u32, 114.0)?;
        let xs_ = ops::normalize(xs_, 0.0, 255.0);
        let ys = self.engine.run(&[xs_])?;
        let ys = self.postprocess(ys, xs)?;
        Ok(ys)
    }

    pub fn postprocess(&self, xs: Vec<Array<f32, IxDyn>>, xs0: &[DynamicImage]) -> Result<Vec<Ys>> {
        let (xs_da, xs_ll, xs_det) = (&xs[0], &xs[1], &xs[2]);
        let mut ys: Vec<Ys> = Vec::new();
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
            let mut ys_bbox = Vec::new();
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
                ys_bbox.push(Bbox::new(
                    Rect::from_xywh(
                        x.max(0.0f32).min(image_width),
                        y.max(0.0f32).min(image_height),
                        w,
                        h,
                    ),
                    id,
                    conf,
                    None,
                ));
            }
            Ys::non_max_suppression(&mut ys_bbox, self.iou);

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
            let mut y_masks =
                ops::get_masks_from_image(mask_da, 1, 0, Some("Drivable area".to_string()));

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
            let masks = ops::get_masks_from_image(mask_ll, 1, 5, Some("Lane line".to_string()));
            y_masks.extend(masks);
            ys.push(Ys::default().with_bboxes(&ys_bbox).with_masks(&y_masks));
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
