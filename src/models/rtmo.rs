use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, Axis, IxDyn};

use crate::{ops, Bbox, DynConf, Keypoint, MinOptMax, Options, OrtEngine, Ys};

#[derive(Debug)]
pub struct RTMO {
    engine: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    confs: DynConf,
    kconfs: DynConf,
}

impl RTMO {
    pub fn new(options: &Options) -> Result<Self> {
        let engine = OrtEngine::new(options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );
        let nc = 1;
        let nk = options.nk.unwrap_or(17);
        let confs = DynConf::new(&options.kconfs, nc);
        let kconfs = DynConf::new(&options.kconfs, nk);
        engine.dry_run()?;

        Ok(Self {
            engine,
            confs,
            kconfs,
            height,
            width,
            batch,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Ys>> {
        let xs_ = ops::letterbox(xs, self.height() as u32, self.width() as u32, 114.0)?;
        let ys = self.engine.run(&[xs_])?;
        let ys = self.postprocess(ys, xs)?;
        Ok(ys)
    }

    pub fn postprocess(&self, xs: Vec<Array<f32, IxDyn>>, xs0: &[DynamicImage]) -> Result<Vec<Ys>> {
        let mut ys: Vec<Ys> = Vec::new();
        let (preds_bboxes, preds_kpts) = if xs[0].ndim() == 3 {
            (&xs[0], &xs[1])
        } else {
            (&xs[1], &xs[0])
        };

        for (idx, (batch_bboxes, batch_kpts)) in preds_bboxes
            .axis_iter(Axis(0))
            .zip(preds_kpts.axis_iter(Axis(0)))
            .enumerate()
        {
            let width_original = xs0[idx].width() as f32;
            let height_original = xs0[idx].height() as f32;
            let ratio =
                (self.width() as f32 / width_original).min(self.height() as f32 / height_original);

            let mut y_bboxes = Vec::new();
            let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
            for (xyxyc, kpts) in batch_bboxes
                .axis_iter(Axis(0))
                .zip(batch_kpts.axis_iter(Axis(0)))
            {
                // bbox
                let x1 = xyxyc[0] / ratio;
                let y1 = xyxyc[1] / ratio;
                let x2 = xyxyc[2] / ratio;
                let y2 = xyxyc[3] / ratio;
                let confidence = xyxyc[4];
                if confidence < self.confs[0] {
                    continue;
                }
                let y_bbox = Bbox::new(
                    (
                        (
                            x1.max(0.0f32).min(width_original),
                            y1.max(0.0f32).min(height_original),
                        ),
                        (x2, y2),
                    )
                        .into(),
                    0,
                    confidence,
                    Some(String::from("Person")),
                );
                y_bboxes.push(y_bbox);

                // keypoints
                let mut kpts_ = Vec::new();
                for (i, kpt) in kpts.axis_iter(Axis(0)).enumerate() {
                    let x = kpt[0] / ratio;
                    let y = kpt[1] / ratio;
                    let c = kpt[2];
                    if c < self.kconfs[i] {
                        kpts_.push(Keypoint::default());
                    } else {
                        kpts_.push(Keypoint::new(
                            (
                                x.max(0.0f32).min(width_original),
                                y.max(0.0f32).min(height_original),
                            )
                                .into(),
                            c,
                        ));
                    }
                }
                y_kpts.push(kpts_);
            }
            ys.push(Ys::default().with_bboxes(&y_bboxes).with_keypoints(&y_kpts));
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
