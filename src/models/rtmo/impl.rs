use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;

use crate::{elapsed, DynConf, Engine, Hbb, Image, Keypoint, Options, Processor, Ts, Xs, Y};

#[derive(Builder, Debug)]
pub struct RTMO {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    ts: Ts,
    spec: String,
    processor: Processor,
    confs: DynConf,
    kconfs: DynConf,
}

impl RTMO {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let spec = engine.spec().to_string();
        let (batch, height, width, ts) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&512.into()).opt(),
            engine.try_width().unwrap_or(&512.into()).opt(),
            engine.ts().clone(),
        );
        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let nk = options.nk().unwrap_or(17);
        let confs = DynConf::new(options.class_confs(), 1);
        let kconfs = DynConf::new(options.keypoint_confs(), nk);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            ts,
            spec,
            processor,
            confs,
            kconfs,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        Ok(self.processor.process_images(xs)?.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed!("preprocess", self.ts, { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });

        Ok(ys)
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }

    fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        // let (preds_bboxes, preds_kpts) = (&xs["dets"], &xs["keypoints"]);
        let (preds_bboxes, preds_kpts) = (&xs[0], &xs[1]);

        for (idx, (batch_bboxes, batch_kpts)) in preds_bboxes
            .axis_iter(Axis(0))
            .zip(preds_kpts.axis_iter(Axis(0)))
            .enumerate()
        {
            let (height_original, width_original) = (
                self.processor.images_transform_info[idx].height_src,
                self.processor.images_transform_info[idx].width_src,
            );
            let ratio = self.processor.images_transform_info[idx].height_scale;

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
                y_bboxes.push(
                    Hbb::default()
                        .with_xyxy(
                            x1.max(0.0f32).min(width_original as _),
                            y1.max(0.0f32).min(height_original as _),
                            x2,
                            y2,
                        )
                        .with_confidence(confidence)
                        .with_id(0)
                        .with_name("Person"),
                );

                // keypoints
                let mut kpts_ = Vec::new();
                for (i, kpt) in kpts.axis_iter(Axis(0)).enumerate() {
                    let x = kpt[0] / ratio;
                    let y = kpt[1] / ratio;
                    let c = kpt[2];
                    if c < self.kconfs[i] {
                        kpts_.push(Keypoint::default());
                    } else {
                        kpts_.push(Keypoint::default().with_id(i).with_confidence(c).with_xy(
                            x.max(0.0f32).min(width_original as _),
                            y.max(0.0f32).min(height_original as _),
                        ));
                    }
                }
                y_kpts.push(kpts_);
            }
            ys.push(Y::default().with_hbbs(&y_bboxes).with_keypointss(&y_kpts));
        }

        Ok(ys)
    }
}
