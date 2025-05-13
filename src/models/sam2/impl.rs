use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};

use crate::{
    elapsed, DynConf, Engine, Image, Mask, Ops, Options, Processor, SamPrompt, Ts, Xs, X, Y,
};

#[derive(Builder, Debug)]
pub struct SAM2 {
    encoder: Engine,
    decoder: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: Processor,
    conf: DynConf,
    ts: Ts,
    spec: String,
}

impl SAM2 {
    pub fn new(options_encoder: Options, options_decoder: Options) -> Result<Self> {
        let encoder = options_encoder.to_engine()?;
        let decoder = options_decoder.to_engine()?;
        let (batch, height, width) = (
            encoder.batch().opt(),
            encoder.try_height().unwrap_or(&1024.into()).opt(),
            encoder.try_width().unwrap_or(&1024.into()).opt(),
        );
        let ts = Ts::merge(&[encoder.ts(), decoder.ts()]);
        let spec = encoder.spec().to_owned();
        let processor = options_encoder
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let conf = DynConf::new(options_encoder.class_confs(), 1);

        Ok(Self {
            encoder,
            decoder,
            conf,
            batch,
            height,
            width,
            ts,
            processor,
            spec,
        })
    }

    pub fn forward(&mut self, xs: &[Image], prompts: &[SamPrompt]) -> Result<Vec<Y>> {
        let ys = elapsed!("encode", self.ts, { self.encode(xs)? });
        let ys = elapsed!("decode", self.ts, { self.decode(&ys, prompts)? });

        Ok(ys)
    }

    pub fn encode(&mut self, xs: &[Image]) -> Result<Xs> {
        let xs_ = self.processor.process_images(xs)?;
        self.encoder.run(Xs::from(xs_))
    }

    pub fn decode(&mut self, xs: &Xs, prompts: &[SamPrompt]) -> Result<Vec<Y>> {
        let (image_embeddings, high_res_features_0, high_res_features_1) = (&xs[0], &xs[1], &xs[2]);

        let mut ys: Vec<Y> = Vec::new();
        for (idx, image_embedding) in image_embeddings.axis_iter(Axis(0)).enumerate() {
            let (image_height, image_width) = (
                self.processor.images_transform_info[idx].height_src,
                self.processor.images_transform_info[idx].width_src,
            );
            let ratio = self.processor.images_transform_info[idx].height_scale;

            let ys_ = self.decoder.run(Xs::from(vec![
                X::from(image_embedding.into_dyn().into_owned())
                    .insert_axis(0)?
                    .repeat(0, self.batch)?,
                X::from(
                    high_res_features_0
                        .slice(s![idx, .., .., ..])
                        .into_dyn()
                        .into_owned(),
                )
                .insert_axis(0)?
                .repeat(0, self.batch)?,
                X::from(
                    high_res_features_1
                        .slice(s![idx, .., .., ..])
                        .into_dyn()
                        .into_owned(),
                )
                .insert_axis(0)?
                .repeat(0, self.batch)?,
                prompts[idx].point_coords(ratio)?,
                prompts[idx].point_labels()?,
                // TODO
                X::zeros(&[1, 1, self.height_low_res() as _, self.width_low_res() as _]),
                X::zeros(&[1]),
                X::from(vec![self.width as _, self.height as _]),
            ]))?;

            let mut y_masks: Vec<Mask> = Vec::new();

            // masks & confs
            let (masks, confs) = (&ys_[0], &ys_[1]);

            for (id, (mask, iou)) in masks
                .axis_iter(Axis(0))
                .zip(confs.axis_iter(Axis(0)))
                .enumerate()
            {
                let (i, conf) = match iou
                    .to_owned()
                    .into_raw_vec_and_offset()
                    .0
                    .into_iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                {
                    Some((i, c)) => (i, c),
                    None => continue,
                };

                if conf < self.conf[0] {
                    continue;
                }
                let mask = mask.slice(s![i, .., ..]);

                let (h, w) = mask.dim();
                let luma = Ops::resize_lumaf32_u8(
                    &mask.into_owned().into_raw_vec_and_offset().0,
                    w as _,
                    h as _,
                    image_width as _,
                    image_height as _,
                    true,
                    "Bilinear",
                )?;

                // contours
                let mask = Mask::new(&luma, image_width, image_height)?.with_id(id);
                y_masks.push(mask);
            }

            let mut y = Y::default();
            if !y_masks.is_empty() {
                y = y.with_masks(&y_masks);
            }

            ys.push(y);
        }

        Ok(ys)
    }

    pub fn width_low_res(&self) -> usize {
        self.width / 4
    }

    pub fn height_low_res(&self) -> usize {
        self.height / 4
    }
}
