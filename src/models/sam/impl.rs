use aksr::Builder;
use anyhow::Result;
use rand::{prelude::*, rng};
use std::str::FromStr;

use crate::{
    elapsed_module, Config, DynConf, Engine, Image, Mask, Ops, Polygon, Processor, SamPrompt,
    Tensor, Xs, Y,
};

/// SAM model variants for different use cases.
#[derive(Debug, Clone)]
pub enum SamKind {
    /// Original SAM model
    Sam,
    /// SAM 2.0 with hierarchical architecture
    Sam2,
    /// Mobile optimized SAM
    MobileSam,
    /// High quality SAM with better segmentation
    SamHq,
    /// Efficient SAM with edge-based segmentation
    EdgeSam,
}

impl FromStr for SamKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sam" => Ok(Self::Sam),
            "sam2" => Ok(Self::Sam2),
            "mobilesam" | "mobile-sam" => Ok(Self::MobileSam),
            "samhq" | "sam-hq" => Ok(Self::SamHq),
            "edgesam" | "edge-sam" => Ok(Self::EdgeSam),
            x => anyhow::bail!("Unsupported SamKind: {}", x),
        }
    }
}

/// Segment Anything Model (SAM) for image segmentation.
///
/// A foundation model for generating high-quality object masks from input prompts such as points or boxes.
/// Supports multiple variants including the original SAM, SAM2, MobileSAM, SAM-HQ and EdgeSAM.
#[derive(Builder, Debug)]
pub struct SAM {
    encoder: Engine,
    decoder: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: Processor,
    conf: DynConf,
    find_contours: bool,
    kind: SamKind,
    use_low_res_mask: bool,

    spec: String,
}

impl SAM {
    /// Creates a new SAM model instance from the provided configuration.
    ///
    /// Initializes the model based on the specified SAM variant (original SAM, SAM2, MobileSAM etc.)
    /// and configures its encoder-decoder architecture.
    pub fn new(config: Config) -> Result<Self> {
        let encoder = Engine::try_from_config(&config.encoder)?;
        let decoder = Engine::try_from_config(&config.decoder)?;

        let (batch, height, width) = (
            encoder.batch().opt(),
            encoder.try_height().unwrap_or(&1024.into()).opt(),
            encoder.try_width().unwrap_or(&1024.into()).opt(),
        );

        let spec = encoder.spec().to_owned();

        let conf = DynConf::new_or_default(config.class_confs(), 1);
        let find_contours = config.find_contours;
        let kind = match config.sam_kind {
            Some(x) => x,
            None => anyhow::bail!("Error: no clear `SamKind` specified."),
        };
        let use_low_res_mask = match kind {
            SamKind::Sam | SamKind::MobileSam | SamKind::SamHq => {
                config.sam_low_res_mask.unwrap_or(false)
            }
            SamKind::EdgeSam | SamKind::Sam2 => true,
        };

        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            encoder,
            decoder,
            conf,
            batch,
            height,
            width,
            processor,
            kind,
            find_contours,
            use_low_res_mask,
            spec,
        })
    }

    /// Runs the complete segmentation pipeline on a batch of images.
    ///
    /// The pipeline consists of:
    /// 1. Encoding the images into embeddings
    /// 2. Decoding the embeddings with input prompts to generate segmentation masks
    pub fn forward(&mut self, xs: &[Image], prompts: &[SamPrompt]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("sam", "encode", self.encode(xs)?);
        let ys = elapsed_module!("sam", "decode", self.decode(&ys, prompts)?);

        Ok(ys)
    }

    /// Encodes input images into image embeddings.
    pub fn encode(&mut self, xs: &[Image]) -> Result<Xs> {
        let xs_ = self.processor.process_images(xs)?;
        self.encoder.run(Xs::from(xs_))
    }

    /// Generates segmentation masks from image embeddings and input prompts.
    ///
    /// Takes the image embeddings from the encoder and input prompts (points or boxes)
    /// to generate binary segmentation masks for the prompted objects.
    pub fn decode(&mut self, xs: &Xs, prompts: &[SamPrompt]) -> Result<Vec<Y>> {
        let (image_embeddings, high_res_features_0, high_res_features_1) = match self.kind {
            SamKind::Sam2 => (&xs[0], Some(&xs[1]), Some(&xs[2])),
            _ => (&xs[0], None, None),
        };

        let mut ys: Vec<Y> = Vec::new();
        for (idx, image_embedding) in image_embeddings.iter_dim(0).enumerate() {
            let (image_height, image_width) = (
                self.processor.images_transform_info[idx].height_src,
                self.processor.images_transform_info[idx].width_src,
            );
            let ratio = self.processor.images_transform_info[idx].height_scale;

            let (mut point_coords, mut point_labels) = (
                prompts[idx].point_coords(ratio)?,
                prompts[idx].point_labels()?,
            );

            if point_coords.shape()[0] != 1 {
                let last_idx = point_coords.shape()[0] - 1;
                point_coords = point_coords
                    .slice(&[
                        last_idx..last_idx + 1,
                        0..point_coords.shape()[1],
                        0..point_coords.shape()[2],
                    ])?
                    .to_owned()?
                    .unsqueeze(0)?;
            }
            if point_labels.shape()[0] != 1 {
                let last_idx = point_labels.shape()[0] - 1;
                point_labels = point_labels
                    .slice(&[last_idx..last_idx + 1, 0..point_labels.shape()[1]])?
                    .to_owned()?
                    .unsqueeze(0)?;
            }

            let args = match self.kind {
                SamKind::Sam | SamKind::MobileSam => {
                    vec![
                        image_embedding
                            .to_owned()
                            .unsqueeze(0)?
                            .repeat(0, self.batch)?, // image_embedding
                        point_coords,
                        point_labels,
                        Tensor::zeros(vec![
                            1,
                            1,
                            self.height_low_res() as _,
                            self.width_low_res() as _,
                        ]), // mask_input
                        Tensor::zeros(vec![1]), // has_mask_input
                        Tensor::from(vec![image_height as f32, image_width as f32]), // orig_im_size
                    ]
                }
                SamKind::SamHq => {
                    vec![
                        image_embedding
                            .to_owned()
                            .unsqueeze(0)?
                            .repeat(0, self.batch)?, // image_embedding
                        xs[1]
                            .slice(&[
                                idx..idx + 1,
                                0..xs[1].shape()[1],
                                0..xs[1].shape()[2],
                                0..xs[1].shape()[3],
                            ])?
                            .to_owned()?
                            .unsqueeze(0)?
                            .repeat(0, self.batch)?, // intern_embedding
                        point_coords,
                        point_labels,
                        Tensor::zeros(vec![
                            1,
                            1,
                            self.height_low_res() as _,
                            self.width_low_res() as _,
                        ]), // mask_input
                        Tensor::zeros(vec![1]), // has_mask_input
                        Tensor::from(vec![image_height as f32, image_width as f32]), // orig_im_size
                    ]
                }
                SamKind::EdgeSam => {
                    vec![
                        image_embedding
                            .to_owned()
                            .unsqueeze(0)?
                            .repeat(0, self.batch)?,
                        point_coords,
                        point_labels,
                    ]
                }
                SamKind::Sam2 => {
                    vec![
                        image_embedding
                            .to_owned()
                            .unsqueeze(0)?
                            .repeat(0, self.batch)?,
                        high_res_features_0
                            .unwrap()
                            .slice(&[
                                idx..idx + 1,
                                0..high_res_features_0.as_ref().unwrap().shape()[1],
                                0..high_res_features_0.as_ref().unwrap().shape()[2],
                                0..high_res_features_0.as_ref().unwrap().shape()[3],
                            ])?
                            .to_owned()?
                            .repeat(0, self.batch)?,
                        high_res_features_1
                            .unwrap()
                            .slice(&[
                                idx..idx + 1,
                                0..high_res_features_1.as_ref().unwrap().shape()[1],
                                0..high_res_features_1.as_ref().unwrap().shape()[2],
                                0..high_res_features_1.as_ref().unwrap().shape()[3],
                            ])?
                            .to_owned()?
                            .repeat(0, self.batch)?,
                        point_coords,
                        point_labels,
                        Tensor::zeros(vec![
                            1,
                            1,
                            self.height_low_res() as _,
                            self.width_low_res() as _,
                        ]), // mask_input
                        Tensor::zeros(vec![1]),
                        Tensor::from(vec![image_height as f32, image_width as f32]),
                    ]
                }
            };

            let ys_ = self.decoder.run(Xs::from(args))?;

            let mut y_masks: Vec<Mask> = Vec::new();
            let mut y_polygons: Vec<Polygon> = Vec::new();

            // masks & confs
            let (masks, confs) = match self.kind {
                SamKind::Sam | SamKind::MobileSam | SamKind::SamHq => {
                    if !self.use_low_res_mask {
                        (&ys_[0], &ys_[1])
                        // (&ys_["masks"], &ys_["iou_predictions"])
                    } else {
                        (&ys_[2], &ys_[1])
                        // (&ys_["low_res_masks"], &ys_["iou_predictions"])
                    }
                }
                SamKind::Sam2 => (&ys_[0], &ys_[1]),
                SamKind::EdgeSam => (&ys_["masks"], &ys_["scores"]),
            };

            for (mask, iou) in masks.iter_dim(0).zip(confs.iter_dim(0)) {
                let iou_vec = iou.to_vec::<f32>().unwrap();
                let (i, conf) =
                    match iou_vec.iter().enumerate().max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        Some((i, c)) => (i, c),
                        None => continue,
                    };

                if *conf < self.conf[0] {
                    continue;
                }
                let mask = mask.slice(&[i..i + 1, 0..mask.shape()[1], 0..mask.shape()[2]])?;

                let shape = mask.shape();
                let (_, h, w) = (shape[0], shape[1], shape[2]);
                let luma = if self.use_low_res_mask {
                    Ops::resize_lumaf32_u8(
                        &mask.to_vec::<f32>()?,
                        w as _,
                        h as _,
                        image_width as _,
                        image_height as _,
                        true,
                        "Bilinear",
                    )?
                } else {
                    mask.to_vec::<f32>()?
                        .into_iter()
                        .map(|x| if x > 0.0 { 255u8 } else { 0u8 })
                        .collect::<Vec<u8>>()
                };

                // contours
                let mut rng = rng();
                let id = rng.random_range(0..20);
                let mask = Mask::new(&luma, image_width, image_height)?.with_id(id);
                if self.find_contours {
                    for polygon in mask.polygons().into_iter() {
                        y_polygons.push(polygon.with_confidence(iou[0]).with_id(id));
                    }
                }
                y_masks.push(mask);
            }

            let mut y = Y::default();
            if !y_masks.is_empty() {
                y = y.with_masks(&y_masks);
            }
            if !y_polygons.is_empty() {
                y = y.with_polygons(&y_polygons);
            }

            ys.push(y);
        }

        Ok(ys)
    }

    /// Returns the width of the low-resolution feature maps.
    pub fn width_low_res(&self) -> usize {
        self.width / 4
    }

    /// Returns the height of the low-resolution feature maps.  
    pub fn height_low_res(&self) -> usize {
        self.height / 4
    }
}
