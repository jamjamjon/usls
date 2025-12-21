use anyhow::Result;
use rayon::prelude::*;

use crate::{
    elapsed_module, models::vision::yolo::YOLO, ort_inputs, Config, Engine, FromConfig, Hbb, Image,
    TextProcessor, X, Y,
};

pub struct YOLOE {
    yolo: YOLO,
    textual_encoder: Option<Engine>,
    visual_encoder: Option<Engine>,
    text_processor: TextProcessor,
}

impl TryFrom<Config> for YOLOE {
    type Error = anyhow::Error;

    fn try_from(config: Config) -> Result<Self, Self::Error> {
        Self::new(config)
    }
}

impl YOLOE {
    pub fn new(mut config: Config) -> Result<Self> {
        let textual_encoder = config
            .take_module(&crate::Module::TextualEncoder)
            .ok()
            .map(Engine::from_config)
            .transpose()?;
        let visual_encoder = config
            .take_module(&crate::Module::VisualEncoder)
            .ok()
            .map(Engine::from_config)
            .transpose()?;

        if textual_encoder.is_some() && visual_encoder.is_some() {
            anyhow::bail!("YOLOE supports either visual or textual encoder, not both");
        }

        // Extract text_processor before moving config
        #[cfg(feature = "vlm")]
        let text_processor = crate::TextProcessor::from_config(config.text_processor.clone())?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = crate::TextProcessor::default();

        let yolo = YOLO::new(config)?;

        if let Some(visual_encoder) = &visual_encoder {
            let (height, width) = yolo.dims();
            if visual_encoder.try_height().unwrap_or(&640.into()).opt() != height {
                anyhow::bail!(
                    "Visual encoder height mismatch: {} vs model {}",
                    visual_encoder.try_height().unwrap_or(&640.into()).opt(),
                    height
                );
            }
            if visual_encoder.try_width().unwrap_or(&640.into()).opt() != width {
                anyhow::bail!(
                    "Visual encoder width mismatch: {} vs model {}",
                    visual_encoder.try_width().unwrap_or(&640.into()).opt(),
                    width
                );
            }
        }

        Ok(Self {
            yolo,
            textual_encoder,
            visual_encoder,
            text_processor,
        })
    }

    #[inline]
    pub fn yolo(&self) -> &YOLO {
        &self.yolo
    }

    #[inline]
    pub fn yolo_mut(&mut self) -> &mut YOLO {
        &mut self.yolo
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        self.yolo.forward(xs)
    }

    pub fn encode_class_names(&mut self, class_names: &[&str]) -> Result<X> {
        if class_names.len() > self.yolo.nc() {
            anyhow::bail!(
                "The length of provided class names: {} exceeds the configured number of classes: {}.",
                class_names.len(),
                self.yolo.nc(),
            );
        }

        let names = self.yolo.names_mut();
        names.clear();
        names.extend(class_names.iter().map(|x| x.to_string()));

        let x = elapsed_module!("YOLO", "textual-encoder-preprocess", {
            let texts: Vec<&str> = self.yolo.names().iter().map(|x| x.as_str()).collect();
            let encodings: Vec<f32> = self
                .text_processor
                .encode_texts_ids(&texts, true)?
                .into_iter()
                .flatten()
                .collect();
            let shape = &[texts.len(), encodings.len() / texts.len()];
            X::from_shape_vec(shape, encodings)?
        });

        let xs = elapsed_module!(
            "YOLO",
            "textual-encoder-inference",
            self.textual_encoder
                .as_mut()
                .expect("YOLOE textual encoder not loaded")
                .run([x].as_slice())?
        );

        let result = elapsed_module!("YOLO", "textual-encoder-postprocess", {
            let x = xs.get::<f32>(0)?;
            let (n, dim) = (x.dims()[0], x.dims()[1]);
            let n_pad = self.yolo.nc().saturating_sub(n);
            let x = if n_pad > 0 {
                let x_owned = x.to_owned();
                let x_zeros = X::zeros(&[n_pad, dim]);
                X::cat(&[x_owned, x_zeros], 0)?
            } else {
                x.to_owned()
            };
            self.yolo.embedding = Some(x.clone());
            Ok::<X, anyhow::Error>(x)
        })?;

        Ok(result)
    }

    pub fn encode_visual_prompt(
        &mut self,
        prompt_image: Image,
        hbbs: &[Hbb],
        // _masks: &[Mask],  // TODO
    ) -> Result<X> {
        let (image_embedding, mask) = elapsed_module!("YOLO", "visual-encoder-preprocess", {
            let image_embedding = self
                .yolo
                .processor_mut()
                .process(&[prompt_image])?
                .as_host()?;
            let ratio = self.yolo.processor().images_transform_info()[0].height_scale;

            let (height, width) = self.yolo.dims();

            let downsample_scale = 8.0;
            let scale_factor = ratio / downsample_scale;
            let (prompt_height, prompt_width) = (
                height as f32 / downsample_scale,
                width as f32 / downsample_scale,
            );

            let mask_h = prompt_height as usize;
            let mask_w = prompt_width as usize;
            let mask_w_f32 = mask_w as f32;
            let mask_h_f32 = mask_h as f32;
            let min_size = 1.0;

            use std::collections::HashMap;

            #[allow(clippy::type_complexity)]
            let mut class_groups: HashMap<&str, Vec<(usize, usize, usize, usize)>> =
                HashMap::with_capacity(hbbs.len());

            let names = self.yolo.names_mut();
            names.clear();
            names.reserve(hbbs.len());

            for hbb in hbbs {
                let class_name = hbb.name().unwrap_or("untitled");
                let (x, y, w, h) = hbb.xywh();
                let x1_f = (x * scale_factor).max(0.0).min(mask_w_f32 - 1.0);
                let y1_f = (y * scale_factor).max(0.0).min(mask_h_f32 - 1.0);
                let x2_f = ((x + w) * scale_factor).max(0.0).min(mask_w_f32);
                let y2_f = ((y + h) * scale_factor).max(0.0).min(mask_h_f32);
                let x2_f = x2_f.max(x1_f + min_size);
                let y2_f = y2_f.max(y1_f + min_size);

                let coords = (x1_f as usize, y1_f as usize, x2_f as usize, y2_f as usize);

                class_groups.entry(class_name).or_default().push(coords);
            }

            let nc = class_groups.len();
            self.yolo.set_nc(nc);

            let mask_size = mask_h * mask_w;
            let mut mask_data = vec![0.0f32; nc * mask_size];

            let class_groups_vec: Vec<_> = class_groups.into_iter().collect();
            mask_data
                .par_chunks_mut(mask_size)
                .zip(class_groups_vec.par_iter())
                .for_each(|(mask_slice, (_class_name, boxes))| {
                    for &(x1, y1, x2, y2) in boxes {
                        for row in y1..y2 {
                            let row_start = row * mask_w + x1;
                            let row_end = row * mask_w + x2;
                            mask_slice[row_start..row_end].fill(1.0);
                        }
                    }
                });

            (
                image_embedding,
                X::from_shape_vec(&[1, nc, mask_h, mask_w], mask_data)?,
            )
        });

        elapsed_module!("YOLO", "visual-encoder-inference", {
            let xs = self
                .visual_encoder
                .as_mut()
                .expect("YOLOE visual encoder not loaded")
                .run([image_embedding, mask].as_slice())?;
            Ok(xs.get::<f32>(0)?.to_owned())
        })
    }

    pub fn forward_with_embedding(&mut self, images: &[Image], embedding: &X) -> Result<Vec<Y>> {
        let xs_images = elapsed_module!(
            "YOLO",
            "preprocess",
            self.yolo.processor_mut().process(images)?
        );

        let (preds, protos) = elapsed_module!("YOLO", "inference", {
            let ys = match (&self.visual_encoder, &self.textual_encoder) {
                (Some(_), None) => {
                    let xs_images_x = xs_images.as_host()?;
                    self.yolo
                        .engine_mut()
                        .run(ort_inputs![xs_images_x.view(), embedding.view()]?)?
                }
                (None, Some(_)) => {
                    let xs_images_x = xs_images.as_host()?;
                    let dim = embedding.dims()[1];
                    let embedding = embedding.unsqueeze(0)?.broadcast_to((
                        images.len(),
                        self.yolo.nc(),
                        dim,
                    ))?;
                    self.yolo
                        .engine_mut()
                        .run(ort_inputs![xs_images_x.view(), embedding.view()]?)?
                }
                (Some(_), Some(_)) => {
                    anyhow::bail!("YOLOE supports either visual or textual encoder, not both")
                }
                (None, None) => anyhow::bail!("YOLOE requires either visual or textual encoder"),
            };

            let preds = ys.get::<f32>(0)?.to_owned();
            let protos = ys.try_get::<f32>(1).map(|p| p.to_owned());
            Ok::<_, anyhow::Error>((preds, protos))
        })?;

        elapsed_module!("YOLO", "postprocess", self.yolo.postprocess(preds, protos))
    }

    pub fn batch(&self) -> usize {
        self.yolo.batch()
    }

    pub fn spec(&self) -> &str {
        self.yolo.spec()
    }

    // TOOD: macro to get yolo's methods
}
