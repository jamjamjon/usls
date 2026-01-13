use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{
    elapsed_module, inputs, Config, DynConf, Engine, Engines, FromConfig, Hbb, Image,
    ImageProcessor, Model, Module, TextProcessor, Xs, X, Y,
};

/// OwlViT v2 model for open-vocabulary object detection.
#[derive(Debug, Builder)]
pub struct OWLv2 {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub names: Vec<String>,
    pub names_with_prompt: Vec<String>,
    pub confs: DynConf,
    pub image_processor: ImageProcessor,
    pub text_processor: TextProcessor,
    pub spec: String,
    pub input_ids: X,
    pub attention_mask: X,
}

impl Model for OWLv2 {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&960.into()).opt(),
            engine.try_width().unwrap_or(&960.into()).opt(),
        );
        let spec = engine.spec().to_owned();
        let names: Vec<String> = config.text_names().to_vec();
        if names.is_empty() {
            anyhow::bail!(
                "No valid class names were provided in the config. Ensure the 'text_names' field is non-empty and contains valid class names."
            );
        }
        let names_with_prompt: Vec<String> =
            names.iter().map(|x| format!("a photo of {x}")).collect();
        let n = names.len();
        let confs = DynConf::new_or_default(config.class_confs(), n);
        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        #[cfg(feature = "vlm")]
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = TextProcessor::default();
        let input_ids: Vec<f32> = text_processor
            .encode_texts_ids(
                &names_with_prompt
                    .iter()
                    .map(|x| x.as_str())
                    .collect::<Vec<_>>(),
                false,
            )?
            .into_iter()
            .flatten()
            .collect();
        let input_ids: X = ndarray::Array2::from_shape_vec((n, input_ids.len() / n), input_ids)?
            .into_dyn()
            .into();
        let attention_mask = X::ones_like(&input_ids);

        let model = Self {
            height,
            width,
            batch,
            spec,
            names,
            names_with_prompt,
            confs,
            image_processor,
            text_processor,
            input_ids,
            attention_mask,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let image_embeddings =
            elapsed_module!("OWLv2", "preprocess", self.image_processor.process(images)?);
        let ys = elapsed_module!(
            "OWLv2",
            "inference",
            engines.run(
                &Module::Model,
                inputs![
                    self.input_ids.view(),
                    image_embeddings,
                    self.attention_mask.view()
                ]?
            )?
        );
        elapsed_module!("OWLv2", "postprocess", self.postprocess(&ys))
    }
}

impl OWLv2 {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let clss_all = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 0"))?;
        let bboxes_all = outputs
            .get::<f32>(1)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 1"))?;
        let ys: Vec<Y> = clss_all
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(bboxes_all.axis_iter(Axis(0)).into_par_iter())
            .enumerate()
            .filter_map(|(idx, (clss, bboxes))| {
                let info = &self.image_processor.images_transform_info[idx];
                let (image_height, image_width) = (info.height_src, info.width_src);

                let ratio = image_height.max(image_width) as f32;
                let y_bboxes: Vec<Hbb> = clss
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, clss_)| {
                        let (class_id, &confidence) = clss_
                            .into_iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))?;

                        let confidence = 1. / ((-confidence).exp() + 1.);
                        if confidence < self.confs[class_id] {
                            return None;
                        }

                        let bbox = bboxes.slice(s![i, ..]).mapv(|x| x * ratio);
                        let (x, y, w, h) = (
                            (bbox[0] - bbox[2] / 2.).max(0.0f32),
                            (bbox[1] - bbox[3] / 2.).max(0.0f32),
                            bbox[2],
                            bbox[3],
                        );

                        Some(
                            Hbb::default()
                                .with_xywh(x, y, w, h)
                                .with_confidence(confidence)
                                .with_id(class_id)
                                .with_name(&self.names[class_id]),
                        )
                    })
                    .collect();

                Some(Y::default().with_hbbs(&y_bboxes))
            })
            .collect();

        Ok(ys)
    }
}
