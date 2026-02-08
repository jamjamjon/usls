use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{
    Config, Engine, Engines, FromConfig, Image, ImageProcessor, Keypoint, Model, Module, Text, Xs,
    Y,
};

/// SLANet: Structure Layout Analysis Network
#[derive(Builder, Debug)]
pub struct SLANet {
    pub processor: ImageProcessor,
    pub td_tokens: Vec<&'static str>,
    pub eos: usize,
    pub sos: usize,
    pub spec: String,
    pub vocab: Vec<String>,
    pub height: usize,
    pub width: usize,
    pub batch: usize,
}

impl Model for SLANet {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let spec = engine.spec().to_owned();

        let err_msg = "You need to specify the image height and image width for visual model.";
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().expect(err_msg).opt(),
            engine.try_width().expect(err_msg).opt(),
        );

        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let vocab = config.inference.class_names;
        tracing::info!("Vacab size: {}", vocab.len());

        let sos = 0;
        let eos = vocab.len() - 1;
        let td_tokens = vec!["<td>", "<td", "<td></td>"];

        let model = Self {
            processor,
            td_tokens,
            eos,
            sos,
            spec,
            vocab,
            height,
            width,
            batch,
        };
        let engines = Engines::from(engine);

        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let xs = crate::perf!("SLANet::preprocess", self.processor.process(images)?);
        let ys = crate::perf!("SLANet::inference", engines.run(&Module::Model, &xs)?);
        crate::perf!("SLANet::postprocess", self.postprocess(&ys))
    }
}

impl SLANet {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let bboxes_all = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 0"))?;
        let structures_all = outputs
            .get::<f32>(1)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 1"))?;

        let ys: Vec<Y> = bboxes_all
            .axis_iter(Axis(0))
            .zip(structures_all.axis_iter(Axis(0)))
            .enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(bid, (bboxes, structures))| {
                let mut y_texts: Vec<&str> = vec!["<html>", "<body>", "<table>"];
                let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
                let info = &self.processor.images_transform_info[bid];

                let (image_height, image_width) = (info.height_src, info.width_src);

                for (i, structure) in structures.axis_iter(Axis(0)).enumerate() {
                    let (token_id, &_confidence) = match structure
                        .into_iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                    {
                        None => continue,
                        Some((id, conf)) => (id, conf),
                    };
                    if token_id == self.eos {
                        break;
                    }
                    if token_id == self.sos {
                        continue;
                    }

                    // token
                    let token = self.vocab[token_id].as_str();

                    // keypoint
                    if self.td_tokens.contains(&token) {
                        let slice_bboxes = bboxes.slice(s![i, ..]);
                        let x14 = slice_bboxes
                            .slice(s![0..;2])
                            .mapv(|x| x * image_width as f32);
                        let y14 = slice_bboxes
                            .slice(s![1..;2])
                            .mapv(|x| x * image_height as f32);
                        y_kpts.push(
                            (0..=3)
                                .map(|i| {
                                    Keypoint::from((x14[i], y14[i]))
                                        .with_id(i)
                                        .with_confidence(1.)
                                })
                                .collect(),
                        );
                    }

                    y_texts.push(token);
                }

                // clean up text
                if y_texts.len() == 3 {
                    y_texts.clear();
                } else {
                    y_texts.extend_from_slice(&["</table>", "</body>", "</html>"]);
                }

                Y::default()
                    .with_keypointss(&y_kpts)
                    .with_texts(&y_texts.into_iter().map(Text::from).collect::<Vec<_>>())
            })
            .collect();

        Ok(ys)
    }
}
