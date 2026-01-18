use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};

use crate::{
    elapsed_module, inputs, Config, Engine, Engines, FromConfig, Image, ImageProcessor,
    LogitsSampler, Model, Module, TextProcessor, X, Y,
};

/// BLIP: Bootstrapping Language-Image Pre-training
#[derive(Debug, Builder)]
pub struct Blip {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub image_processor: ImageProcessor,
    pub text_processor: TextProcessor,
    pub max_length: usize,
    pub eos_token_id: u32,
    pub spec: String,
}

impl Model for Blip {
    type Input<'a> = (&'a [Image], Option<&'a str>);

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let visual = Engine::from_config(config.take_module(&Module::Visual)?)?;
        let textual = Engine::from_config(config.take_module(&Module::Textual)?)?;
        let (batch, height, width) = (
            visual.batch().opt(),
            visual.try_height().unwrap_or(&384.into()).opt(),
            visual.try_width().unwrap_or(&384.into()).opt(),
        );
        let spec = visual.spec().to_owned();

        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        #[cfg(feature = "vlm")]
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = TextProcessor::default();
        let max_length = 512;
        let eos_token_id = 102;

        let model = Self {
            max_length,
            eos_token_id,
            batch,
            height,
            width,
            image_processor,
            text_processor,
            spec,
        };

        let mut engines = Engines::default();
        engines.insert(Module::Visual, visual);
        engines.insert(Module::Textual, textual);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, (images, text): Self::Input<'_>) -> Result<Vec<Y>> {
        let image_embeds = elapsed_module!(
            "BLIP",
            "encode_images",
            self.encode_images_internal(engines, images)?
        );
        elapsed_module!(
            "BLIP",
            "generate",
            self.generate_internal(engines, &image_embeds, text)
        )
    }
}

impl Blip {
    fn encode_images_internal(&mut self, engines: &mut Engines, xs: &[Image]) -> Result<X> {
        let ys = self.image_processor.process(xs)?;
        self.batch = xs.len();
        let output = engines.run(&Module::Visual, inputs![&ys]?)?;

        let x = output
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get vision output"))?;
        Ok(X::from(x))
    }

    fn encode_texts_internal(&self, text: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let input_ids = self
            .text_processor
            .encode_text_ids(text.unwrap_or_default(), false)?;
        Ok(vec![input_ids.clone(); self.batch()])
    }

    fn generate_internal(
        &mut self,
        engines: &mut Engines,
        image_embeds: &X,
        text: Option<&str>,
    ) -> Result<Vec<Y>> {
        // encode texts
        let mut token_ids = self.encode_texts_internal(text)?;

        // generate
        let logits_sampler = LogitsSampler::new();
        let mut finished = vec![false; self.batch()];
        for _ in 0..self.max_length {
            let input_ids_nd = token_ids
                .iter()
                .map(|tokens| X::from(tokens.clone()).insert_axis(0))
                .collect::<Result<Vec<_>, _>>()?;

            let input_ids_nd = X::concat(&input_ids_nd, 0)?;
            let input_ids_attn_mask = X::ones(input_ids_nd.dims());

            // decode
            let logits = {
                let image_attn_mask = X::ones(&[self.batch(), image_embeds.dims()[1]]);
                let outputs = engines.run(
                    &Module::Textual,
                    inputs![
                        input_ids_nd,
                        input_ids_attn_mask,
                        image_embeds.clone(),
                        image_attn_mask
                    ]?,
                )?;
                X::from(
                    outputs
                        .get::<f32>(0)
                        .ok_or_else(|| anyhow::anyhow!("Failed to get decoder output"))?,
                )
            };

            // decode each token for each batch
            for (i, logit) in logits.axis_iter(Axis(0)).enumerate() {
                if !finished[i] {
                    let token_id = logits_sampler.decode(
                        &logit
                            .slice(s![-1, ..])
                            .into_owned()
                            .into_raw_vec_and_offset()
                            .0,
                    )?;
                    if token_id == self.eos_token_id {
                        finished[i] = true;
                    }
                    token_ids[i].push(token_id as f32);
                } else {
                    token_ids[i].push(self.eos_token_id as f32);
                }
            }

            if finished.iter().all(|&x| x) {
                break;
            }
        }

        // batch decode
        let texts = self.text_processor.decode_tokens_batch(
            &token_ids
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as u32).collect::<Vec<_>>())
                .collect::<Vec<Vec<_>>>(),
            true,
        )?;

        let ys = texts
            .into_iter()
            .map(|x| Y::default().with_texts(&[x.into()]))
            .collect::<Vec<_>>();

        Ok(ys)
    }
}
