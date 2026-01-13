use aksr::Builder;
use anyhow::Result;
use ndarray::s;

use crate::{
    inputs, AnyResStrategy, Config, Engine, Engines, FromConfig, Image, ImageProcessor,
    LogitsSampler, Model, Module, Scale, TextProcessor, X, Y,
};

/// SmolVLM - A compact Vision-Language Model.
#[derive(Debug, Builder)]
pub struct SmolVLM {
    scale: Scale,
    image_token: String,
    global_img_token: String,
    fake_image_token: String,
    bos_token: String,
    eos_token: String,
    eos_token_id: u32,
    image_token_id: u32,
    max_length: u64,
    ignore_eos: bool,
    image_seq_len: usize,
    num_hidden_layers: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    num_patch: usize,
    batch: usize,
    width: usize,
    height: usize,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
}

impl SmolVLM {
    fn generate_one(&mut self, engines: &mut Engines, image: &Image, text: &str) -> Result<String> {
        let bs = 1; // TODO

        // patches and pixel_attention_mask
        let (patches, nw_nh) = self.process_one(image)?;
        let dims = patches.dims();
        let pixel_attention_mask = X::ones(&[dims[0], dims[1], dims[3], dims[4]]);

        // input ids
        let prompt = self.image_prompt_string(nw_nh, text);
        let mut input_ids: Vec<f32> = self.text_processor.encode_text_ids(&prompt, true)?;

        // position ids
        let mut position_ids = X::from(
            (1..input_ids.len() + 1)
                .map(|x| x as f32)
                .collect::<Vec<f32>>(),
        )
        .insert_axis(0)?;

        // past key_values
        let mut past_key_values = vec![
            X::zeros(&[bs, self.num_key_value_heads, 0, self.head_dim]);
            self.num_hidden_layers * 2
        ];

        // generate
        let logits_sampler = LogitsSampler::new();
        let mut token_ids: Vec<u32> = vec![];
        for ii in 0..self.max_length {
            // inputs embeds
            let input_ids_x = X::from(input_ids.clone()).insert_axis(0)?;
            let mut inputs_embeds = {
                let ys = engines.run(&Module::Textual, inputs![&input_ids_x]?)?;
                X::from(
                    ys.get::<f32>(0)
                        .ok_or_else(|| anyhow::anyhow!("Failed to get text embeddings"))?,
                )
            };

            // encode image and merge
            if ii == 0 {
                let image_features = {
                    let ys = engines.run(
                        &Module::Visual,
                        inputs![patches.clone(), pixel_attention_mask.clone()]?,
                    )?;
                    X::from(
                        ys.get::<f32>(0)
                            .ok_or_else(|| anyhow::anyhow!("Failed to get image features"))?,
                    )
                };
                let dims = image_features.dim();
                let image_features = image_features.to_shape((dims[0] * dims[1], dims[2]))?;

                // merge
                let mut r = 0;
                for (i, &token_id) in input_ids_x.indexed_iter() {
                    if token_id == self.image_token_id as f32 {
                        inputs_embeds
                            .0
                            .slice_mut(s![0, i[1], ..])
                            .assign(&image_features.slice(s![r, ..]));
                        r += 1;
                    }
                }
            }

            // inputs
            let mut xs = vec![
                inputs_embeds.clone(),
                X::ones_like(&input_ids_x),
                position_ids.clone(),
            ];
            for i in 0..self.num_hidden_layers {
                xs.push(past_key_values[i * 2].clone());
                xs.push(past_key_values[i * 2 + 1].clone());
            }

            // decode
            let (logits, new_past_key_values) = {
                let decoder_outputs = engines.run(&Module::TextualDecoderMerged, &xs)?;
                let logits = X::from(
                    decoder_outputs
                        .get::<f32>(0)
                        .ok_or_else(|| anyhow::anyhow!("Failed to get logits"))?,
                );
                let new_past_key_values: Vec<X> = (1..decoder_outputs.len())
                    .step_by(2)
                    .flat_map(|i| [i, i + 1])
                    .map(|i| X::from(decoder_outputs.get::<f32>(i).unwrap()))
                    .collect();
                (logits, new_past_key_values)
            };
            past_key_values = new_past_key_values;
            let token_id = logits_sampler.decode(
                &logits
                    .slice(s![0, -1, ..])
                    .into_owned()
                    .into_raw_vec_and_offset()
                    .0,
            )?;

            // early return
            if !self.ignore_eos && token_id == self.eos_token_id {
                break;
            } else {
                token_ids.push(token_id);
            }

            // update
            input_ids = vec![token_id as f32];
            position_ids = X::from(
                position_ids
                    .slice(s![.., -1..])
                    .mapv(|x| x + 1.0)
                    .into_owned()
                    .into_dyn(),
            );
        }

        // decode tokens
        let text = self.text_processor.decode_tokens(&token_ids, true)?;

        Ok(text)
    }

    fn image_prompt_string(&self, nw_nh: (u32, u32), text: &str) -> String {
        let (nw, nh) = nw_nh;
        let image_tokens = self.image_token.repeat(self.image_seq_len);
        let s1 = format!("{}User:", self.bos_token);
        let s_global = format!(
            "{}{}{}{}{}{}\nAssistant:",
            self.fake_image_token,
            self.global_img_token,
            image_tokens,
            self.fake_image_token,
            text,
            self.eos_token
        );

        match nw_nh {
            (1, 1) => format!("{s1}{s_global}"),
            _ => {
                let mut s = String::with_capacity(
                    s1.len()
                        + (nw as usize
                            * nh as usize
                            * (self.fake_image_token.len() + image_tokens.len() + 20))
                        + 10,
                );
                s.push_str(&s1);
                // let mut s = s1;
                for r in 1..=nh {
                    for c in 1..=nw {
                        s.push_str(&format!(
                            "{}<row_{}_col_{}>{}",
                            self.fake_image_token, r, c, image_tokens
                        ));
                    }
                    s.push('\n');
                }
                format!("{s}\n{s_global}")
            }
        }
    }

    pub fn process_one(&mut self, x: &Image) -> Result<(X, (u32, u32))> {
        // ImageProcessor handles AnyRes transform internally
        // It splits the image into patches and resizes each patch
        let (patches, infos) = self.image_processor.process_with_info(&[x.clone()])?;
        let patches = patches.as_host()?.insert_axis(0)?;

        // Extract grid size from transform info
        let nw_nh = if let Some(info) = infos.first() {
            // Calculate grid size from original image and patch size
            let src_w = info.width_src();
            let src_h = info.height_src();
            let patch_w = self.width as u32;
            let patch_h = self.height as u32;
            if src_w > patch_w || src_h > patch_h {
                (src_w.div_ceil(patch_w), src_h.div_ceil(patch_h))
            } else {
                (1, 1)
            }
        } else {
            (1, 1)
        };

        Ok((patches, nw_nh))
    }
}

/// Implement the Model trait for SmolVLM.
impl Model for SmolVLM {
    type Input<'a> = (&'a [Image], &'a str);

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        "smolvlm"
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let vision = Engine::from_config(config.take_module(&Module::Visual)?)?;
        let text_embed = Engine::from_config(config.take_module(&Module::Textual)?)?;
        let decoder = Engine::from_config(config.take_module(&Module::TextualDecoderMerged)?)?;
        // TODO: fetch from file
        let fake_image_token = "<fake_token_around_image>".to_string();
        let image_token = "<image>".to_string();
        let global_img_token = "<global-img>".to_string();
        let bos_token = "<|im_start|>".to_string(); // id: 0
        let eos_token = "<end_of_utterance>".to_string();
        let eos_token_id = 49279;
        let image_token_id = 49190;
        let image_seq_len = 64;
        let max_length = config.inference.max_tokens.unwrap_or(1024);
        let ignore_eos = config.inference.ignore_eos;
        let scale = config
            .scale
            .take()
            .ok_or_else(|| anyhow::anyhow!("Scale configuration is required for SmolVLM model"))?;
        let (num_hidden_layers, head_dim, num_key_value_heads) = match &scale {
            Scale::Million(256.) => (30, 64, 3),
            Scale::Million(500.) => (32, 64, 5),
            _ => unimplemented!(),
        };
        let (batch, num_patch, height, width) = (
            vision.batch().opt(),
            vision.inputs.minoptmax[0][1].opt(),
            vision.inputs.minoptmax[0][3].opt(),
            vision.inputs.minoptmax[0][4].opt(),
        );
        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _)
            .with_dynres_strategy(Some(AnyResStrategy::SmolVLM {
                patch_width: width as u32,
                patch_height: height as u32,
                include_global: true,
            }));
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        // println!("text_processor: {:#?}", text_processor.tokenizer.as_ref().unwrap().get_added_tokens_decoder());

        let model = Self {
            scale,
            max_length,
            ignore_eos,
            eos_token_id,
            image_token,
            image_token_id,
            global_img_token,
            fake_image_token,
            num_hidden_layers,
            head_dim,
            num_key_value_heads,
            bos_token,
            eos_token,
            image_seq_len,
            batch,
            num_patch,
            height,
            width,
            image_processor,
            text_processor,
        };

        // Build engines collection
        let mut engines = Engines::new();
        engines.insert(Module::Visual, vision);
        engines.insert(Module::Textual, text_embed);
        engines.insert(Module::TextualDecoderMerged, decoder);

        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, (images, text): Self::Input<'_>) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for image in images.iter() {
            let y = self.generate_one(engines, image, text)?;
            ys.push(Y::default().with_texts(&[y.into()]));
        }
        Ok(ys)
    }
}
