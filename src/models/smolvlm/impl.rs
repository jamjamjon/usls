use aksr::Builder;
use anyhow::Result;
use image::GenericImageView;
use ndarray::s;

use crate::{Config, Engine, Image, LogitsSampler, Processor, Scale, Xs, X, Y};

#[derive(Debug, Builder)]
pub struct SmolVLM {
    vision: Engine,
    text_embed: Engine,
    decoder: Engine,
    scale: Scale,
    image_token: String,
    global_img_token: String,
    fake_image_token: String,
    bos_token: String,
    eos_token: String,
    eos_token_id: u32,
    image_token_id: u32,
    max_length: usize,
    image_seq_len: usize,
    num_hidden_layers: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    num_patch: usize,
    batch: usize,
    width: usize,
    height: usize,
    processor: Processor,
}

impl SmolVLM {
    pub fn new(config: Config) -> Result<Self> {
        let vision = Engine::try_from_config(&config.visual)?;
        let text_embed = Engine::try_from_config(&config.textual)?;
        let decoder = Engine::try_from_config(&config.textual_decoder_merged)?;
        let fake_image_token = "<fake_token_around_image>".to_string();
        let image_token = "<image>".to_string();
        let global_img_token = "<global-img>".to_string();
        let bos_token = "<|im_start|>".to_string();
        let eos_token = "<end_of_utterance>".to_string();
        let eos_token_id = 2;
        let image_token_id = 49190;
        let image_seq_len = 64;
        let max_length = 1024;
        let (num_hidden_layers, head_dim, num_key_value_heads) = match &config.scale {
            Some(Scale::Million(256.)) => (30, 64, 3),
            Some(Scale::Million(500.)) => (32, 64, 5),
            _ => unimplemented!(),
        };
        let scale = config
            .scale
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Scale configuration is required for SmolVLM model"))?;
        let (batch, num_patch, height, width) = (
            vision.batch().opt(),
            vision.inputs_minoptmax()[0][1].opt(),
            vision.inputs_minoptmax()[0][3].opt(),
            vision.inputs_minoptmax()[0][4].opt(),
        );
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            vision,
            text_embed,
            decoder,
            scale,
            max_length,
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
            processor,
        })
    }

    pub fn forward(&mut self, images: &[Image], text: &str) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for image in images.iter() {
            let y = self.generate_one(image, text)?;
            ys.push(Y::default().with_texts(&[y.into()]));
        }

        Ok(ys)
    }

    fn generate_one(&mut self, image: &Image, text: &str) -> Result<String> {
        let bs = 1; // TODO

        // patches and pixel_attention_mask
        let (patches, nw_nh) = self.process_one(image)?;
        let dims = patches.dims();
        let pixel_attention_mask = X::ones(&[dims[0], dims[1], dims[3], dims[4]]);

        // input ids
        let prompt = self.image_prompt_string(nw_nh, text);
        let mut input_ids: Vec<f32> = self.processor.encode_text_ids(&prompt, true)?;

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
            let mut inputs_embeds = self.text_embed.run(input_ids_x.clone().into())?[0].clone();

            // encode image and merge
            if ii == 0 {
                let image_features = self.vision.run(Xs::from(vec![
                    patches.clone(),
                    pixel_attention_mask.clone(),
                ]))?[0]
                    .clone();
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
            let decoder_outputs = self.decoder.run(xs.into())?;
            let logits = &decoder_outputs[0];
            past_key_values = (1..decoder_outputs.len())
                .step_by(2)
                .flat_map(|i| [i, i + 1])
                .map(|i| decoder_outputs[i].clone())
                .collect();
            let token_id = logits_sampler.decode(
                &logits
                    .slice(s![0, -1, ..])
                    .into_owned()
                    .into_raw_vec_and_offset()
                    .0,
            )?;

            // early return
            if token_id == self.eos_token_id {
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
        let text = self.processor.decode_tokens(&token_ids, true)?;

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
            (1, 1) => format!("{}{}", s1, s_global),
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
                format!("{}\n{}", s, s_global)
            }
        }
    }

    fn create_patches(image: &Image, patch_size: (u32, u32)) -> (Vec<Image>, (u32, u32)) {
        let mut patches = vec![];
        let image_rgb8 = image.to_rgb8();
        let (image_width, image_height) = image_rgb8.dimensions();
        let (patch_width, patch_height) = patch_size;

        let (nw, nh) = if image_width > patch_width || image_height > patch_height {
            // calculate the number of splits
            let nw = image_width.div_ceil(patch_width);
            let nh = image_height.div_ceil(patch_height);

            // calculate the optimal width and height for the sub-images
            let optimal_height = image_height.div_ceil(nh);
            let optimal_width = image_width.div_ceil(nw);

            // SubImage
            for r in 0..nh {
                for c in 0..nw {
                    let x0 = c * optimal_width;
                    let y0 = r * optimal_height;
                    let x1 = (x0 + optimal_width).min(image_width);
                    let y1 = (y0 + optimal_height).min(image_height);
                    let sub_image = image_rgb8.view(x0, y0, x1 - x0, y1 - y0).to_image();
                    patches.push(Image::from(sub_image));
                }
            }
            (nw, nh)
        } else {
            (1, 1)
        };
        patches.push(image.clone());

        (patches, (nw, nh))
    }

    pub fn process_one(&mut self, x: &Image) -> Result<(X, (u32, u32))> {
        let (patches, nw_nh) = Self::create_patches(x, (self.width as _, self.height as _));
        let patches = self.processor.process_images(&patches)?.insert_axis(0)?;

        Ok((patches, (nw_nh)))
    }
}
