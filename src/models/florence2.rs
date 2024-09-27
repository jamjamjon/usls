use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Axis};
use rayon::prelude::*;
use std::collections::BTreeMap;
use tokenizers::Tokenizer;

use crate::{
    build_progress_bar, Bbox, LogitsSampler, MinOptMax, Ops, Options, OrtEngine, Polygon,
    Quantizer, Task, Xs, X, Y,
};

#[derive(Debug)]
pub struct Florence2 {
    pub vision_encoder: OrtEngine,
    pub text_embed: OrtEngine,
    pub encoder: OrtEngine,
    pub decoder: OrtEngine,
    pub decoder_merged: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    tokenizer: Tokenizer,
    max_length: usize,
    quantizer: Quantizer,
}

impl Florence2 {
    pub fn new(
        options_vision_encoder: Options,
        options_text_embed: Options,
        options_encoder: Options,
        options_decoder: Options,
        options_decoder_merged: Options,
    ) -> Result<Self> {
        let mut vision_encoder = OrtEngine::new(&options_vision_encoder)?;
        let mut text_embed = OrtEngine::new(&options_text_embed)?;
        let mut encoder = OrtEngine::new(&options_encoder)?;
        let mut decoder = OrtEngine::new(&options_decoder)?;
        let mut decoder_merged = OrtEngine::new(&options_decoder_merged)?;
        let (batch, height, width) = (
            vision_encoder.batch().to_owned(),
            vision_encoder.height().to_owned(),
            vision_encoder.width().to_owned(),
        );
        let tokenizer = options_text_embed
            .tokenizer
            .ok_or(anyhow::anyhow!("No tokenizer file found"))?;
        let tokenizer = match Tokenizer::from_file(tokenizer) {
            Err(err) => anyhow::bail!("Failed to build tokenizer: {:?}", err),
            Ok(x) => x,
        };

        let quantizer = Quantizer::default();
        let max_length = 1024;

        // dry run
        vision_encoder.dry_run()?;
        text_embed.dry_run()?;
        encoder.dry_run()?;
        decoder.dry_run()?;
        decoder_merged.dry_run()?;

        Ok(Self {
            vision_encoder,
            text_embed,
            encoder,
            decoder,
            decoder_merged,
            height,
            width,
            batch,
            tokenizer,
            max_length,
            quantizer,
        })
    }

    pub fn encode_images(&mut self, xs: &[DynamicImage]) -> Result<X> {
        let xs_ = X::apply(&[
            Ops::Resize(
                xs,
                self.height.opt() as u32,
                self.width.opt() as u32,
                "Bilinear",
            ),
            Ops::Normalize(0., 255.),
            Ops::Standardize(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225], 3),
            Ops::Nhwc2nchw,
        ])?;
        let ys = self.vision_encoder.run(Xs::from(xs_))?[0].to_owned();
        Ok(ys)
    }

    pub fn run_with_tasks(
        &mut self,
        xs: &[DynamicImage],
        tasks: &[Task],
    ) -> Result<BTreeMap<Task, Vec<Y>>> {
        let mut ys: BTreeMap<Task, Vec<Y>> = BTreeMap::new();

        // encode images
        let image_embeddings = self.encode_images(xs)?;

        // note: the length of xs is not always equal to batch size
        self.batch.update_opt(xs.len() as _);

        // build pb
        let pb = build_progress_bar(
            tasks.len() as u64,
            "  Working On",
            None,
            crate::PROGRESS_BAR_STYLE_CYAN_2,
        )?;

        // tasks
        for task in tasks.iter() {
            pb.inc(1);
            pb.set_message(format!("{:?}", task));

            // construct prompt and encode
            let input_ids = self
                .encode_prompt(task)?
                .insert_axis(0)?
                .repeat(0, self.batch())?;
            let text_embeddings = self.text_embed.run(Xs::from(input_ids))?[0].clone();

            // run
            let texts = self.run_batch(&image_embeddings, &text_embeddings)?;

            // tasks iteration
            let ys_task = (0..self.batch())
                .into_par_iter()
                .map(|batch| {
                    // image size
                    let image_width = xs[batch].width() as usize;
                    let image_height = xs[batch].height() as usize;

                    // texts cleanup
                    let text = texts[batch]
                        .as_str()
                        .replace("<s>", "")
                        .replace("</s>", "")
                        .replace("<pad>", "");

                    // postprocess
                    let mut y = Y::default();
                    if let Task::Caption(_) | Task::Ocr = task {
                        y = y.with_texts(&[text]);
                    } else {
                        let elems = Self::loc_parse(&text)?;
                        match task {
                            Task::RegionToCategory(..) | Task::RegionToDescription(..) => {
                                let text = elems[0][0].clone();
                                y = y.with_texts(&[text]);
                            }
                            Task::ObjectDetection
                            | Task::OpenSetDetection(_)
                            | Task::DenseRegionCaption
                            | Task::CaptionToPhraseGrounding(_) => {
                                let y_bboxes: Vec<Bbox> = elems
                                    .par_iter()
                                    .enumerate()
                                    .flat_map(|(i, elem)| {
                                        Self::process_bboxes(
                                            &elem[1..],
                                            &self.quantizer,
                                            image_width,
                                            image_height,
                                            Some((&elem[0], i)),
                                        )
                                    })
                                    .collect();
                                y = y.with_bboxes(&y_bboxes);
                            }
                            Task::RegionProposal => {
                                let y_bboxes: Vec<Bbox> = Self::process_bboxes(
                                    &elems[0],
                                    &self.quantizer,
                                    image_width,
                                    image_height,
                                    None,
                                );
                                y = y.with_bboxes(&y_bboxes);
                            }
                            Task::ReferringExpressionSegmentation(_)
                            | Task::RegionToSegmentation(..) => {
                                let points = Self::process_polygons(
                                    &elems[0],
                                    &self.quantizer,
                                    image_width,
                                    image_height,
                                );
                                y = y.with_polygons(&[Polygon::default()
                                    .with_points(&points)
                                    .with_id(0)]);
                            }
                            Task::OcrWithRegion => {
                                let y_polygons: Vec<Polygon> = elems
                                    .par_iter()
                                    .enumerate()
                                    .map(|(i, elem)| {
                                        let points = Self::process_polygons(
                                            &elem[1..],
                                            &self.quantizer,
                                            image_width,
                                            image_height,
                                        );
                                        Polygon::default()
                                            .with_name(&elem[0])
                                            .with_points(&points)
                                            .with_id(i as _)
                                    })
                                    .collect();
                                y = y.with_polygons(&y_polygons);
                            }
                            _ => anyhow::bail!("Unsupported Florence2 task."),
                        };
                    }
                    Ok(y)
                })
                .collect::<Result<Vec<Y>>>()?;

            ys.insert(task.clone(), ys_task);
        }

        // update pb
        pb.set_prefix("   Completed");
        pb.set_message("Florence2 tasks");
        pb.set_style(indicatif::ProgressStyle::with_template(
            crate::PROGRESS_BAR_STYLE_FINISH_2,
        )?);
        pb.finish();

        Ok(ys)
    }

    fn run_batch(&mut self, image_embeddings: &X, text_embeddings: &X) -> Result<Vec<String>> {
        // concate image_embeddings and prompt embeddings
        let inputs_embeds = image_embeddings.clone().concatenate(text_embeddings, 1)?;
        let attention_mask = X::ones(&[self.batch(), inputs_embeds.dims()[1]]);

        // encoder
        let last_hidden_state = self.encoder.run(Xs::from(vec![
            attention_mask.clone(),
            inputs_embeds.clone(),
        ]))?[0]
            .clone();

        // decoder
        let inputs_embeds = inputs_embeds.slice(s![.., -1.., ..]);
        let inputs_embeds = X::from(inputs_embeds.to_owned().into_dyn());
        let mut decoder_outputs = self.decoder.run(Xs::from(vec![
            attention_mask.clone(),
            last_hidden_state.clone(),
            inputs_embeds,
        ]))?;

        let encoder_k0 = decoder_outputs[3].clone();
        let encoder_v0 = decoder_outputs[4].clone();
        let encoder_k1 = decoder_outputs[7].clone();
        let encoder_v1 = decoder_outputs[8].clone();
        let encoder_k2 = decoder_outputs[11].clone();
        let encoder_v2 = decoder_outputs[12].clone();
        let encoder_k3 = decoder_outputs[15].clone();
        let encoder_v3 = decoder_outputs[16].clone();
        let encoder_k4 = decoder_outputs[19].clone();
        let encoder_v4 = decoder_outputs[20].clone();
        let encoder_k5 = decoder_outputs[23].clone();
        let encoder_v5 = decoder_outputs[24].clone();

        let mut generated_tokens: Vec<Vec<u32>> = vec![vec![]; self.batch()];
        let mut finished = vec![false; self.batch()];

        // save last batch tokens
        let mut last_tokens: Vec<f32> = vec![0.; self.batch()];
        let mut logits_sampler = LogitsSampler::new();

        // generate
        for _ in 0..self.max_length {
            let logits = &decoder_outputs["logits"];
            let decoder_k0 = &decoder_outputs[1];
            let decoder_v0 = &decoder_outputs[2];
            let decoder_k1 = &decoder_outputs[5];
            let decoder_v1 = &decoder_outputs[6];
            let decoder_k2 = &decoder_outputs[9];
            let decoder_v2 = &decoder_outputs[10];
            let decoder_k3 = &decoder_outputs[13];
            let decoder_v3 = &decoder_outputs[14];
            let decoder_k4 = &decoder_outputs[17];
            let decoder_v4 = &decoder_outputs[18];
            let decoder_k5 = &decoder_outputs[21];
            let decoder_v5 = &decoder_outputs[22];

            // decode each token for each batch
            for (i, logit) in logits.axis_iter(Axis(0)).enumerate() {
                if !finished[i] {
                    let token_id = logits_sampler.decode(
                        &logit
                            .slice(s![-1, ..])
                            .into_owned()
                            .into_raw_vec_and_offset()
                            .0,
                    )?; //
                    generated_tokens[i].push(token_id);

                    // update last_tokens
                    last_tokens[i] = token_id as f32;

                    if token_id == 2 {
                        finished[i] = true;
                    }
                }
            }

            // all finished?
            if finished.iter().all(|&x| x) {
                break;
            }

            // next input text embedding
            let next_tokens = X::from(last_tokens.clone()).insert_axis(1)?;

            // decode
            let inputs_embeds = &self.text_embed.run(Xs::from(next_tokens))?[0].clone();
            let use_cache = X::ones(&[1]);
            decoder_outputs = self.decoder_merged.run(Xs::from(vec![
                attention_mask.clone(),
                last_hidden_state.clone(),
                inputs_embeds.clone(),
                decoder_k0.clone(),
                decoder_v0.clone(),
                encoder_k0.clone(),
                encoder_v0.clone(),
                decoder_k1.clone(),
                decoder_v1.clone(),
                encoder_k1.clone(),
                encoder_v1.clone(),
                decoder_k2.clone(),
                decoder_v2.clone(),
                encoder_k2.clone(),
                encoder_v2.clone(),
                decoder_k3.clone(),
                decoder_v3.clone(),
                encoder_k3.clone(),
                encoder_v3.clone(),
                decoder_k4.clone(),
                decoder_v4.clone(),
                encoder_k4.clone(),
                encoder_v4.clone(),
                decoder_k5.clone(),
                decoder_v5.clone(),
                encoder_k5.clone(),
                encoder_v5.clone(),
                use_cache,
            ]))?;
        }

        // batch decode
        let texts = match self.tokenizer.decode_batch(
            &generated_tokens
                .iter()
                .map(|tokens| tokens.as_slice())
                .collect::<Vec<_>>(),
            false,
        ) {
            Err(err) => anyhow::bail!("{:?}", err),
            Ok(xs) => xs,
        };

        Ok(texts)
    }

    pub fn encode_prompt(&self, task: &Task) -> Result<X> {
        let prompt = task.prompt_for_florence2()?;
        let encodings = match self.tokenizer.encode(prompt, true) {
            Err(err) => anyhow::bail!("{}", err),
            Ok(x) => x,
        };
        let ids: Vec<f32> = encodings.get_ids().iter().map(|x| *x as f32).collect();

        Ok(X::from(ids))
    }

    fn process_polygons(
        elems: &[String],
        quantizer: &Quantizer,
        image_width: usize,
        image_height: usize,
    ) -> Vec<Vec<f32>> {
        elems
            .par_chunks(2)
            .map(|chunk| {
                let coord: Vec<_> = chunk.iter().map(|s| s.parse::<usize>().unwrap()).collect();
                quantizer.dequantize(&coord, (image_width, image_height))
            })
            .collect()
    }

    fn process_bboxes(
        elems: &[String],
        quantizer: &Quantizer,
        image_width: usize,
        image_height: usize,
        class_name: Option<(&str, usize)>,
    ) -> Vec<Bbox> {
        elems
            .par_chunks(4)
            .enumerate()
            .map(|(i, chunk)| {
                let bbox: Vec<_> = chunk.iter().map(|s| s.parse::<usize>().unwrap()).collect();
                let dequantized_bbox = quantizer.dequantize(&bbox, (image_width, image_height));

                let mut bbox = Bbox::default().with_xyxy(
                    dequantized_bbox[0].max(0.0f32).min(image_width as f32),
                    dequantized_bbox[1].max(0.0f32).min(image_height as f32),
                    dequantized_bbox[2],
                    dequantized_bbox[3],
                );
                if let Some((class_name, i)) = class_name {
                    bbox = bbox.with_name(class_name).with_id(i as _);
                } else {
                    bbox = bbox.with_id(i as _);
                }

                bbox
            })
            .collect()
    }

    fn loc_parse(hay: &str) -> Result<Vec<Vec<String>>> {
        let pattern = r"(?i)(<loc_(?<coord>\d+)>)|(?<name>[^<]+)";
        let re = regex::Regex::new(pattern)?;
        let mut ys: Vec<Vec<String>> = Vec::new();
        let mut y = Vec::new();

        for cap in re.captures_iter(hay) {
            if let Some(loc) = cap.name("coord") {
                y.push(loc.as_str().to_string());
            } else if let Some(text) = cap.name("name") {
                if !text.as_str().is_empty() {
                    if !y.is_empty() {
                        ys.push(y);
                        y = Vec::new();
                    }
                    y.push(text.as_str().to_string());
                }
            }
        }
        if !y.is_empty() {
            ys.push(y);
        }
        Ok(ys)
    }

    pub fn batch(&self) -> usize {
        self.batch.opt()
    }
}
