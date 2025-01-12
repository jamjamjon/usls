use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{
    elapsed,
    models::{BaseModelTextual, BaseModelVisual, Quantizer},
    Bbox, LogitsSampler, Options, Polygon, Scale, Task, Ts, Xs, Ys, X, Y,
};

#[derive(Debug, Builder)]
pub struct Florence2 {
    pub vision_encoder: BaseModelVisual,
    pub text_embed: BaseModelTextual,
    pub encoder: BaseModelTextual,
    pub decoder: BaseModelTextual,
    pub decoder_merged: BaseModelTextual,
    ts: Ts,
    quantizer: Quantizer,
    max_length: usize,
    eos_token_id: u32,
    decoder_start_token_id: u32,
    n_kvs: usize,
}

impl Florence2 {
    pub fn new(
        options_vision_encoder: Options,
        options_text_embed: Options,
        options_encoder: Options,
        options_decoder: Options,
        options_decoder_merged: Options,
    ) -> Result<Self> {
        let vision_encoder = BaseModelVisual::new(options_vision_encoder)?;
        let text_embed = BaseModelTextual::new(options_text_embed)?;
        let encoder = BaseModelTextual::new(options_encoder)?;
        let decoder = BaseModelTextual::new(options_decoder)?;
        let decoder_merged = BaseModelTextual::new(options_decoder_merged)?;
        let quantizer = Quantizer::default();
        let ts = Ts::merge(&[
            vision_encoder.engine().ts(),
            text_embed.engine().ts(),
            encoder.engine().ts(),
            decoder.engine().ts(),
            decoder_merged.engine().ts(),
        ]);
        let max_length = 1024;
        let eos_token_id = 2;
        let decoder_start_token_id = 2;
        let n_kvs = match decoder.scale() {
            Some(Scale::B) => 6,
            Some(Scale::L) => 12,
            _ => unimplemented!(),
        };

        Ok(Self {
            vision_encoder,
            text_embed,
            encoder,
            decoder,
            decoder_merged,
            max_length,
            quantizer,
            ts,
            eos_token_id,
            decoder_start_token_id,
            n_kvs,
        })
    }

    fn process_task(task: &Task, image_height: usize, image_width: usize) -> Task {
        // region-related tasks
        match task {
            Task::RegionToSegmentation(x0, y0, x1, y1) => {
                let xyxy = Quantizer::default()
                    .quantize(&[*x0, *y0, *x1, *y1], (image_width, image_height));
                Task::RegionToSegmentation(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            }
            Task::RegionToCategory(x0, y0, x1, y1) => {
                let xyxy = Quantizer::default()
                    .quantize(&[*x0, *y0, *x1, *y1], (image_width, image_height));
                Task::RegionToCategory(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            }
            Task::RegionToDescription(x0, y0, x1, y1) => {
                let xyxy = Quantizer::default()
                    .quantize(&[*x0, *y0, *x1, *y1], (image_width, image_height));
                Task::RegionToDescription(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            }
            _ => *task,
        }
    }

    fn encode_text(&mut self, task: &Task, images: &[DynamicImage]) -> Result<X> {
        let xs = images
            .par_iter()
            .map(|im| {
                let text = Self::process_task(task, im.height() as _, im.width() as _)
                    .prompt_for_florence2()?;
                let ids = self.text_embed.processor().encode_text_ids(&text, true)?;
                X::from(ids).insert_axis(0)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let x = X::concat(&xs, 0)?;
        let xs = self.text_embed.inference(x.into())?;
        let x = xs[0].to_owned();

        Ok(x)
    }

    pub fn forward(&mut self, xs_visual: &[DynamicImage], x_textual: &Task) -> Result<Ys> {
        let visual_embeddings = elapsed!("visual-encode", self.ts, {
            self.vision_encoder.encode(xs_visual)?
        });

        let textual_embedding = elapsed!("textual-encode", self.ts, {
            self.encode_text(x_textual, xs_visual)?
        });

        let generated = elapsed!("generate-then-decode", self.ts, {
            self.generate_then_decode(&visual_embeddings, &textual_embedding)?
        });

        let ys = elapsed!("postprocess", self.ts, {
            self.postprocess(&generated, xs_visual, x_textual)?
        });

        Ok(ys)
    }

    // decode or postprocess, batch images and one text
    fn generate_then_decode(
        &mut self,
        visual_embeddings: &X,
        textual_embedding: &X,
    ) -> Result<Vec<String>> {
        // concate image embeddings and prompt embeddings
        let inputs_embeds = visual_embeddings
            .clone()
            .concatenate(textual_embedding, 1)?;
        let attention_mask = X::ones(&[self.batch(), inputs_embeds.dims()[1]]);

        // encoder
        let last_hidden_state = self.encoder.inference(Xs::from(vec![
            attention_mask.clone(),
            inputs_embeds.clone(),
        ]))?[0]
            .clone();

        // decoder
        let inputs_embeds = inputs_embeds.slice(s![.., -1.., ..]);
        let inputs_embeds = X::from(inputs_embeds.to_owned().into_dyn());
        let mut decoder_outputs = self.decoder.inference(Xs::from(vec![
            attention_mask.clone(),
            last_hidden_state.clone(),
            inputs_embeds,
        ]))?;

        // encoder kvs
        let encoder_kvs: Vec<_> = (3..4 * self.n_kvs)
            .step_by(4)
            .flat_map(|i| [i, i + 1])
            .map(|i| decoder_outputs[i].clone())
            .collect();

        // token ids
        let mut token_ids: Vec<Vec<u32>> = vec![vec![]; self.batch()];
        let mut finished = vec![false; self.batch()];
        let mut last_tokens: Vec<f32> = vec![0.; self.batch()];
        let logits_sampler = LogitsSampler::new();

        // generate
        for _ in 0..self.max_length {
            let logits = &decoder_outputs[0];
            let decoder_kvs: Vec<_> = (1..(4 * self.n_kvs) - 2)
                .step_by(4)
                .flat_map(|i| [i, i + 1])
                .map(|i| decoder_outputs[i].clone())
                .collect();

            // decode each token for each batch
            // let (finished, last_tokens) = self.decoder_merged.processor().par_generate(
            //     logits,
            //     &mut token_ids,
            //     self.eos_token_id,
            // )?;

            // if finished {
            //     break;
            // }

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
                    } else {
                        token_ids[i].push(token_id);
                    }
                    // update
                    last_tokens[i] = token_id as f32;
                }
            }

            // all finished?
            if finished.iter().all(|&x| x) {
                break;
            }

            // decode
            let next_tokens = X::from(last_tokens.clone()).insert_axis(1)?;
            let inputs_embeds = &self.text_embed.inference(Xs::from(next_tokens))?[0].clone();
            let use_cache = X::ones(&[1]);
            let mut xs = vec![
                attention_mask.clone(),
                last_hidden_state.clone(),
                inputs_embeds.clone(),
            ];
            for i in 0..self.n_kvs {
                xs.push(decoder_kvs[i * 2].clone());
                xs.push(decoder_kvs[i * 2 + 1].clone());
                xs.push(encoder_kvs[i * 2].clone());
                xs.push(encoder_kvs[i * 2 + 1].clone());
            }
            xs.push(use_cache);
            decoder_outputs = self.decoder_merged.inference(xs.into())?;
        }

        // batch decode
        let texts = self
            .text_embed
            .processor()
            .decode_tokens_batch(&token_ids, false)?;

        Ok(texts)
    }

    fn postprocess(
        &mut self,
        generated_text: &[String],
        xs_visual: &[DynamicImage],
        x_textual: &Task,
    ) -> Result<Ys> {
        let mut ys = Vec::new();
        let ys_task = (0..self.batch())
            .into_par_iter()
            .map(|batch| {
                // image size
                let image_width = xs_visual[batch].width() as usize;
                let image_height = xs_visual[batch].height() as usize;

                // texts cleanup
                let text = generated_text[batch]
                    .as_str()
                    .replace("<s>", "")
                    .replace("</s>", "")
                    .replace("<pad>", "");

                // postprocess
                let mut y = Y::default();
                if let Task::Caption(_) | Task::Ocr = x_textual {
                    y = y.with_texts(&[text.into()]);
                } else {
                    let elems = Self::loc_parse(&text)?;
                    match x_textual {
                        Task::RegionToCategory(..) | Task::RegionToDescription(..) => {
                            let text = elems[0][0].clone();
                            y = y.with_texts(&[text.into()]);
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

        ys.extend_from_slice(&ys_task);

        Ok(ys.into())
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
        self.vision_encoder.batch() as _
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }
}
