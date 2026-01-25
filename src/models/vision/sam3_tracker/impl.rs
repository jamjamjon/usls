use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array4};

use crate::{
    elapsed_module, Config, Engine, Engines, FromConfig, Image, ImageProcessor, Mask, Model,
    Module, Ops, Sam3Prompt, X, Y,
};

/// Sam3Tracker - Segment Anything Model 3 with point and box prompts.
///
/// This model implements the unified `Model` trait with multi-engine support.
/// It uses 2 engines:
/// - `VisualEncoder`: Image encoder
/// - `Decoder`: Mask decoder
///
/// ONNX Model shapes:
/// - Vision encoder input: [B, 3, 1008, 1008]
/// - Vision encoder outputs: embeddings\[0\]=\[B,32,288,288\], \[1\]=\[B,64,144,144\], \[2\]=\[B,256,72,72\]
/// - Decoder inputs:
///   - input_points: [B, num_objects, num_points, 2]
///   - input_labels: [B, num_objects, num_points] (int64)
///   - input_boxes: [B, num_objects, 4]
///   - image_embeddings.0/1/2: from vision encoder
/// - Decoder outputs:
///   - iou_scores: [B, num_objects, 3]
///   - pred_masks: [B, num_objects, 3, H, W]
///   - object_score_logits: [B, num_objects, 1]
///
/// Note: Each box represents one object to segment. Points can be combined
/// with a box to refine the segmentation of that object.
#[derive(Builder, Debug)]
pub struct Sam3Tracker {
    pub vision_batch: usize,
    pub decoder_batch: usize,
    pub image_processor: ImageProcessor,
    pub spec: String,
}

impl Sam3Tracker {
    fn encode_images(&mut self, engines: &mut Engines, xs: &[Image]) -> Result<Vec<X>> {
        let xs_ = self.image_processor.process(xs)?;
        let output = engines.run(&Module::VisualEncoder, &xs_)?;
        let xs_out: Vec<X> = (0..output.len())
            .map(|i| X::from(output.get::<f32>(i).unwrap()))
            .collect();
        Ok(xs_out)
    }

    fn forward_tracker(
        &mut self,
        engines: &mut Engines,
        xs: &[Image],
        prompts: &[Sam3Prompt],
    ) -> Result<Vec<Y>> {
        if xs.is_empty() || prompts.is_empty() {
            return Ok(vec![]);
        }

        let image_embeddings = elapsed_module!(
            "Sam3Tracker",
            "vision-encoder",
            self.encode_images(engines, xs)?
        );

        let image_metas: Vec<_> = self
            .image_processor
            .images_transform_info()
            .iter()
            .map(|info| {
                let h_src = info.height_src as f32;
                let w_src = info.width_src as f32;
                let h_dst = info.height_dst as f32;
                let w_dst = info.width_dst as f32;
                (h_src as usize, w_src as usize, w_dst / w_src, h_dst / h_src)
            })
            .collect();

        let mut results = Vec::with_capacity(xs.len());

        for (img_idx, (orig_h, orig_w, scale_x, scale_y)) in image_metas.iter().enumerate() {
            let emb0: X = image_embeddings[0]
                .slice(s![img_idx..img_idx + 1, .., .., ..])
                .to_owned()
                .into_dyn()
                .into();
            let emb1: X = image_embeddings[1]
                .slice(s![img_idx..img_idx + 1, .., .., ..])
                .to_owned()
                .into_dyn()
                .into();
            let emb2: X = image_embeddings[2]
                .slice(s![img_idx..img_idx + 1, .., .., ..])
                .to_owned()
                .into_dyn()
                .into();

            let mut image_masks: Vec<Mask> = Vec::new();
            let mut object_idx = 0usize;

            for prompt in prompts.iter() {
                let scaled_boxes = prompt.scaled_boxes_xyxy(*scale_x, *scale_y);
                let scaled_points = prompt.scaled_points(*scale_x, *scale_y);
                let point_labels = prompt.point_labels();

                if scaled_boxes.is_empty() && scaled_points.is_empty() {
                    continue;
                }

                // Case 1: Only points, no boxes - single object with all points
                if scaled_boxes.is_empty() {
                    let masks = self.decode_single_object(
                        engines,
                        &scaled_points,
                        &point_labels,
                        None,
                        &emb0,
                        &emb1,
                        &emb2,
                    )?;

                    for (iou, obj_score, mask_data, mask_h, mask_w) in masks {
                        if let Some(mask) = self.create_mask(
                            &mask_data,
                            mask_w,
                            mask_h,
                            *orig_w,
                            *orig_h,
                            iou,
                            obj_score,
                            object_idx,
                            prompt.class_name(),
                        )? {
                            image_masks.push(mask);
                            object_idx += 1;
                        }
                    }
                }
                // Case 2: Has boxes - each box is a separate object
                else {
                    for (box_idx, box_coords) in scaled_boxes.iter().enumerate() {
                        let obj_points = if scaled_boxes.len() == 1 {
                            scaled_points.clone()
                        } else {
                            vec![]
                        };
                        let obj_labels = if scaled_boxes.len() == 1 {
                            point_labels.clone()
                        } else {
                            vec![]
                        };

                        let masks = self.decode_single_object(
                            engines,
                            &obj_points,
                            &obj_labels,
                            Some(box_coords),
                            &emb0,
                            &emb1,
                            &emb2,
                        )?;

                        for (iou, obj_score, mask_data, mask_h, mask_w) in masks {
                            let name = if scaled_boxes.len() > 1 {
                                format!("{}_{}", prompt.class_name(), box_idx)
                            } else {
                                prompt.class_name().to_string()
                            };

                            if let Some(mask) = self.create_mask(
                                &mask_data, mask_w, mask_h, *orig_w, *orig_h, iou, obj_score,
                                object_idx, &name,
                            )? {
                                image_masks.push(mask);
                                object_idx += 1;
                            }
                        }
                    }
                }
            }

            results.push(Y::default().with_masks(&image_masks));
        }

        Ok(results)
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn decode_single_object(
        &mut self,
        engines: &mut Engines,
        points: &[[f32; 2]],
        labels: &[i64],
        box_coords: Option<&[f32; 4]>,
        emb0: &X,
        emb1: &X,
        emb2: &X,
    ) -> Result<Vec<(f32, f32, Vec<f32>, usize, usize)>> {
        let (input_points, input_labels) = if !points.is_empty() {
            let num_pts = points.len();
            let pts_flat: Vec<f32> = points.iter().flat_map(|p| p.iter()).copied().collect();
            let pts_arr = Array4::from_shape_vec((1, 1, num_pts, 2), pts_flat)?;
            let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
            let labels_arr = ndarray::Array3::from_shape_vec((1, 1, num_pts), labels_f32)?;
            (X::from(pts_arr.into_dyn()), X::from(labels_arr.into_dyn()))
        } else {
            let pts_arr = Array4::<f32>::zeros((1, 1, 1, 2));
            let labels_arr = ndarray::Array3::from_elem((1, 1, 1), -1.0f32);
            (X::from(pts_arr.into_dyn()), X::from(labels_arr.into_dyn()))
        };

        let input_boxes = if let Some(coords) = box_coords {
            let boxes_arr = ndarray::Array3::from_shape_vec((1, 1, 4), coords.to_vec())?;
            X::from(boxes_arr.into_dyn())
        } else {
            let boxes_arr = ndarray::Array3::<f32>::zeros((1, 0, 4));
            X::from(boxes_arr.into_dyn())
        };

        let args = vec![
            input_points,
            input_labels,
            input_boxes,
            emb0.clone(),
            emb1.clone(),
            emb2.clone(),
        ];
        let decoder_outputs = elapsed_module!(
            "Sam3Tracker",
            "decoder",
            engines.run(&Module::Decoder, &args)?
        );

        let iou_scores = X::from(
            decoder_outputs
                .get::<f32>(0)
                .ok_or_else(|| anyhow::anyhow!("Failed to get iou_scores"))?,
        );
        let pred_masks = X::from(
            decoder_outputs
                .get::<f32>(1)
                .ok_or_else(|| anyhow::anyhow!("Failed to get pred_masks"))?,
        );
        let obj_scores = X::from(
            decoder_outputs
                .get::<f32>(2)
                .ok_or_else(|| anyhow::anyhow!("Failed to get obj_scores"))?,
        );

        let iou = iou_scores.slice(s![0, 0, ..]).to_owned();
        let masks = pred_masks.slice(s![0, 0, .., .., ..]).to_owned();
        let obj_logit = obj_scores[[0, 0, 0]];
        let obj_score = 1.0 / (1.0 + (-obj_logit).exp());

        let best_idx = iou
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_iou = iou[best_idx];
        let best_mask = masks.slice(s![best_idx, .., ..]).to_owned();
        let (mask_h, mask_w) = best_mask.dim();

        let mask_probs: Vec<f32> = best_mask
            .iter()
            .map(|&v| {
                let v_clipped = v.clamp(-50.0, 50.0);
                1.0 / (1.0 + (-v_clipped).exp())
            })
            .collect();

        Ok(vec![(best_iou, obj_score, mask_probs, mask_h, mask_w)])
    }

    #[allow(clippy::too_many_arguments)]
    fn create_mask(
        &self,
        mask_probs: &[f32],
        mask_w: usize,
        mask_h: usize,
        orig_w: usize,
        orig_h: usize,
        iou: f32,
        obj_score: f32,
        object_idx: usize,
        name: &str,
    ) -> Result<Option<Mask>> {
        let luma: Vec<u8> = Ops::interpolate_1d_u8(
            mask_probs,
            mask_w as _,
            mask_h as _,
            orig_w as _,
            orig_h as _,
            false,
        )?;

        let luma_binary: Vec<u8> = luma
            .iter()
            .map(|&v| if v > 127 { 255 } else { 0 })
            .collect();

        if let Ok(mask) = Mask::new(&luma_binary, orig_w as u32, orig_h as u32) {
            Ok(Some(
                mask.with_id(object_idx)
                    .with_name(name)
                    .with_confidence(iou * obj_score),
            ))
        } else {
            Ok(None)
        }
    }
}

/// Implement the Model trait for Sam3Tracker.
impl Model for Sam3Tracker {
    type Input<'a> = (&'a [Image], &'a [Sam3Prompt]);

    fn batch(&self) -> usize {
        self.vision_batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let visual_encoder = Engine::from_config(config.take_module(&Module::VisualEncoder)?)?;
        let decoder = Engine::from_config(config.take_module(&Module::Decoder)?)?;

        let vision_batch = visual_encoder.batch().opt();
        let decoder_batch = decoder.batch().opt();
        let height = visual_encoder.try_height().unwrap_or(&1008.into()).opt();
        let width = visual_encoder.try_width().unwrap_or(&1008.into()).opt();

        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            vision_batch,
            decoder_batch,
            image_processor,
            spec: "sam3-tracker".to_string(),
        };

        let mut engines = Engines::new();
        engines.insert(Module::VisualEncoder, visual_encoder);
        engines.insert(Module::Decoder, decoder);

        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, (images, prompts): Self::Input<'_>) -> Result<Vec<Y>> {
        self.forward_tracker(engines, images, prompts)
    }
}
