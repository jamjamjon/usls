use aksr::Builder;
use anyhow::{ensure, Result};
use rayon::prelude::*;

use crate::{
    inputs, Config, DynConf, Engine, Engines, FromConfig, Image, ImageProcessor, Keypoint, Model,
    Module, ResizeModeType, XAny, XView, Xs, X, Y,
};

/// ECPose: EdgeCrafter multi-person pose estimation.
///
/// The ONNX export accepts `images` `[N, 3, 640, 640]` and `orig_target_sizes`
/// `[N, 2]` in `[width, height]` order, then returns fixed-query pose tensors:
/// `scores [N, 60]`, `labels [N, 60]`, and `keypoints [N, 60, 17, 2]`.
#[derive(Debug, Builder)]
pub struct ECPose {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub max_queries: usize,
    pub num_keypoints: usize,
    pub confs: DynConf,
    pub names: Vec<String>,
    pub processor: ImageProcessor,
    pub spec: String,
}

impl Model for ECPose {
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
            engine.try_height().unwrap_or(&640.into()).opt(),
            engine.try_width().unwrap_or(&640.into()).opt(),
        );
        let spec = engine.spec().to_owned();
        let num_keypoints = config.inference.num_keypoints.unwrap_or(17);
        let names = config.inference.keypoint_names;
        let confs = DynConf::new_or(&config.inference.class_confs, 1, 0.5);
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            max_queries: 60,
            num_keypoints,
            confs,
            names,
            processor,
            spec,
        };

        Ok((model, Engines::from(engine)))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let xs_images = crate::perf!("ECPose::preprocess", self.processor.process(images)?);
        self.validate_image_input(&xs_images, images.len())?;

        let orig_target_sizes =
            X::<i64>::from_shape_vec_generic([images.len(), 2], self.orig_target_sizes(images))?;
        self.validate_orig_target_sizes(&orig_target_sizes, images.len())?;

        let ys = crate::perf!(
            "ECPose::inference",
            engines.run(&Module::Model, inputs![&xs_images, orig_target_sizes]?)?
        );
        crate::perf!("ECPose::postprocess", self.postprocess(&ys))
    }
}

impl ECPose {
    fn orig_target_sizes(&self, images: &[Image]) -> Vec<i64> {
        images
            .iter()
            .flat_map(|image| [image.width() as i64, image.height() as i64])
            .collect()
    }

    fn validate_image_input(&self, xs_images: &XAny, batch: usize) -> Result<()> {
        ensure!(
            matches!(
                self.processor.resize_mode_type(),
                Some(ResizeModeType::FitExact)
            ),
            "ECPose expects FitExact preprocessing, got {:?}",
            self.processor.resize_mode_type()
        );

        let shape = xs_images.shape();
        ensure!(
            shape == [batch, 3, self.height, self.width],
            "ECPose expects image input shape [{batch}, 3, {}, {}], got {shape:?}",
            self.height,
            self.width
        );
        Ok(())
    }

    fn validate_orig_target_sizes(&self, orig_target_sizes: &X<i64>, batch: usize) -> Result<()> {
        let shape = orig_target_sizes.dims();
        ensure!(
            shape == [batch, 2],
            "ECPose expects orig_target_sizes shape [{batch}, 2], got {shape:?}"
        );
        Ok(())
    }

    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        ensure!(
            outputs.len() == 3,
            "ECPose expects exactly 3 outputs (scores, labels, keypoints), got {}",
            outputs.len()
        );

        let scores = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get ECPose scores output"))?;
        let labels = outputs
            .get::<i64>(1)
            .ok_or_else(|| anyhow::anyhow!("Failed to get ECPose labels output"))?;
        let keypoints = outputs
            .get::<f32>(2)
            .ok_or_else(|| anyhow::anyhow!("Failed to get ECPose keypoints output"))?;

        self.postprocess_tensors(scores, labels, keypoints)
    }

    fn postprocess_tensors(
        &self,
        scores: XView<'_, f32>,
        labels: XView<'_, i64>,
        keypoints: XView<'_, f32>,
    ) -> Result<Vec<Y>> {
        let (batch, queries, num_keypoints) =
            self.validate_output_shapes(&scores, &labels, &keypoints)?;
        ensure!(
            self.processor.transform_info().len() == batch,
            "ECPose expected {batch} image transform records, got {}",
            self.processor.transform_info().len()
        );

        let has_names = !self.names.is_empty();
        let ys = (0..batch)
            .into_par_iter()
            .map(|batch_idx| {
                let info = &self.processor.transform_info()[batch_idx];
                let width = info.width_src as f32;
                let height = info.height_src as f32;

                let keypointss = (0..queries)
                    .filter_map(|query_idx| {
                        let score = scores[[batch_idx, query_idx]];
                        let label = labels[[batch_idx, query_idx]];
                        if label < 0 {
                            return None;
                        }

                        let class_id = label as usize;
                        if score < self.class_conf(class_id) {
                            return None;
                        }

                        let pose = (0..num_keypoints)
                            .map(|keypoint_idx| {
                                let x = keypoints[[batch_idx, query_idx, keypoint_idx, 0]]
                                    .clamp(0.0, width);
                                let y = keypoints[[batch_idx, query_idx, keypoint_idx, 1]]
                                    .clamp(0.0, height);
                                let mut keypoint = Keypoint::default()
                                    .with_xy(x, y)
                                    .with_id(keypoint_idx)
                                    .with_confidence(score);
                                if has_names {
                                    keypoint = keypoint.with_name(&self.names[keypoint_idx]);
                                }
                                keypoint
                            })
                            .collect::<Vec<_>>();

                        Some(pose)
                    })
                    .collect::<Vec<_>>();

                Y::default().with_keypointss(&keypointss)
            })
            .collect();

        Ok(ys)
    }

    fn validate_output_shapes(
        &self,
        scores: &XView<'_, f32>,
        labels: &XView<'_, i64>,
        keypoints: &XView<'_, f32>,
    ) -> Result<(usize, usize, usize)> {
        let scores_shape = scores.dims();
        let labels_shape = labels.dims();
        let keypoints_shape = keypoints.dims();

        ensure!(
            scores_shape.len() == 2,
            "ECPose expects scores shape [batch, {}], got {scores_shape:?}",
            self.max_queries
        );
        ensure!(
            labels_shape.len() == 2,
            "ECPose expects labels shape [batch, {}], got {labels_shape:?}",
            self.max_queries
        );
        ensure!(
            keypoints_shape.len() == 4,
            "ECPose expects keypoints shape [batch, {}, {}, 2], got {keypoints_shape:?}",
            self.max_queries,
            self.num_keypoints
        );

        let batch = scores_shape[0];
        let queries = scores_shape[1];
        ensure!(
            labels_shape == scores_shape,
            "ECPose expects labels shape to match scores shape, got scores {scores_shape:?} and labels {labels_shape:?}"
        );
        ensure!(
            queries == self.max_queries,
            "ECPose expects {} pose queries, got {}",
            self.max_queries,
            queries
        );
        ensure!(
            keypoints_shape[0] == batch,
            "ECPose expects keypoints batch dimension {batch}, got {}",
            keypoints_shape[0]
        );
        ensure!(
            keypoints_shape[1] == queries,
            "ECPose expects keypoints query dimension {queries}, got {}",
            keypoints_shape[1]
        );
        ensure!(
            keypoints_shape[2] == self.num_keypoints,
            "ECPose expects {} keypoints per pose, got {}",
            self.num_keypoints,
            keypoints_shape[2]
        );
        ensure!(
            keypoints_shape[3] == 2,
            "ECPose expects 2 coordinates per keypoint, got {}",
            keypoints_shape[3]
        );
        if has_names_mismatch(&self.names, self.num_keypoints) {
            anyhow::bail!(
                "ECPose expected {} keypoint names or none, got {}",
                self.num_keypoints,
                self.names.len()
            );
        }

        Ok((batch, queries, self.num_keypoints))
    }

    fn class_conf(&self, class_id: usize) -> f32 {
        self.confs
            .values()
            .get(class_id)
            .copied()
            .or_else(|| self.confs.values().first().copied())
            .unwrap_or(0.5)
    }
}

fn has_names_mismatch(names: &[String], expected: usize) -> bool {
    !names.is_empty() && names.len() != expected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageProcessorConfig, ImageTransformInfo, ResizeModeType};

    fn test_model() -> ECPose {
        let mut processor = ImageProcessor::from_config(
            ImageProcessorConfig::default().with_resize_mode_type(ResizeModeType::FitExact),
        )
        .expect("processor");
        processor.images_transform_info = vec![ImageTransformInfo::default()
            .with_width_src(100)
            .with_height_src(80)];

        ECPose {
            height: 640,
            width: 640,
            batch: 1,
            max_queries: 60,
            num_keypoints: 17,
            confs: DynConf::new_or(&[0.5], 1, 0.5),
            names: crate::NAMES_COCO_KEYPOINTS_17
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            processor,
            spec: "ecpose/test".to_string(),
        }
    }

    #[test]
    fn postprocess_filters_and_clamps_keypoints() {
        let model = test_model();

        let mut scores = vec![0.1f32; 60];
        scores[0] = 0.9;
        scores[1] = 0.49;
        let scores = X::from_shape_vec(&[1, 60], scores).expect("scores");

        let labels = X::<i64>::from_shape_vec_generic([1, 60], vec![0i64; 60]).expect("labels");

        let mut keypoints = vec![0.0f32; 60 * 17 * 2];
        for idx in 0..17 {
            let base = idx * 2;
            keypoints[base] = if idx == 0 { -4.0 } else { 10.0 + idx as f32 };
            keypoints[base + 1] = if idx == 0 { 120.0 } else { 20.0 + idx as f32 };
        }
        let keypoints = X::from_shape_vec(&[1, 60, 17, 2], keypoints).expect("synthetic keypoints");

        let ys = model
            .postprocess_tensors(scores.view(), labels.view(), keypoints.view())
            .expect("postprocess");

        assert_eq!(ys.len(), 1);
        assert_eq!(ys[0].keypointss.len(), 1);
        assert_eq!(ys[0].hbbs.len(), 0);

        let pose = &ys[0].keypointss[0];
        assert_eq!(pose.len(), 17);
        assert_eq!(pose[0].id(), Some(0));
        assert_eq!(pose[0].name(), Some("nose"));
        assert_eq!(pose[0].confidence(), Some(0.9));
        assert_eq!(pose[0].xy(), (0.0, 80.0));
        assert_eq!(pose[16].confidence(), Some(0.9));
    }

    #[test]
    fn postprocess_rejects_invalid_shapes() {
        let model = test_model();
        let scores = X::from_shape_vec(&[1, 60], vec![0.5f32; 60]).expect("scores");
        let labels = X::<i64>::from_shape_vec_generic([1, 60], vec![0i64; 60]).expect("labels");
        let keypoints =
            X::from_shape_vec(&[1, 59, 17, 2], vec![0.0f32; 59 * 17 * 2]).expect("keypoints");

        let err = model
            .postprocess_tensors(scores.view(), labels.view(), keypoints.view())
            .expect_err("shape validation should fail");

        assert!(
            err.to_string()
                .contains("ECPose expects keypoints query dimension 60"),
            "unexpected error: {err}"
        );
    }
}
