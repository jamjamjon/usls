use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::{
    elapsed_module, Config, Engine, Engines, FromConfig, Image, ImageProcessor, Model, Module, Xs,
    X, Y,
};

/// MODNet: Trimap-Free Portrait Matting in Real Time
#[derive(Builder, Debug)]
pub struct MobileGaze {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub spec: String,
    pub processor: ImageProcessor,
    bins: usize,
    binwidth: usize,
    angle_offset: usize,
}

impl Model for MobileGaze {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&448.into()).opt(),
            engine.try_width().unwrap_or(&448.into()).opt(),
        );
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let bins = 90;
        let binwidth = 4;
        let angle_offset = 180;

        let model = Self {
            height,
            width,
            batch,
            spec,
            processor,
            bins,
            binwidth,
            angle_offset,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let xs = elapsed_module!("MobileGaze", "preprocess", self.processor.process(images)?);
        let ys = elapsed_module!("MobileGaze", "inference", engines.run(&Module::Model, &xs)?);
        elapsed_module!("MobileGaze", "postprocess", self.postprocess(&ys))
    }
}

impl MobileGaze {
    fn postprocess(&self, xs: &Xs) -> Result<Vec<Y>> {
        let pitches = xs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get pitch output tensor"))?;
        let yaws = xs
            .get::<f32>(1)
            .ok_or_else(|| anyhow::anyhow!("Failed to get yaw output tensor"))?;

        let batch_size = pitches.shape()[0];

        (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let pitch_dist = pitches.index_axis(Axis(0), i);
                let yaw_dist = yaws.index_axis(Axis(0), i);

                // Convert to 1D arrays with proper error handling
                let pitch_dist_1d = pitch_dist
                    .into_dimensionality()
                    .map_err(|e| anyhow::anyhow!("Failed to convert pitch tensor to 1D: {e}"))?;
                let yaw_dist_1d = yaw_dist
                    .into_dimensionality()
                    .map_err(|e| anyhow::anyhow!("Failed to convert yaw tensor to 1D: {e}"))?;

                // Convert probability distributions to angles
                let pitch_angle = self.compute_angle_from_bins(&pitch_dist_1d)?;
                let yaw_angle = self.compute_angle_from_bins(&yaw_dist_1d)?;

                // Convert degrees to radians
                let pitch_rad = pitch_angle.to_radians();
                let yaw_rad = yaw_angle.to_radians();

                // Create gaze result with angles stored in extra
                let mut extra = HashMap::new();
                extra.insert("pitch".to_string(), X::from(vec![pitch_rad]));
                extra.insert("yaw".to_string(), X::from(vec![yaw_rad]));

                Ok(Y::default().with_extra(extra))
            })
            .collect::<Result<Vec<Y>>>()
    }
}

impl MobileGaze {
    fn softmax(&self, logits: &ndarray::ArrayView1<f32>) -> ndarray::Array1<f32> {
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();

        if sum_exp == 0.0 {
            let uniform_prob = 1.0 / logits.len() as f32;
            ndarray::Array1::from_vec(vec![uniform_prob; logits.len()])
        } else {
            ndarray::Array1::from_vec(exp_vals.iter().map(|&x| x / sum_exp).collect())
        }
    }

    fn compute_angle_from_bins(&self, logits: &ndarray::ArrayView1<f32>) -> Result<f32> {
        let probs = self.softmax(logits);

        let mut weighted_sum = 0.0f32;
        let mut prob_sum = 0.0f32;

        for (idx, &prob) in probs.iter().enumerate() {
            let bin_value = idx as f32;
            weighted_sum += bin_value * prob;
            prob_sum += prob;
        }

        if prob_sum == 0.0 {
            anyhow::bail!("Probability sum is zero");
        }

        let angle_degrees =
            (weighted_sum / prob_sum) * self.binwidth as f32 - self.angle_offset as f32;

        Ok(angle_degrees)
    }
}
