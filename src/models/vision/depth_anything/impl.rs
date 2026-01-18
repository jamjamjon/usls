use aksr::Builder;
use anyhow::Result;
use rayon::prelude::*;

use crate::{
    elapsed_module, inputs, Config, Engine, Engines, FromConfig, Image, ImageProcessor, Mask,
    Model, Module, Ops, Task, Version, XView, Xs, Y,
};

/// Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data
#[derive(Debug, Builder)]
pub struct DepthAnything {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub spec: String,
    pub task: Task,
    pub processor: ImageProcessor,
    pub version: Version,
}

impl Model for DepthAnything {
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
        let task = config.task.unwrap_or(Task::MonocularDepthEstimation);
        let version = config.version.unwrap_or(1.into());
        let (batch, height, width) = match task {
            Task::MultiocularDepthEstimation => (
                1, // Note: batch is not supported for multiocular depth estimation
                engine.inputs.minoptmax[0][3].opt(),
                engine.inputs.minoptmax[0][4].opt(),
            ),
            _ => (
                engine.batch().opt(),
                engine.try_height().unwrap_or(&518.into()).opt(),
                engine.try_width().unwrap_or(&518.into()).opt(),
            ),
        };
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let model = Self {
            height,
            width,
            batch,
            spec,
            task,
            processor,
            version,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x = elapsed_module!(
            "DepthAnything",
            "preprocess",
            self.processor.process(images)?
        );

        // Note: For multi-view depth estimation (v3), all input images are treated as views of the same scene.
        let x = if let Task::MultiocularDepthEstimation = self.task {
            x.insert_axis(0)? // Insert batch dimension to get [1, N, C, H, W] where N is the number of views.
        } else {
            x
        };

        let ys = elapsed_module!(
            "DepthAnything",
            "inference",
            engines.run(&Module::Model, inputs![&x]?)?
        );
        elapsed_module!("DepthAnything", "postprocess", self.postprocess(&ys))
    }
}

impl DepthAnything {
    fn postprocess_one(
        &self,
        idx: usize,
        luma: &XView<f32>,
        sky: Option<&XView<f32>>,
        use_percentile: bool,
        invert_depth: bool,
    ) -> Result<Mask> {
        let info = &self.processor.images_transform_info[idx];
        let (h1, w1) = (info.height_src, info.width_src);

        // Get data as slice for processing
        let luma_slice = luma.as_slice().unwrap_or(&[]);

        // Apply 1.0/x transformation for depth maps if needed
        let depth_data: Vec<f32> = if invert_depth {
            luma_slice
                .iter()
                .map(|&x| if x != 0.0 { 1.0 / x } else { 0.0 })
                .collect()
        } else {
            luma_slice.to_vec()
        };

        // Find min/max values with proper handling of NaN/Inf
        let (min_, max_) = if use_percentile {
            self.find_percentile_range(&depth_data, 2.0, 98.0)?
        } else {
            self.find_min_max(&depth_data)?
        };

        let range = max_ - min_;
        if range <= f32::EPSILON {
            return Err(anyhow::anyhow!(
                "Invalid depth range: min={min_}, max={max_}"
            ));
        }

        // Process depth data with or without sky mask
        let normalized = if let Some(sky_view) = sky {
            let sky_slice = sky_view.as_slice().unwrap_or(&[]);
            depth_data
                .par_iter()
                .zip(sky_slice.par_iter())
                .map(|(&depth, &sky_val)| {
                    if sky_val >= 0.5 {
                        0u8
                    } else if depth.is_finite() {
                        (((depth - min_) / range) * 255.0).clamp(0.0, 255.0) as u8
                    } else {
                        0u8
                    }
                })
                .collect::<Vec<_>>()
        } else {
            depth_data
                .par_iter()
                .map(|&x| {
                    if x.is_finite() {
                        (((x - min_) / range) * 255.0).clamp(0.0, 255.0) as u8
                    } else {
                        0u8
                    }
                })
                .collect::<Vec<_>>()
        };

        // Resize and create mask
        let resized = Ops::resize_luma8_u8(
            &normalized,
            self.width as _,
            self.height as _,
            w1 as _,
            h1 as _,
            false,
            "Bilinear",
        )?;

        Mask::new(&resized, w1, h1)
    }

    /// Find min and max values in a slice, handling NaN and Inf values
    fn find_min_max(&self, values: &[f32]) -> Result<(f32, f32)> {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut has_finite = false;

        for &x in values {
            if x.is_finite() {
                has_finite = true;
                min_val = min_val.min(x);
                max_val = max_val.max(x);
            }
        }

        if !has_finite {
            return Err(anyhow::anyhow!("No finite values found in depth data"));
        }

        Ok((min_val, max_val))
    }

    /// Find percentile range in a slice, handling NaN and Inf values
    fn find_percentile_range(
        &self,
        values: &[f32],
        lower_pct: f32,
        upper_pct: f32,
    ) -> Result<(f32, f32)> {
        // Filter out non-finite values
        let mut finite_values: Vec<f32> = values
            .iter()
            .filter_map(|&x| if x.is_finite() { Some(x) } else { None })
            .collect();

        if finite_values.is_empty() {
            return Err(anyhow::anyhow!("No finite values found in depth data"));
        }

        // Sort values for percentile calculation
        finite_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx = ((finite_values.len() as f32 * lower_pct / 100.0) as usize)
            .min(finite_values.len() - 1);
        let upper_idx = ((finite_values.len() as f32 * upper_pct / 100.0) as usize)
            .min(finite_values.len() - 1);

        Ok((finite_values[lower_idx], finite_values[upper_idx]))
    }

    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = match self.version {
            Version(1, _, _) | Version(2, _, _) => {
                let output = outputs
                    .get::<f32>(0)
                    .ok_or_else(|| anyhow::anyhow!("Failed to get depth output"))?;

                output
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(idx, luma)| {
                        self.postprocess_one(idx, &XView::from(luma.view()), None, false, false)
                            .ok()
                            .map(|mask| Y::default().with_masks(&[mask]))
                    })
                    .collect()
            }
            Version(3, _, _) => {
                match self.task {
                    Task::MonocularDepthEstimation | Task::MetricDepthEstimation => {
                        let depth = outputs
                            .get::<f32>(0)
                            .ok_or_else(|| anyhow::anyhow!("Failed to get depth map"))?;

                        let sky = outputs
                            .get::<f32>(1)
                            .ok_or_else(|| anyhow::anyhow!("Failed to get sky-seg map"))?;

                        // Process each batch item in parallel
                        let depth_views: Vec<_> = depth.axis_iter(ndarray::Axis(0)).collect();
                        let sky_views: Vec<_> = sky.axis_iter(ndarray::Axis(0)).collect();

                        depth_views
                            .into_par_iter()
                            .zip(sky_views.into_par_iter())
                            .enumerate()
                            .filter_map(|(idx, (depth_map, sky_mask))| {
                                self.postprocess_one(
                                    idx,
                                    &XView::from(depth_map.view()),
                                    Some(&XView::from(sky_mask.view())),
                                    true,
                                    true,
                                )
                                .ok()
                                .map(|mask| Y::default().with_masks(&[mask]))
                            })
                            .collect()
                    }
                    Task::MultiocularDepthEstimation => {
                        let depth = outputs
                            .get::<f32>(0)
                            .ok_or_else(|| anyhow::anyhow!("Failed to get depth output"))?;

                        let conf = outputs
                            .get::<f32>(1)
                            .ok_or_else(|| anyhow::anyhow!("Failed to get confidence output"))?;

                        let intrinsics = outputs
                            .get::<f32>(2)
                            .ok_or_else(|| anyhow::anyhow!("Failed to get intrinsics output"))?;

                        let extrinsics = outputs
                            .get::<f32>(3)
                            .ok_or_else(|| anyhow::anyhow!("Failed to get extrinsics output"))?;

                        depth
                            .slice(ndarray::s![0, .., .., ..])
                            .into_dyn() // Note: batch = 1
                            .axis_iter(ndarray::Axis(0))
                            .into_par_iter()
                            .enumerate()
                            .filter_map(|(idx, luma)| {
                                self.postprocess_one(
                                    idx,
                                    &XView::from(luma.view()),
                                    None,
                                    true,
                                    true,
                                )
                                .ok()
                                .map(|mask| {
                                    let mut extras = std::collections::HashMap::new();

                                    // Extract view-specific conf, intrinsics, extrinsics
                                    // conf: [1, N, H, W] -> [H, W]
                                    let c_view = conf
                                        .slice(ndarray::s![0, idx, .., ..])
                                        .to_owned()
                                        .into_dyn();
                                    extras.insert("conf".to_string(), crate::X::from(c_view));

                                    // intrinsics: [1, N, 3, 3] -> [3, 3]
                                    let i_view = intrinsics
                                        .slice(ndarray::s![0, idx, .., ..])
                                        .to_owned()
                                        .into_dyn();
                                    extras.insert("intrinsics".to_string(), crate::X::from(i_view));

                                    // extrinsics: [1, N, 3, 4] -> [3, 4]
                                    let e_view = extrinsics
                                        .slice(ndarray::s![0, idx, .., ..])
                                        .to_owned()
                                        .into_dyn();
                                    extras.insert("extrinsics".to_string(), crate::X::from(e_view));

                                    Y::default().with_masks(&[mask]).with_extras(extras)
                                })
                            })
                            .collect()
                    }
                    _ => todo!(),
                }
            }
            _ => unimplemented!(),
        };

        Ok(ys)
    }
}
