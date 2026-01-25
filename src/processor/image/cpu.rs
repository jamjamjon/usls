use anyhow::Result;
use ndarray::Array;
use rayon::prelude::*;
use std::sync::Mutex;

use crate::{
    AnyResStrategy, CropMode, Image, ImagePlan, ImageTransform, ImageTransformInfo, PadMode,
    ResizeMode, TransformExecutor, XAny, X,
};

#[derive(Default)]
pub struct CpuTransformExecutor;

impl CpuTransformExecutor {
    pub fn new() -> Self {
        Self
    }

    fn execute_resize(
        &self,
        images: &[Image],
        mode: &ResizeMode,
        width: u32,
        height: u32,
        alg: crate::ResizeAlg,
        padding_value: u8,
    ) -> Result<(Vec<Image>, Vec<ImageTransformInfo>)> {
        images
            .par_iter()
            .map(|img| img.resize_with_info(width, height, alg, mode, padding_value))
            .collect::<Result<Vec<_>>>()
            .map(|results| {
                let (imgs, infos): (Vec<_>, Vec<_>) = results.into_iter().unzip();
                (imgs, infos)
            })
    }

    fn execute_pad(
        &self,
        images: &[Image],
        mode: &PadMode,
    ) -> Result<(Vec<Image>, Vec<ImageTransformInfo>)> {
        match mode {
            PadMode::ToMultiple {
                window_size,
                fill_mode,
            } => images
                .par_iter()
                .map(|img| img.pad_with_mode(*window_size, *fill_mode))
                .collect::<Result<Vec<_>>>()
                .map(|results| {
                    let (imgs, infos): (Vec<_>, Vec<_>) = results.into_iter().unzip();
                    (imgs, infos)
                }),
            PadMode::ToSize { .. } => {
                todo!("PadMode::ToSize not yet implemented")
            }
            PadMode::Fixed {
                top,
                bottom,
                left,
                right,
                fill_mode,
            } => images
                .par_iter()
                .map(|img| img.pad_fixed(*top, *bottom, *left, *right, *fill_mode))
                .collect::<Result<Vec<_>>>()
                .map(|results| {
                    let (imgs, infos): (Vec<_>, Vec<_>) = results.into_iter().unzip();
                    (imgs, infos)
                }),
        }
    }

    fn execute_crop(
        &self,
        images: &[Image],
        mode: &CropMode,
    ) -> Result<(Vec<Image>, Vec<ImageTransformInfo>)> {
        let CropMode::Center { width, height } = mode else {
            todo!()
        };
        images
            .par_iter()
            .map(|img| img.crop_center(*width, *height))
            .collect::<Result<Vec<_>>>()
            .map(|results| {
                let (imgs, infos): (Vec<_>, Vec<_>) = results.into_iter().unzip();
                (imgs, infos)
            })
    }

    fn execute_dynres(
        &self,
        images: &[Image],
        strategy: &AnyResStrategy,
    ) -> Result<(Vec<Image>, Vec<ImageTransformInfo>)> {
        match strategy {
            AnyResStrategy::SmolVLM {
                patch_width,
                patch_height,
                include_global,
            } => {
                let results: Vec<(Vec<Image>, ImageTransformInfo)> = images
                    .iter()
                    .map(|img| img.dynres_smolvlm(*patch_width, *patch_height, *include_global))
                    .collect::<Result<Vec<_>>>()?;

                let mut all_patches = Vec::new();
                let mut all_infos = Vec::new();

                for (patches, info) in results {
                    for _ in 0..patches.len() {
                        all_infos.push(info.clone());
                    }
                    all_patches.extend(patches);
                }

                Ok((all_patches, all_infos))
            }
            AnyResStrategy::Moondream2 {
                patch_size,
                templates,
            } => {
                let results: Vec<(Vec<Image>, ImageTransformInfo)> = images
                    .iter()
                    .map(|img| img.dynres_moondream2(*patch_size, templates))
                    .collect::<Result<Vec<_>>>()?;

                let mut all_patches = Vec::new();
                let mut all_infos = Vec::new();

                for (patches, info) in results {
                    for _ in 0..patches.len() {
                        all_infos.push(info.clone());
                    }
                    all_patches.extend(patches);
                }

                Ok((all_patches, all_infos))
            }
            AnyResStrategy::Grid { .. } => {
                todo!("AnyResStrategy::Grid not yet implemented")
            }
        }
    }
}

impl TransformExecutor for CpuTransformExecutor {
    fn execute_plan(
        &self,
        images: &[Image],
        plan: &ImagePlan,
    ) -> Result<(XAny, Vec<ImageTransformInfo>)> {
        plan.validate()?;

        if images.is_empty() {
            anyhow::bail!("No input images provided");
        }

        let mut current_images = images.to_vec();

        // Initialize transform_infos with source image dimensions
        // Extract width and height from transforms
        let (out_width, out_height) = plan
            .transforms
            .iter()
            .find_map(|t| {
                if let ImageTransform::Resize(mode) = t {
                    Some((mode.width(), mode.height()))
                } else {
                    None
                }
            })
            .unwrap_or((640, 640));

        let mut transform_infos: Vec<ImageTransformInfo> = images
            .iter()
            .map(|img| {
                let (w, h) = img.dimensions();
                ImageTransformInfo::default()
                    .with_width_src(w)
                    .with_height_src(h)
                    .with_width_dst(out_width)
                    .with_height_dst(out_height)
                    .with_width_scale(1.0)
                    .with_height_scale(1.0)
            })
            .collect();

        // Execute transforms in sequence
        for transform in &plan.transforms {
            // TODO: sequence matters
            let (new_images, new_infos) = match transform {
                ImageTransform::Pad(mode) => {
                    // TODO: pad only support simple image!
                    if current_images.len() != 1 {
                        anyhow::bail!("When pad_image is true, only one image is allowed.");
                    }
                    self.execute_pad(&current_images, mode)?
                }
                ImageTransform::Resize(mode) => self.execute_resize(
                    &current_images,
                    mode,
                    mode.width(),
                    mode.height(),
                    mode.alg(),
                    mode.padding_value(),
                )?,
                ImageTransform::Crop(mode) => self.execute_crop(&current_images, mode)?,
                ImageTransform::AnyRes(strategy) => {
                    self.execute_dynres(&current_images, strategy)?
                }
            };

            current_images = new_images;

            if new_infos.len() != transform_infos.len() {
                transform_infos = new_infos;
            } else {
                for (existing, new) in transform_infos.iter_mut().zip(new_infos.iter()) {
                    *existing = existing.merge(new);
                }
            }
        }

        let mut x = self.images_to_tensor(&current_images)?;

        // Post-processing
        if plan.normalize {
            x = x.normalize(0., 255.)?;
        }

        if let (Some(mean), Some(std)) = (plan.mean.as_ref(), plan.std.as_ref()) {
            x = x.standardize(mean, std, 3)?;
        }

        // Apply layout transformation (always NHWC -> NCHW if needed)
        if plan.layout.is_channels_first() {
            x = x.nhwc2nchw()?;
        }

        if plan.unsigned {
            x = x.unsigned();
        }

        Ok((XAny::from_host(x), transform_infos))
    }
}

impl CpuTransformExecutor {
    fn images_to_tensor(&self, images: &[Image]) -> Result<X> {
        if images.is_empty() {
            anyhow::bail!("No images provided for tensor conversion");
        }

        let width = images[0].width();
        let height = images[0].height();

        match images.len() {
            1 => {
                // Single image: always return NHWC (1, H, W, 3)
                let y = images[0].to_ndarray()?.insert_axis(0)?;
                Ok(y)
            }
            _ => {
                // Multi-image: always create NHWC tensor (N, H, W, 3)
                let ys = Mutex::new(
                    Array::zeros((images.len(), height as usize, width as usize, 3)).into_dyn(),
                );

                images.par_iter().enumerate().try_for_each(|(i, img)| {
                    let y = img.to_ndarray()?;
                    let mut ys = ys.lock().unwrap();
                    ys.slice_mut(ndarray::s![i, .., .., ..])
                        .assign(&y.slice(ndarray::s![.., .., ..]));
                    Ok::<_, anyhow::Error>(())
                })?;

                let tensor = ys
                    .into_inner()
                    .map_err(|_| anyhow::anyhow!("Mutex into_inner error"))?;
                Ok(X::from(tensor))
            }
        }
    }
}
