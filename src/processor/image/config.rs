//! Image processor configuration module.

use crate::{
    AnyResStrategy, Device, ImagePlan, ImageTensorLayout, ResizeAlg, ResizeFilter, ResizeMode,
    ResizeModeType,
};

/// Image processor configuration.
#[derive(aksr::Builder, Debug, Clone, PartialEq)]
pub struct ImageProcessorConfig {
    /// Device for image processing.
    pub device: Device,
    /// Target image width for resizing.
    pub image_width: u32,
    /// Target image height for resizing.
    pub image_height: u32,
    /// Image tensor layout format.
    pub image_tensor_layout: ImageTensorLayout,
    /// Whether to resize the image.
    pub do_resize: bool,
    /// Image resizing mode.
    pub resize_mode: ResizeMode,
    /// Image resize algorithm (includes filter type).
    pub resize_alg: ResizeAlg,
    /// Padding value for image borders.
    pub padding_value: u8,
    /// Whether to normalize image values.
    pub normalize: bool,
    /// Standard deviation values for normalization (RGB).
    pub image_std: Option<[f32; 3]>,
    /// Mean values for normalization (RGB).
    pub image_mean: Option<[f32; 3]>,
    /// Whether to use unsigned integer format.
    pub unsigned: bool,
    /// Whether to pad image for super resolution.
    pub pad_image: bool,
    /// Padding size for super resolution.
    pub pad_size: usize,
    /// Dynamic resolution strategy for VLM models.
    pub anyres_strategy: Option<AnyResStrategy>,
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu(0),
            image_width: 640,
            image_height: 640,
            image_tensor_layout: ImageTensorLayout::default(),
            do_resize: true,
            resize_mode: ResizeMode::default(),
            resize_alg: ResizeAlg::default(),
            padding_value: 114,
            normalize: true,
            image_std: None,
            image_mean: None,
            unsigned: false,
            pad_image: false,
            pad_size: 8,
            anyres_strategy: None,
        }
    }
}

impl ImageProcessorConfig {
    pub(crate) fn compile_image_plan(&self) -> ImagePlan {
        let (mean, std) = match (self.image_mean, self.image_std) {
            (Some(mean), Some(std)) => (Some(mean), Some(std)),
            _ => (None, None),
        };

        let resize_mode_type = if self.do_resize || self.anyres_strategy.is_some() {
            Some(match &self.resize_mode {
                ResizeMode::FitExact { .. } => ResizeModeType::FitExact,
                ResizeMode::FitWidth { .. } => ResizeModeType::FitWidth,
                ResizeMode::FitHeight { .. } => ResizeModeType::FitHeight,
                ResizeMode::FitAdaptive { .. } => ResizeModeType::FitAdaptive,
                ResizeMode::Letterbox { .. } => ResizeModeType::Letterbox,
            })
        } else {
            None
        };

        let mut plan = ImagePlan {
            transforms: vec![],
            layout: self.image_tensor_layout,
            normalize: self.normalize,
            mean,
            std,
            unsigned: self.unsigned,
            pad_image: self.pad_image,
            pad_size: self.pad_size,
            do_resize: self.do_resize,
            resize_mode_type,
        };

        // AnyRes transform takes precedence (for VLM models)
        if let Some(strategy) = &self.anyres_strategy {
            plan.add_transform(crate::ImageTransform::AnyRes(strategy.clone()));
            // After AnyRes, apply resize to each patch
            plan.add_transform({
                let mode = match self.resize_mode {
                    ResizeMode::FitExact { .. } => ResizeMode::FitExact {
                        width: self.image_width,
                        height: self.image_height,
                        alg: self.resize_alg,
                    },
                    ResizeMode::FitWidth { .. } => ResizeMode::FitWidth {
                        width: self.image_width,
                        height: self.image_height,
                        alg: self.resize_alg,
                        padding_value: self.padding_value,
                    },
                    ResizeMode::FitHeight { .. } => ResizeMode::FitHeight {
                        width: self.image_width,
                        height: self.image_height,
                        alg: self.resize_alg,
                        padding_value: self.padding_value,
                    },
                    ResizeMode::FitAdaptive { .. } => ResizeMode::FitAdaptive {
                        width: self.image_width,
                        height: self.image_height,
                        alg: self.resize_alg,
                        padding_value: self.padding_value,
                    },
                    ResizeMode::Letterbox { .. } => ResizeMode::Letterbox {
                        width: self.image_width,
                        height: self.image_height,
                        alg: self.resize_alg,
                        padding_value: self.padding_value,
                    },
                };
                crate::ImageTransform::Resize(mode)
            });
        } else if self.pad_image {
            // Super-resolution path: pad to multiple
            plan.add_transform(crate::ImageTransform::Pad(crate::PadMode::ToMultiple {
                window_size: self.pad_size,
                fill_mode: crate::PadFillMode::Reflect,
            }));
        } else if self.do_resize {
            plan.add_transform({
                let mode = match self.resize_mode {
                    ResizeMode::FitExact { .. } => ResizeMode::FitExact {
                        width: self.image_width,
                        height: self.image_height,
                        alg: self.resize_alg,
                    },
                    ResizeMode::FitWidth { .. } => ResizeMode::FitWidth {
                        width: self.image_width,
                        height: self.image_height,
                        alg: self.resize_alg,
                        padding_value: self.padding_value,
                    },
                    ResizeMode::FitHeight { .. } => ResizeMode::FitHeight {
                        width: self.image_width,
                        height: self.image_height,
                        alg: self.resize_alg,
                        padding_value: self.padding_value,
                    },
                    ResizeMode::FitAdaptive { .. } => ResizeMode::FitAdaptive {
                        width: self.image_width,
                        height: self.image_height,
                        alg: self.resize_alg,
                        padding_value: self.padding_value,
                    },
                    ResizeMode::Letterbox { .. } => ResizeMode::Letterbox {
                        width: self.image_width,
                        height: self.image_height,
                        alg: self.resize_alg,
                        padding_value: self.padding_value,
                    },
                };
                crate::ImageTransform::Resize(mode)
            });
        }

        plan
    }

    pub fn with_resize_filter(mut self, resize_filter: ResizeFilter) -> Self {
        match self.resize_alg {
            ResizeAlg::Convolution(f) if f != resize_filter => {
                self.resize_alg = ResizeAlg::Convolution(resize_filter);
            }
            ResizeAlg::Interpolation(f) if f != resize_filter => {
                self.resize_alg = ResizeAlg::Interpolation(resize_filter);
            }
            ResizeAlg::SuperSampling(f, m) if f != resize_filter => {
                self.resize_alg = ResizeAlg::SuperSampling(resize_filter, m);
            }
            _ => {}
        }

        self
    }

    pub fn with_resize_mode_type(mut self, mode_type: ResizeModeType) -> Self {
        self.resize_mode = match mode_type {
            ResizeModeType::FitExact => ResizeMode::FitExact {
                width: self.image_width,
                height: self.image_height,
                alg: self.resize_alg,
            },
            ResizeModeType::FitWidth => ResizeMode::FitWidth {
                width: self.image_width,
                height: self.image_height,
                alg: self.resize_alg,
                padding_value: self.padding_value,
            },
            ResizeModeType::FitHeight => ResizeMode::FitHeight {
                width: self.image_width,
                height: self.image_height,
                alg: self.resize_alg,
                padding_value: self.padding_value,
            },
            ResizeModeType::FitAdaptive => ResizeMode::FitAdaptive {
                width: self.image_width,
                height: self.image_height,
                alg: self.resize_alg,
                padding_value: self.padding_value,
            },
            ResizeModeType::Letterbox => ResizeMode::Letterbox {
                width: self.image_width,
                height: self.image_height,
                alg: self.resize_alg,
                padding_value: self.padding_value,
            },
        };
        self
    }
}
