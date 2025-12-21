//! Image processor configuration module.

use crate::{Device, ImagePlan, ImageTensorLayout, ResizeAlg, ResizeFilter, ResizeMode};

/// Image processor configuration.
///
/// Contains all settings for image processing including resizing, normalization,
/// and tensor layout.
#[derive(aksr::Builder, Debug, Clone)]
pub struct ImageProcessorConfig {
    /// Target image width for resizing.
    pub image_width: u32,
    /// Target image height for resizing.
    pub image_height: u32,
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
    /// Up-scaling factor for super resolution.
    pub up_scale: f32,
    /// Image tensor layout format.
    pub image_tensor_layout: ImageTensorLayout,
    /// Device for image processing.
    pub device: Device,
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            device: Device::Cuda(0),
            #[cfg(not(feature = "cuda"))]
            device: Device::Cpu(0),
            image_width: 640,
            image_height: 640,
            do_resize: true,
            resize_mode: ResizeMode::FitExact,
            resize_alg: ResizeAlg::default(),
            padding_value: 114,
            image_tensor_layout: ImageTensorLayout::NCHW,
            normalize: true,
            image_std: None,
            image_mean: None,
            unsigned: false,
            pad_image: false,
            pad_size: 8,
            up_scale: 2.0,
        }
    }
}

impl ImageProcessorConfig {
    /// Compile image processing configuration into an ImagePlan.
    pub fn compile_image_plan(&self) -> ImagePlan {
        let (mean, std) = match (self.image_mean, self.image_std) {
            (Some(mean), Some(std)) => (Some(mean), Some(std)),
            _ => (None, None),
        };

        ImagePlan {
            width: self.image_width,
            height: self.image_height,
            resize_mode: self.resize_mode.clone(),
            resize_alg: self.resize_alg,
            layout: self.image_tensor_layout,
            normalize: self.normalize,
            mean,
            std,
            padding_value: self.padding_value,
            unsigned: self.unsigned,
        }
    }

    pub fn with_resize_filter(mut self, resize_filter: ResizeFilter) -> Self {
        self.resize_alg = ResizeAlg::Convolution(resize_filter);
        self
    }

    pub fn with_resize_filter_str(mut self, resize_filter: &'static str) -> Self {
        let s = resize_filter.to_lowercase();
        if s == "nearest" {
            self.resize_alg = ResizeAlg::Nearest;
            return self;
        }
        let filter: ResizeFilter = resize_filter.parse().unwrap_or_default();
        self.resize_alg = ResizeAlg::Convolution(filter);
        self
    }
}
