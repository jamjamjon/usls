//! processing plan definitions.
//!
//! This module defines platform-agnostic processing types that describe
//! "what to do" without specifying "how to do it". Backends (CPU/CUDA/Metal)
//! consume these plans to execute the actual processing.

use crate::{ImageTensorLayout, ResizeAlg, ResizeFilter, ResizeMode};

/// Unified image processing plan.
///
/// This is the single source of truth for ALL image processing parameters.
/// Used by CPU, CUDA, and Metal backends. Backends consume this plan without
/// needing to interpret raw config.
#[derive(Debug, Clone)]
pub struct ImagePlan {
    /// Target output width.
    pub width: u32,
    /// Target output height.
    pub height: u32,
    /// Resize mode (FitExact, Letterbox, FitAdaptive, etc.).
    pub resize_mode: ResizeMode,
    /// Resize algorithm (includes filter type).
    pub resize_alg: ResizeAlg,
    /// Output tensor layout (NCHW, NHWC, CHW, HWC).
    pub layout: ImageTensorLayout,
    /// Whether to normalize pixel values to [0, 1].
    pub normalize: bool,
    /// Mean values for standardization (RGB).
    pub mean: Option<[f32; 3]>,
    /// Std values for standardization (RGB).
    pub std: Option<[f32; 3]>,
    /// Padding value for letterbox/adaptive modes.
    pub padding_value: u8,
    /// Whether to apply unsigned (ReLU-like clamp to 0).
    pub unsigned: bool,
}

// /// Type alias for backward compatibility.
// pub type PreprocessPlan = ImagePlan;

impl Default for ImagePlan {
    fn default() -> Self {
        Self {
            width: 640,
            height: 640,
            resize_mode: ResizeMode::FitExact,
            resize_alg: ResizeAlg::default(),
            layout: ImageTensorLayout::NCHW,
            normalize: true,
            mean: None,
            std: None,
            padding_value: 114,
            unsigned: false,
        }
    }
}

impl ImagePlan {
    /// Create a new plan with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Set the resize mode.
    pub fn with_resize_mode(mut self, mode: ResizeMode) -> Self {
        self.resize_mode = mode;
        self
    }

    /// Set the resize algorithm.
    pub fn with_resize_alg(mut self, alg: ResizeAlg) -> Self {
        self.resize_alg = alg;
        self
    }

    /// Set the resize filter (convenience for Convolution algorithm).
    pub fn with_resize_filter(mut self, filter: ResizeFilter) -> Self {
        self.resize_alg = ResizeAlg::Convolution(filter);
        self
    }

    /// Set the output tensor layout.
    pub fn with_layout(mut self, layout: ImageTensorLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Set whether to normalize.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set mean/std for standardization from arrays.
    pub fn with_mean_std(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.mean = Some(mean);
        self.std = Some(std);
        self
    }

    /// Set padding value.
    pub fn with_padding_value(mut self, value: u8) -> Self {
        self.padding_value = value;
        self
    }

    /// Set unsigned flag.
    pub fn with_unsigned(mut self, unsigned: bool) -> Self {
        self.unsigned = unsigned;
        self
    }

    /// Check if standardization is enabled.
    pub fn has_standardize(&self) -> bool {
        self.mean.is_some() && self.std.is_some()
    }

    /// Get the resize filter (if applicable).
    pub fn filter(&self) -> Option<ResizeFilter> {
        self.resize_alg.filter()
    }
}
