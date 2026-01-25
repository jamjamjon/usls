//! processing plan definitions.
//!
//! This module defines platform-agnostic processing types that describe
//! "what to do" without specifying "how to do it". Backends (CPU/CUDA/Metal)
//! consume these plans to execute the actual processing.

use crate::{ImageTensorLayout, ImageTransform, ResizeModeType};

/// Unified image processing plan.
///
/// This is the single source of truth for ALL image processing parameters.
/// Used by CPU, CUDA, and Metal backends. Backends consume this plan without
/// needing to interpret raw config.
#[derive(aksr::Builder, Debug, Clone, Default)]
pub struct ImagePlan {
    /// Transform operations to apply in sequence.
    pub transforms: Vec<ImageTransform>,
    /// Output tensor layout (NCHW, NHWC, CHW, HWC).
    pub layout: ImageTensorLayout,
    /// Whether to normalize pixel values to [0, 1].
    pub normalize: bool,
    /// Mean values for standardization (RGB).
    pub mean: Option<[f32; 3]>,
    /// Std values for standardization (RGB).
    pub std: Option<[f32; 3]>,
    /// Whether to apply unsigned (ReLU-like clamp to 0).
    pub unsigned: bool,
    /// Super Resolution: whether to pad image.
    pub pad_image: bool,
    /// Super Resolution: padding size.
    pub pad_size: usize,
    /// Whether to perform resize (can skip for SR workflows).
    pub do_resize: bool,
    /// Resize mode type for easy access in post-processing.
    pub resize_mode_type: Option<ResizeModeType>,
}

impl ImagePlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mean_std(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.mean = Some(mean);
        self.std = Some(std);
        self
    }

    pub(crate) fn add_transform(&mut self, transform: ImageTransform) {
        self.transforms.push(transform);
    }

    pub(crate) fn insert_transform_first(&mut self, transform: ImageTransform) {
        self.transforms.insert(0, transform);
    }

    /// Validate the plan (check for conflicting transforms).
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        // AnyRes + Resize is valid: AnyRes splits image, then Resize each patch
        // Multiple AnyRes is not allowed
        let dynres_count = self
            .transforms
            .iter()
            .filter(|t| matches!(t, ImageTransform::AnyRes(_)))
            .count();
        if dynres_count > 1 {
            anyhow::bail!("Cannot have multiple AnyRes transforms in the same plan");
        }

        Ok(())
    }
}
