//! Runtime processing context for CUDA backend.

use crate::ImagePlan;
use crate::{ImageTensorLayout, ResizeFilter, ResizeMode};

/// Runtime processing context for a single image.
///
/// Combines the static `ImagePlan` with runtime input dimensions.
#[derive(Debug, Clone)]
pub struct CudaImageProcessContext<'a> {
    pub plan: &'a ImagePlan,
    pub in_width: u32,
    pub in_height: u32,
}

impl<'a> CudaImageProcessContext<'a> {
    #[inline]
    pub fn new(plan: &'a ImagePlan, in_width: u32, in_height: u32) -> Self {
        Self {
            plan,
            in_width,
            in_height,
        }
    }

    #[inline]
    pub fn out_width(&self) -> u32 {
        self.plan.width
    }

    #[inline]
    pub fn out_height(&self) -> u32 {
        self.plan.height
    }

    #[inline]
    pub fn resize_mode(&self) -> &ResizeMode {
        &self.plan.resize_mode
    }

    #[inline]
    pub fn resize_filter(&self) -> ResizeFilter {
        self.plan.resize_alg.filter().unwrap_or_default()
    }

    #[inline]
    pub fn layout(&self) -> ImageTensorLayout {
        self.plan.layout
    }

    #[inline]
    pub fn normalize(&self) -> bool {
        self.plan.normalize
    }

    #[inline]
    #[allow(dead_code)]
    pub fn mean(&self) -> Option<&[f32; 3]> {
        self.plan.mean.as_ref()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn std(&self) -> Option<&[f32; 3]> {
        self.plan.std.as_ref()
    }

    #[inline]
    pub fn padding_value(&self) -> u8 {
        self.plan.padding_value
    }

    #[inline]
    pub fn unsigned(&self) -> bool {
        self.plan.unsigned
    }

    #[inline]
    #[allow(dead_code)]
    pub fn has_standardize(&self) -> bool {
        self.plan.has_standardize()
    }
}
