//! Runtime processing context for CUDA backend.

use crate::{ImagePlan, ImageTensorLayout, ResizeAlg, ResizeFilter, ResizeMode};

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
        // Extract from first Resize transform
        self.plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    Some(mode.width())
                } else {
                    None
                }
            })
            .unwrap_or(self.in_width)
    }

    #[inline]
    pub fn out_height(&self) -> u32 {
        // Extract from first Resize transform
        self.plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    Some(mode.height())
                } else {
                    None
                }
            })
            .unwrap_or(self.in_height)
    }

    #[inline]
    pub fn resize_mode(&self) -> ResizeMode {
        // Extract from first Resize transform
        self.plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    Some(mode.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default()
    }

    #[inline]
    pub fn resize_filter(&self) -> ResizeFilter {
        self.plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    mode.alg().filter()
                } else {
                    None
                }
            })
            .unwrap_or_default()
    }

    #[inline]
    pub fn resize_alg(&self) -> ResizeAlg {
        self.plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    Some(mode.alg())
                } else {
                    None
                }
            })
            .unwrap_or_default()
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
        // Extract from first Resize transform if it's Letterbox
        self.plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    Some(mode.padding_value())
                } else {
                    None
                }
            })
            .unwrap_or(114)
    }

    #[inline]
    pub fn unsigned(&self) -> bool {
        self.plan.unsigned
    }
}
